"""Keyword (lexical) retrieval -- the second arm of hybrid search.

Why keep a lexical arm at all when we have embeddings?  Because dense
retrieval is bad at exactly the things enterprise documents are full of:
identifiers, product codes, acronyms, policy numbers, dates, and rare proper
nouns.  A query for "form 16A" or "clause 7.3.2" is a near-exact string match
problem, and an embedding will happily return something *about* tax forms
instead.  Lexical search nails those; dense search nails paraphrase.  Fusing
both is strictly better than either alone.

Two backends, same interface:

``PostgresKeywordSearcher``
    Uses PostgreSQL full-text search.  ``ts_rank_cd`` is a real ranking
    function (cover density, term frequency, document length), and the query
    expression matches the functional GIN index built in migration 0001.

``Bm25KeywordSearcher``
    Okapi BM25 implemented over the caller's chunks, for the SQLite fallback.
    Written out rather than pulled from a library: it is ~40 lines, it removes
    a dependency, and being able to explain the saturation and length-
    normalisation terms is worth more than importing them.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.models.document import DocumentChunk
from app.rag.retrieval.types import AccessScope, ScoredChunk
from app.rag.retrieval.vector_store import apply_access_scope

_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")

_STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those of in on at to for
    from by with as is are was were be been being it its do does did not no
    what which who whom how when where why can could should would may might
    will shall must have has had about into over under some such only same so
    too very just i you he she they we my your our their there here me him her
    """.split()
)

# BM25 constants. k1 controls how quickly term frequency saturates; b controls
# how strongly length normalisation is applied. 1.5/0.75 are the standard
# defaults from the TREC literature and behave well on prose.
BM25_K1 = 1.5
BM25_B = 0.75


def tokenize(text_value: str) -> list[str]:
    return [
        token.lower()
        for token in _TOKEN.findall(text_value)
        if token.lower() not in _STOPWORDS and len(token) > 1
    ]


class KeywordSearcher(ABC):
    backend: str = "abstract"

    @abstractmethod
    def search(
        self, db: Session, query: str, scope: AccessScope, limit: int
    ) -> list[ScoredChunk]: ...


class PostgresKeywordSearcher(KeywordSearcher):
    backend = "postgres_fts"

    @staticmethod
    def _lexemes(query: str) -> list[str]:
        """Reduce a query to bare alphanumeric runs safe for ``to_tsquery``.

        Hyphenated terms are split into their parts on purpose:
        ``to_tsvector`` indexes "multi-factor" as the lexemes ``multi-factor``,
        ``multi`` and ``factor``, so searching for the parts matches, whereas a
        de-punctuated "multifactor" matches nothing.

        Because the result contains only ``[a-z0-9]`` runs, it cannot express a
        tsquery operator, so a malformed or hostile query degrades to a plain
        term list instead of a syntax error.
        """
        lexemes: list[str] = []
        for term in tokenize(query):
            lexemes.extend(re.findall(r"[a-z0-9]+", term.lower()))
        # Preserve order, drop duplicates and single characters.
        seen: set[str] = set()
        return [
            lexeme
            for lexeme in lexemes
            if len(lexeme) > 1 and not (lexeme in seen or seen.add(lexeme))
        ]

    def search(
        self, db: Session, query: str, scope: AccessScope, limit: int
    ) -> list[ScoredChunk]:
        lexemes = self._lexemes(query)
        if not lexemes:
            return []

        # OR, not AND. `plainto_tsquery` conjoins every term, which turns this
        # arm into an all-terms-present *filter* rather than a ranking function:
        # "How many days of annual leave..." becomes
        # 'mani' & 'day' & 'annual' & 'leav' & ... and matches nothing, because
        # no document contains "many". Natural-language questions almost always
        # carry a word the document lacks, so on PostgreSQL the keyword arm
        # silently returned zero results and hybrid retrieval quietly degraded
        # to vector-only.
        #
        # Disjunction restores the semantics the BM25 fallback already had:
        # partial matches score rather than being excluded, and `ts_rank_cd`
        # ranks documents matching more terms higher. Keeping the two backends
        # semantically identical is the point -- otherwise a test that passes on
        # SQLite says nothing about production.
        #
        # The lexemes are bound as a single parameter (no SQL interpolation) and
        # are reduced to bare alphanumeric runs, so they cannot express tsquery
        # operators either.
        tsquery = func.to_tsquery(text("'english'"), " | ".join(lexemes))
        tsvector = func.to_tsvector(text("'english'"), DocumentChunk.content)
        rank = func.ts_rank_cd(tsvector, tsquery)

        stmt = select(DocumentChunk, rank.label("rank"))
        stmt = apply_access_scope(stmt, scope)
        stmt = stmt.where(tsvector.op("@@")(tsquery))
        stmt = stmt.order_by(rank.desc()).limit(limit)

        rows = db.execute(stmt).all()
        if not rows:
            return []

        # ts_rank_cd is unbounded; normalise against the best hit so the score
        # is comparable with the vector arm's cosine similarity in the UI.
        best = max(float(rank_value) for _, rank_value in rows) or 1.0
        return [
            ScoredChunk.from_model(
                chunk,
                score=float(rank_value) / best,
                keyword_score=float(rank_value) / best,
                rank_sources={"keyword": position + 1},
            )
            for position, (chunk, rank_value) in enumerate(rows)
        ]


class Bm25KeywordSearcher(KeywordSearcher):
    """Okapi BM25 over the caller's authorised chunks."""

    backend = "bm25_python"

    def __init__(self, max_scan: int = 20_000) -> None:
        self._max_scan = max_scan

    def search(
        self, db: Session, query: str, scope: AccessScope, limit: int
    ) -> list[ScoredChunk]:
        terms = tokenize(query)
        if not terms:
            return []

        stmt = apply_access_scope(select(DocumentChunk), scope).limit(self._max_scan)
        chunks = list(db.execute(stmt).scalars().all())
        if not chunks:
            return []

        tokenised = [tokenize(chunk.content) for chunk in chunks]
        lengths = [len(tokens) for tokens in tokenised]
        average_length = (sum(lengths) / len(lengths)) or 1.0

        # Document frequency across the caller's own corpus. IDF computed
        # per-user is the correct scope here: relevance is relative to the
        # documents this user can actually see.
        document_frequency: Counter[str] = Counter()
        for tokens in tokenised:
            document_frequency.update(set(tokens))

        total_documents = len(chunks)
        idf = {
            term: math.log(
                1
                + (total_documents - document_frequency.get(term, 0) + 0.5)
                / (document_frequency.get(term, 0) + 0.5)
            )
            for term in set(terms)
        }

        scored: list[tuple[float, DocumentChunk]] = []
        for index, chunk in enumerate(chunks):
            counts = Counter(tokenised[index])
            length = lengths[index] or 1
            score = 0.0
            for term in terms:
                frequency = counts.get(term, 0)
                if frequency == 0:
                    continue
                numerator = frequency * (BM25_K1 + 1)
                denominator = frequency + BM25_K1 * (
                    1 - BM25_B + BM25_B * length / average_length
                )
                score += idf[term] * numerator / denominator
            if score > 0:
                scored.append((score, chunk))

        if not scored:
            return []

        scored.sort(key=lambda pair: pair[0], reverse=True)
        best = scored[0][0] or 1.0
        return [
            ScoredChunk.from_model(
                chunk,
                score=score / best,
                keyword_score=score / best,
                rank_sources={"keyword": position + 1},
            )
            for position, (score, chunk) in enumerate(scored[:limit])
        ]


def get_keyword_searcher(db: Session) -> KeywordSearcher:
    dialect = db.bind.dialect.name if db.bind is not None else ""
    if dialect == "postgresql":
        return PostgresKeywordSearcher()
    return Bm25KeywordSearcher()
