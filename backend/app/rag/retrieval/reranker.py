"""Reranking.

Retrieval optimises for *recall* over a large corpus; reranking optimises for
*precision* over the ~20 candidates that survived.  This matters more in a
security-first RAG system than in a normal one: every chunk placed in the
context is another chunk that could carry an embedded instruction, so a
tighter top-k is a smaller attack surface, not just a cheaper prompt.

Two implementations:

``HeuristicReranker`` (default)
    Combines retrieval score with query-term coverage, exact-phrase presence,
    a mild length prior and an injection-risk penalty.  It is deterministic,
    costs microseconds, needs no model download, and is fully explainable in
    the UI -- which is the point: a security reviewer can read why a chunk was
    ranked where it was.

``CrossEncoderReranker`` (optional)
    A real cross-encoder (``sentence-transformers``) that scores each
    (query, chunk) pair jointly.  Materially better ordering, but it pulls in
    torch (~2 GB) and adds tens to hundreds of milliseconds per query, so it is
    opt-in via ``RERANKER=cross_encoder``.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from app.core.config import settings
from app.core.logging import get_logger
from app.rag.retrieval.keyword import tokenize
from app.rag.retrieval.types import ScoredChunk

logger = get_logger("app.rag.reranker")


class Reranker(ABC):
    name: str = "abstract"

    @abstractmethod
    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]: ...


class NoOpReranker(Reranker):
    name = "none"

    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        return candidates[:top_k]


class HeuristicReranker(Reranker):
    """Feature-based reranking with transparent, inspectable weights."""

    name = "heuristic"

    # Weights sum to 1.0 for the positive features; the injection penalty is
    # subtractive so it can only ever push a risky chunk down.
    W_RETRIEVAL = 0.55
    W_COVERAGE = 0.25
    W_PHRASE = 0.15
    W_LENGTH = 0.05
    INJECTION_PENALTY = 0.35

    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        if not candidates:
            return []

        query_terms = set(tokenize(query))
        phrases = _significant_phrases(query)
        best_retrieval = max((c.score for c in candidates), default=1.0) or 1.0

        for candidate in candidates:
            content_lower = candidate.content.lower()
            content_terms = set(tokenize(candidate.content))

            # 1. Normalised retrieval score from the fusion stage.
            retrieval = candidate.score / best_retrieval

            # 2. What fraction of the query's content words appear at all.
            coverage = (
                len(query_terms & content_terms) / len(query_terms)
                if query_terms
                else 0.0
            )

            # 3. Exact multi-word phrase hits: a strong precision signal that
            #    neither embeddings nor bag-of-words scoring captures.
            phrase = (
                sum(1 for p in phrases if p in content_lower) / len(phrases)
                if phrases
                else 0.0
            )

            # 4. Mild preference for substantial chunks. Very short chunks
            #    often match on a single token without carrying the answer.
            length = min(len(candidate.content) / 600.0, 1.0)

            score = (
                self.W_RETRIEVAL * retrieval
                + self.W_COVERAGE * coverage
                + self.W_PHRASE * phrase
                + self.W_LENGTH * length
            )
            # 5. Demote anything the ingest scan found suspicious. Chunks above
            #    the quarantine threshold never reach here, but a chunk in the
            #    grey zone should lose to an equally relevant clean one.
            score -= self.INJECTION_PENALTY * candidate.injection_risk

            candidate.rerank_score = round(max(score, 0.0), 6)
            candidate.meta["rerank_features"] = {
                "retrieval": round(retrieval, 4),
                "coverage": round(coverage, 4),
                "phrase": round(phrase, 4),
                "length": round(length, 4),
                "injection_penalty": round(
                    self.INJECTION_PENALTY * candidate.injection_risk, 4
                ),
            }

        ordered = sorted(
            candidates,
            key=lambda c: (-(c.rerank_score or 0.0), c.source_filename, c.chunk_index),
        )
        return ordered[:top_k]


class CrossEncoderReranker(Reranker):
    """Optional cross-encoder reranking (``RERANKER=cross_encoder``)."""

    name = "cross_encoder"

    def __init__(self, model_name: str | None = None) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "RERANKER=cross_encoder requires sentence-transformers "
                "(see backend/requirements-optional.txt)"
            ) from exc
        self._model = CrossEncoder(model_name or settings.CROSS_ENCODER_MODEL)

    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:  # pragma: no cover - requires the optional model
        if not candidates:
            return []
        pairs = [(query, candidate.content) for candidate in candidates]
        scores = self._model.predict(pairs)
        for candidate, score in zip(candidates, scores, strict=True):
            adjusted = float(score) - (
                HeuristicReranker.INJECTION_PENALTY * candidate.injection_risk
            )
            candidate.rerank_score = adjusted
        return sorted(
            candidates,
            key=lambda c: (-(c.rerank_score or 0.0), c.source_filename, c.chunk_index),
        )[:top_k]


_PHRASE_SPLIT = re.compile(r"[^a-z0-9]+")


def _significant_phrases(query: str, min_words: int = 2) -> list[str]:
    """Contiguous content-word phrases from the query, longest first."""
    words = tokenize(query)
    phrases: list[str] = []
    for size in (3, 2):
        if size < min_words:
            continue
        for start in range(len(words) - size + 1):
            phrases.append(" ".join(words[start : start + size]))
    return phrases[:12]


_RERANKER_CACHE: dict[str, Reranker] = {}


def get_reranker(name: str | None = None) -> Reranker:
    """Return the configured reranker, caching model-backed instances."""
    key = (name or settings.RERANKER).lower()
    if key in _RERANKER_CACHE:
        return _RERANKER_CACHE[key]

    reranker: Reranker
    if key == "none":
        reranker = NoOpReranker()
    elif key == "cross_encoder":
        try:
            reranker = CrossEncoderReranker()
        except RuntimeError as exc:  # pragma: no cover - optional dependency
            # Falling back is right here: a missing optional model should
            # degrade ranking quality, not take the service down.
            logger.error("cross_encoder_unavailable", extra={"error": str(exc)})
            reranker = HeuristicReranker()
    else:
        reranker = HeuristicReranker()

    _RERANKER_CACHE[key] = reranker
    return reranker
