"""Vector similarity search.

Two backends behind one interface:

``PgVectorStore``
    Executes the search inside PostgreSQL using pgvector's ``<=>`` cosine
    distance operator, backed by the HNSW index created in migration 0001.
    The authorisation predicate is part of the same statement, so the database
    never materialises a row the caller is not allowed to see.

``FallbackVectorStore``
    Loads the caller's chunks and scores them in Python.  Exact, but O(n) per
    query.  It exists so the project runs and its tests pass with no external
    services; it is not intended for a real corpus.

Both apply :class:`~app.rag.retrieval.types.AccessScope` *before* ranking.
Pre-filtering matters: post-filtering a top-k list means an unauthorised
document can consume a slot and silently reduce the results the user should
have received -- a correctness bug and an information leak at the same time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.models.document import Document, DocumentChunk, DocumentStatus
from app.rag.embeddings.base import cosine_similarity
from app.rag.retrieval.types import AccessScope, ScoredChunk

logger = get_logger("app.rag.vector_store")


def apply_access_scope(stmt: Select, scope: AccessScope) -> Select:
    """Attach the authorisation predicate to a chunk query.

    Single choke point: every retrieval path (vector, keyword, direct fetch)
    routes through this function, so there is exactly one place where the
    ownership rule can be got wrong, and exactly one place to audit.
    """
    stmt = stmt.where(DocumentChunk.owner_id == scope.user_id)

    if not scope.include_quarantined:
        stmt = stmt.where(DocumentChunk.is_quarantined.is_(False))

    if scope.document_ids:
        stmt = stmt.where(DocumentChunk.document_id.in_(scope.document_ids))

    # Only chunks of fully-ingested documents are searchable; a half-processed
    # document would otherwise answer questions from a partial corpus.
    stmt = stmt.join(Document, Document.id == DocumentChunk.document_id).where(
        Document.status == DocumentStatus.READY.value
    )
    return stmt


class VectorStore(ABC):
    backend: str = "abstract"

    @abstractmethod
    def search(
        self,
        db: Session,
        query_vector: Sequence[float],
        scope: AccessScope,
        limit: int,
        min_similarity: float = 0.0,
    ) -> list[ScoredChunk]: ...


class PgVectorStore(VectorStore):
    backend = "pgvector"

    def search(
        self,
        db: Session,
        query_vector: Sequence[float],
        scope: AccessScope,
        limit: int,
        min_similarity: float = 0.0,
    ) -> list[ScoredChunk]:
        # Vectors are L2-normalised by every provider, so cosine distance and
        # similarity are exact complements.
        distance = DocumentChunk.embedding.cosine_distance(list(query_vector))
        stmt = select(DocumentChunk, distance.label("distance"))
        stmt = apply_access_scope(stmt, scope)
        stmt = stmt.where(DocumentChunk.embedding.isnot(None))
        stmt = stmt.order_by(distance).limit(limit)

        results: list[ScoredChunk] = []
        for rank, (chunk, dist) in enumerate(db.execute(stmt).all()):
            similarity = 1.0 - float(dist)
            if similarity < min_similarity:
                continue
            results.append(
                ScoredChunk.from_model(
                    chunk,
                    score=similarity,
                    vector_score=similarity,
                    rank_sources={"vector": rank + 1},
                )
            )
        return results


class FallbackVectorStore(VectorStore):
    """Brute-force cosine similarity computed in Python.

    Used when the configured database has no pgvector (i.e. SQLite).  The scan
    is bounded by ``max_scan`` so a pathological corpus cannot turn one request
    into an unbounded amount of work.
    """

    backend = "python_fallback"

    def __init__(self, max_scan: int = 20_000) -> None:
        self._max_scan = max_scan

    def search(
        self,
        db: Session,
        query_vector: Sequence[float],
        scope: AccessScope,
        limit: int,
        min_similarity: float = 0.0,
    ) -> list[ScoredChunk]:
        stmt = select(DocumentChunk)
        stmt = apply_access_scope(stmt, scope)
        stmt = stmt.where(DocumentChunk.embedding.isnot(None))
        stmt = stmt.limit(self._max_scan)

        chunks = list(db.execute(stmt).scalars().all())
        if len(chunks) >= self._max_scan:
            logger.warning(
                "fallback_vector_scan_truncated",
                extra={"scanned": len(chunks), "limit": self._max_scan},
            )

        scored: list[tuple[float, DocumentChunk]] = []
        for chunk in chunks:
            if not chunk.embedding:
                continue
            similarity = cosine_similarity(query_vector, chunk.embedding)
            if similarity >= min_similarity:
                scored.append((similarity, chunk))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            ScoredChunk.from_model(
                chunk,
                score=similarity,
                vector_score=similarity,
                rank_sources={"vector": rank + 1},
            )
            for rank, (similarity, chunk) in enumerate(scored[:limit])
        ]


def get_vector_store(db: Session) -> VectorStore:
    """Choose the backend from the live connection's dialect."""
    dialect = db.bind.dialect.name if db.bind is not None else ""
    if dialect == "postgresql":
        return PgVectorStore()
    return FallbackVectorStore()
