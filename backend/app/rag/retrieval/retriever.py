"""Retrieval orchestration.

    query -> embed -> [vector arm | keyword arm] -> RRF -> rerank -> top-k

Every arm is scoped by :class:`~app.rag.retrieval.types.AccessScope` before it
ranks anything, so authorisation is enforced once per arm at the SQL level and
never depends on a later filtering step.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import get_logger
from app.rag.embeddings.base import EmbeddingProvider
from app.rag.embeddings.factory import get_embedding_provider
from app.rag.retrieval.fusion import reciprocal_rank_fusion
from app.rag.retrieval.keyword import get_keyword_searcher
from app.rag.retrieval.reranker import get_reranker
from app.rag.retrieval.types import AccessScope, ScoredChunk
from app.rag.retrieval.vector_store import get_vector_store

logger = get_logger("app.rag.retriever")


@dataclass
class RetrievalResult:
    chunks: list[ScoredChunk]
    mode: str
    vector_candidates: int = 0
    keyword_candidates: int = 0
    fused_candidates: int = 0
    timings_ms: dict[str, float] = field(default_factory=dict)
    backends: dict[str, str] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.chunks

    @property
    def total_latency_ms(self) -> float:
        return round(sum(self.timings_ms.values()), 2)


class Retriever:
    def __init__(self, embedder: EmbeddingProvider | None = None) -> None:
        self._embedder = embedder

    @property
    def embedder(self) -> EmbeddingProvider:
        if self._embedder is None:
            self._embedder = get_embedding_provider()
        return self._embedder

    def retrieve(
        self,
        db: Session,
        query: str,
        scope: AccessScope,
        *,
        top_k: int | None = None,
        candidate_k: int | None = None,
        mode: str | None = None,
    ) -> RetrievalResult:
        top_k = top_k or settings.RETRIEVAL_TOP_K
        candidate_k = candidate_k or settings.RETRIEVAL_CANDIDATE_K
        mode = (mode or settings.RETRIEVAL_MODE).lower()
        # Always over-fetch relative to top_k: the reranker can only improve
        # ordering if it has more to choose from than it will return.
        candidate_k = max(candidate_k, top_k * 2)

        timings: dict[str, float] = {}
        backends: dict[str, str] = {}
        vector_hits: list[ScoredChunk] = []
        keyword_hits: list[ScoredChunk] = []

        if mode in {"vector", "hybrid"}:
            started = time.perf_counter()
            query_vector = self.embedder.embed_query(query)
            timings["embed_ms"] = round((time.perf_counter() - started) * 1000, 2)

            store = get_vector_store(db)
            backends["vector"] = store.backend
            started = time.perf_counter()
            vector_hits = store.search(
                db,
                query_vector,
                scope,
                limit=candidate_k,
                min_similarity=settings.MIN_SIMILARITY,
            )
            timings["vector_ms"] = round((time.perf_counter() - started) * 1000, 2)

        if mode in {"keyword", "hybrid"}:
            searcher = get_keyword_searcher(db)
            backends["keyword"] = searcher.backend
            started = time.perf_counter()
            keyword_hits = searcher.search(db, query, scope, limit=candidate_k)
            timings["keyword_ms"] = round((time.perf_counter() - started) * 1000, 2)

        started = time.perf_counter()
        if mode == "hybrid":
            candidates = reciprocal_rank_fusion(
                [vector_hits, keyword_hits], k=settings.HYBRID_RRF_K
            )
        elif mode == "keyword":
            candidates = keyword_hits
        else:
            candidates = vector_hits
        timings["fusion_ms"] = round((time.perf_counter() - started) * 1000, 2)

        started = time.perf_counter()
        reranker = get_reranker()
        backends["reranker"] = reranker.name
        chunks = reranker.rerank(query, candidates, top_k)
        timings["rerank_ms"] = round((time.perf_counter() - started) * 1000, 2)

        logger.info(
            "retrieval_completed",
            extra={
                "mode": mode,
                "vector_candidates": len(vector_hits),
                "keyword_candidates": len(keyword_hits),
                "fused_candidates": len(candidates),
                "returned": len(chunks),
                **timings,
            },
        )

        return RetrievalResult(
            chunks=chunks,
            mode=mode,
            vector_candidates=len(vector_hits),
            keyword_candidates=len(keyword_hits),
            fused_candidates=len(candidates),
            timings_ms=timings,
            backends=backends,
        )
