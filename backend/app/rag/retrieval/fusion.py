"""Reciprocal Rank Fusion.

The vector arm returns cosine similarities; the keyword arm returns BM25 or
``ts_rank_cd`` values.  These live on different, unbounded, non-comparable
scales, so combining them by weighted sum requires calibration that drifts the
moment you change the embedding model.

RRF sidesteps the problem entirely by discarding the scores and fusing on
*rank*:

    score(d) = sum over arms of  1 / (k + rank_arm(d))

Properties that make it the right default here:

* scale-free -- no tuning when the embedding model or corpus changes;
* robust to one arm being badly calibrated;
* rewards documents that both arms surface, which is exactly the signal we want;
* one parameter, ``k``, that damps the influence of the very top ranks.
  ``k = 60`` is the value from Cormack et al. (2009) and is a sane default.

Reference: Cormack, Clarke & Buettcher, "Reciprocal Rank Fusion outperforms
Condorcet and individual rank learning methods", SIGIR 2009.
"""

from __future__ import annotations

from collections.abc import Iterable

from app.rag.retrieval.types import ScoredChunk


def reciprocal_rank_fusion(
    result_sets: Iterable[list[ScoredChunk]],
    *,
    k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[ScoredChunk]:
    """Fuse several ranked lists into one.

    ``weights`` optionally scales an arm's contribution by its ``rank_sources``
    key (e.g. ``{"vector": 1.0, "keyword": 0.7}``).  Left unset, arms count
    equally.
    """
    fused: dict[str, ScoredChunk] = {}
    scores: dict[str, float] = {}

    for results in result_sets:
        for position, candidate in enumerate(results, start=1):
            arm = next(iter(candidate.rank_sources), "unknown")
            weight = (weights or {}).get(arm, 1.0)
            contribution = weight / (k + position)

            existing = fused.get(candidate.chunk_id)
            if existing is None:
                # Copy so fusing does not mutate the arm's own result objects.
                merged = ScoredChunk(
                    chunk_id=candidate.chunk_id,
                    document_id=candidate.document_id,
                    content=candidate.content,
                    source_filename=candidate.source_filename,
                    chunk_index=candidate.chunk_index,
                    page_number=candidate.page_number,
                    section=candidate.section,
                    owner_id=candidate.owner_id,
                    injection_risk=candidate.injection_risk,
                    vector_score=candidate.vector_score,
                    keyword_score=candidate.keyword_score,
                    rank_sources=dict(candidate.rank_sources),
                    meta=dict(candidate.meta),
                )
                fused[candidate.chunk_id] = merged
                scores[candidate.chunk_id] = contribution
            else:
                existing.rank_sources.update(candidate.rank_sources)
                # Keep whichever per-arm score each arm reported.
                if candidate.vector_score is not None:
                    existing.vector_score = candidate.vector_score
                if candidate.keyword_score is not None:
                    existing.keyword_score = candidate.keyword_score
                scores[candidate.chunk_id] += contribution

    for chunk_id, score in scores.items():
        fused[chunk_id].score = score

    return sorted(
        fused.values(),
        # Tie-break deterministically so results are reproducible: identical
        # RRF scores are common with small corpora.
        key=lambda c: (-c.score, c.source_filename, c.chunk_index),
    )
