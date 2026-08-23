"""Evaluation API schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

VALID_KINDS = (
    "answerable",
    "unanswerable",
    "ambiguous",
    "direct_injection",
    "indirect_injection",
    "pii",
    "authorization",
    "benign_control",
)


class EvaluationRunRequest(BaseModel):
    kinds: list[str] | None = Field(
        default=None,
        max_length=len(VALID_KINDS),
        description=(
            "Restrict the run to these case kinds. Omit to run everything. "
            f"Valid values: {', '.join(VALID_KINDS)}"
        ),
    )
    include_cases: bool = Field(
        default=False,
        description=(
            "Include the per-case detail. Off by default because the full "
            "corpus makes for a large response; the aggregates are usually "
            "what you want."
        ),
    )


class EvaluationRunResponse(BaseModel):
    """The same report the CLI writes, returned inline.

    The nested sections are passed through as-is rather than being re-typed
    here: `app.evaluation.metrics` is the single definition of what each
    number means, and duplicating that shape in a schema would create two
    places for it to drift.
    """

    started_at: str
    finished_at: str
    duration_seconds: float
    configuration: dict[str, Any]
    dataset: dict[str, int]
    ingestion: dict[str, Any]
    security: dict[str, Any]
    retrieval: dict[str, Any]
    quality: dict[str, Any]
    latency: dict[str, Any]
    totals: dict[str, Any]
    cases: list[dict[str, Any]] = Field(default_factory=list)
