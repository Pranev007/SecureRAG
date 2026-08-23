"""Shared retrieval data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.models.document import DocumentChunk


@dataclass
class AccessScope:
    """The authorisation envelope for a retrieval.

    Every search function takes one of these as a *required* argument.  Making
    the scope impossible to omit is the point: an unscoped query cannot be
    written by accident, because there is no overload that accepts none.

    SECURITY PRINCIPLE 4: authorisation is applied here, in the SQL predicate --
    not by asking the model to be careful with the documents it was given.
    """

    user_id: str
    is_admin: bool = False
    # Optional narrowing chosen by the *user* (e.g. "search only these files").
    # This never widens access; it intersects with ownership.
    document_ids: list[str] | None = None
    include_quarantined: bool = False

    def __post_init__(self) -> None:
        if not self.user_id:
            raise ValueError("AccessScope requires a user_id")


@dataclass
class ScoredChunk:
    """A retrieved chunk plus the scores that produced its rank."""

    chunk_id: str
    document_id: str
    content: str
    source_filename: str
    chunk_index: int
    page_number: int | None = None
    section: str | None = None
    owner_id: str = ""
    injection_risk: float = 0.0

    score: float = 0.0
    vector_score: float | None = None
    keyword_score: float | None = None
    rerank_score: float | None = None
    # Which arm produced this candidate and at what rank; drives RRF and makes
    # the retrieval explainable in the UI.
    rank_sources: dict[str, int] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_model(cls, chunk: DocumentChunk, **scores: Any) -> ScoredChunk:
        return cls(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            content=chunk.content,
            source_filename=chunk.source_filename,
            chunk_index=chunk.chunk_index,
            page_number=chunk.page_number,
            section=chunk.section,
            owner_id=chunk.owner_id,
            injection_risk=chunk.injection_risk,
            **scores,
        )

    def citation_label(self) -> str:
        if self.page_number:
            return f"{self.source_filename} - page {self.page_number}"
        if self.section:
            return f"{self.source_filename} - {self.section}"
        return self.source_filename
