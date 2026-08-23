"""Chat API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.core.config import settings
from app.schemas.common import ORMModel


class ChatRequest(BaseModel):
    """A question about the caller's documents.

    Bounds are enforced here so a malformed request is rejected by FastAPI
    before any application code runs.  The substantive security checks live in
    :mod:`app.security.input_guard`, which every entry point shares -- these
    limits are the cheap outer envelope, not the guardrail.
    """

    question: str = Field(
        ...,
        min_length=1,
        max_length=settings.MAX_QUERY_LENGTH * 2,
        description="The question to answer from your documents",
    )
    session_id: str | None = Field(
        default=None, description="Continue an existing chat session"
    )
    document_ids: list[str] | None = Field(
        default=None,
        max_length=50,
        description="Restrict retrieval to these documents (never widens access)",
    )
    top_k: int | None = Field(
        default=None, ge=1, le=20, description="Number of chunks to retrieve"
    )

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value


class CitationResponse(BaseModel):
    index: int
    document_id: str
    chunk_id: str
    filename: str
    page: int | None = None
    section: str | None = None
    quote: str = ""
    verified: bool = True
    label: str = ""


class SecurityStatus(BaseModel):
    """What the guardrails did to this request, shown in the UI."""

    blocked: bool = False
    refused: bool = False
    reason: str | None = None
    risk_score: float = 0.0
    grounding_score: float = 0.0
    confidence: float = 0.0
    pii_detected: bool = False
    pii_types: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    message_id: str
    sources: list[CitationResponse] = Field(default_factory=list)
    security: SecurityStatus
    retrieved_chunk_count: int = 0
    latency_ms: float = 0.0
    timings_ms: dict[str, float] = Field(default_factory=dict)
    request_id: str | None = None


class MessageResponse(ORMModel):
    id: str
    role: str
    content: str
    was_blocked: bool
    block_reason: str | None
    risk_score: float
    grounding_score: float | None
    pii_detected: bool
    citations: list[dict[str, Any]]
    retrieved_chunk_count: int
    latency_ms: float
    created_at: datetime
    meta: dict[str, Any]


class ChatSessionResponse(ORMModel):
    id: str
    title: str
    message_count: int
    created_at: datetime
    updated_at: datetime


class ChatSessionDetailResponse(ChatSessionResponse):
    messages: list[MessageResponse] = Field(default_factory=list)
