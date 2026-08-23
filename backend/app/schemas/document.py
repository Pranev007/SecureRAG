"""Document schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.schemas.common import ORMModel


class DocumentResponse(ORMModel):
    id: str
    filename: str
    extension: str
    content_type: str
    file_size_bytes: int
    status: str
    error_message: str | None
    visibility: str
    page_count: int
    chunk_count: int
    char_count: int
    # Surfaced so a user can see that a file they uploaded carries embedded
    # instructions -- the alternative is silently quarantining their content.
    max_injection_risk: float
    quarantined_chunk_count: int
    created_at: datetime


class DocumentChunkResponse(ORMModel):
    id: str
    chunk_index: int
    content: str
    page_number: int | None
    section: str | None
    token_count: int
    char_count: int
    injection_risk: float
    is_quarantined: bool
    injection_labels: list[str]


class DocumentDetailResponse(DocumentResponse):
    chunks: list[DocumentChunkResponse] = Field(default_factory=list)


class DocumentUploadResponse(BaseModel):
    document: DocumentResponse
    message: str
    warnings: list[str] = Field(default_factory=list)
