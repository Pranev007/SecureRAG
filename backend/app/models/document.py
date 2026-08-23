"""Document and chunk models."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import settings
from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.db.types import GUID, JSONBType, VectorType

if TYPE_CHECKING:
    from app.models.user import User


class DocumentStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class DocumentVisibility(StrEnum):
    """Access-control metadata carried by every document and chunk.

    ``PRIVATE`` documents are visible only to their owner.  ``ORGANISATION``
    is the extension point for team sharing; today it behaves as owner-only
    plus admin read, and is enforced in
    :mod:`app.security.access_control` rather than in the prompt.
    """

    PRIVATE = "private"
    ORGANISATION = "organisation"


class Document(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "documents"
    __table_args__ = (
        # Per-owner deduplication: the same bytes uploaded twice by one user is
        # a no-op, but two different users may legitimately hold the same file
        # and must get independent, separately-authorised copies.
        UniqueConstraint("owner_id", "content_sha256", name="uq_document_owner_sha"),
        Index("ix_documents_owner_status", "owner_id", "status"),
    )

    owner_id: Mapped[str] = mapped_column(
        GUID(), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    extension: Mapped[str] = mapped_column(String(16), nullable=False)
    content_type: Mapped[str] = mapped_column(String(128), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    storage_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    status: Mapped[str] = mapped_column(
        String(32), default=DocumentStatus.PENDING.value, nullable=False, index=True
    )
    error_message: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    visibility: Mapped[str] = mapped_column(
        String(32), default=DocumentVisibility.PRIVATE.value, nullable=False
    )

    page_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Highest indirect-injection score seen across this document's chunks at
    # ingest time, and how many chunks were quarantined.  Surfaced in the UI so
    # a user can see that an uploaded file contains embedded instructions.
    max_injection_risk: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    quarantined_chunk_count: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )

    meta: Mapped[dict[str, Any]] = mapped_column(JSONBType, default=dict, nullable=False)

    owner: Mapped[User] = relationship(back_populates="documents")
    chunks: Mapped[list[DocumentChunk]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Document id={self.id} status={self.status} chunks={self.chunk_count}>"


class DocumentChunk(Base, UUIDPrimaryKeyMixin):
    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunk_document_index"),
        # The composite index mirrors the retrieval predicate exactly:
        # every search is scoped by owner first, then filtered on quarantine
        # state.  Authorisation is part of the query plan, not an afterthought.
        Index("ix_chunks_owner_quarantine", "owner_id", "is_quarantined"),
    )

    document_id: Mapped[str] = mapped_column(
        GUID(),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # Denormalised from documents.owner_id on purpose.  Retrieval filters on
    # ownership in the same WHERE clause as the vector search, so the ANN index
    # and the authorisation predicate are evaluated together; a join would
    # either defeat the index or tempt a post-filter (which leaks results).
    owner_id: Mapped[str] = mapped_column(GUID(), nullable=False, index=True)
    visibility: Mapped[str] = mapped_column(
        String(32), default=DocumentVisibility.PRIVATE.value, nullable=False
    )

    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)

    token_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section: Mapped[str | None] = mapped_column(String(512), nullable=True)
    source_filename: Mapped[str] = mapped_column(String(512), nullable=False)

    embedding: Mapped[list[float] | None] = mapped_column(
        VectorType(settings.EMBEDDING_DIMENSIONS), nullable=True
    )
    embedding_model: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # Indirect prompt-injection score computed at ingest time.  Quarantined
    # chunks are excluded from retrieval entirely.
    injection_risk: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    is_quarantined: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    injection_labels: Mapped[list[str]] = mapped_column(
        JSONBType, default=list, nullable=False
    )

    meta: Mapped[dict[str, Any]] = mapped_column(JSONBType, default=dict, nullable=False)

    document: Mapped[Document] = relationship(back_populates="chunks")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<DocumentChunk doc={self.document_id} idx={self.chunk_index}>"
