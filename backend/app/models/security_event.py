"""Security event model -- the audit trail for every guardrail decision."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from sqlalchemy import Float, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.db.types import GUID, JSONBType


class SecurityEventType(StrEnum):
    # --- input layer ---
    INPUT_VALIDATION_FAILED = "input_validation_failed"
    PROMPT_INJECTION_DETECTED = "prompt_injection_detected"
    PROMPT_INJECTION_SUSPECTED = "prompt_injection_suspected"
    INPUT_PII_DETECTED = "input_pii_detected"
    DUPLICATE_QUERY_FLOOD = "duplicate_query_flood"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"

    # --- auth layer ---
    LOGIN_SUCCEEDED = "login_succeeded"
    LOGIN_FAILED = "login_failed"
    REGISTRATION = "registration"
    AUTHORIZATION_DENIED = "authorization_denied"

    # --- retrieval / context layer ---
    DOCUMENT_UPLOADED = "document_uploaded"
    DOCUMENT_DELETED = "document_deleted"
    INGESTION_FAILED = "ingestion_failed"
    INDIRECT_INJECTION_DETECTED = "indirect_injection_detected"
    CHUNK_QUARANTINED = "chunk_quarantined"
    RETRIEVAL_PERFORMED = "retrieval_performed"

    # --- output layer ---
    GROUNDING_FAILED = "grounding_failed"
    CITATION_INVALID = "citation_invalid"
    OUTPUT_PII_DETECTED = "output_pii_detected"
    OUTPUT_SCHEMA_INVALID = "output_schema_invalid"
    UNSAFE_OUTPUT_DETECTED = "unsafe_output_detected"

    # --- system ---
    GUARDRAIL_ERROR = "guardrail_error"
    PROVIDER_ERROR = "provider_error"


class SecuritySeverity(StrEnum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityAction(StrEnum):
    ALLOW = "allow"
    FLAG = "flag"
    SANITISE = "sanitise"
    QUARANTINE = "quarantine"
    REDACT = "redact"
    BLOCK = "block"


class SecurityLayer(StrEnum):
    AUTH = "auth"
    INPUT = "input"
    RETRIEVAL = "retrieval"
    CONTEXT = "context"
    OUTPUT = "output"
    SYSTEM = "system"


class SecurityEvent(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """One guardrail decision.

    SECURITY: this table stores *decisions and metadata*, never content.  User
    queries and document text are represented by a salted-free SHA-256 prefix
    (see :func:`app.core.logging.redact_for_log`) which is enough to correlate
    repeat attacks without persisting the payload.
    """

    __tablename__ = "security_events"
    __table_args__ = (
        Index("ix_events_type_created", "event_type", "created_at"),
        Index("ix_events_user_created", "user_id", "created_at"),
    )

    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    user_id: Mapped[str | None] = mapped_column(GUID(), nullable=True, index=True)

    event_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    layer: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    risk_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    # Which detector fired, e.g. "pattern:instruction_override".
    detector: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # Non-reversible reference to the offending input.
    content_ref: Mapped[str | None] = mapped_column(String(64), nullable=True)

    resource_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    resource_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Truncated hash of the client IP: enough to spot a single abusive source,
    # not enough to be a durable identifier.
    client_ref: Mapped[str | None] = mapped_column(String(32), nullable=True)

    detail: Mapped[dict[str, Any]] = mapped_column(
        JSONBType, default=dict, nullable=False
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<SecurityEvent {self.event_type} action={self.action}>"
