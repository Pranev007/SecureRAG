"""Chat session and message models."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

from sqlalchemy import Boolean, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.db.types import GUID, JSONBType

if TYPE_CHECKING:
    from app.models.user import User


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class ChatSession(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "chat_sessions"

    user_id: Mapped[str] = mapped_column(
        GUID(), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(255), default="New chat", nullable=False)
    message_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    user: Mapped[User] = relationship(back_populates="chat_sessions")
    messages: Mapped[list[Message]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Message.created_at",
    )


class Message(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "messages"
    __table_args__ = (Index("ix_messages_session_created", "session_id", "created_at"),)

    session_id: Mapped[str] = mapped_column(
        GUID(),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[str] = mapped_column(GUID(), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # --- guardrail outcome, denormalised for the dashboard -------------
    was_blocked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    block_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    risk_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    grounding_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    pii_detected: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    citations: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONBType, default=list, nullable=False
    )
    retrieved_chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    meta: Mapped[dict[str, Any]] = mapped_column(JSONBType, default=dict, nullable=False)

    session: Mapped[ChatSession] = relationship(back_populates="messages")
