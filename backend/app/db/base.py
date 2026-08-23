"""Declarative base and shared model mixins."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.db.types import GUID


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    """Base class for all ORM models."""


class UUIDPrimaryKeyMixin:
    """UUID primary keys.

    Chosen over auto-increment integers because document and user identifiers
    are exposed in API paths: sequential integers leak corpus size and invite
    enumeration (``/documents/1``, ``/documents/2``, ...).  Authorisation is
    still enforced server-side -- unguessable ids are defence in depth, not the
    control itself.
    """

    id: Mapped[str] = mapped_column(GUID(), primary_key=True, default=new_uuid)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
