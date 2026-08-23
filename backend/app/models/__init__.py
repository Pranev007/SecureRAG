"""ORM models.

Imported as a package so that Alembic autogenerate and ``Base.metadata`` see
every table.
"""

from app.db.base import Base
from app.models.chat import ChatSession, Message, MessageRole
from app.models.document import (
    Document,
    DocumentChunk,
    DocumentStatus,
    DocumentVisibility,
)
from app.models.security_event import (
    SecurityAction,
    SecurityEvent,
    SecurityEventType,
    SecurityLayer,
    SecuritySeverity,
)
from app.models.user import User, UserRole

__all__ = [
    "Base",
    "ChatSession",
    "Document",
    "DocumentChunk",
    "DocumentStatus",
    "DocumentVisibility",
    "Message",
    "MessageRole",
    "SecurityAction",
    "SecurityEvent",
    "SecurityEventType",
    "SecurityLayer",
    "SecuritySeverity",
    "User",
    "UserRole",
]
