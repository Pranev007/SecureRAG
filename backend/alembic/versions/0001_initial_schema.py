"""Initial schema: users, documents, chunks, chat, security events.

Revision ID: 0001_initial
Revises:
Create Date: 2026-01-01

This migration is dialect-aware.  On PostgreSQL it enables the ``vector``
extension, stores embeddings in a native ``vector(n)`` column, and creates an
HNSW index for approximate nearest-neighbour search plus a GIN index over
``to_tsvector('english', content)`` for the keyword arm of hybrid retrieval.
On SQLite (tests / offline development) embeddings are stored as JSON and both
specialised indexes are skipped.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from app.core.config import settings
from app.db.types import GUID, JSONBType, VectorType

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _is_postgres() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def upgrade() -> None:
    if _is_postgres():
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ------------------------------------------------------------------
    # users
    # ------------------------------------------------------------------
    op.create_table(
        "users",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("full_name", sa.String(255), nullable=True),
        sa.Column("role", sa.String(32), nullable=False, server_default="user"),
        sa.Column(
            "is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)
    op.create_index("ix_users_role", "users", ["role"])
    op.create_index("ix_users_created_at", "users", ["created_at"])

    # ------------------------------------------------------------------
    # documents
    # ------------------------------------------------------------------
    op.create_table(
        "documents",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column(
            "owner_id",
            GUID(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("filename", sa.String(512), nullable=False),
        sa.Column("extension", sa.String(16), nullable=False),
        sa.Column("content_type", sa.String(128), nullable=False),
        sa.Column("file_size_bytes", sa.Integer(), nullable=False),
        sa.Column("content_sha256", sa.String(64), nullable=False),
        sa.Column("storage_path", sa.String(1024), nullable=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("error_message", sa.String(1024), nullable=True),
        sa.Column(
            "visibility", sa.String(32), nullable=False, server_default="private"
        ),
        sa.Column("page_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("char_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "max_injection_risk", sa.Float(), nullable=False, server_default="0"
        ),
        sa.Column(
            "quarantined_chunk_count", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column("meta", JSONBType(), nullable=False, server_default="{}"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("owner_id", "content_sha256", name="uq_document_owner_sha"),
    )
    op.create_index("ix_documents_owner_id", "documents", ["owner_id"])
    op.create_index("ix_documents_status", "documents", ["status"])
    op.create_index("ix_documents_content_sha256", "documents", ["content_sha256"])
    op.create_index("ix_documents_created_at", "documents", ["created_at"])
    op.create_index("ix_documents_owner_status", "documents", ["owner_id", "status"])

    # ------------------------------------------------------------------
    # document_chunks
    # ------------------------------------------------------------------
    op.create_table(
        "document_chunks",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column(
            "document_id",
            GUID(),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("owner_id", GUID(), nullable=False),
        sa.Column(
            "visibility", sa.String(32), nullable=False, server_default="private"
        ),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_sha256", sa.String(64), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("char_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("section", sa.String(512), nullable=True),
        sa.Column("source_filename", sa.String(512), nullable=False),
        sa.Column(
            "embedding", VectorType(settings.EMBEDDING_DIMENSIONS), nullable=True
        ),
        sa.Column("embedding_model", sa.String(128), nullable=True),
        sa.Column("injection_risk", sa.Float(), nullable=False, server_default="0"),
        sa.Column(
            "is_quarantined",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column("injection_labels", JSONBType(), nullable=False, server_default="[]"),
        sa.Column("meta", JSONBType(), nullable=False, server_default="{}"),
        sa.UniqueConstraint(
            "document_id", "chunk_index", name="uq_chunk_document_index"
        ),
    )
    op.create_index("ix_document_chunks_document_id", "document_chunks", ["document_id"])
    op.create_index("ix_document_chunks_owner_id", "document_chunks", ["owner_id"])
    op.create_index(
        "ix_chunks_owner_quarantine",
        "document_chunks",
        ["owner_id", "is_quarantined"],
    )

    if _is_postgres():
        # Approximate nearest-neighbour index. HNSW is preferred over IVFFlat
        # here because it needs no training pass and stays accurate as rows are
        # inserted incrementally, which is exactly the upload-driven pattern of
        # this application. vector_cosine_ops matches the `<=>` operator used
        # by the retriever.
        op.execute(
            "CREATE INDEX ix_chunks_embedding_hnsw ON document_chunks "
            "USING hnsw (embedding vector_cosine_ops) "
            "WITH (m = 16, ef_construction = 64)"
        )
        # Functional GIN index backing the BM25-style keyword arm of hybrid
        # retrieval. Expression must match the query in
        # app/rag/retrieval/keyword.py exactly for the planner to use it.
        op.execute(
            "CREATE INDEX ix_chunks_content_fts ON document_chunks "
            "USING gin (to_tsvector('english', content))"
        )

    # ------------------------------------------------------------------
    # chat_sessions / messages
    # ------------------------------------------------------------------
    op.create_table(
        "chat_sessions",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column(
            "user_id",
            GUID(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("title", sa.String(255), nullable=False, server_default="New chat"),
        sa.Column("message_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_chat_sessions_user_id", "chat_sessions", ["user_id"])
    op.create_index("ix_chat_sessions_created_at", "chat_sessions", ["created_at"])

    op.create_table(
        "messages",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column(
            "session_id",
            GUID(),
            sa.ForeignKey("chat_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("user_id", GUID(), nullable=False),
        sa.Column("role", sa.String(32), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "was_blocked",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column("block_reason", sa.String(64), nullable=True),
        sa.Column("risk_score", sa.Float(), nullable=False, server_default="0"),
        sa.Column("grounding_score", sa.Float(), nullable=True),
        sa.Column(
            "pii_detected",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column("citations", JSONBType(), nullable=False, server_default="[]"),
        sa.Column(
            "retrieved_chunk_count", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column("latency_ms", sa.Float(), nullable=False, server_default="0"),
        sa.Column("request_id", sa.String(64), nullable=True),
        sa.Column("meta", JSONBType(), nullable=False, server_default="{}"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_messages_session_id", "messages", ["session_id"])
    op.create_index("ix_messages_user_id", "messages", ["user_id"])
    op.create_index("ix_messages_request_id", "messages", ["request_id"])
    op.create_index("ix_messages_created_at", "messages", ["created_at"])
    op.create_index(
        "ix_messages_session_created", "messages", ["session_id", "created_at"]
    )

    # ------------------------------------------------------------------
    # security_events
    # ------------------------------------------------------------------
    op.create_table(
        "security_events",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column("request_id", sa.String(64), nullable=True),
        sa.Column("user_id", GUID(), nullable=True),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column("layer", sa.String(32), nullable=False),
        sa.Column("severity", sa.String(16), nullable=False),
        sa.Column("action", sa.String(32), nullable=False),
        sa.Column("risk_score", sa.Float(), nullable=False, server_default="0"),
        sa.Column("detector", sa.String(128), nullable=True),
        sa.Column("content_ref", sa.String(64), nullable=True),
        sa.Column("resource_type", sa.String(32), nullable=True),
        sa.Column("resource_id", sa.String(64), nullable=True),
        sa.Column("client_ref", sa.String(32), nullable=True),
        sa.Column("detail", JSONBType(), nullable=False, server_default="{}"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_security_events_request_id", "security_events", ["request_id"])
    op.create_index("ix_security_events_user_id", "security_events", ["user_id"])
    op.create_index("ix_security_events_event_type", "security_events", ["event_type"])
    op.create_index("ix_security_events_layer", "security_events", ["layer"])
    op.create_index("ix_security_events_severity", "security_events", ["severity"])
    op.create_index("ix_security_events_action", "security_events", ["action"])
    op.create_index("ix_security_events_created_at", "security_events", ["created_at"])
    op.create_index(
        "ix_events_type_created", "security_events", ["event_type", "created_at"]
    )
    op.create_index(
        "ix_events_user_created", "security_events", ["user_id", "created_at"]
    )


def downgrade() -> None:
    op.drop_table("security_events")
    op.drop_table("messages")
    op.drop_table("chat_sessions")
    if _is_postgres():
        op.execute("DROP INDEX IF EXISTS ix_chunks_content_fts")
        op.execute("DROP INDEX IF EXISTS ix_chunks_embedding_hnsw")
    op.drop_table("document_chunks")
    op.drop_table("documents")
    op.drop_table("users")
