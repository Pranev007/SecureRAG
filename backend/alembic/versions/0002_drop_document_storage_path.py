"""Drop documents.storage_path, a column nothing ever wrote.

Revision ID: 0002_drop_storage_path
Revises: 0001_initial
Create Date: 2026-08-29

The column was created by ``0001`` for a feature that was never built: keeping
the uploaded file on disk alongside the parsed chunks. Ingestion parses an
upload in memory and writes chunks and embeddings to the database, the original
bytes are never written anywhere, and no route serves them back. Nothing in the
codebase ever assigned the column, so every row has held NULL since the schema
existed.

That made the delete path in ``DocumentService`` unreachable: it guarded an
``unlink`` behind ``if document.storage_path``, which is never true. Dropping
the column and that branch together keeps the schema honest about what the
service actually stores.

The downgrade restores the column as nullable, which is faithful -- the data it
would have held never existed.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_drop_storage_path"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # batch_alter_table so this also works on SQLite, which cannot DROP COLUMN
    # directly on older versions and is the default for tests and offline dev.
    with op.batch_alter_table("documents") as batch:
        batch.drop_column("storage_path")


def downgrade() -> None:
    with op.batch_alter_table("documents") as batch:
        batch.add_column(sa.Column("storage_path", sa.String(1024), nullable=True))
