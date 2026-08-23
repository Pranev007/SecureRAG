"""Liveness and readiness endpoints."""

from __future__ import annotations

import re

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db

router = APIRouter(tags=["health"])


@router.get("/health", summary="Liveness probe")
def health() -> dict:
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
    }


def _check_embedding_dimensions(db: Session) -> str:
    """Compare the configured vector size against the actual column.

    Returns ``"ok"``, ``"not_migrated"``, or a message naming both numbers so an
    operator can see immediately which side is wrong.
    """
    try:
        declared = db.execute(
            text(
                "SELECT format_type(a.atttypid, a.atttypmod) "
                "FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid "
                "WHERE c.relname = 'document_chunks' AND a.attname = 'embedding'"
            )
        ).scalar_one_or_none()
    except Exception as exc:  # pragma: no cover - infrastructure failure path
        return f"error: {type(exc).__name__}"

    if not declared:
        return "not_migrated"

    match = re.fullmatch(r"vector\((\d+)\)", declared)
    if not match:
        return f"unexpected column type: {declared}"

    column_dimensions = int(match.group(1))
    if column_dimensions != settings.EMBEDDING_DIMENSIONS:
        return (
            f"mismatch: EMBEDDING_DIMENSIONS={settings.EMBEDDING_DIMENSIONS} "
            f"but the column is vector({column_dimensions}); "
            "re-migrate and re-embed, or restore the previous setting"
        )
    return "ok"


@router.get("/health/ready", summary="Readiness probe")
def readiness(db: Session = Depends(get_db)) -> dict:
    """Report whether dependencies are usable.

    Also reports which vector backend is active, because the difference between
    pgvector and the SQLite fallback matters operationally.
    """
    checks: dict[str, str] = {}
    try:
        db.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:  # pragma: no cover - infrastructure failure path
        checks["database"] = f"error: {type(exc).__name__}"

    dialect = db.bind.dialect.name if db.bind is not None else "unknown"
    if dialect == "postgresql":
        try:
            row = db.execute(
                text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            ).first()
            checks["pgvector"] = "ok" if row else "missing"
        except Exception as exc:  # pragma: no cover
            checks["pgvector"] = f"error: {type(exc).__name__}"

        # A configured EMBEDDING_DIMENSIONS that disagrees with the migrated
        # column is silent until the first upload, which then fails deep inside
        # the driver with "expected 256 dimensions, not 384" and a 500. Surface
        # it here instead: it is a configuration fault, and readiness is exactly
        # where configuration faults belong.
        checks["embedding_dimensions"] = _check_embedding_dimensions(db)
    else:
        checks["pgvector"] = "not_applicable"
        checks["embedding_dimensions"] = "not_applicable"

    healthy = all(v in {"ok", "not_applicable"} for v in checks.values())
    return {
        "status": "ready" if healthy else "degraded",
        "checks": checks,
        "vector_backend": "pgvector" if dialect == "postgresql" else "python_fallback",
        "llm_provider": settings.LLM_PROVIDER,
        "embedding_provider": settings.EMBEDDING_PROVIDER,
    }
