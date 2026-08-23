"""Aggregations for the security dashboard.

Scoping rule: a normal user sees statistics for **their own** activity; an
admin sees the whole system.  The scope is applied in SQL, not by filtering
results afterwards, for the same reason it is in retrieval -- a post-filter is
one forgotten branch away from a leak.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import Select, case, func, select
from sqlalchemy.orm import Session

from app.models.chat import Message
from app.models.document import Document, DocumentChunk
from app.models.security_event import SecurityEvent, SecurityEventType
from app.models.user import User


def _scoped(stmt: Select, column, user: User | None) -> Select:
    """Restrict a query to one user unless the caller is an admin."""
    if user is not None and not user.is_admin:
        return stmt.where(column == user.id)
    return stmt


def _count(db: Session, model, column, user: User | None, *filters) -> int:
    stmt = select(func.count()).select_from(model).where(*filters)
    return int(db.execute(_scoped(stmt, column, user)).scalar_one())


# Event types that represent an attack being stopped, as opposed to routine
# activity. Used for the headline "attacks blocked" figure.
ATTACK_EVENT_TYPES = (
    SecurityEventType.PROMPT_INJECTION_DETECTED.value,
    SecurityEventType.INDIRECT_INJECTION_DETECTED.value,
    SecurityEventType.UNSAFE_OUTPUT_DETECTED.value,
    SecurityEventType.AUTHORIZATION_DENIED.value,
)


def dashboard_stats(
    db: Session, *, user: User | None = None, window_days: int = 30
) -> dict:
    """Everything the dashboard needs, in one round of aggregates."""
    since = datetime.now(UTC) - timedelta(days=window_days)
    scope_user = user if (user is not None and not user.is_admin) else None

    total_messages = _count(
        db, Message, Message.user_id, scope_user, Message.role == "user"
    )
    # Filtered to role="user" so it is comparable with `total_messages`. A
    # blocked exchange writes *two* rows (the question and the refusal), both
    # flagged; counting them all would report a block rate of exactly double
    # the truth.
    blocked = _count(
        db,
        Message,
        Message.user_id,
        scope_user,
        Message.role == "user",
        Message.was_blocked.is_(True),
    )

    injection_attempts = _count(
        db,
        SecurityEvent,
        SecurityEvent.user_id,
        scope_user,
        SecurityEvent.event_type.in_(
            [
                SecurityEventType.PROMPT_INJECTION_DETECTED.value,
                SecurityEventType.PROMPT_INJECTION_SUSPECTED.value,
            ]
        ),
    )
    indirect_injections = _count(
        db,
        SecurityEvent,
        SecurityEvent.user_id,
        scope_user,
        SecurityEvent.event_type == SecurityEventType.INDIRECT_INJECTION_DETECTED.value,
    )
    grounding_failures = _count(
        db,
        SecurityEvent,
        SecurityEvent.user_id,
        scope_user,
        SecurityEvent.event_type == SecurityEventType.GROUNDING_FAILED.value,
    )
    pii_detections = _count(
        db,
        SecurityEvent,
        SecurityEvent.user_id,
        scope_user,
        SecurityEvent.event_type.in_(
            [
                SecurityEventType.OUTPUT_PII_DETECTED.value,
                SecurityEventType.INPUT_PII_DETECTED.value,
            ]
        ),
    )
    rate_limited = _count(
        db,
        SecurityEvent,
        SecurityEvent.user_id,
        scope_user,
        SecurityEvent.event_type == SecurityEventType.RATE_LIMIT_EXCEEDED.value,
    )
    authorization_denials = _count(
        db,
        SecurityEvent,
        SecurityEvent.user_id,
        scope_user,
        SecurityEvent.event_type == SecurityEventType.AUTHORIZATION_DENIED.value,
    )

    documents = _count(db, Document, Document.owner_id, scope_user)
    chunks = _count(db, DocumentChunk, DocumentChunk.owner_id, scope_user)
    quarantined = _count(
        db,
        DocumentChunk,
        DocumentChunk.owner_id,
        scope_user,
        DocumentChunk.is_quarantined.is_(True),
    )

    latency_stmt = select(func.avg(Message.latency_ms)).where(
        Message.role == "assistant", Message.latency_ms > 0
    )
    average_latency = db.execute(
        _scoped(latency_stmt, Message.user_id, scope_user)
    ).scalar()

    grounding_stmt = select(func.avg(Message.grounding_score)).where(
        Message.role == "assistant", Message.grounding_score.isnot(None)
    )
    average_grounding = db.execute(
        _scoped(grounding_stmt, Message.user_id, scope_user)
    ).scalar()

    retrieved_stmt = select(func.avg(Message.retrieved_chunk_count)).where(
        Message.role == "assistant"
    )
    average_retrieved = db.execute(
        _scoped(retrieved_stmt, Message.user_id, scope_user)
    ).scalar()

    severity_stmt = (
        select(SecurityEvent.severity, func.count())
        .where(SecurityEvent.created_at >= since)
        .group_by(SecurityEvent.severity)
    )
    by_severity = {
        row[0]: int(row[1])
        for row in db.execute(_scoped(severity_stmt, SecurityEvent.user_id, scope_user))
    }

    type_stmt = (
        select(SecurityEvent.event_type, func.count())
        .where(SecurityEvent.created_at >= since)
        .group_by(SecurityEvent.event_type)
        .order_by(func.count().desc())
        .limit(15)
    )
    by_type = {
        row[0]: int(row[1])
        for row in db.execute(_scoped(type_stmt, SecurityEvent.user_id, scope_user))
    }

    return {
        "scope": "system" if scope_user is None else "user",
        "window_days": window_days,
        "queries": {
            "total": total_messages,
            "blocked": blocked,
            "block_rate": round(blocked / total_messages, 4) if total_messages else 0.0,
        },
        "security": {
            "prompt_injection_attempts": injection_attempts,
            "indirect_injection_detections": indirect_injections,
            "grounding_failures": grounding_failures,
            "pii_detections": pii_detections,
            "rate_limit_violations": rate_limited,
            "authorization_denials": authorization_denials,
        },
        "documents": {
            "total": documents,
            "chunks": chunks,
            "quarantined_chunks": quarantined,
        },
        "performance": {
            "average_latency_ms": round(float(average_latency), 2)
            if average_latency
            else 0.0,
            "average_grounding_score": round(float(average_grounding), 4)
            if average_grounding
            else 0.0,
            "average_retrieved_chunks": round(float(average_retrieved), 2)
            if average_retrieved
            else 0.0,
        },
        "events": {
            "by_severity": by_severity,
            "by_type": by_type,
            "total": sum(by_severity.values()),
        },
    }


def event_timeseries(
    db: Session, *, user: User | None = None, days: int = 14
) -> list[dict]:
    """Daily counts of total and attack-related events."""
    since = datetime.now(UTC) - timedelta(days=days)
    scope_user = user if (user is not None and not user.is_admin) else None

    # date() works identically on SQLite and PostgreSQL for a timestamp column.
    day = func.date(SecurityEvent.created_at)
    stmt = (
        select(
            day.label("day"),
            func.count().label("total"),
            func.sum(
                case((SecurityEvent.event_type.in_(ATTACK_EVENT_TYPES), 1), else_=0)
            ).label("attacks"),
        )
        .where(SecurityEvent.created_at >= since)
        .group_by(day)
        .order_by(day)
    )

    return [
        {
            "date": str(row.day),
            "total": int(row.total or 0),
            "attacks": int(row.attacks or 0),
        }
        for row in db.execute(_scoped(stmt, SecurityEvent.user_id, scope_user))
    ]
