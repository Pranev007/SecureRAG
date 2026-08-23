"""Recording of security events.

Every guardrail decision lands here and nowhere else, which gives one place to
enforce the rule that matters most: **no user query text and no document text
is ever written to the audit trail**.  Callers pass content through
:func:`~app.core.logging.redact_for_log`, which yields a short SHA-256 prefix.
That is enough to recognise the same payload arriving fifty times (an attack
campaign) while keeping the payload itself out of the database and the logs.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.core.request_context import get_request_id
from app.models.security_event import (
    SecurityAction,
    SecurityEvent,
    SecurityEventType,
    SecurityLayer,
    SecuritySeverity,
)

logger = get_logger("app.security.audit")

# Keys that must never appear in a security event's detail payload, even if a
# caller passes them by mistake. Defence against accidental leakage.
_FORBIDDEN_DETAIL_KEYS = frozenset(
    """
    content text query question answer prompt password token chunk
    document_text context
    """.split()
)


def hash_client(value: str | None) -> str | None:
    """Truncated hash of a client identifier (usually an IP address)."""
    if not value:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _scrub(detail: dict[str, Any] | None) -> dict[str, Any]:
    if not detail:
        return {}
    scrubbed: dict[str, Any] = {}
    for key, value in detail.items():
        if key.lower() in _FORBIDDEN_DETAIL_KEYS:
            logger.warning(
                "security_event_detail_dropped",
                extra={"dropped_key": key},
            )
            continue
        scrubbed[key] = value
    return scrubbed


def record_event(
    db: Session,
    *,
    event_type: SecurityEventType,
    layer: SecurityLayer,
    severity: SecuritySeverity,
    action: SecurityAction,
    user_id: str | None = None,
    risk_score: float = 0.0,
    detector: str | None = None,
    content_ref: str | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
    client_ref: str | None = None,
    detail: dict[str, Any] | None = None,
    commit: bool = True,
) -> SecurityEvent:
    """Persist one security event and mirror it to the structured log."""
    event = SecurityEvent(
        request_id=get_request_id(),
        user_id=user_id,
        event_type=event_type.value,
        layer=layer.value,
        severity=severity.value,
        action=action.value,
        risk_score=round(float(risk_score), 4),
        detector=detector,
        content_ref=content_ref,
        resource_type=resource_type,
        resource_id=resource_id,
        client_ref=client_ref,
        detail=_scrub(detail),
    )
    db.add(event)
    if commit:
        db.commit()

    log = logger.warning if severity in _LOUD else logger.info
    log(
        event_type.value,
        extra={
            "layer": layer.value,
            "severity": severity.value,
            "action": action.value,
            "risk_score": event.risk_score,
            "detector": detector,
            "content_ref": content_ref,
        },
    )
    return event


_LOUD = {SecuritySeverity.HIGH, SecuritySeverity.CRITICAL}


def query_events(
    db: Session,
    *,
    user_id: str | None = None,
    event_types: list[str] | None = None,
    layers: list[str] | None = None,
    severities: list[str] | None = None,
    actions: list[str] | None = None,
    since: datetime | None = None,
    limit: int = 100,
    offset: int = 0,
) -> tuple[list[SecurityEvent], int]:
    """Return a filtered page of events and the total matching count."""
    filters = []
    if user_id is not None:
        filters.append(SecurityEvent.user_id == user_id)
    if event_types:
        filters.append(SecurityEvent.event_type.in_(event_types))
    if layers:
        filters.append(SecurityEvent.layer.in_(layers))
    if severities:
        filters.append(SecurityEvent.severity.in_(severities))
    if actions:
        filters.append(SecurityEvent.action.in_(actions))
    if since is not None:
        filters.append(SecurityEvent.created_at >= since)

    total = db.execute(
        select(func.count()).select_from(SecurityEvent).where(*filters)
    ).scalar_one()

    rows = (
        db.execute(
            select(SecurityEvent)
            .where(*filters)
            .order_by(SecurityEvent.created_at.desc())
            .limit(min(limit, 500))
            .offset(max(offset, 0))
        )
        .scalars()
        .all()
    )
    return list(rows), int(total)


def purge_expired_events(db: Session, retention_days: int) -> int:
    """Delete events older than the retention window."""
    cutoff = datetime.now(UTC) - timedelta(days=retention_days)
    deleted = db.query(SecurityEvent).filter(SecurityEvent.created_at < cutoff).delete()
    db.commit()
    return int(deleted)
