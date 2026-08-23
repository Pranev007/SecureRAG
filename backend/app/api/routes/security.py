"""Security dashboard, audit trail and attack playground endpoints."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Query

from app.api.deps import AdminUser, CurrentUser, DbSession
from app.schemas.common import Page
from app.schemas.security import (
    AttackScenarioResponse,
    PlaygroundRequest,
    PlaygroundResultResponse,
    PlaygroundSuiteResponse,
    SecurityEventResponse,
    SecurityStatsResponse,
    TimeseriesPoint,
)
from app.security.playground import (
    SCENARIOS,
    SCENARIOS_BY_ID,
    AttackSurface,
    analyse,
    run_full_suite,
)
from app.services.security_event_service import query_events
from app.services.stats_service import dashboard_stats, event_timeseries

router = APIRouter(prefix="/security", tags=["security"])


@router.get(
    "/stats",
    response_model=SecurityStatsResponse,
    summary="Dashboard statistics",
)
def stats(
    db: DbSession,
    current_user: CurrentUser,
    window_days: int = Query(default=30, ge=1, le=365),
) -> SecurityStatsResponse:
    """Aggregate security statistics.

    Scope follows the caller: an ordinary user sees their own activity, an
    administrator sees the whole system. The scope is applied in SQL.
    """
    return SecurityStatsResponse.model_validate(
        dashboard_stats(db, user=current_user, window_days=window_days)
    )


@router.get(
    "/timeseries",
    response_model=list[TimeseriesPoint],
    summary="Daily event counts",
)
def timeseries(
    db: DbSession,
    current_user: CurrentUser,
    days: int = Query(default=14, ge=1, le=90),
) -> list[TimeseriesPoint]:
    return [
        TimeseriesPoint(**point)
        for point in event_timeseries(db, user=current_user, days=days)
    ]


@router.get(
    "/events",
    response_model=Page[SecurityEventResponse],
    summary="Security event audit trail",
)
def events(
    db: DbSession,
    current_user: CurrentUser,
    event_type: list[str] | None = Query(default=None),
    layer: list[str] | None = Query(default=None),
    severity: list[str] | None = Query(default=None),
    action: list[str] | None = Query(default=None),
    hours: int | None = Query(default=None, ge=1, le=24 * 365),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> Page[SecurityEventResponse]:
    """List security events.

    Non-admins see only their own events; there is no query parameter that
    widens the scope, so it cannot be widened by a crafted request.
    """
    since = datetime.now(UTC) - timedelta(hours=hours) if hours else None
    rows, total = query_events(
        db,
        user_id=None if current_user.is_admin else current_user.id,
        event_types=event_type,
        layers=layer,
        severities=severity,
        actions=action,
        since=since,
        limit=limit,
        offset=offset,
    )
    return Page[SecurityEventResponse](
        items=[SecurityEventResponse.model_validate(row) for row in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/playground/scenarios",
    response_model=list[AttackScenarioResponse],
    summary="Catalogue of attack scenarios",
)
def scenarios(current_user: CurrentUser) -> list[AttackScenarioResponse]:
    return [
        AttackScenarioResponse(
            id=s.id,
            category=s.category.value,
            surface=s.surface.value,
            name=s.name,
            payload=s.payload,
            description=s.description,
            expected=s.expected,
        )
        for s in SCENARIOS
    ]


@router.post(
    "/playground/run",
    response_model=PlaygroundResultResponse,
    summary="Run one attack against the live guardrails",
)
def run_scenario(
    payload: PlaygroundRequest, db: DbSession, current_user: CurrentUser
) -> PlaygroundResultResponse:
    """Analyse a payload without executing it.

    No retrieval and no model call happen here: the payload is scored by the
    same detector objects the real request path uses, and the result reports
    what would have happened.
    """
    scenario = SCENARIOS_BY_ID.get(payload.scenario_id or "")
    text = payload.payload or (scenario.payload if scenario else "")
    surface = (
        AttackSurface(payload.surface)
        if payload.surface
        else (scenario.surface if scenario else AttackSurface.USER_INPUT)
    )

    result = analyse(db, text, user=current_user, surface=surface, scenario=scenario)
    return PlaygroundResultResponse.from_result(result)


@router.post(
    "/playground/run-all",
    response_model=PlaygroundSuiteResponse,
    summary="Run the whole attack catalogue",
)
def run_all(db: DbSession, current_user: CurrentUser) -> PlaygroundSuiteResponse:
    results = run_full_suite(db, user=current_user)
    responses = [PlaygroundResultResponse.from_result(r) for r in results]

    attacks = [r for r in results if r.category != "benign_control"]
    controls = [r for r in results if r.category == "benign_control"]

    return PlaygroundSuiteResponse(
        results=responses,
        total=len(responses),
        matched_expectation=sum(1 for r in results if r.matched_expectation),
        attack_scenarios=len(attacks),
        control_scenarios=len(controls),
    )


@router.get(
    "/admin/overview",
    response_model=SecurityStatsResponse,
    summary="System-wide statistics (admin only)",
)
def admin_overview(
    db: DbSession,
    admin: AdminUser,
    window_days: int = Query(default=30, ge=1, le=365),
) -> SecurityStatsResponse:
    return SecurityStatsResponse.model_validate(
        dashboard_stats(db, user=None, window_days=window_days)
    )
