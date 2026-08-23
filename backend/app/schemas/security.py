"""Security dashboard and playground schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator

from app.schemas.common import ORMModel


class SecurityEventResponse(ORMModel):
    id: str
    request_id: str | None
    user_id: str | None
    event_type: str
    layer: str
    severity: str
    action: str
    risk_score: float
    detector: str | None
    resource_type: str | None
    resource_id: str | None
    detail: dict[str, Any]
    created_at: datetime


class QueryStats(BaseModel):
    total: int = 0
    blocked: int = 0
    block_rate: float = 0.0


class SecurityCounters(BaseModel):
    prompt_injection_attempts: int = 0
    indirect_injection_detections: int = 0
    grounding_failures: int = 0
    pii_detections: int = 0
    rate_limit_violations: int = 0
    authorization_denials: int = 0


class DocumentStats(BaseModel):
    total: int = 0
    chunks: int = 0
    quarantined_chunks: int = 0


class PerformanceStats(BaseModel):
    average_latency_ms: float = 0.0
    average_grounding_score: float = 0.0
    average_retrieved_chunks: float = 0.0


class EventStats(BaseModel):
    by_severity: dict[str, int] = Field(default_factory=dict)
    by_type: dict[str, int] = Field(default_factory=dict)
    total: int = 0


class SecurityStatsResponse(BaseModel):
    scope: str
    window_days: int
    queries: QueryStats
    security: SecurityCounters
    documents: DocumentStats
    performance: PerformanceStats
    events: EventStats


class TimeseriesPoint(BaseModel):
    date: str
    total: int
    attacks: int


# ----------------------------------------------------------------------
# Playground
# ----------------------------------------------------------------------


class AttackScenarioResponse(BaseModel):
    id: str
    category: str
    surface: str
    name: str
    payload: str
    description: str
    expected: str


class PlaygroundRequest(BaseModel):
    """Run a catalogued scenario, or a custom payload of your own."""

    scenario_id: str | None = None
    payload: str | None = Field(default=None, max_length=8000)
    surface: str | None = Field(
        default=None, description="user_input | document | model_output"
    )

    @model_validator(mode="after")
    def _require_something_to_run(self) -> PlaygroundRequest:
        if not self.scenario_id and not (self.payload or "").strip():
            raise ValueError("provide either scenario_id or payload")
        return self


class DetectorFindingResponse(BaseModel):
    layer: str
    detector: str
    score: float
    detail: str = ""


class PlaygroundResultResponse(BaseModel):
    scenario_id: str | None
    category: str
    surface: str
    name: str
    payload_preview: str
    decision: str
    risk_score: float
    classification: str
    findings: list[DetectorFindingResponse]
    explanation: str
    expected: str
    matched_expectation: bool | None
    thresholds: dict[str, float]

    @classmethod
    def from_result(cls, result) -> PlaygroundResultResponse:
        return cls(
            scenario_id=result.scenario_id,
            category=result.category,
            surface=result.surface,
            name=result.name,
            payload_preview=result.payload_preview,
            decision=result.decision,
            risk_score=result.risk_score,
            classification=result.classification,
            findings=[
                DetectorFindingResponse(
                    layer=f.layer, detector=f.detector, score=f.score, detail=f.detail
                )
                for f in result.findings
            ],
            explanation=result.explanation,
            expected=result.expected,
            matched_expectation=result.matched_expectation,
            thresholds=result.thresholds,
        )


class PlaygroundSuiteResponse(BaseModel):
    results: list[PlaygroundResultResponse]
    total: int
    matched_expectation: int
    attack_scenarios: int
    control_scenarios: int
