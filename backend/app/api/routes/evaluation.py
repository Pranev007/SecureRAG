"""Evaluation endpoint.

Runs the evaluation suite on demand and returns the report.

Two properties matter here, and both are security properties rather than
conveniences:

* **Admin only.** The suite reports system-level guardrail behaviour — detection
  rates, thresholds in force, which cases fail. That is operator information,
  not user information.

* **It never touches the live database.** ``EvaluationRunner`` creates its own
  users and ingests its own corpus, including a deliberately poisoned document.
  Running that against the application's database would inject evaluation
  fixtures into real data. The run is therefore given a throwaway SQLite
  database that is deleted when it finishes, exactly as the CLI does.

The run is synchronous and takes a few seconds. That is acceptable for an
occasional admin action and keeps the endpoint honest: the response *is* the
result, rather than a job id that has to be polled.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.api.deps import AdminUser, chat_rate_limit
from app.core.logging import get_logger
from app.evaluation.datasets import cases_for, dataset_summary
from app.evaluation.runner import EvaluationRunner
from app.models import Base
from app.schemas.evaluation import EvaluationRunRequest, EvaluationRunResponse

logger = get_logger("app.api.evaluation")

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.post(
    "/run",
    response_model=EvaluationRunResponse,
    summary="Run the evaluation suite (admin only)",
    dependencies=[Depends(chat_rate_limit)],
)
def run_evaluation(
    payload: EvaluationRunRequest, admin: AdminUser
) -> EvaluationRunResponse:
    """Execute the evaluation suite against a throwaway database.

    Optionally restrict to particular case kinds, e.g.
    ``{"kinds": ["direct_injection", "benign_control"]}`` — useful for a quick
    guardrail check without paying for the full corpus.
    """
    cases = cases_for(set(payload.kinds)) if payload.kinds else None

    with tempfile.TemporaryDirectory(prefix="securerag-eval-api-") as workspace:
        database_path = Path(workspace) / "evaluation.db"
        engine = create_engine(
            f"sqlite:///{database_path.as_posix()}",
            connect_args={"check_same_thread": False},
        )
        # `create_all` rather than Alembic: this database exists for the
        # duration of one request, and rebinding the global settings URL to run
        # a migration would race any concurrent request.
        Base.metadata.create_all(engine)

        session = sessionmaker(bind=engine, expire_on_commit=False)()
        try:
            report = EvaluationRunner(session).run(cases)
        finally:
            session.close()
            engine.dispose()

    logger.info(
        "evaluation_run_via_api",
        extra={
            "requested_by": admin.id,
            "cases": report.totals["cases"],
            "pass_rate": report.totals["pass_rate"],
            "detection_rate": report.security["detection_rate"],
        },
    )

    data = report.as_dict()
    return EvaluationRunResponse(
        started_at=data["started_at"],
        finished_at=data["finished_at"],
        duration_seconds=data["duration_seconds"],
        configuration=data["configuration"],
        dataset=data["dataset"],
        ingestion=data["ingestion"],
        security=data["security"],
        retrieval=data["retrieval"],
        quality=data["quality"],
        latency=data["latency"],
        totals=data["totals"],
        cases=data["cases"] if payload.include_cases else [],
    )


@router.get(
    "/dataset",
    summary="Describe the evaluation dataset without running it",
)
def describe_dataset(admin: AdminUser) -> dict:
    """Case counts per category.

    Cheap enough to call freely, and it makes the shape of the suite visible
    without waiting for a run — in particular the ratio of attacks to benign
    controls, which is what makes the detection rate meaningful.
    """
    summary = dataset_summary()
    return {
        "total": summary.pop("total"),
        "by_kind": summary,
        "note": (
            "Benign controls are inputs engineered to look suspicious. A "
            "detection rate reported without the matching false-positive rate "
            "is not a meaningful number."
        ),
    }
