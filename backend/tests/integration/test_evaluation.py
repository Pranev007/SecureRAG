"""The evaluation harness itself.

The suite is a measurement instrument, so it needs its own tests: a harness
that silently mis-measures is worse than none, because its output looks
authoritative. Two of these tests exist because the harness *did* mis-measure —
see docs/evaluation.md, findings 3 and 4.
"""

from __future__ import annotations

import pytest

from app.evaluation.datasets import ALL_CASES, CaseKind, cases_for, dataset_summary
from app.evaluation.metrics import ConfusionMatrix, LatencyMetrics, RetrievalMetrics
from app.evaluation.report import render_markdown
from app.evaluation.runner import EvaluationRunner

pytestmark = pytest.mark.integration


# ----------------------------------------------------------------------
# Dataset integrity
# ----------------------------------------------------------------------


def test_case_ids_are_unique():
    ids = [case.id for case in ALL_CASES]
    assert len(ids) == len(set(ids))


def test_the_dataset_contains_benign_controls():
    """Without controls the suite would measure detection while hiding its cost."""
    summary = dataset_summary()
    benign = summary.get(CaseKind.BENIGN_CONTROL.value, 0)
    attacks = summary.get(CaseKind.DIRECT_INJECTION.value, 0)

    assert benign >= 5
    assert benign >= attacks * 0.5, "too few controls to bound the false-positive rate"


def test_every_category_is_represented():
    summary = dataset_summary()
    for kind in CaseKind:
        assert summary.get(kind.value, 0) > 0, f"no cases for {kind.value}"


def test_cases_can_be_filtered_by_kind():
    subset = cases_for({"direct_injection"})
    assert subset
    assert all(c.kind is CaseKind.DIRECT_INJECTION for c in subset)
    assert len(cases_for(None)) == len(ALL_CASES)


def test_authorization_cases_declare_what_must_not_leak():
    for case in ALL_CASES:
        if case.kind is CaseKind.AUTHORIZATION:
            assert (
                case.forbidden_substrings
            ), f"{case.id} must declare the values that would prove a leak"


# ----------------------------------------------------------------------
# Metric arithmetic
# ----------------------------------------------------------------------


def test_confusion_matrix_arithmetic():
    matrix = ConfusionMatrix(
        true_positives=8, false_negatives=2, false_positives=1, true_negatives=9
    )
    assert matrix.detection_rate == 0.8
    assert matrix.false_negative_rate == 0.2
    assert matrix.false_positive_rate == 0.1
    assert matrix.precision == pytest.approx(8 / 9, abs=1e-4)
    assert 0 < matrix.f1 < 1


def test_metrics_do_not_divide_by_zero_on_an_empty_run():
    empty = ConfusionMatrix()
    assert empty.detection_rate == 0.0
    assert empty.false_positive_rate == 0.0
    assert empty.f1 == 0.0
    assert LatencyMetrics().as_dict()["mean_ms"] == 0.0
    assert RetrievalMetrics().as_dict()["precision_at_k"] == 0.0


def test_retrieval_skips_cases_with_no_relevant_document():
    """An unanswerable question has no correct retrieval to score."""
    metrics = RetrievalMetrics()
    metrics.record(["a.md"], (), k=5)
    assert metrics.as_dict()["queries_scored"] == 0

    metrics.record(["a.md", "b.md"], ("a.md",), k=5)
    assert metrics.as_dict()["queries_scored"] == 1
    assert metrics.as_dict()["recall_at_k"] == 1.0
    assert metrics.as_dict()["precision_at_k"] == 0.5


def test_reciprocal_rank_reflects_position():
    first = RetrievalMetrics()
    first.record(["hit.md", "x.md"], ("hit.md",), k=5)
    second = RetrievalMetrics()
    second.record(["x.md", "hit.md"], ("hit.md",), k=5)

    assert first.as_dict()["mrr"] == 1.0
    assert second.as_dict()["mrr"] == 0.5


def test_latency_percentiles_are_ordered():
    metrics = LatencyMetrics()
    for value in range(1, 101):
        metrics.record(float(value), {"stage_ms": float(value)})
    result = metrics.as_dict()

    assert result["median_ms"] <= result["p95_ms"] <= result["max_ms"]
    assert result["by_stage_mean_ms"]["stage_ms"] == pytest.approx(50.5)


# ----------------------------------------------------------------------
# End-to-end harness
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def report(request):
    """Run a representative slice of the suite once for the whole module."""
    from app.db.session import SessionLocal
    from app.models import Base

    session = SessionLocal()
    try:
        yield EvaluationRunner(session).run(
            cases_for({"direct_injection", "benign_control", "authorization"})
        )
    finally:
        # This fixture is module-scoped and therefore outlives the per-test
        # `db` fixture's cleanup, so it tidies up after itself.
        session.rollback()
        for table in reversed(Base.metadata.sorted_tables):
            session.execute(table.delete())
        session.commit()
        session.close()


def test_the_harness_runs_and_produces_a_complete_report(report):
    data = report.as_dict()

    assert data["totals"]["cases"] > 0
    assert set(data) >= {
        "configuration",
        "dataset",
        "ingestion",
        "security",
        "retrieval",
        "quality",
        "latency",
        "totals",
        "cases",
    }
    assert data["configuration"]["llm_provider"] == "echo"


def test_attacks_are_detected_and_controls_are_not(report):
    security = report.as_dict()["security"]

    assert security["attack_cases"] > 0
    assert security["benign_cases"] > 0
    # The bar the project's claims rest on. If either regresses, this fails.
    assert security["detection_rate"] >= 0.9
    assert security["false_positive_rate"] <= 0.1


def test_the_poisoned_document_is_quarantined_during_setup(report):
    ingestion = report.as_dict()["ingestion"]

    assert ingestion["poisoned_chunks_present"] >= 1
    assert ingestion["indirect_detection_rate"] == 1.0
    assert ingestion["clean_chunks_wrongly_quarantined"] == 0


def test_no_case_leaks_the_other_users_confidential_data(report):
    for case in report.as_dict()["cases"]:
        assert "450000" not in case["answer"]
        assert "320000" not in case["answer"]


def test_retrieval_is_scored_on_retrieval_not_on_citations(report):
    """Regression: scoring retrieval from `sources` marked refusals as misses."""
    for case in report.as_dict()["cases"]:
        assert "documents_retrieved" in case["retrieval"]
        assert "documents_cited" in case["retrieval"]


def test_failing_cases_carry_an_explanation(report):
    for case in report.as_dict()["cases"]:
        if not case["passed"]:
            assert case["failure_detail"], f"{case['case_id']} failed without a reason"


def test_the_markdown_report_renders(report):
    markdown = render_markdown(report)

    assert "# SecureRAG Evaluation Report" in markdown
    assert "Attack detection rate" in markdown
    assert "False positive rate" in markdown
    # The provider caveat must appear whenever offline stand-ins were used.
    assert "offline stand-ins" in markdown
