"""Answer relevance scoring.

The metric's job is to catch answers that are grounded but off-target, so the
tests are built around that contrast: same context, same faithfulness, one
answer that responds and one that does not.
"""

from __future__ import annotations

import pytest

from app.core.config import settings
from app.evaluation.metrics import AnswerRelevanceMetrics
from app.evaluation.relevance import (
    RelevanceScore,
    relevance_caveat,
    score_relevance,
)

pytestmark = pytest.mark.unit


QUESTION = "How many days of annual leave do employees get?"
ON_TARGET = "Employees receive 24 days of paid annual leave per calendar year."
OFF_TARGET = "Passwords must be at least 14 characters and rotate every 180 days."


def test_a_responsive_answer_outscores_an_unrelated_one():
    """The whole point of the metric, in one assertion."""
    good = score_relevance(QUESTION, ON_TARGET)
    bad = score_relevance(QUESTION, OFF_TARGET)

    assert good.score > bad.score


def test_a_grounded_but_off_topic_answer_is_caught():
    """Faithfulness cannot see this failure; relevance is why it exists.

    Both answers below could be quoted verbatim from the corpus and so score
    identically on grounding. Only one answers the question.
    """
    off_topic = score_relevance(QUESTION, OFF_TARGET)
    assert off_topic.score < settings.ANSWER_RELEVANCE_MIN_SCORE
    assert not off_topic.relevant


def test_a_how_many_question_wants_a_number():
    with_number = score_relevance(QUESTION, "Employees get 24 days.")
    without = score_relevance(QUESTION, "Employees get a generous leave allowance.")

    assert with_number.answer_type == "quantity"
    assert with_number.type_match == 1.0
    assert without.type_match == 0.0
    assert with_number.score > without.score


def test_the_reported_answer_type_is_a_readable_name():
    """It lands in the JSON report; a regex fragment there helps nobody."""
    assert score_relevance("When is it due?", "Tomorrow.").answer_type == "time"
    assert score_relevance("Who approves it?", "The manager.").answer_type == "agent"
    assert score_relevance("Where is it filed?", "In the portal.").answer_type == "place"


def test_a_when_question_wants_a_time():
    question = "When must a security incident be reported?"
    assert score_relevance(question, "Within one hour of discovery.").type_match == 1.0
    assert score_relevance(question, "It should be reported promptly.").type_match == 0.0


def test_an_untyped_question_is_not_penalised_for_the_type_signal():
    """An open question must not fail a test that never applied to it."""
    result = score_relevance(
        "What is the policy on expense claims?",
        "Expense claims must be submitted within 30 days.",
    )
    assert result.answer_type == "none"
    assert result.type_match == 1.0


def test_citation_markers_do_not_count_as_answer_content():
    """`[1]` is apparatus. Left in, it would satisfy the numeric type check."""
    result = score_relevance(QUESTION, "Employees get a generous allowance [1][2].")
    assert result.type_match == 0.0


def test_a_refusal_is_flagged_not_scored():
    result = score_relevance(QUESTION, "I could not find that.", refused=True)

    assert result.is_refusal
    assert result.score == 0.0


def test_an_empty_answer_is_treated_as_a_refusal():
    assert score_relevance(QUESTION, "").is_refusal
    assert score_relevance(QUESTION, "   ").is_refusal


def test_scores_stay_within_the_unit_interval():
    for answer in (ON_TARGET, OFF_TARGET, QUESTION, "24", "x" * 2000):
        result = score_relevance(QUESTION, answer)
        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.semantic <= 1.0


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------


def test_refusals_are_excluded_from_the_mean_not_scored_zero():
    """Otherwise the metric would track guardrail behaviour, not answer quality.

    A run that (correctly) refused more attacks would show falling "relevance"
    even though every answer it did give was just as good.
    """
    metrics = AnswerRelevanceMetrics()
    metrics.record(RelevanceScore(0.8, 0.8, 0.8, 1.0))
    metrics.record(RelevanceScore(0.6, 0.6, 0.6, 1.0))
    metrics.record(RelevanceScore(0.0, 0.0, 0.0, 0.0, is_refusal=True))

    data = metrics.as_dict()
    assert data["scored_answers"] == 2
    assert data["refusals_excluded"] == 1
    assert data["answer_relevance"] == pytest.approx(0.7, abs=1e-4)


def test_an_empty_run_does_not_divide_by_zero():
    data = AnswerRelevanceMetrics().as_dict()
    assert data["answer_relevance"] == 0.0
    assert data["components_mean"]["semantic"] == 0.0


def test_answers_below_the_threshold_are_counted():
    metrics = AnswerRelevanceMetrics()
    metrics.record(RelevanceScore(0.9, 0.9, 0.9, 1.0))
    metrics.record(RelevanceScore(0.1, 0.1, 0.1, 0.0))

    assert metrics.as_dict()["below_threshold"] == 1


def test_the_hashing_embedder_forces_a_caveat(monkeypatch):
    """A relevance number from a lexical embedder must never travel bare."""
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "hashing")
    caveat = relevance_caveat()
    assert caveat and "hashing" in caveat

    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "openai")
    assert relevance_caveat() is None
