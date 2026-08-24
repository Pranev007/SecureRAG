"""Evaluation metrics.

Every metric here is defined in one place with its formula visible, so that a
number in the report can be traced to the arithmetic that produced it.

The definitions that matter most, stated precisely:

* **Attack detection rate** = blocked attacks / total attacks.  This is
  *recall on the attack class*, and it is meaningless on its own -- a detector
  that blocks everything scores 1.0.
* **False positive rate** = benign inputs blocked / total benign inputs.
  Reported beside detection, always.  A guardrail is characterised by the
  *pair*, never by detection alone.
* **Benign refusal rate** = benign inputs that produced no answer / total
  benign inputs.  Distinct from the false-positive rate because a refusal is
  not a block: the case still passes, so this cost does not appear in the pass
  rate.  Reported because a guardrail can buy faithfulness by answering less,
  and that trade has to be visible.
* **False negative rate** = 1 - detection rate.  Reported explicitly rather
  than left for the reader to subtract, because it is the number that gets
  quietly omitted.
* **Precision@k / Recall@k** at the document level: of the k chunks retrieved,
  how many came from a document marked relevant, and how many of the relevant
  documents were reached at all.
* **Faithfulness** = mean grounding score over answered cases.
* **Citation accuracy** = verified citations / emitted citations.
* **Answer relevance** = mean relevance score over *answered* cases only.
  Refusals are excluded rather than scored zero -- see
  ``app.evaluation.relevance`` for why mixing them in would make the metric
  track guardrail behaviour instead of answer quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, median


def _safe_ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


@dataclass
class ConfusionMatrix:
    """Attack detection framed as binary classification.

    "Positive" = the system took a protective action (block/flag/quarantine).
    """

    true_positives: int = 0
    false_negatives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    # Benign inputs the system answered nothing for. Counted separately from
    # false positives because a refusal is not a block: the judge accepts it
    # for a benign case, and it is therefore invisible in the pass rate. It is
    # still a cost the reader has to see -- a guardrail that refuses half the
    # benign traffic scores a perfect confusion matrix.
    benign_refusals: int = 0

    @property
    def attacks(self) -> int:
        return self.true_positives + self.false_negatives

    @property
    def benign(self) -> int:
        return self.false_positives + self.true_negatives

    @property
    def detection_rate(self) -> float:
        """Recall on the attack class."""
        return _safe_ratio(self.true_positives, self.attacks)

    @property
    def false_negative_rate(self) -> float:
        return _safe_ratio(self.false_negatives, self.attacks)

    @property
    def false_positive_rate(self) -> float:
        return _safe_ratio(self.false_positives, self.benign)

    @property
    def precision(self) -> float:
        predicted_positive = self.true_positives + self.false_positives
        return _safe_ratio(self.true_positives, predicted_positive)

    @property
    def benign_refusal_rate(self) -> float:
        """Benign inputs that produced no answer, over all benign inputs."""
        return _safe_ratio(self.benign_refusals, self.benign)

    @property
    def f1(self) -> float:
        precision, recall = self.precision, self.detection_rate
        if precision + recall == 0:
            return 0.0
        return round(2 * precision * recall / (precision + recall), 4)

    def as_dict(self) -> dict:
        return {
            "true_positives": self.true_positives,
            "false_negatives": self.false_negatives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "attack_cases": self.attacks,
            "benign_cases": self.benign,
            "detection_rate": self.detection_rate,
            "false_negative_rate": self.false_negative_rate,
            "false_positive_rate": self.false_positive_rate,
            "benign_refusals": self.benign_refusals,
            "benign_refusal_rate": self.benign_refusal_rate,
            "precision": self.precision,
            "f1": self.f1,
        }


@dataclass
class RetrievalMetrics:
    precision_at_k: list[float] = field(default_factory=list)
    recall_at_k: list[float] = field(default_factory=list)
    reciprocal_ranks: list[float] = field(default_factory=list)
    k: int = 0

    def record(
        self, retrieved_filenames: list[str], relevant: tuple[str, ...], k: int
    ) -> None:
        """Record one query's retrieval outcome.

        Cases with no declared relevant document are skipped rather than
        counted as zero: an unanswerable question has no correct retrieval, and
        scoring it as a miss would understate retrieval quality for a reason
        that has nothing to do with retrieval.
        """
        if not relevant:
            return
        self.k = max(self.k, k)
        relevant_set = set(relevant)
        top_k = retrieved_filenames[:k]

        hits = sum(1 for name in top_k if name in relevant_set)
        self.precision_at_k.append(_safe_ratio(hits, len(top_k)))

        found = len({name for name in top_k if name in relevant_set})
        self.recall_at_k.append(_safe_ratio(found, len(relevant_set)))

        rank = next((i + 1 for i, name in enumerate(top_k) if name in relevant_set), None)
        self.reciprocal_ranks.append(round(1.0 / rank, 4) if rank else 0.0)

    def as_dict(self) -> dict:
        return {
            "k": self.k,
            "queries_scored": len(self.precision_at_k),
            "precision_at_k": round(mean(self.precision_at_k), 4)
            if self.precision_at_k
            else 0.0,
            "recall_at_k": round(mean(self.recall_at_k), 4) if self.recall_at_k else 0.0,
            "mrr": round(mean(self.reciprocal_ranks), 4)
            if self.reciprocal_ranks
            else 0.0,
        }


@dataclass
class LatencyMetrics:
    samples: list[float] = field(default_factory=list)
    stages: dict[str, list[float]] = field(default_factory=dict)

    def record(self, total_ms: float, timings: dict[str, float]) -> None:
        self.samples.append(total_ms)
        for stage, value in timings.items():
            self.stages.setdefault(stage, []).append(value)

    @staticmethod
    def _percentile(values: list[float], fraction: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(int(round(fraction * (len(ordered) - 1))), len(ordered) - 1)
        return round(ordered[index], 2)

    def as_dict(self) -> dict:
        return {
            "samples": len(self.samples),
            "mean_ms": round(mean(self.samples), 2) if self.samples else 0.0,
            "median_ms": round(median(self.samples), 2) if self.samples else 0.0,
            "p95_ms": self._percentile(self.samples, 0.95),
            "max_ms": round(max(self.samples), 2) if self.samples else 0.0,
            "by_stage_mean_ms": {
                stage: round(mean(values), 2)
                for stage, values in sorted(self.stages.items())
            },
        }


@dataclass
class AnswerRelevanceMetrics:
    """Does the answer address the question asked?

    Kept separate from :class:`QualityMetrics` because the denominators differ:
    faithfulness is defined over every answered case, while relevance is only
    defined where an answer was actually attempted. Folding them together
    would silently change what "mean" meant depending on how many cases the
    guardrails refused.
    """

    scores: list[float] = field(default_factory=list)
    semantic: list[float] = field(default_factory=list)
    coverage: list[float] = field(default_factory=list)
    type_match: list[float] = field(default_factory=list)
    refusals_excluded: int = 0
    caveat: str | None = None

    def record(self, result) -> None:
        """Record one :class:`~app.evaluation.relevance.RelevanceScore`."""
        if result.is_refusal:
            self.refusals_excluded += 1
            return
        self.scores.append(result.score)
        self.semantic.append(result.semantic)
        self.coverage.append(result.coverage)
        self.type_match.append(result.type_match)

    @property
    def below_threshold(self) -> int:
        from app.core.config import settings

        return sum(1 for s in self.scores if s < settings.ANSWER_RELEVANCE_MIN_SCORE)

    def as_dict(self) -> dict:
        return {
            "scored_answers": len(self.scores),
            "refusals_excluded": self.refusals_excluded,
            "answer_relevance": round(mean(self.scores), 4) if self.scores else 0.0,
            "below_threshold": self.below_threshold,
            "components_mean": {
                "semantic": round(mean(self.semantic), 4) if self.semantic else 0.0,
                "coverage": round(mean(self.coverage), 4) if self.coverage else 0.0,
                "type_match": round(mean(self.type_match), 4) if self.type_match else 0.0,
            },
            "caveat": self.caveat,
        }


@dataclass
class QualityMetrics:
    grounding_scores: list[float] = field(default_factory=list)
    citations_emitted: int = 0
    citations_verified: int = 0
    answers_with_citations: int = 0
    answered: int = 0
    correct_substring_hits: int = 0
    substring_cases: int = 0

    def as_dict(self) -> dict:
        return {
            "answered_cases": self.answered,
            "faithfulness": round(mean(self.grounding_scores), 4)
            if self.grounding_scores
            else 0.0,
            "answers_with_citations": _safe_ratio(
                self.answers_with_citations, self.answered
            ),
            "citation_accuracy": _safe_ratio(
                self.citations_verified, self.citations_emitted
            ),
            "answer_correctness": _safe_ratio(
                self.correct_substring_hits, self.substring_cases
            ),
        }
