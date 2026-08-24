"""Natural Language Inference for grounding verification.

WHY THIS EXISTS
---------------
``grounding.py`` scores a claim by *lexical* overlap with the retrieved
context.  That is fast, dependency-free and explainable, and it catches the
bulk of real hallucination -- but it has two failure modes it cannot fix from
inside, both documented there:

* a fluent **contradiction** that reuses the source's vocabulary scores highly,
  because overlap is polarity-blind;
* a correct answer that **paraphrases** heavily scores poorly, because
  different words are different words.

An NLI cross-encoder addresses exactly those two.  It reads a (premise,
hypothesis) pair jointly and returns a distribution over
*entailment / neutral / contradiction*, which is the relation grounding
actually cares about -- "does the source support this sentence?" -- rather
than a proxy for it.

WHAT THIS IS NOT
----------------
NLI is not a correctness oracle:

* **Numbers.** Better than the folklore, worse than sufficient. Measured on
  ``cross-encoder/nli-deberta-v3-base`` against 13 fabricated figures, the
  model caught 12 -- including off-by-one, transposed digits and unit swaps.
  It confidently *entailed* the thirteenth (``23 days`` against a source
  saying ``24``) at p=0.94. One confident miss is still a fabricated figure
  served to a user, and the lexical numeric check costs nothing on the twelve
  it already catches, so the check stays. See
  ``tests/security/test_nli_numeric_behaviour.py``, which measures this rather
  than assuming it.
* **Arithmetic.** It cannot do any. A claim correctly derived from the source
  ("12 days every six months" from "two days per month") is scored as a
  contradiction, and so is a wrong one. Both signals fail here, so a correct
  derived figure is penalised -- a real false-refusal cost, recorded rather
  than hidden.
* **Long premises.** Entailment quality degrades as the premise grows, which
  is why this module scores against selected sentences rather than the whole
  concatenated context.

So this does not *replace* the lexical checker.  ``grounding.py`` combines
them so that each signal vetoes only in the region where it is trustworthy --
see ``combine_signals`` there.  The honest summary is that NLI raises recall on
paraphrase and adds real contradiction detection, while the lexical numeric
check remains a cheap backstop for the cases it gets confidently wrong.

COST
----
This is an optional dependency (``sentence-transformers``, which pulls torch).
When it is absent or the model cannot be loaded, :func:`get_nli_verifier`
returns ``None`` and grounding falls back to lexical scoring with the
degradation recorded in the report's ``method`` field.  A guardrail that
silently changed strength based on what happened to be installed would make
every measured number unattributable.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from app.core.config import settings
from app.core.logging import get_logger
from app.rag.ingestion.chunker import split_sentences
from app.rag.retrieval.keyword import tokenize

logger = get_logger("app.security.output.nli")


@dataclass
class EntailmentResult:
    """The NLI verdict for one claim against its best-matching premise."""

    entailment: float
    contradiction: float
    neutral: float
    premise: str = ""

    @property
    def label(self) -> str:
        pairs = (
            ("entailment", self.entailment),
            ("contradiction", self.contradiction),
            ("neutral", self.neutral),
        )
        return max(pairs, key=lambda item: item[1])[0]


@dataclass
class NLIUnavailable:
    """Why the verifier could not be constructed. Recorded, never swallowed."""

    reason: str
    detail: str = ""


# ----------------------------------------------------------------------
# Premise selection
# ----------------------------------------------------------------------
#
# A cross-encoder must be given a premise, and the choice matters more than
# the model does. Feeding the whole context as one premise degrades accuracy
# and blows up the sequence length; feeding single sentences loses claims that
# are supported across a sentence boundary. So candidates are built at two
# granularities and the model picks the winner by taking the maximum.


def build_premises(context: str, max_window: int = 2) -> list[str]:
    """Candidate premises: every sentence, plus every adjacent window."""
    sentences = [s.strip() for s in split_sentences(context) if s.strip()]
    if not sentences:
        return []

    premises: list[str] = list(sentences)
    for size in range(2, max_window + 1):
        for start in range(len(sentences) - size + 1):
            premises.append(" ".join(sentences[start : start + size]))
    return premises


def select_premises(claim: str, premises: list[str], limit: int) -> list[str]:
    """Pre-filter premises lexically before spending cross-encoder time.

    Scoring every claim against every premise is quadratic and, on CPU, the
    dominant cost of the whole request.  Lexical overlap is a cheap and
    high-recall *ranker* even though it is a poor *judge*: the premise that
    entails a claim nearly always shares some vocabulary with it.  Using it to
    shortlist -- and letting the cross-encoder decide among the shortlist --
    keeps the accurate model where it matters.

    A claim sharing no vocabulary with any premise still gets the highest-
    ranked candidates rather than none, so an entirely-paraphrased claim is
    still judged by the model rather than being failed by the pre-filter.
    """
    if not premises:
        return []

    claim_tokens = set(tokenize(claim))
    if not claim_tokens:
        return premises[:limit]

    scored = []
    for premise in premises:
        premise_tokens = set(tokenize(premise))
        if not premise_tokens:
            continue
        overlap = len(claim_tokens & premise_tokens) / len(claim_tokens)
        # Mild preference for the shorter premise at equal overlap: NLI
        # accuracy falls off with premise length, so the tightest premise that
        # covers the claim is the best one to ask about.
        scored.append((overlap, -len(premise), premise))

    scored.sort(reverse=True)
    return [premise for _, _, premise in scored[:limit]]


# ----------------------------------------------------------------------
# The verifier
# ----------------------------------------------------------------------


class CrossEncoderNLIVerifier:
    """Entailment scoring via a sentence-transformers ``CrossEncoder``.

    Loaded once per process and guarded by a lock: the model is several
    hundred megabytes and two concurrent requests must not each build one.
    """

    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name or settings.NLI_MODEL
        self._model = CrossEncoder(self.model_name, max_length=settings.NLI_MAX_LENGTH)
        self._label_order = self._resolve_label_order()

    def _resolve_label_order(self) -> dict[str, int]:
        """Map entailment/neutral/contradiction to output column indices.

        Different NLI checkpoints order their labels differently -- the
        ``cross-encoder/nli-*`` family uses contradiction/entailment/neutral
        while most MNLI ports use contradiction/neutral/entailment.  Reading
        ``id2label`` from the checkpoint instead of assuming an order is the
        difference between a working verifier and one that silently reports
        entailment scores as contradictions.
        """
        id2label: dict = {}
        for holder in (self._model, getattr(self._model, "model", None)):
            config = getattr(holder, "config", None)
            candidate = getattr(config, "id2label", None)
            if candidate:
                id2label = dict(candidate)
                break

        order: dict[str, int] = {}
        for index, label in id2label.items():
            key = str(label).strip().lower()
            for wanted in ("entailment", "neutral", "contradiction"):
                if key.startswith(wanted[:5]):
                    order[wanted] = int(index)

        if set(order) != {"entailment", "neutral", "contradiction"}:
            raise ValueError(
                f"{self.model_name!r} does not expose a 3-way NLI label map "
                f"(id2label={id2label!r})"
            )
        return order

    def entails(self, pairs: list[tuple[str, str]]) -> list[EntailmentResult]:
        """Score ``(premise, hypothesis)`` pairs in a single batch."""
        if not pairs:
            return []

        raw = self._model.predict(
            pairs, batch_size=settings.NLI_BATCH_SIZE, show_progress_bar=False
        )
        results: list[EntailmentResult] = []
        for (premise, _hypothesis), row in zip(pairs, raw, strict=True):
            probabilities = _softmax([float(v) for v in row])
            results.append(
                EntailmentResult(
                    entailment=probabilities[self._label_order["entailment"]],
                    contradiction=probabilities[self._label_order["contradiction"]],
                    neutral=probabilities[self._label_order["neutral"]],
                    premise=premise,
                )
            )
        return results

    def verify_claims(
        self, claims: list[str], context: str
    ) -> dict[str, EntailmentResult]:
        """Best entailment result for each claim against the context.

        Every (claim, premise) pair across every claim goes through the model
        in **one** batch. Per-claim calls would pay the framework's fixed
        overhead once per sentence, which on CPU dominates the actual compute.
        """
        premises = build_premises(context)
        if not premises or not claims:
            return {}

        pairs: list[tuple[str, str]] = []
        spans: dict[str, tuple[int, int]] = {}
        for claim in claims:
            if claim in spans:
                continue
            selected = select_premises(claim, premises, settings.NLI_TOP_PREMISES)
            if not selected:
                continue
            start = len(pairs)
            pairs.extend((premise, claim) for premise in selected)
            spans[claim] = (start, len(pairs))

        scored = self.entails(pairs)

        best: dict[str, EntailmentResult] = {}
        for claim, (start, end) in spans.items():
            window = scored[start:end]
            if not window:
                continue
            # The supporting premise is the one that entails most strongly.
            # Contradiction is then read from *that same* premise rather than
            # from the most-contradicting one anywhere in the document: a
            # corpus almost always contains some sentence contradicting any
            # given claim, and scoring against it would fail every answer.
            best[claim] = max(window, key=lambda r: r.entailment)
        return best


def _softmax(values: list[float]) -> list[float]:
    """Numerically stable softmax over raw logits.

    ``CrossEncoder.predict`` returns logits for multi-class checkpoints, so the
    columns are not comparable as probabilities until they are normalised.
    Subtracting the max before exponentiating avoids overflow on the large
    logits these models produce.
    """
    if not values:
        return []
    import math

    largest = max(values)
    exponentials = [math.exp(v - largest) for v in values]
    total = sum(exponentials) or 1.0
    return [e / total for e in exponentials]


# ----------------------------------------------------------------------
# Process-wide accessor
# ----------------------------------------------------------------------

_verifier: CrossEncoderNLIVerifier | None = None
_unavailable: NLIUnavailable | None = None
_lock = threading.Lock()


def get_nli_verifier() -> CrossEncoderNLIVerifier | None:
    """Return the shared verifier, or ``None`` if NLI is off or unavailable.

    Failure to load is cached alongside success. Retrying a missing dependency
    on every request would add a multi-second import attempt to each answer
    for a condition that cannot change without a restart.
    """
    global _verifier, _unavailable

    if settings.GROUNDING_METHOD == "lexical":
        return None
    if _verifier is not None:
        return _verifier
    if _unavailable is not None:
        return None

    with _lock:
        if _verifier is not None:
            return _verifier
        if _unavailable is not None:
            return None
        try:
            _verifier = CrossEncoderNLIVerifier()
            logger.info(
                "nli_verifier_ready",
                extra={"model": _verifier.model_name},
            )
            return _verifier
        except ImportError as exc:
            _unavailable = NLIUnavailable("dependency_missing", str(exc)[:200])
        except Exception as exc:  # model download failure, bad checkpoint, OOM
            _unavailable = NLIUnavailable(type(exc).__name__, str(exc)[:200])

    logger.warning(
        "nli_verifier_unavailable",
        extra={
            "reason": _unavailable.reason,
            "grounding_method": settings.GROUNDING_METHOD,
        },
    )
    return None


def nli_status() -> dict[str, object]:
    """Whether NLI is active, for the report's configuration block."""
    verifier = get_nli_verifier()
    return {
        "requested": settings.GROUNDING_METHOD,
        "active": verifier is not None,
        "model": verifier.model_name if verifier else None,
        "unavailable_reason": _unavailable.reason if _unavailable else None,
    }


def reset_nli_verifier_cache() -> None:
    """Drop the cached verifier (used by tests that switch configuration)."""
    global _verifier, _unavailable
    with _lock:
        _verifier = None
        _unavailable = None
