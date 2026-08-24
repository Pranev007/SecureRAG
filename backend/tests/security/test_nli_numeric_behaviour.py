"""Why the lexical numeric gate survives the arrival of NLI.

`combine_signals` lets the lexical numeric check overrule the cross-encoder on
any claim containing a figure. That is a real cost -- it caps otherwise-valid
answers -- so the justification has to be measured rather than asserted, and it
has to be re-measurable when the checkpoint changes.

These tests run against real weights and are skipped when the checkpoint is not
cached. They fall into two groups:

* **Characterisation** -- what the NLI model actually does on adversarial
  numeric pairs. Deliberately *not* asserted pair-by-pair: a better checkpoint
  should not fail the suite, and pinning a model's mistakes as expectations is
  how a test suite becomes a museum.
* **The invariant** -- what the *combined* verifier must guarantee no matter
  what the model says. This is the part that is allowed to fail the build.

Measured on ``cross-encoder/nli-deberta-v3-base`` (2026-08-24): of the 13
fabricated figures below, NLI alone entailed **one** -- `23 days` against a
source saying `24`, at p=0.94. Twelve were caught, so the model is far better
at numbers than its reputation suggests; one confident miss is still one
fabricated figure served to a user, and the gate costs nothing on the other
twelve because they were already going to fail.

That single case also exposed a defect in the *combination*, not the model:
with the ceiling set independently at 0.50 and the support floor at 0.45, the
cap left the claim above the floor and it passed. Lexical scoring alone had
given it 0.42 and blocked it -- so adding the stronger model made that case
strictly worse until the ceiling was tied to the floor.
"""

from __future__ import annotations

import pytest

from app.core.config import settings
from app.rag.retrieval.types import ScoredChunk
from app.security.output.grounding import (
    CLAIM_SUPPORT_FLOOR,
    NUMERIC_MISMATCH_CEILING,
    verify_grounding,
)
from app.security.output.nli import get_nli_verifier, reset_nli_verifier_cache

# The inequality the whole gate rests on. Asserted at import so that editing
# either constant fails loudly here rather than quietly letting figures through.
assert NUMERIC_MISMATCH_CEILING < CLAIM_SUPPORT_FLOOR


def _nli_reason() -> str | None:
    try:
        import sentence_transformers  # noqa: F401
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return "sentence-transformers is not installed"

    if not any(
        isinstance(try_to_load_from_cache(settings.NLI_MODEL, name), str)
        for name in ("model.safetensors", "pytorch_model.bin")
    ):
        return f"{settings.NLI_MODEL} weights are not in the local cache"
    return None


pytestmark = [pytest.mark.security, pytest.mark.requires_nli]
requires_nli = pytest.mark.skipif(_nli_reason() is not None, reason=_nli_reason() or "")

LEAVE = (
    "Full-time employees accrue two days of paid annual leave per month, for a "
    "total of 24 days per calendar year."
)
CARRY = (
    "Unused leave may be carried forward to the next calendar year up to a "
    "maximum of 10 days."
)
EXPENSES = (
    "Expense claims must be submitted within 30 days of the expense being "
    "incurred. Claims submitted after 60 days will not be reimbursed."
)
VENDOR = "Delivery times averaged 4.2 days against a target of 5 days."
PASSWORDS = "Passwords must be at least 14 characters and are rotated every 180 days."

# (source, fabricated claim, what makes it hard)
FABRICATED_FIGURES = [
    (LEAVE, "Employees receive 25 days of paid annual leave per year.", "off-by-one"),
    (LEAVE, "Employees receive 23 days of paid annual leave per year.", "off-by-one"),
    (LEAVE, "Employees accrue three days of leave per month.", "spelled out"),
    (LEAVE, "Employees receive 240 days of paid annual leave per year.", "magnitude"),
    (LEAVE, "Employees receive 24 weeks of paid annual leave per year.", "unit swap"),
    (CARRY, "Up to 12 days of unused leave carries over.", "off-by-two"),
    (CARRY, "Unused leave carries over for up to 18 months.", "fabricated unit"),
    (EXPENSES, "Expense claims must be submitted within 45 days.", "between two real"),
    (EXPENSES, "Claims submitted after 90 days will not be reimbursed.", "off-by-30"),
    (VENDOR, "Delivery times averaged 4.5 days.", "decimal near-miss"),
    (VENDOR, "Delivery times averaged 2.4 days.", "transposed digits"),
    (PASSWORDS, "Passwords rotate every 190 days.", "off-by-ten"),
    (PASSWORDS, "Passwords must be at least 41 characters.", "transposed digits"),
]


def _chunk(content: str) -> ScoredChunk:
    return ScoredChunk(
        chunk_id="c1",
        document_id="d1",
        content=content,
        source_filename="handbook.md",
        chunk_index=0,
    )


@pytest.fixture(autouse=True)
def _clean_verifier_cache():
    reset_nli_verifier_cache()
    yield
    reset_nli_verifier_cache()


# ----------------------------------------------------------------------
# The invariant: this is what is allowed to fail the build
# ----------------------------------------------------------------------


@requires_nli
@pytest.mark.parametrize(
    ("source", "claim", "difficulty"),
    FABRICATED_FIGURES,
    ids=[f"{d}:{c[:28]}" for _, c, d in FABRICATED_FIGURES],
)
def test_no_fabricated_figure_survives_hybrid_grounding(
    monkeypatch, source, claim, difficulty
):
    """Every claim here states a number its source does not.

    Whatever the cross-encoder concludes, the combined verifier must not mark
    the claim supported. This is the guarantee the numeric gate exists to
    provide, asserted against real weights.
    """
    monkeypatch.setattr(settings, "GROUNDING_METHOD", "hybrid")

    report = verify_grounding(claim, [_chunk(source)])

    assert report.method == "hybrid_v1", "NLI did not actually run"
    assert not report.claims[0].supported, difficulty
    assert report.score < settings.GROUNDING_MIN_SCORE, difficulty


@requires_nli
def test_a_correct_figure_is_not_capped(monkeypatch):
    """The gate must not fire on numbers that *are* in the source.

    Without this the previous test could be satisfied by capping everything
    numeric, which would make the guardrail useless rather than strict.
    """
    monkeypatch.setattr(settings, "GROUNDING_METHOD", "hybrid")

    report = verify_grounding(
        "Employees receive 24 days of paid annual leave per calendar year.",
        [_chunk(LEAVE)],
    )

    assert report.score > CLAIM_SUPPORT_FLOOR
    assert report.claims[0].supported


@requires_nli
def test_nli_rescues_a_paraphrase_with_no_numbers(monkeypatch):
    """The gate is scoped to numeric claims and must not suppress the upside."""
    monkeypatch.setattr(settings, "GROUNDING_METHOD", "hybrid")

    report = verify_grounding(
        "Staff may roll unused holiday over into the following year.",
        [_chunk(CARRY)],
    )

    assert report.claims[0].entailment is not None
    assert report.claims[0].supported


# ----------------------------------------------------------------------
# Characterisation: reported, not asserted pair-by-pair
# ----------------------------------------------------------------------


@requires_nli
def test_the_model_is_not_reliable_enough_to_be_trusted_alone(monkeypatch):
    """Records how often NLI alone would have accepted a fabricated figure.

    The threshold is loose on purpose. The claim being defended is "not
    reliable enough to be the sole authority on numbers", not a specific
    accuracy figure, and a checkpoint that improved should not fail the build.
    """
    monkeypatch.setattr(settings, "GROUNDING_METHOD", "nli")
    verifier = get_nli_verifier()
    assert verifier is not None

    verdicts = verifier.entails(
        [(source, claim) for source, claim, _ in FABRICATED_FIGURES]
    )
    wrongly_entailed = [
        (claim, round(v.entailment, 3))
        for (_, claim, _), v in zip(FABRICATED_FIGURES, verdicts, strict=True)
        if v.entailment >= settings.NLI_ENTAILMENT_FLOOR
    ]

    # Not "must be > 0" -- that would pin the model's mistakes as a requirement.
    # The point is that any non-zero count justifies keeping the gate, and the
    # list is printed so a checkpoint change can be re-read rather than guessed.
    print(
        f"\nNLI alone accepted {len(wrongly_entailed)}/{len(FABRICATED_FIGURES)} "
        f"fabricated figures: {wrongly_entailed}"
    )
    assert len(wrongly_entailed) < len(FABRICATED_FIGURES), (
        "the model entailed every fabricated figure -- check the label mapping, "
        "which is the failure that looks like this"
    )
