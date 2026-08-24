"""NLI-backed grounding verification.

Most of these run without the model. That is deliberate: the parts most likely
to break silently are the *plumbing* -- label ordering, premise selection,
signal combination, and the fallback when the dependency is absent -- and all
of them can be tested with a stub verifier. The tests that need real weights
are marked ``requires_nli`` and skip cleanly when they are not installed.
"""

from __future__ import annotations

import pytest

from app.core.config import settings
from app.rag.retrieval.types import ScoredChunk
from app.security.output import nli as nli_module
from app.security.output.grounding import (
    CLAIM_SUPPORT_FLOOR,
    NUMERIC_MISMATCH_CEILING,
    ClaimScore,
    combine_signals,
    verify_grounding,
)
from app.security.output.nli import (
    EntailmentResult,
    _softmax,
    build_premises,
    get_nli_verifier,
    nli_status,
    reset_nli_verifier_cache,
)

pytestmark = pytest.mark.security


CONTEXT = (
    "Full-time employees accrue two days of paid annual leave per month, for a "
    "total of 24 days per calendar year. Unused leave may be carried forward "
    "to the next calendar year up to a maximum of 10 days."
)


def make_chunk(content: str = CONTEXT) -> ScoredChunk:
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
# Premise construction and selection
# ----------------------------------------------------------------------


def test_premises_include_sentences_and_adjacent_windows():
    """A claim supported across a sentence boundary needs the pair as premise."""
    premises = build_premises(CONTEXT)

    assert any("24 days per calendar year" in p for p in premises)
    # The two-sentence window must exist, or a claim combining accrual and
    # carry-forward could never be entailed by any single candidate.
    assert any("24 days" in p and "carried forward" in p for p in premises)


def test_premise_selection_prefers_overlap_then_brevity():
    premises = [
        "Unrelated text about vendor delivery times.",
        "Employees get 24 days of annual leave.",
        "Employees get 24 days of annual leave. " + "Padding sentence. " * 20,
    ]
    chosen = nli_module.select_premises("How many days of annual leave?", premises, 2)

    assert chosen
    # Equal overlap: the shorter premise wins, because NLI accuracy falls off
    # as the premise grows.
    assert chosen[0] == "Employees get 24 days of annual leave."


def test_a_claim_sharing_no_vocabulary_still_gets_premises():
    """The lexical pre-filter must never starve the model of candidates."""
    chosen = nli_module.select_premises("zzzz qqqq", build_premises(CONTEXT), 3)
    assert len(chosen) == 3


def test_empty_context_yields_no_premises():
    assert build_premises("") == []
    assert nli_module.select_premises("anything", [], 4) == []


# ----------------------------------------------------------------------
# Softmax
# ----------------------------------------------------------------------


def test_softmax_normalises_logits():
    probabilities = _softmax([2.0, 1.0, 0.1])
    assert sum(probabilities) == pytest.approx(1.0)
    assert probabilities[0] > probabilities[1] > probabilities[2]


def test_softmax_survives_large_logits():
    """Without the max-subtraction this overflows to inf/inf = nan."""
    probabilities = _softmax([1000.0, 999.0, -1000.0])
    assert sum(probabilities) == pytest.approx(1.0)
    assert all(p == p for p in probabilities)  # no NaN


def test_softmax_of_empty_is_empty():
    assert _softmax([]) == []


# ----------------------------------------------------------------------
# Signal combination -- the security-relevant logic
# ----------------------------------------------------------------------


def _claim(**overrides) -> ClaimScore:
    base = {
        "sentence": "Employees receive 24 days of leave.",
        "score": 0.8,
        "overlap": 0.9,
        "numeric": 1.0,
        "ngram": 0.7,
        "lexical_score": 0.8,
    }
    base.update(overrides)
    return ClaimScore(**base)


def test_entailment_rescues_a_paraphrase_the_lexical_check_penalised(monkeypatch):
    """The failure NLI was added to fix: correct answer, different words."""
    monkeypatch.setattr(settings, "GROUNDING_METHOD", "hybrid")
    claim = _claim(score=0.30, lexical_score=0.30, overlap=0.2, ngram=0.0)

    combine_signals(
        claim, EntailmentResult(entailment=0.95, contradiction=0.02, neutral=0.03)
    )

    assert claim.score >= CLAIM_SUPPORT_FLOOR
    assert claim.supported
    assert claim.entailment == 0.95


def test_a_confident_contradiction_is_penalised_even_at_high_overlap(monkeypatch):
    """The other failure NLI was added to fix: fluent negation of the source."""
    monkeypatch.setattr(settings, "GROUNDING_METHOD", "hybrid")
    claim = _claim(score=0.85, lexical_score=0.85, contradicts=False)

    combine_signals(
        claim, EntailmentResult(entailment=0.05, contradiction=0.90, neutral=0.05)
    )

    assert claim.contradicts
    assert not claim.supported


def test_the_numeric_gate_outranks_a_confident_model(monkeypatch):
    """The core of the design: NLI never gets the final word on numbers.

    MNLI-trained models routinely call a wrong figure "entailed". If NLI could
    override the numeric check, the worst hallucination class in document QA
    would be re-opened by the very feature meant to reduce hallucination.
    """
    monkeypatch.setattr(settings, "GROUNDING_METHOD", "hybrid")
    claim = _claim(numeric=0.0, sentence="Employees receive 28 days of leave.")

    combine_signals(
        claim, EntailmentResult(entailment=0.99, contradiction=0.0, neutral=0.01)
    )

    assert claim.score <= NUMERIC_MISMATCH_CEILING


def test_nli_mode_ignores_lexical_support_but_hybrid_keeps_it(monkeypatch):
    verdict = EntailmentResult(entailment=0.10, contradiction=0.05, neutral=0.85)

    monkeypatch.setattr(settings, "GROUNDING_METHOD", "nli")
    pure = combine_signals(_claim(score=0.90, lexical_score=0.90), verdict)

    monkeypatch.setattr(settings, "GROUNDING_METHOD", "hybrid")
    hybrid = combine_signals(_claim(score=0.90, lexical_score=0.90), verdict)

    assert pure.score == pytest.approx(0.10, abs=1e-4)
    assert hybrid.score == pytest.approx(0.90, abs=1e-4)


def test_a_lexical_contradiction_survives_a_confident_entailment(monkeypatch):
    """Contradiction is a veto from *either* signal, not a vote."""
    monkeypatch.setattr(settings, "GROUNDING_METHOD", "hybrid")
    claim = _claim(contradicts=True)

    combine_signals(
        claim, EntailmentResult(entailment=0.99, contradiction=0.01, neutral=0.0)
    )

    assert claim.contradicts
    assert claim.score == pytest.approx(0.99 * 0.4, abs=1e-3)


def test_hedges_are_left_alone(monkeypatch):
    monkeypatch.setattr(settings, "GROUNDING_METHOD", "hybrid")
    hedge = ClaimScore("I could not find that.", 1.0, 1.0, 1.0, 1.0, is_hedge=True)

    combine_signals(
        hedge, EntailmentResult(entailment=0.0, contradiction=0.9, neutral=0.1)
    )

    assert hedge.score == 1.0
    assert hedge.entailment is None


# ----------------------------------------------------------------------
# Degradation
# ----------------------------------------------------------------------


def test_lexical_mode_never_constructs_a_verifier(monkeypatch):
    monkeypatch.setattr(settings, "GROUNDING_METHOD", "lexical")
    assert get_nli_verifier() is None


def test_a_missing_dependency_degrades_to_lexical_and_says_so(monkeypatch):
    monkeypatch.setattr(settings, "GROUNDING_METHOD", "nli")
    monkeypatch.setattr(
        nli_module,
        "CrossEncoderNLIVerifier",
        lambda *a, **k: (_ for _ in ()).throw(ImportError("no sentence-transformers")),
    )

    report = verify_grounding("Employees receive 24 days of leave [1].", [make_chunk()])

    # The answer is still scored -- lexically -- and the substitution is
    # disclosed rather than silently changing what the number means.
    assert report.method == "lexical_v1"
    assert any("unavailable" in note for note in report.notes)
    assert nli_status()["active"] is False


def test_an_inference_error_falls_back_without_failing_the_request(monkeypatch):
    class Exploding:
        model_name = "stub"

        def verify_claims(self, claims, context):
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(settings, "GROUNDING_METHOD", "hybrid")
    monkeypatch.setattr(nli_module, "get_nli_verifier", lambda: Exploding())
    monkeypatch.setattr(
        "app.security.output.grounding.get_nli_verifier", lambda: Exploding()
    )

    report = verify_grounding("Employees receive 24 days of leave [1].", [make_chunk()])

    assert report.method == "lexical_v1"
    assert any("nli error" in note for note in report.notes)
    assert report.score > 0


def test_the_method_field_records_what_ran_not_what_was_configured(monkeypatch):
    class Stub:
        model_name = "stub"

        def verify_claims(self, claims, context):
            return {
                claim: EntailmentResult(entailment=0.9, contradiction=0.02, neutral=0.08)
                for claim in claims
            }

    monkeypatch.setattr(settings, "GROUNDING_METHOD", "hybrid")
    monkeypatch.setattr("app.security.output.grounding.get_nli_verifier", lambda: Stub())

    report = verify_grounding("Employees receive 24 days of leave [1].", [make_chunk()])
    assert report.method == "hybrid_v1"

    monkeypatch.setattr(settings, "GROUNDING_METHOD", "nli")
    report = verify_grounding("Employees receive 24 days of leave [1].", [make_chunk()])
    assert report.method == "nli_v1"


def test_a_refusal_short_circuits_before_the_model_is_called(monkeypatch):
    """A refusal must not pay for inference: it asserts nothing to verify."""
    called = False

    class Tracking:
        model_name = "stub"

        def verify_claims(self, claims, context):
            nonlocal called
            called = True
            return {}

    monkeypatch.setattr(settings, "GROUNDING_METHOD", "hybrid")
    monkeypatch.setattr(
        "app.security.output.grounding.get_nli_verifier", lambda: Tracking()
    )

    report = verify_grounding("I could not find that in your documents.", [make_chunk()])

    assert report.score == 1.0
    assert called is False


# ----------------------------------------------------------------------
# With real weights
# ----------------------------------------------------------------------


def _nli_reason() -> str | None:
    """Why the real-weights tests cannot run here, or None if they can.

    Checking the *cache* rather than just the import matters: without it these
    tests would silently trigger a ~750 MB download on a cold machine, turning
    a "skipped" test into a multi-minute one and making CI wall time depend on
    network weather.
    """
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return "sentence-transformers is not installed"

    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:  # pragma: no cover - ships with sentence-transformers
        return "huggingface_hub is not installed"

    # The *weights*, not config.json: the small files land first, so checking
    # the config reports "cached" while the 750 MB download is still running --
    # which is exactly the hang this guard exists to prevent.
    weights = ("model.safetensors", "pytorch_model.bin")
    if not any(
        isinstance(try_to_load_from_cache(settings.NLI_MODEL, name), str)
        for name in weights
    ):
        return (
            f"{settings.NLI_MODEL} weights are not in the local cache "
            "(load the model once to download them)"
        )
    return None


requires_nli = pytest.mark.skipif(_nli_reason() is not None, reason=_nli_reason() or "")


@pytest.mark.requires_nli
@requires_nli
def test_the_real_model_separates_entailment_from_contradiction(monkeypatch):
    monkeypatch.setattr(settings, "GROUNDING_METHOD", "hybrid")
    verifier = get_nli_verifier()
    if verifier is None:
        pytest.skip("NLI checkpoint could not be loaded")

    # The premise is the corpus sentence verbatim, not a hand-written summary
    # of it. That is not cosmetic: `build_premises` only ever yields spans of
    # real documents, and the first draft of this test used a paraphrase
    # ("Employees may carry forward up to 10 days of unused leave.") that the
    # model labels *contradiction* at p=0.99 against the same hypothesis. A
    # test built on inputs the system never produces measures the wrong thing
    # -- and the model's fallibility is characterised in
    # test_nli_numeric_behaviour.py rather than papered over here.
    premise = (
        "Unused leave may be carried forward to the next calendar year up to a "
        "maximum of 10 days."
    )
    entailed, contradicted = verifier.entails(
        [
            (premise, "Unused leave can be rolled over to the following year."),
            (premise, "Employees may not carry forward any unused leave."),
        ]
    )

    assert entailed.label == "entailment"
    assert contradicted.contradiction > contradicted.entailment


@pytest.mark.requires_nli
@requires_nli
def test_the_label_map_is_read_from_the_checkpoint(monkeypatch):
    """Guards the bug this would otherwise hide: transposed label columns.

    A checkpoint ordering its labels differently from the assumed order would
    make the verifier report contradictions as entailments -- inverting the
    guardrail while every test that only checked "a number came back" passed.
    """
    monkeypatch.setattr(settings, "GROUNDING_METHOD", "nli")
    verifier = get_nli_verifier()
    if verifier is None:
        pytest.skip("NLI checkpoint could not be loaded")

    assert set(verifier._label_order) == {"entailment", "neutral", "contradiction"}
    assert sorted(verifier._label_order.values()) == [0, 1, 2]
