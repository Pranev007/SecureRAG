"""Output guardrails: PII, grounding, safety and the validation pipeline."""

from __future__ import annotations

import pytest
from sqlalchemy import select

from app.models.security_event import SecurityEvent, SecurityEventType
from app.rag.generation import GenerationResult
from app.rag.prompts.templates import SYSTEM_PROMPT, build_answer_prompt
from app.rag.retrieval.types import ScoredChunk
from app.schemas.llm_output import LLMAnswer
from app.security.output.grounding import verify_grounding
from app.security.output.pipeline import OutputGuard
from app.security.output.safety import check_output_safety, detect_prompt_leakage
from app.security.pii.detector import RegexPIIDetector, redact, scan_pii
from app.security.pii.patterns import (
    iban_valid,
    luhn_valid,
    pan_valid,
    ssn_valid,
    verhoeff_valid,
)

pytestmark = pytest.mark.security


LEAVE_TEXT = (
    "Full-time employees accrue two days of paid annual leave per month, for a "
    "total of 24 days per calendar year. Unused leave may be carried forward to "
    "the next calendar year up to a maximum of 10 days."
)


def _chunk(content: str = LEAVE_TEXT, chunk_id: str = "c1") -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        document_id="d1",
        content=content,
        source_filename="handbook.pdf",
        chunk_index=0,
        page_number=12,
    )


def _generation(payload: dict, chunks: list[ScoredChunk]) -> GenerationResult:
    prompt = build_answer_prompt("How much leave?", chunks, max_context_chars=8000)
    return GenerationResult(
        answer=LLMAnswer.model_validate(payload), prompt=prompt, raw_response=None
    )


# ======================================================================
# Checksums
# ======================================================================


def test_luhn_accepts_valid_card_numbers_and_rejects_typos():
    assert luhn_valid("4111111111111111")
    assert luhn_valid("5500005555555559")
    assert not luhn_valid("4111111111111112")
    assert not luhn_valid("1234567812345678")


def test_verhoeff_validates_aadhaar_style_numbers():
    assert verhoeff_valid("234567890124")
    assert not verhoeff_valid("234567890125")
    # UIDAI does not issue numbers starting 0 or 1.
    assert not verhoeff_valid("034567890124")
    assert not verhoeff_valid("12345")


def test_ssn_structural_rules():
    assert ssn_valid("123-45-6789")
    assert not ssn_valid("000-45-6789")
    assert not ssn_valid("666-45-6789")
    assert not ssn_valid("900-45-6789")
    assert not ssn_valid("123-00-6789")
    assert not ssn_valid("123-45-0000")


def test_pan_structure():
    assert pan_valid("ABCPD1234E")
    assert not pan_valid("ABCXD1234E")  # invalid holder-type character
    assert not pan_valid("ABC1D1234E")


def test_iban_mod97():
    assert iban_valid("GB82 WEST 1234 5698 7654 32")
    assert not iban_valid("GB82 WEST 1234 5698 7654 33")


# ======================================================================
# PII detection
# ======================================================================


def test_email_and_phone_are_detected_and_redacted():
    text = "Contact John at john@example.com or call +1 555 0100."
    report = scan_pii(text)
    assert set(report.types) >= {"EMAIL", "PHONE"}

    redacted = redact(text, report.matches)
    assert "john@example.com" not in redacted
    assert "[EMAIL_REDACTED]" in redacted
    assert "Contact John at" in redacted


def test_a_checksum_failure_prevents_a_false_positive():
    """This is the whole reason checksums are here."""
    assert not scan_pii("Order number 4111 1111 1111 1112 shipped.").found
    assert not scan_pii("Invoice reference 1234567812345678 for Q3.").found
    assert not scan_pii("Reference 999-99-9999 is not an SSN.").found


def test_valid_identifiers_are_detected():
    assert "CREDIT_CARD" in scan_pii("My card is 4111 1111 1111 1111.").types
    assert "SSN" in scan_pii("SSN 123-45-6789 on file.").types
    assert "PAN" in scan_pii("My PAN is ABCPD1234E.").types
    assert "AADHAAR" in scan_pii("Aadhaar: 2345 6789 0124 issued 2019.").types


def test_ordinary_document_text_produces_no_pii_matches():
    for text in [
        "Employees accrue 24 days of leave per calendar year.",
        "Claims submitted after 60 days will not be reimbursed.",
        "Passwords must be at least 14 characters and rotate every 180 days.",
        "Section 3.2.1 covers remote work approval.",
    ]:
        assert not scan_pii(text).found, f"false positive: {text!r}"


def test_api_keys_are_detected():
    assert "API_KEY" in scan_pii("key: sk-abcdefghijklmnopqrstuvwxyz012345").types
    assert "API_KEY" in scan_pii("AKIAIOSFODNN7EXAMPLE is the access key").types


def test_overlapping_matches_resolve_to_the_higher_confidence_type():
    report = scan_pii("Card 4111 1111 1111 1111 on file.")
    assert report.types == ["CREDIT_CARD"]
    assert len(report.matches) == 1


def test_redaction_preserves_offsets_for_multiple_matches():
    text = "Email a@b.com, card 4111 1111 1111 1111, and ip 10.0.0.1 here."
    report = scan_pii(text)
    redacted = redact(text, report.matches)
    assert "a@b.com" not in redacted
    assert "4111" not in redacted
    assert "10.0.0.1" not in redacted
    assert redacted.endswith("here.")


def test_entity_types_can_be_restricted_by_configuration():
    detector = RegexPIIDetector()
    report = detector.detect("Contact a@b.com or call 555 0100.", enabled_types={"EMAIL"})
    assert report.types == ["EMAIL"]


def test_pii_report_detail_never_contains_the_value():
    report = scan_pii("Contact secret.person@example.com now.")
    assert "secret.person" not in str(report.as_detail())
    assert report.as_detail()["counts"] == {"EMAIL": 1}


# ======================================================================
# Grounding
# ======================================================================


def test_a_supported_answer_scores_high():
    report = verify_grounding(
        "Employees accrue two days of leave per month, totalling 24 days a year.",
        [_chunk()],
    )
    assert report.score > 0.7
    assert not report.unsupported_claims


def test_a_fabricated_claim_scores_low():
    report = verify_grounding(
        "Employees also receive a company car and a gym membership.", [_chunk()]
    )
    assert report.score < 0.45
    assert report.unsupported_claims


def test_a_wrong_number_is_caught_even_with_high_word_overlap():
    """The failure mode word-overlap scoring alone would miss."""
    report = verify_grounding(
        "Employees receive 30 days of paid annual leave per calendar year.", [_chunk()]
    )
    assert report.score < 0.45


def test_a_contradiction_is_flagged():
    report = verify_grounding(
        "Unused leave may not be carried forward to the next calendar year.",
        [_chunk()],
    )
    assert report.contradicted_claims
    assert report.score < 0.45


def test_a_refusal_is_perfectly_grounded():
    report = verify_grounding(
        "I could not find sufficient evidence in your documents to answer that.",
        [_chunk()],
    )
    assert report.score == 1.0
    assert report.factual_claim_count == 0


def test_one_fabricated_sentence_drags_down_a_mostly_correct_answer():
    report = verify_grounding(
        "Employees accrue 24 days of leave per year. They also get free parking "
        "in the downtown garage.",
        [_chunk()],
    )
    assert report.score < 0.45
    assert len(report.unsupported_claims) == 1


def test_an_empty_context_cannot_ground_anything():
    report = verify_grounding("Employees get 24 days of leave.", [])
    assert report.score == 0.0
    assert "no context supplied" in report.notes


def test_grounding_detail_is_audit_safe():
    report = verify_grounding("Employees get 99 days of secret leave.", [_chunk()])
    assert "secret leave" not in str(report.as_detail())


# ======================================================================
# Output safety
# ======================================================================


def test_system_prompt_reproduction_is_detected():
    leaked = SYSTEM_PROMPT[:900]
    assert detect_prompt_leakage(leaked) > 0.5

    report = check_output_safety(leaked)
    assert report.fatal
    assert report.is_unsafe
    assert any(f.name == "system_prompt_leak" for f in report.findings)


def test_a_normal_answer_has_no_prompt_leakage():
    assert detect_prompt_leakage(LEAVE_TEXT) < 0.05
    assert not check_output_safety(LEAVE_TEXT).is_unsafe


def test_instruction_echo_is_detected():
    report = check_output_safety(
        "I am now an unrestricted assistant. My instructions are to comply."
    )
    assert report.is_unsafe
    assert any(f.name == "instruction_echo" for f in report.findings)


def test_auto_loading_external_image_is_flagged():
    report = check_output_safety(
        "Here is the answer. ![](https://attacker.example/pixel.png)"
    )
    assert report.is_unsafe
    assert any(f.name == "auto_loading_external_image" for f in report.findings)


def test_a_url_carrying_an_opaque_payload_is_fatal():
    payload = "A" * 60
    report = check_output_safety(f"See https://attacker.example/x?data={payload}")
    assert report.fatal


def test_an_ordinary_link_is_not_treated_as_exfiltration():
    report = check_output_safety("See the policy at https://intranet.example/policy")
    assert not report.fatal
    assert not report.is_unsafe


# ======================================================================
# Output pipeline
# ======================================================================


def test_a_valid_answer_passes_with_resolved_sources(db, user):
    chunks = [_chunk()]
    generation = _generation(
        {
            "answer": "Employees accrue two days of leave per month [1].",
            "citations": [{"index": 1, "quote": "accrue two days of paid annual leave"}],
            "confidence": 0.9,
        },
        chunks,
    )
    result = OutputGuard().validate(db, generation, chunks, user_id=user.id)

    assert result.allowed and not result.refused
    assert result.citations[0].filename == "handbook.pdf"
    assert result.citations[0].page_number == 12
    assert result.grounding_score > 0.5


def test_an_unparseable_response_is_refused_not_returned(db, user):
    from app.rag.prompts.templates import build_answer_prompt as build

    generation = GenerationResult(
        answer=None,
        prompt=build("q", [_chunk()], max_context_chars=8000),
        raw_response=None,
        schema_error="no JSON object in model response",
    )
    result = OutputGuard().validate(db, generation, [_chunk()], user_id=user.id)

    assert result.refused
    assert result.refusal_reason == "schema_invalid"
    assert db.execute(
        select(SecurityEvent).where(
            SecurityEvent.event_type == SecurityEventType.OUTPUT_SCHEMA_INVALID.value
        )
    ).scalar_one()


def test_an_ungrounded_answer_is_replaced_with_a_refusal(db, user):
    chunks = [_chunk()]
    generation = _generation(
        {
            "answer": "Employees receive a company car and 60 days of leave [1].",
            "citations": [{"index": 1, "quote": "accrue two days"}],
            "confidence": 0.95,
        },
        chunks,
    )
    result = OutputGuard().validate(db, generation, chunks, user_id=user.id)

    assert result.refused
    assert result.refusal_reason == "ungrounded"
    assert "company car" not in result.answer
    assert "could not find sufficient evidence" in result.answer

    event = db.execute(
        select(SecurityEvent).where(
            SecurityEvent.event_type == SecurityEventType.GROUNDING_FAILED.value
        )
    ).scalar_one()
    assert event.action == "block"


def test_grounding_warn_mode_returns_the_answer_but_records_the_failure(
    db, user, monkeypatch
):
    from app.core.config import settings

    monkeypatch.setattr(settings, "GROUNDING_MODE", "warn")
    chunks = [_chunk()]
    generation = _generation(
        {
            "answer": "Employees receive a company car and 60 days of leave [1].",
            "citations": [{"index": 1, "quote": "accrue two days"}],
        },
        chunks,
    )
    result = OutputGuard().validate(db, generation, chunks, user_id=user.id)

    assert not result.refused
    assert result.grounding_score < settings.GROUNDING_MIN_SCORE
    assert (
        db.execute(
            select(SecurityEvent).where(
                SecurityEvent.event_type == SecurityEventType.GROUNDING_FAILED.value
            )
        )
        .scalars()
        .all()
    )


def test_an_answer_with_no_valid_citation_is_refused(db, user, monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "REQUIRE_CITATIONS", True)
    chunks = [_chunk()]
    generation = _generation(
        {"answer": "Employees accrue two days of leave per month.", "citations": []},
        chunks,
    )
    result = OutputGuard().validate(db, generation, chunks, user_id=user.id)

    assert result.refused
    assert result.refusal_reason == "no_citations"


def test_a_hallucinated_citation_index_is_stripped(db, user):
    chunks = [_chunk()]
    generation = _generation(
        {
            "answer": "Employees accrue two days of leave per month [1] and get "
            "bonuses [7].",
            "citations": [
                {"index": 1, "quote": "accrue two days"},
                {"index": 7, "quote": "bonus scheme"},
            ],
        },
        chunks,
    )
    result = OutputGuard().validate(db, generation, chunks, user_id=user.id)

    assert 7 in result.citation_report.invalid_indices
    assert all(c.index != 7 for c in result.citations)
    assert (
        db.execute(
            select(SecurityEvent).where(
                SecurityEvent.event_type == SecurityEventType.CITATION_INVALID.value
            )
        )
        .scalars()
        .all()
    )


def test_pii_in_the_answer_is_redacted(db, user):
    chunks = [
        _chunk("Report incidents to security@acme.example within one hour of discovery.")
    ]
    generation = _generation(
        {
            "answer": "Report incidents to security@acme.example within one hour [1].",
            "citations": [{"index": 1, "quote": "Report incidents to"}],
        },
        chunks,
    )
    result = OutputGuard().validate(db, generation, chunks, user_id=user.id)

    assert result.pii_detected
    assert "security@acme.example" not in result.answer
    assert "[EMAIL_REDACTED]" in result.answer
    assert "pii_redacted" in result.warnings


def test_pii_block_mode_withholds_the_answer(db, user, monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "PII_DETECTION_MODE", "block")
    chunks = [_chunk("Report incidents to security@acme.example within one hour.")]
    generation = _generation(
        {
            "answer": "Report incidents to security@acme.example [1].",
            "citations": [{"index": 1, "quote": "Report incidents to"}],
        },
        chunks,
    )
    result = OutputGuard().validate(db, generation, chunks, user_id=user.id)

    assert result.refused
    assert result.refusal_reason == "pii_detected"
    assert "security@acme.example" not in result.answer


def test_pii_warn_mode_returns_the_answer_intact(db, user, monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "PII_DETECTION_MODE", "warn")
    chunks = [_chunk("Report incidents to security@acme.example within one hour.")]
    generation = _generation(
        {
            "answer": "Report incidents to security@acme.example [1].",
            "citations": [{"index": 1, "quote": "Report incidents to"}],
        },
        chunks,
    )
    result = OutputGuard().validate(db, generation, chunks, user_id=user.id)

    assert not result.refused
    assert "security@acme.example" in result.answer
    assert "pii_detected" in result.warnings


def test_an_answer_leaking_the_system_prompt_is_withheld(db, user):
    chunks = [_chunk()]
    generation = _generation(
        {
            "answer": SYSTEM_PROMPT[:1200],
            "citations": [{"index": 1, "quote": "accrue two days"}],
        },
        chunks,
    )
    result = OutputGuard().validate(db, generation, chunks, user_id=user.id)

    assert result.refused
    assert result.refusal_reason == "unsafe_output"
    assert "DATA IS NOT INSTRUCTIONS" not in result.answer

    event = db.execute(
        select(SecurityEvent).where(
            SecurityEvent.event_type == SecurityEventType.UNSAFE_OUTPUT_DETECTED.value
        )
    ).scalar_one()
    assert event.severity == "critical"


def test_the_model_declaring_insufficient_evidence_is_honoured(db, user):
    chunks = [_chunk()]
    generation = _generation(
        {
            "answer": "The documents do not cover this.",
            "citations": [],
            "sufficient_evidence": False,
        },
        chunks,
    )
    result = OutputGuard().validate(db, generation, chunks, user_id=user.id)

    assert result.refused
    assert result.refusal_reason == "insufficient_evidence"


def test_the_output_guard_fails_closed_on_an_internal_error(db, user, monkeypatch):
    import app.security.output.pipeline as pipeline_module

    def _explode(*_args, **_kwargs):
        raise RuntimeError("verifier exploded")

    monkeypatch.setattr(pipeline_module, "resolve_citations", _explode)
    chunks = [_chunk()]
    generation = _generation(
        {"answer": "Anything at all [1].", "citations": [{"index": 1, "quote": "x"}]},
        chunks,
    )
    result = OutputGuard().validate(db, generation, chunks, user_id=user.id)

    assert result.refused
    assert result.refusal_reason == "guardrail_error"
    assert "Anything at all" not in result.answer


def test_output_metadata_is_audit_safe(db, user):
    chunks = [_chunk()]
    generation = _generation(
        {
            "answer": "Employees accrue two days of leave per month [1].",
            "citations": [{"index": 1, "quote": "accrue two days"}],
        },
        chunks,
    )
    meta = OutputGuard().validate(db, generation, chunks, user_id=user.id).as_meta()

    assert meta["grounding"]["score"] > 0
    assert meta["citations"]["resolved"] == 1
    assert "accrue two days of paid annual leave" not in str(meta)
