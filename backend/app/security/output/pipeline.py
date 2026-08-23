"""The output guardrail stage.

    schema -> citations -> safety -> grounding -> PII -> final answer

Ordering is deliberate and is the interesting part of this module:

1. **Schema first.**  Everything downstream needs a parsed answer, and a
   response that will not parse is itself a signal that generation went wrong.
2. **Citations before everything else.**  Citation resolution produces the
   sanitised answer text (invalid markers stripped), and the later stages
   should judge the text the user will actually see.
3. **Safety before grounding**, even though grounding is the cheaper check.
   A leaked system prompt is also, incidentally, ungrounded -- so running
   grounding first would catch it, refuse the answer, and record the event as
   a *hallucination*. That is the correct outcome for the user and the wrong
   one for the operator: an extraction attempt that succeeded must be recorded
   as ``unsafe_output`` at CRITICAL, not filed under grounding noise. When two
   checks would both fire, the more specific and more severe one has to run
   first or its signal is lost.
4. **PII last.**  Redaction rewrites the text, and rewriting before the
   grounding check would score a string full of ``[EMAIL_REDACTED]`` against a
   context containing real addresses -- lowering the grounding score because of
   the guardrail's own edits.

Every failure path returns a *replacement answer*, never a partially-validated
one.  There is no code path that returns raw model output.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import get_logger
from app.models.security_event import (
    SecurityAction,
    SecurityEventType,
    SecurityLayer,
    SecuritySeverity,
)
from app.rag.generation import GenerationResult
from app.rag.retrieval.types import ScoredChunk
from app.security.output.citations import CitationReport, resolve_citations
from app.security.output.grounding import GroundingReport, verify_grounding
from app.security.output.safety import SafetyReport, check_output_safety
from app.security.pii.detector import PIIReport, redact, scan_pii
from app.services.security_event_service import record_event

logger = get_logger("app.security.output")


@dataclass
class OutputDecision:
    """The final, validated answer plus everything that shaped it."""

    answer: str
    allowed: bool = True
    refused: bool = False
    refusal_reason: str | None = None

    confidence: float = 0.0
    grounding_score: float = 0.0
    citations: list = field(default_factory=list)
    pii_detected: bool = False
    pii_types: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    citation_report: CitationReport | None = None
    grounding_report: GroundingReport | None = None
    safety_report: SafetyReport | None = None
    pii_report: PIIReport | None = None
    latency_ms: float = 0.0

    def as_meta(self) -> dict:
        """Audit-safe summary of the whole output stage."""
        return {
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "grounding": self.grounding_report.as_detail()
            if self.grounding_report
            else None,
            "citations": {
                "resolved": len(self.citations),
                "invalid": len(self.citation_report.invalid_indices)
                if self.citation_report
                else 0,
                "unverified_quotes": len(self.citation_report.unverified_quotes)
                if self.citation_report
                else 0,
            },
            "pii": self.pii_report.as_detail() if self.pii_report else None,
            "safety": self.safety_report.as_detail() if self.safety_report else None,
        }


class OutputGuard:
    def validate(
        self,
        db: Session,
        generation: GenerationResult,
        chunks: list[ScoredChunk],
        *,
        user_id: str,
        client_ref: str | None = None,
    ) -> OutputDecision:
        started = time.perf_counter()
        try:
            decision = self._run(
                db, generation, chunks, user_id=user_id, client_ref=client_ref
            )
        except Exception as exc:
            logger.exception("output_guard_error", extra={"error": type(exc).__name__})
            record_event(
                db,
                event_type=SecurityEventType.GUARDRAIL_ERROR,
                layer=SecurityLayer.OUTPUT,
                severity=SecuritySeverity.HIGH,
                action=SecurityAction.BLOCK,
                user_id=user_id,
                detector="output_guard",
                client_ref=client_ref,
                detail={"error": type(exc).__name__},
            )
            # Fail closed: an unverified answer is never returned.
            decision = self._refuse(
                "guardrail_error",
                "The answer could not be verified and has been withheld.",
            )

        decision.latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return decision

    # ------------------------------------------------------------------
    def _run(
        self,
        db: Session,
        generation: GenerationResult,
        chunks: list[ScoredChunk],
        *,
        user_id: str,
        client_ref: str | None,
    ) -> OutputDecision:
        # --- 1. schema ------------------------------------------------
        if not generation.succeeded or generation.answer is None:
            record_event(
                db,
                event_type=SecurityEventType.OUTPUT_SCHEMA_INVALID,
                layer=SecurityLayer.OUTPUT,
                severity=SecuritySeverity.MEDIUM,
                action=SecurityAction.BLOCK,
                user_id=user_id,
                risk_score=0.5,
                detector="schema_validation",
                client_ref=client_ref,
                detail={"error": generation.schema_error or "unknown"},
            )
            return self._refuse(
                "schema_invalid",
                "The answer could not be produced in a verifiable form. "
                "Please try again.",
            )

        answer = generation.answer

        # An explicit "insufficient evidence" from the model is the desired
        # behaviour, not a failure, and short-circuits the rest of the checks.
        if not answer.sufficient_evidence or not chunks:
            return OutputDecision(
                answer=settings.INSUFFICIENT_EVIDENCE_MESSAGE,
                refused=True,
                refusal_reason="insufficient_evidence",
                confidence=0.0,
                grounding_score=0.0,
                warnings=["no_supporting_documents"],
            )

        # --- 2. citations ---------------------------------------------
        citation_report = resolve_citations(answer, generation.prompt.index_to_chunk)
        text = citation_report.sanitised_answer or answer.answer

        if citation_report.invalid_indices or citation_report.unverified_quotes:
            record_event(
                db,
                event_type=SecurityEventType.CITATION_INVALID,
                layer=SecurityLayer.OUTPUT,
                severity=SecuritySeverity.MEDIUM,
                action=SecurityAction.SANITISE,
                user_id=user_id,
                risk_score=0.4,
                detector="citation_verification",
                client_ref=client_ref,
                detail={
                    "invalid_indices": citation_report.invalid_indices[:10],
                    "unverified_quotes": citation_report.unverified_quotes[:10],
                    "accuracy": citation_report.accuracy,
                },
            )

        if settings.REQUIRE_CITATIONS and not citation_report.has_valid_citations:
            record_event(
                db,
                event_type=SecurityEventType.GROUNDING_FAILED,
                layer=SecurityLayer.OUTPUT,
                severity=SecuritySeverity.MEDIUM,
                action=SecurityAction.BLOCK,
                user_id=user_id,
                risk_score=0.6,
                detector="missing_citations",
                client_ref=client_ref,
                detail={"reason": "answer cited no supplied source"},
            )
            return self._refuse(
                "no_citations",
                settings.INSUFFICIENT_EVIDENCE_MESSAGE,
                citation_report=citation_report,
            )

        # --- 3. safety ------------------------------------------------
        safety_report = SafetyReport()
        if settings.OUTPUT_SAFETY_ENABLED:
            safety_report = check_output_safety(text)
            if safety_report.is_unsafe:
                record_event(
                    db,
                    event_type=SecurityEventType.UNSAFE_OUTPUT_DETECTED,
                    layer=SecurityLayer.OUTPUT,
                    severity=SecuritySeverity.CRITICAL
                    if safety_report.fatal
                    else SecuritySeverity.HIGH,
                    action=SecurityAction.BLOCK,
                    user_id=user_id,
                    risk_score=safety_report.risk_score,
                    detector="output_safety",
                    client_ref=client_ref,
                    detail=safety_report.as_detail(),
                )
                return self._refuse(
                    "unsafe_output",
                    "The generated answer failed a safety check and has been "
                    "withheld.",
                    citation_report=citation_report,
                    safety_report=safety_report,
                )

        # --- 4. grounding ---------------------------------------------
        grounding_report = GroundingReport(score=1.0, method="disabled")
        if settings.GROUNDING_ENABLED:
            grounding_report = verify_grounding(text, chunks)
            if grounding_report.score < settings.GROUNDING_MIN_SCORE:
                record_event(
                    db,
                    event_type=SecurityEventType.GROUNDING_FAILED,
                    layer=SecurityLayer.OUTPUT,
                    severity=SecuritySeverity.MEDIUM,
                    action=SecurityAction.BLOCK
                    if settings.GROUNDING_MODE == "block"
                    else SecurityAction.FLAG,
                    user_id=user_id,
                    risk_score=round(1.0 - grounding_report.score, 4),
                    detector="grounding_verification",
                    client_ref=client_ref,
                    detail=grounding_report.as_detail(),
                )
                if settings.GROUNDING_MODE == "block":
                    return self._refuse(
                        "ungrounded",
                        settings.INSUFFICIENT_EVIDENCE_MESSAGE,
                        citation_report=citation_report,
                        grounding_report=grounding_report,
                        safety_report=safety_report,
                    )

        # --- 5. PII ---------------------------------------------------
        warnings: list[str] = []
        pii_report = PIIReport()
        if settings.PII_DETECTION_MODE != "off":
            pii_report = scan_pii(text)
            if pii_report.found:
                record_event(
                    db,
                    event_type=SecurityEventType.OUTPUT_PII_DETECTED,
                    layer=SecurityLayer.OUTPUT,
                    severity=SecuritySeverity.MEDIUM,
                    action={
                        "warn": SecurityAction.FLAG,
                        "redact": SecurityAction.REDACT,
                        "block": SecurityAction.BLOCK,
                    }[settings.PII_DETECTION_MODE],
                    user_id=user_id,
                    risk_score=0.5,
                    detector=f"pii:{pii_report.engine}",
                    client_ref=client_ref,
                    detail=pii_report.as_detail(),
                )

                if settings.PII_DETECTION_MODE == "block":
                    return self._refuse(
                        "pii_detected",
                        "The answer contained personal data and has been withheld.",
                        citation_report=citation_report,
                        grounding_report=grounding_report,
                        pii_report=pii_report,
                    )
                if settings.PII_DETECTION_MODE == "redact":
                    text = redact(text, pii_report.matches)
                    pii_report.redacted_text = text
                    warnings.append("pii_redacted")
                else:
                    warnings.append("pii_detected")

        if answer.observed_injection_attempt:
            warnings.append("injection_attempt_in_sources")

        return OutputDecision(
            answer=text,
            allowed=True,
            confidence=answer.confidence,
            grounding_score=grounding_report.score,
            citations=citation_report.citations,
            pii_detected=pii_report.found,
            pii_types=pii_report.types,
            warnings=warnings,
            citation_report=citation_report,
            grounding_report=grounding_report,
            safety_report=safety_report,
            pii_report=pii_report,
        )

    @staticmethod
    def _refuse(
        reason: str,
        message: str,
        *,
        citation_report: CitationReport | None = None,
        grounding_report: GroundingReport | None = None,
        safety_report: SafetyReport | None = None,
        pii_report: PIIReport | None = None,
    ) -> OutputDecision:
        return OutputDecision(
            answer=message,
            allowed=False,
            refused=True,
            refusal_reason=reason,
            confidence=0.0,
            grounding_score=grounding_report.score if grounding_report else 0.0,
            citation_report=citation_report,
            grounding_report=grounding_report,
            safety_report=safety_report,
            pii_report=pii_report,
        )


_guard: OutputGuard | None = None


def get_output_guard() -> OutputGuard:
    global _guard
    if _guard is None:
        _guard = OutputGuard()
    return _guard
