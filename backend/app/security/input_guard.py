"""The input guardrail stage.

    normalise -> validate -> detect injection -> decide -> audit

One entry point, :meth:`InputGuard.check`, so that there is exactly one place
where an input can be admitted to the RAG pipeline.  Callers get a
:class:`InputDecision`; they never see the detector's internals, and the
end-user response never varies with which rule fired (SECURITY PRINCIPLE 8).

Fail-closed behaviour: if a guardrail raises an unexpected exception and
``FAIL_CLOSED`` is set, the request is blocked rather than admitted unchecked.
A guardrail that silently disables itself on error is worse than no guardrail,
because it creates the *appearance* of protection.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.exceptions import SecurityBlockError
from app.core.logging import get_logger, redact_for_log
from app.models.security_event import (
    SecurityAction,
    SecurityEventType,
    SecurityLayer,
    SecuritySeverity,
)
from app.security.injection.detector import (
    InjectionAction,
    InjectionDetector,
    InjectionVerdict,
    get_injection_detector,
)
from app.security.input_validation import (
    ValidationFailure,
    ValidationResult,
    validate_query,
)
from app.services.security_event_service import record_event

logger = get_logger("app.security.input_guard")

# Validation failures that indicate deliberate probing rather than a mistake.
_ADVERSARIAL_FAILURES = {
    ValidationFailure.CONTROL_CHARACTERS,
    ValidationFailure.HIDDEN_CHARACTERS,
    ValidationFailure.DEGENERATE_REPETITION,
    ValidationFailure.QUERY_FLOOD,
}


@dataclass
class InputDecision:
    allowed: bool
    text: str
    risk_score: float = 0.0
    reason: str | None = None
    classification: str = "benign"
    flagged: bool = False
    detectors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    verdict: InjectionVerdict | None = None


class InputGuard:
    def __init__(self, detector: InjectionDetector | None = None) -> None:
        self._detector = detector or get_injection_detector()

    def check(
        self,
        db: Session,
        raw_text: str,
        *,
        user_id: str,
        client_ref: str | None = None,
        raise_on_block: bool = True,
    ) -> InputDecision:
        started = time.perf_counter()
        try:
            decision = self._evaluate(
                db, raw_text, user_id=user_id, client_ref=client_ref
            )
        except SecurityBlockError:
            raise
        except Exception as exc:
            logger.exception("input_guard_error", extra={"error": type(exc).__name__})
            record_event(
                db,
                event_type=SecurityEventType.GUARDRAIL_ERROR,
                layer=SecurityLayer.INPUT,
                severity=SecuritySeverity.HIGH,
                action=SecurityAction.BLOCK
                if settings.FAIL_CLOSED
                else SecurityAction.ALLOW,
                user_id=user_id,
                detector="input_guard",
                client_ref=client_ref,
                detail={"error": type(exc).__name__},
            )
            if settings.FAIL_CLOSED:
                raise SecurityBlockError(
                    reason="guardrail_error",
                    risk_score=1.0,
                    internal_detail=f"input guard raised {type(exc).__name__}",
                ) from exc
            decision = InputDecision(allowed=True, text=raw_text)

        decision.latency_ms = round((time.perf_counter() - started) * 1000, 2)

        if not decision.allowed and raise_on_block:
            raise SecurityBlockError(
                reason=decision.reason or "input_policy",
                risk_score=decision.risk_score,
                internal_detail=decision.classification,
            )
        return decision

    # ------------------------------------------------------------------
    def _evaluate(
        self,
        db: Session,
        raw_text: str,
        *,
        user_id: str,
        client_ref: str | None,
    ) -> InputDecision:
        validation = validate_query(raw_text, user_id=user_id)
        if not validation.valid:
            return self._handle_validation_failure(
                db, validation, user_id=user_id, client_ref=client_ref, raw=raw_text
            )

        text = validation.normalised_text

        # PII in the *input* is recorded but never blocked: a user pasting
        # their own phone number into their own question is not an attack, and
        # refusing it would be baffling. The event exists so an operator can
        # see that sensitive data is flowing in, which is a data-governance
        # question rather than a security one.
        if settings.PII_SCAN_INPUT and settings.PII_DETECTION_MODE != "off":
            from app.security.pii.detector import scan_pii

            pii = scan_pii(text)
            if pii.found:
                record_event(
                    db,
                    event_type=SecurityEventType.INPUT_PII_DETECTED,
                    layer=SecurityLayer.INPUT,
                    severity=SecuritySeverity.LOW,
                    action=SecurityAction.FLAG,
                    user_id=user_id,
                    risk_score=0.2,
                    detector=f"pii:{pii.engine}",
                    content_ref=redact_for_log(text),
                    client_ref=client_ref,
                    detail=pii.as_detail(),
                )
                validation.warnings.append("input_contains_pii")

        verdict = self._detector.detect(text)

        if verdict.action is InjectionAction.BLOCK:
            record_event(
                db,
                event_type=SecurityEventType.PROMPT_INJECTION_DETECTED,
                layer=SecurityLayer.INPUT,
                severity=SecuritySeverity.HIGH,
                action=SecurityAction.BLOCK,
                user_id=user_id,
                risk_score=verdict.risk_score,
                detector=",".join(verdict.detectors[:5]) or "heuristics",
                content_ref=redact_for_log(text),
                client_ref=client_ref,
                detail={
                    "classification": verdict.classification,
                    "categories": verdict.categories,
                    "signal_count": len(verdict.detectors),
                    "classifier_consulted": verdict.classifier_consulted,
                },
            )
            return InputDecision(
                allowed=False,
                text=text,
                risk_score=verdict.risk_score,
                reason="prompt_injection",
                classification=verdict.classification,
                detectors=verdict.detectors,
                verdict=verdict,
            )

        if verdict.action is InjectionAction.FLAG:
            # Allowed, but recorded and marked. Blocking every borderline input
            # would make the product unusable; ignoring them would lose the
            # signal that matters most when tuning thresholds.
            record_event(
                db,
                event_type=SecurityEventType.PROMPT_INJECTION_SUSPECTED,
                layer=SecurityLayer.INPUT,
                severity=SecuritySeverity.MEDIUM,
                action=SecurityAction.FLAG,
                user_id=user_id,
                risk_score=verdict.risk_score,
                detector=",".join(verdict.detectors[:5]) or "heuristics",
                content_ref=redact_for_log(text),
                client_ref=client_ref,
                detail={
                    "classification": verdict.classification,
                    "categories": verdict.categories,
                },
            )
            return InputDecision(
                allowed=True,
                text=text,
                risk_score=verdict.risk_score,
                classification=verdict.classification,
                flagged=True,
                detectors=verdict.detectors,
                warnings=validation.warnings,
                verdict=verdict,
            )

        return InputDecision(
            allowed=True,
            text=text,
            risk_score=verdict.risk_score,
            classification=verdict.classification,
            warnings=validation.warnings,
            verdict=verdict,
        )

    def _handle_validation_failure(
        self,
        db: Session,
        validation: ValidationResult,
        *,
        user_id: str,
        client_ref: str | None,
        raw: str,
    ) -> InputDecision:
        failure = validation.failure
        adversarial = failure in _ADVERSARIAL_FAILURES

        event_type = (
            SecurityEventType.DUPLICATE_QUERY_FLOOD
            if failure is ValidationFailure.QUERY_FLOOD
            else SecurityEventType.INPUT_VALIDATION_FAILED
        )

        record_event(
            db,
            event_type=event_type,
            layer=SecurityLayer.INPUT,
            severity=SecuritySeverity.MEDIUM if adversarial else SecuritySeverity.LOW,
            action=SecurityAction.BLOCK,
            user_id=user_id,
            risk_score=0.6 if adversarial else 0.1,
            detector=f"validation:{failure.value if failure else 'unknown'}",
            content_ref=redact_for_log(raw),
            client_ref=client_ref,
            detail={"failure": failure.value if failure else "unknown"},
        )

        return InputDecision(
            allowed=False,
            text=validation.normalised_text,
            risk_score=0.6 if adversarial else 0.1,
            reason="input_validation",
            classification=failure.value if failure else "invalid",
            detectors=[f"validation:{failure.value if failure else 'unknown'}"],
        )


_guard: InputGuard | None = None


def get_input_guard() -> InputGuard:
    global _guard
    if _guard is None:
        _guard = InputGuard()
    return _guard
