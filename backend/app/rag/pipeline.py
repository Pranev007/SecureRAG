"""The full SecureRAG request pipeline.

    input guard -> retrieval (access-scoped) -> context sanitisation
                -> prompt assembly -> LLM -> output guard -> answer

This module is deliberately the only place that wires the stages together, so
"what actually happens to a request" is answerable by reading one file.

Two properties are structural rather than conventional:

* **Guardrails cannot be skipped.**  The stages are sequenced here, and the
  service layer has no way to reach the generator directly.
* **Every exit is a validated exit.**  A block, a refusal and a successful
  answer all return the same :class:`RagResponse` type, so a caller cannot
  accidentally handle only the happy path.

Cost discipline (SECTION 25): the expensive stage -- the LLM call -- runs only
after the cheap deterministic stages have had their say.  A blocked query costs
zero model calls, and an empty retrieval costs zero model calls.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.exceptions import ProviderError, SecurityBlockError
from app.core.logging import get_logger
from app.core.request_context import get_request_id
from app.models.security_event import (
    SecurityAction,
    SecurityEventType,
    SecurityLayer,
    SecuritySeverity,
)
from app.models.user import User
from app.rag.generation import Generator
from app.rag.retrieval.retriever import Retriever
from app.rag.retrieval.types import AccessScope
from app.security.context_sanitizer import SanitisationReport, sanitise_chunks
from app.security.input_guard import InputDecision, InputGuard, get_input_guard
from app.security.output.pipeline import OutputDecision, OutputGuard, get_output_guard
from app.services.security_event_service import record_event

logger = get_logger("app.rag.pipeline")


@dataclass
class Source:
    """A citation as presented to the client."""

    index: int
    document_id: str
    chunk_id: str
    filename: str
    page_number: int | None
    section: str | None
    quote: str
    verified: bool
    label: str


@dataclass
class RagResponse:
    answer: str
    blocked: bool = False
    refused: bool = False
    reason: str | None = None

    sources: list[Source] = field(default_factory=list)
    confidence: float = 0.0
    grounding_score: float = 0.0
    risk_score: float = 0.0
    pii_detected: bool = False
    pii_types: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    retrieved_chunk_count: int = 0
    # Filenames the *retriever* surfaced, recorded before the output guardrails
    # run. Kept separate from `sources` (which lists what the answer actually
    # cited) so that retrieval quality can be measured independently of whether
    # the answer was ultimately refused.
    retrieved_documents: list[str] = field(default_factory=list)
    request_id: str | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def total_latency_ms(self) -> float:
        return round(sum(self.timings_ms.values()), 2)


class RagPipeline:
    def __init__(
        self,
        input_guard: InputGuard | None = None,
        retriever: Retriever | None = None,
        generator: Generator | None = None,
        output_guard: OutputGuard | None = None,
    ) -> None:
        self.input_guard = input_guard or get_input_guard()
        self.retriever = retriever or Retriever()
        self.generator = generator or Generator()
        self.output_guard = output_guard or get_output_guard()

    def answer(
        self,
        db: Session,
        question: str,
        *,
        user: User,
        document_ids: list[str] | None = None,
        top_k: int | None = None,
        client_ref: str | None = None,
    ) -> RagResponse:
        timings: dict[str, float] = {}

        # ---------------------------------------------------------------
        # 1. Input guardrails
        # ---------------------------------------------------------------
        try:
            decision = self.input_guard.check(
                db, question, user_id=user.id, client_ref=client_ref
            )
        except SecurityBlockError as exc:
            # A blocked request is a normal outcome, not an error: it is
            # returned as a response so the caller records it like any other.
            return RagResponse(
                answer=(
                    "This request was rejected by the security policy and was "
                    "not sent to the language model."
                ),
                blocked=True,
                reason=exc.reason,
                risk_score=exc.risk_score,
                request_id=get_request_id(),
                timings_ms=timings,
            )

        timings["input_guard_ms"] = decision.latency_ms
        question = decision.text

        # ---------------------------------------------------------------
        # 2. Access-scoped retrieval
        # ---------------------------------------------------------------
        scope = AccessScope(
            user_id=user.id,
            is_admin=user.is_admin,
            document_ids=document_ids,
        )
        retrieval = self.retriever.retrieve(db, question, scope, top_k=top_k)
        timings.update(retrieval.timings_ms)

        record_event(
            db,
            event_type=SecurityEventType.RETRIEVAL_PERFORMED,
            layer=SecurityLayer.RETRIEVAL,
            severity=SecuritySeverity.INFO,
            action=SecurityAction.ALLOW,
            user_id=user.id,
            client_ref=client_ref,
            detail={
                "mode": retrieval.mode,
                "returned": len(retrieval.chunks),
                "vector_candidates": retrieval.vector_candidates,
                "keyword_candidates": retrieval.keyword_candidates,
                "scoped_documents": len(document_ids) if document_ids else 0,
            },
        )

        if retrieval.is_empty:
            # No evidence means no answer. Calling the model here would invite
            # exactly the ungrounded response the guardrails exist to catch,
            # and would cost a request for a result we would then discard.
            return RagResponse(
                answer=settings.INSUFFICIENT_EVIDENCE_MESSAGE,
                refused=True,
                reason="no_results",
                risk_score=decision.risk_score,
                warnings=[*decision.warnings, "no_matching_documents"],
                request_id=get_request_id(),
                timings_ms=timings,
                meta={"retrieval_mode": retrieval.mode},
            )

        # ---------------------------------------------------------------
        # 3. Context sanitisation (indirect injection defence)
        # ---------------------------------------------------------------
        retrieved_documents = sorted({c.source_filename for c in retrieval.chunks})

        started = time.perf_counter()
        chunks, sanitisation = sanitise_chunks(retrieval.chunks)
        timings["sanitise_ms"] = round((time.perf_counter() - started) * 1000, 2)

        warnings = list(decision.warnings)
        if sanitisation.any_action_taken:
            warnings.append("context_sanitised")
            self._record_sanitisation(
                db, sanitisation, user_id=user.id, client_ref=client_ref
            )

        if not chunks:
            return RagResponse(
                answer=settings.INSUFFICIENT_EVIDENCE_MESSAGE,
                refused=True,
                reason="all_context_quarantined",
                warnings=[*warnings, "all_matching_content_was_quarantined"],
                retrieved_documents=retrieved_documents,
                request_id=get_request_id(),
                timings_ms=timings,
            )

        # ---------------------------------------------------------------
        # 4. Generation
        # ---------------------------------------------------------------
        generation_started = time.perf_counter()
        try:
            generation = self.generator.generate(question, chunks)
        except ProviderError as exc:
            # Time the failed attempt too. `llm_ms` used to be recorded only on
            # the success path, so a provider failure contributed nothing to the
            # total and the response reported just the retrieval stages. A
            # provider that hung for 46 seconds displayed as "945 ms total",
            # which is precisely backwards: the number is least trustworthy in
            # the situation where an operator most needs it, and it sent this
            # investigation looking for a fast rejection when the truth was a
            # timeout.
            timings["llm_ms"] = round(
                (time.perf_counter() - generation_started) * 1000, 2
            )
            record_event(
                db,
                event_type=SecurityEventType.PROVIDER_ERROR,
                layer=SecurityLayer.SYSTEM,
                severity=SecuritySeverity.MEDIUM,
                action=SecurityAction.BLOCK,
                user_id=user.id,
                detector="llm_provider",
                client_ref=client_ref,
                detail={"error": exc.internal_detail[:200]},
            )
            # Retrieval already succeeded here -- the provider is what failed --
            # so the response must still report what was retrieved. Omitting
            # these made every provider failure claim zero chunks, which reads
            # as a retrieval bug: it was logged as a separate open item and cost
            # two debugging sessions chasing a fault that did not exist. The
            # timings gave it away in the end (`sanitise` cannot run on an empty
            # retrieval), but a response should not need forensics to be
            # believed. `retrieved_documents` exists precisely so retrieval
            # stays measurable across a refusal -- see its field comment.
            return RagResponse(
                answer=(
                    "The language model service is currently unavailable. "
                    "Please try again shortly."
                ),
                refused=True,
                reason="provider_unavailable",
                risk_score=decision.risk_score,
                warnings=warnings,
                retrieved_chunk_count=len(chunks),
                retrieved_documents=retrieved_documents,
                request_id=get_request_id(),
                timings_ms=timings,
                meta={"retrieval_mode": retrieval.mode},
            )

        timings["llm_ms"] = generation.latency_ms

        # ---------------------------------------------------------------
        # 5. Output guardrails
        # ---------------------------------------------------------------
        validated = self.output_guard.validate(
            db, generation, chunks, user_id=user.id, client_ref=client_ref
        )
        timings["output_guard_ms"] = validated.latency_ms

        response = self._build_response(
            validated,
            decision,
            warnings=warnings,
            retrieved=len(chunks),
            retrieved_documents=retrieved_documents,
            timings=timings,
            meta={
                "retrieval_mode": retrieval.mode,
                "retrieval_backends": retrieval.backends,
                "sanitisation": sanitisation.as_dict(),
                **generation.meta,
            },
        )

        logger.info(
            "rag_request_completed",
            extra={
                "blocked": response.blocked,
                "refused": response.refused,
                "reason": response.reason,
                "grounding_score": response.grounding_score,
                "sources": len(response.sources),
                "retrieved": response.retrieved_chunk_count,
                "total_ms": response.total_latency_ms,
            },
        )
        return response

    # ------------------------------------------------------------------
    @staticmethod
    def _record_sanitisation(
        db: Session,
        report: SanitisationReport,
        *,
        user_id: str,
        client_ref: str | None,
    ) -> None:
        record_event(
            db,
            event_type=SecurityEventType.INDIRECT_INJECTION_DETECTED,
            layer=SecurityLayer.CONTEXT,
            severity=SecuritySeverity.HIGH
            if report.chunks_dropped
            else SecuritySeverity.MEDIUM,
            action=SecurityAction.QUARANTINE
            if report.chunks_dropped
            else SecurityAction.SANITISE,
            user_id=user_id,
            risk_score=report.max_risk,
            detector="context_sanitiser",
            client_ref=client_ref,
            detail=report.as_dict(),
        )

    @staticmethod
    def _build_response(
        validated: OutputDecision,
        decision: InputDecision,
        *,
        warnings: list[str],
        retrieved: int,
        retrieved_documents: list[str],
        timings: dict[str, float],
        meta: dict,
    ) -> RagResponse:
        sources = [
            Source(
                index=c.index,
                document_id=c.document_id,
                chunk_id=c.chunk_id,
                filename=c.filename,
                page_number=c.page_number,
                section=c.section,
                quote=c.quote,
                verified=c.quote_verified,
                label=c.label,
            )
            for c in validated.citations
        ]

        return RagResponse(
            answer=validated.answer,
            blocked=False,
            refused=validated.refused,
            reason=validated.refusal_reason,
            sources=sources,
            confidence=validated.confidence,
            grounding_score=validated.grounding_score,
            risk_score=decision.risk_score,
            pii_detected=validated.pii_detected,
            pii_types=validated.pii_types,
            warnings=[*warnings, *validated.warnings],
            retrieved_chunk_count=retrieved,
            retrieved_documents=retrieved_documents,
            request_id=get_request_id(),
            timings_ms=timings,
            meta={**meta, "output": validated.as_meta()},
        )


_pipeline: RagPipeline | None = None


def get_rag_pipeline() -> RagPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RagPipeline()
    return _pipeline
