"""Layer 3 (optional): LLM-based injection classification.

Off by default (``INJECTION_USE_LLM_CLASSIFIER=false``) and consulted only for
inputs the cheap layers already found *borderline*.  That ordering is the whole
point: paying for a model call on every request to catch the small band of
cases that deterministic rules genuinely cannot resolve would be poor
engineering, and it would put a network dependency on the hot path of every
query.

Two properties make this safe to use as a security control:

* The classifier prompt is a **separate call with its own system message**.  It
  never receives the retrieval context, so document content cannot steer it.
* The candidate text is **fenced and defanged** before it is shown, and the
  classifier is told explicitly that its input is data to describe, not
  instructions to follow.  A classifier that can be talked out of classifying
  is not a control.

It is advisory: it can only *raise* an already-borderline score, never rescue
an input the deterministic layers blocked outright.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import ValidationError as PydanticValidationError

from app.core.logging import get_logger
from app.rag.llm.base import LLMProvider, extract_json_object
from app.rag.llm.factory import get_llm_provider
from app.rag.prompts.templates import build_classifier_prompt
from app.schemas.llm_output import InjectionClassification

logger = get_logger("app.security.injection.classifier")


@dataclass(frozen=True)
class ClassifierVerdict:
    consulted: bool
    is_injection: bool = False
    confidence: float = 0.0
    category: str = "not_evaluated"
    error: str | None = None


class LLMInjectionClassifier:
    def __init__(self, provider: LLMProvider | None = None) -> None:
        self._provider = provider

    @property
    def provider(self) -> LLMProvider:
        if self._provider is None:
            self._provider = get_llm_provider()
        return self._provider

    def classify(self, candidate: str) -> ClassifierVerdict:
        system_prompt, user_prompt = build_classifier_prompt(candidate)
        try:
            response = self.provider.complete(
                system_prompt,
                user_prompt,
                temperature=0.0,
                max_tokens=120,
                json_mode=True,
            )
        except Exception as exc:
            # An unavailable classifier must not take the request down. The
            # deterministic layers already produced a score; this layer simply
            # abstains, which is recorded so the abstention is visible.
            logger.warning(
                "injection_classifier_unavailable",
                extra={"error": type(exc).__name__},
            )
            return ClassifierVerdict(consulted=False, error=type(exc).__name__)

        payload = extract_json_object(response.text)
        if payload is None:
            return ClassifierVerdict(consulted=False, error="unparseable_response")

        try:
            verdict = InjectionClassification.model_validate(payload)
        except PydanticValidationError:
            return ClassifierVerdict(consulted=False, error="schema_invalid")

        return ClassifierVerdict(
            consulted=True,
            is_injection=verdict.is_injection,
            confidence=verdict.confidence,
            category=verdict.category,
        )
