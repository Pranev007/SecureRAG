"""Answer generation: prompt -> LLM -> parsed structured answer.

Deliberately thin.  It builds the prompt, makes exactly one model call, and
parses the result.  It does **not** decide whether the answer is acceptable --
that is the output guardrail's job, and keeping the two separate means the
verification step cannot be quietly skipped by a change to generation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from pydantic import ValidationError as PydanticValidationError

from app.core.config import settings
from app.core.logging import get_logger
from app.rag.llm.base import LLMProvider, LLMResponse, extract_json_object
from app.rag.llm.factory import get_llm_provider
from app.rag.prompts.templates import BuiltPrompt, build_answer_prompt
from app.rag.retrieval.types import ScoredChunk
from app.schemas.llm_output import LLMAnswer

logger = get_logger("app.rag.generation")


@dataclass
class GenerationResult:
    answer: LLMAnswer | None
    prompt: BuiltPrompt
    raw_response: LLMResponse | None
    schema_error: str | None = None
    latency_ms: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.answer is not None


class Generator:
    def __init__(self, provider: LLMProvider | None = None) -> None:
        self._provider = provider

    @property
    def provider(self) -> LLMProvider:
        if self._provider is None:
            self._provider = get_llm_provider()
        return self._provider

    def generate(
        self,
        question: str,
        chunks: list[ScoredChunk],
        *,
        conversation_summary: str | None = None,
    ) -> GenerationResult:
        prompt = build_answer_prompt(
            question,
            chunks,
            max_context_chars=settings.MAX_CONTEXT_CHARS,
            conversation_summary=conversation_summary,
        )

        started = time.perf_counter()
        response = self.provider.complete(prompt.system, prompt.user, json_mode=True)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)

        payload = extract_json_object(response.text)
        if payload is None:
            detail = (
                "response truncated at the token limit"
                if response.was_truncated
                else "no JSON object in model response"
            )
            logger.warning(
                "llm_output_unparseable",
                extra={
                    "finish_reason": response.finish_reason,
                    "response_chars": len(response.text),
                },
            )
            return GenerationResult(
                answer=None,
                prompt=prompt,
                raw_response=response,
                schema_error=detail,
                latency_ms=latency_ms,
            )

        try:
            answer = LLMAnswer.model_validate(payload)
        except PydanticValidationError as exc:
            logger.warning(
                "llm_output_schema_invalid",
                extra={"error_count": exc.error_count()},
            )
            return GenerationResult(
                answer=None,
                prompt=prompt,
                raw_response=response,
                schema_error=f"schema validation failed: {exc.error_count()} error(s)",
                latency_ms=latency_ms,
            )

        return GenerationResult(
            answer=answer,
            prompt=prompt,
            raw_response=response,
            latency_ms=latency_ms,
            meta={
                "provider": response.provider,
                "model": response.model,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "context_blocks": prompt.block_count,
                "context_chars": prompt.context_chars,
            },
        )
