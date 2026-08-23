"""HTTP LLM providers: OpenAI-compatible and Ollama."""

from __future__ import annotations

import time
from typing import Any

import httpx

from app.core.config import settings
from app.core.exceptions import ConfigurationError, ProviderError
from app.core.logging import get_logger
from app.rag.llm.base import LLMProvider, LLMResponse

logger = get_logger("app.rag.llm")

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def _post(client: httpx.Client, url: str, payload: dict, headers: dict) -> dict:
    """POST with a short bounded retry.

    A generation call is expensive and user-visible, so we retry only what is
    plausibly transient and give up quickly -- two extra attempts, no
    exponential sleep, because the caller is a blocking HTTP request.
    """
    last_status: int | None = None
    for attempt in range(3):
        try:
            response = client.post(url, json=payload, headers=headers)
            if response.status_code in _RETRYABLE_STATUS and attempt < 2:
                last_status = response.status_code
                continue
            if response.status_code >= 400:
                # The provider's error body may echo the prompt back; log only
                # the status code.
                raise ProviderError(
                    internal_detail=f"HTTP {response.status_code} from LLM provider"
                )
            return response.json()
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            if attempt == 2:
                raise ProviderError(
                    internal_detail=f"{type(exc).__name__} contacting LLM provider"
                ) from exc
    raise ProviderError(internal_detail=f"LLM provider unavailable (last={last_status})")


class OpenAICompatibleProvider(LLMProvider):
    """Any endpoint implementing ``POST /chat/completions`` in the OpenAI shape.

    Covers OpenAI itself plus Groq, Together, OpenRouter, vLLM, LM Studio and
    llama.cpp's server -- one implementation, many deployment options, which is
    the point of not hard-coding a vendor.
    """

    name = "openai"

    def __init__(self) -> None:
        needs_key = "api.openai.com" in settings.LLM_BASE_URL
        if needs_key and not settings.LLM_API_KEY:
            raise ConfigurationError(
                internal_detail="LLM_API_KEY is required for api.openai.com"
            )
        self._base_url = settings.LLM_BASE_URL.rstrip("/")
        self._model = settings.LLM_MODEL

    @property
    def model(self) -> str:
        return self._model

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": (
                settings.LLM_TEMPERATURE if temperature is None else temperature
            ),
            "max_tokens": max_tokens or settings.LLM_MAX_TOKENS,
        }
        if json_mode:
            # Constrained decoding where the provider supports it. The output
            # guardrail still validates the result -- JSON mode guarantees
            # syntax, not that the object matches our schema.
            payload["response_format"] = {"type": "json_object"}

        headers = {"Content-Type": "application/json"}
        if settings.LLM_API_KEY:
            headers["Authorization"] = f"Bearer {settings.LLM_API_KEY}"

        started = time.perf_counter()
        with httpx.Client(timeout=settings.LLM_TIMEOUT_SECONDS) as client:
            data = _post(client, f"{self._base_url}/chat/completions", payload, headers)
        latency_ms = (time.perf_counter() - started) * 1000

        try:
            choice = data["choices"][0]
            content = choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                internal_detail="malformed chat completion response"
            ) from exc

        usage = data.get("usage") or {}
        return LLMResponse(
            text=content,
            model=data.get("model", self._model),
            provider=self.name,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            latency_ms=round(latency_ms, 2),
            finish_reason=choice.get("finish_reason", "stop") or "stop",
        )


class OllamaProvider(LLMProvider):
    """Local Ollama server (``POST /api/chat``)."""

    name = "ollama"

    def __init__(self) -> None:
        base = settings.LLM_BASE_URL.rstrip("/")
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        self._base_url = base
        self._model = settings.LLM_MODEL

    @property
    def model(self) -> str:
        return self._model

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": (
                    settings.LLM_TEMPERATURE if temperature is None else temperature
                ),
                "num_predict": max_tokens or settings.LLM_MAX_TOKENS,
            },
        }
        if json_mode:
            payload["format"] = "json"

        started = time.perf_counter()
        with httpx.Client(timeout=settings.LLM_TIMEOUT_SECONDS) as client:
            data = _post(
                client,
                f"{self._base_url}/api/chat",
                payload,
                {"Content-Type": "application/json"},
            )
        latency_ms = (time.perf_counter() - started) * 1000

        content = (data.get("message") or {}).get("content", "")
        if not content:
            raise ProviderError(internal_detail="empty ollama response")

        return LLMResponse(
            text=content,
            model=data.get("model", self._model),
            provider=self.name,
            prompt_tokens=int(data.get("prompt_eval_count", 0)),
            completion_tokens=int(data.get("eval_count", 0)),
            latency_ms=round(latency_ms, 2),
            finish_reason="length" if data.get("done_reason") == "length" else "stop",
        )
