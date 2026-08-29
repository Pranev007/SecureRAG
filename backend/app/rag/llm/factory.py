"""LLM provider selection."""

from __future__ import annotations

from functools import cache

from app.core.config import settings
from app.core.exceptions import ConfigurationError
from app.rag.llm.base import LLMProvider


@cache
def get_llm_provider(provider: str | None = None) -> LLMProvider:
    """Return the configured provider (cached per process)."""
    name = (provider or settings.LLM_PROVIDER).lower()

    if name == "echo":
        from app.rag.llm.echo import EchoLLMProvider

        return EchoLLMProvider()

    if name == "openai":
        from app.rag.llm.remote import OpenAICompatibleProvider

        return OpenAICompatibleProvider()

    if name == "ollama":
        from app.rag.llm.remote import OllamaProvider

        return OllamaProvider()

    raise ConfigurationError(internal_detail=f"unknown LLM_PROVIDER={name!r}")
