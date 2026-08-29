"""Embedding provider selection."""

from __future__ import annotations

from functools import cache

from app.core.config import settings
from app.core.exceptions import ConfigurationError
from app.core.logging import get_logger
from app.rag.embeddings.base import EmbeddingProvider

logger = get_logger("app.rag.embeddings")


@cache
def get_embedding_provider(provider: str | None = None) -> EmbeddingProvider:
    """Return the configured provider (cached per process).

    Cached because constructing a provider validates configuration and, for
    remote providers, is where credential checks happen -- doing that once per
    chunk would be wasteful and would spam the logs.
    """
    name = (provider or settings.EMBEDDING_PROVIDER).lower()

    if name == "hashing":
        from app.rag.embeddings.hashing import HashingEmbeddingProvider

        return HashingEmbeddingProvider()

    if name == "openai":
        from app.rag.embeddings.remote import OpenAIEmbeddingProvider

        return OpenAIEmbeddingProvider()

    if name == "ollama":
        from app.rag.embeddings.remote import OllamaEmbeddingProvider

        return OllamaEmbeddingProvider()

    raise ConfigurationError(internal_detail=f"unknown EMBEDDING_PROVIDER={name!r}")
