"""HTTP embedding providers: OpenAI-compatible and Ollama."""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from app.core.config import settings
from app.core.exceptions import ConfigurationError, ProviderError
from app.core.logging import get_logger
from app.rag.embeddings.base import EmbeddingProvider, l2_normalise

logger = get_logger("app.rag.embeddings")

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def _request_with_retry(
    client: httpx.Client, url: str, payload: dict, headers: dict
) -> dict:
    """POST with a short bounded retry on transient failures.

    Two retries with linear backoff: enough to ride out a rate-limit blip,
    short enough that a failing provider surfaces quickly instead of holding a
    request open until the client times out.
    """
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = client.post(url, json=payload, headers=headers)
            if response.status_code in _RETRYABLE_STATUS and attempt < 2:
                last_error = ProviderError(
                    internal_detail=f"HTTP {response.status_code} from {url}"
                )
                continue
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            # The provider's error body can contain the prompt; never log it.
            raise ProviderError(
                internal_detail=f"HTTP {exc.response.status_code} from embedding provider"
            ) from exc
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_error = exc
            if attempt == 2:
                raise ProviderError(
                    internal_detail=f"{type(exc).__name__} contacting embedding provider"
                ) from exc
    raise ProviderError(internal_detail=f"embedding provider unavailable: {last_error}")


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Any endpoint implementing ``POST /embeddings`` in the OpenAI shape."""

    name = "openai"

    def __init__(self) -> None:
        needs_key = "api.openai.com" in settings.EMBEDDING_BASE_URL
        if needs_key and not settings.EMBEDDING_API_KEY:
            raise ConfigurationError(
                internal_detail="EMBEDDING_API_KEY is required for api.openai.com"
            )
        self._base_url = settings.EMBEDDING_BASE_URL.rstrip("/")
        self._model = settings.EMBEDDING_MODEL
        self._dimensions = settings.EMBEDDING_DIMENSIONS

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model(self) -> str:
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        headers = {"Content-Type": "application/json"}
        if settings.EMBEDDING_API_KEY:
            headers["Authorization"] = f"Bearer {settings.EMBEDDING_API_KEY}"

        batch_size = max(1, settings.EMBEDDING_BATCH_SIZE)
        with httpx.Client(timeout=settings.EMBEDDING_TIMEOUT_SECONDS) as client:
            for start in range(0, len(texts), batch_size):
                batch = list(texts[start : start + batch_size])
                payload: dict = {"model": self._model, "input": batch}
                # Only the v3 models accept an explicit output size; sending it
                # to others is an error, so we ask only when it is meaningful.
                if "text-embedding-3" in self._model:
                    payload["dimensions"] = self._dimensions

                data = _request_with_retry(
                    client, f"{self._base_url}/embeddings", payload, headers
                )
                items = sorted(data.get("data", []), key=lambda d: d.get("index", 0))
                if len(items) != len(batch):
                    raise ProviderError(
                        internal_detail=(
                            f"embedding count mismatch: sent {len(batch)}, "
                            f"received {len(items)}"
                        )
                    )
                for item in items:
                    vector = [float(x) for x in item["embedding"]]
                    self._check_dimensions(len(vector))
                    vectors.append(l2_normalise(vector))
        return vectors

    def _check_dimensions(self, actual: int) -> None:
        if actual != self._dimensions:
            # Silently truncating or padding would corrupt the index in a way
            # that only shows up as mysteriously poor retrieval later.
            raise ConfigurationError(
                internal_detail=(
                    f"EMBEDDING_DIMENSIONS={self._dimensions} but model "
                    f"{self._model} returned {actual}. The database column was "
                    "created with the configured size; re-embed after changing it."
                )
            )


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Local Ollama server (``POST /api/embed``)."""

    name = "ollama"

    def __init__(self) -> None:
        self._base_url = settings.EMBEDDING_BASE_URL.rstrip("/")
        if self._base_url.endswith("/v1"):
            self._base_url = self._base_url[: -len("/v1")]
        self._model = settings.EMBEDDING_MODEL
        self._dimensions = settings.EMBEDDING_DIMENSIONS

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model(self) -> str:
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        batch_size = max(1, settings.EMBEDDING_BATCH_SIZE)
        with httpx.Client(timeout=settings.EMBEDDING_TIMEOUT_SECONDS) as client:
            for start in range(0, len(texts), batch_size):
                batch = list(texts[start : start + batch_size])
                data = _request_with_retry(
                    client,
                    f"{self._base_url}/api/embed",
                    {"model": self._model, "input": batch},
                    {"Content-Type": "application/json"},
                )
                embeddings = data.get("embeddings") or []
                if len(embeddings) != len(batch):
                    raise ProviderError(
                        internal_detail=(
                            f"ollama returned {len(embeddings)} vectors "
                            f"for {len(batch)} inputs"
                        )
                    )
                for raw in embeddings:
                    vector = [float(x) for x in raw]
                    if len(vector) != self._dimensions:
                        raise ConfigurationError(
                            internal_detail=(
                                f"EMBEDDING_DIMENSIONS={self._dimensions} but "
                                f"{self._model} returned {len(vector)}"
                            )
                        )
                    vectors.append(l2_normalise(vector))
        return vectors
