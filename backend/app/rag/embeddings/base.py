"""Embedding provider interface."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence


class EmbeddingProvider(ABC):
    """Turns text into fixed-length vectors.

    Implementations must return **L2-normalised** vectors.  With unit vectors,
    cosine similarity reduces to a dot product and pgvector's ``<=>`` operator
    becomes ``1 - dot``, which keeps the scoring identical across the pgvector
    and Python fallback backends.
    """

    name: str = "base"

    @property
    @abstractmethod
    def dimensions(self) -> int: ...

    @property
    @abstractmethod
    def model(self) -> str: ...

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of chunk texts."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query.

        Separate from :meth:`embed_documents` because some models require an
        asymmetric prefix ("query: " / "passage: ") for retrieval to work.
        """
        return self.embed_documents([text])[0]


def l2_normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0.0:
        return vector
    return [component / norm for component in vector]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity, safe for zero vectors and mismatched lengths."""
    if not a or not b:
        return 0.0
    size = min(len(a), len(b))
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for index in range(size):
        x = a[index]
        y = b[index]
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)
