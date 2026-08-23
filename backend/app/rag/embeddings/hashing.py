"""Deterministic offline embedding provider (the "hashing trick").

WHAT THIS IS
------------
A feature-hashing vectoriser: each n-gram of the input is hashed to a
dimension index and a sign, and its TF-IDF-ish weight is accumulated there.
This is the same technique behind scikit-learn's ``HashingVectorizer``.

WHY IT EXISTS
-------------
So that the repository can be cloned, tested and demonstrated with **no API key
and no model download**, while still producing *real* similarity scores.  The
evaluation suite therefore reports genuine measurements rather than fabricated
ones -- the numbers are simply the numbers for a lexical retriever.

WHAT IT IS NOT
--------------
It is not a semantic model.  It matches words and word-fragments, not meaning:
"annual leave" and "paid time off" score near zero against each other.  Set
``EMBEDDING_PROVIDER=openai`` or ``ollama`` for semantic retrieval.  This
limitation is stated in the README and docs/evaluation.md because reporting a
lexical retriever's recall as if it were a semantic model's would be dishonest.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence
from itertools import pairwise

from app.core.config import settings
from app.rag.embeddings.base import EmbeddingProvider, l2_normalise

_TOKEN = re.compile(r"[a-z0-9]+")

# Very common words carry no retrieval signal but do carry a lot of weight in a
# bag-of-n-grams model.
_STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those of in on at to for
    from by with as is are was were be been being it its do does did not no
    i you he she they we my your our their there here what which who whom how
    when where why can could should would may might will shall must have has
    had about into over under again further more most other some such only own
    same so too very s t just don now
    """.split()
)


def _hash(token: str, salt: str = "") -> int:
    return int.from_bytes(
        hashlib.blake2b((salt + token).encode("utf-8"), digest_size=8).digest(),
        "big",
    )


class HashingEmbeddingProvider(EmbeddingProvider):
    """Feature-hashing embedder over word unigrams, bigrams and char 4-grams."""

    name = "hashing"

    def __init__(self, dimensions: int | None = None) -> None:
        self._dimensions = dimensions or settings.EMBEDDING_DIMENSIONS
        if self._dimensions < 32:
            raise ValueError("hashing embedder needs at least 32 dimensions")

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model(self) -> str:
        return f"hashing-{self._dimensions}d"

    def _features(self, text: str) -> Counter[str]:
        tokens = _TOKEN.findall(text.lower())
        content = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]
        features: Counter[str] = Counter()

        for token in content:
            features[f"w:{token}"] += 1
            # Character 4-grams give partial credit for morphological variants
            # ("policy"/"policies") and typos, which a pure word model misses.
            padded = f"^{token}$"
            if len(padded) >= 4:
                for i in range(len(padded) - 3):
                    features[f"c:{padded[i:i + 4]}"] += 1

        # Word bigrams capture short phrases ("annual leave", "sick leave").
        for first, second in pairwise(content):
            features[f"b:{first}_{second}"] += 2

        return features

    def _vectorise(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        features = self._features(text)
        if not features:
            return vector

        for feature, count in features.items():
            digest = _hash(feature)
            index = digest % self._dimensions
            # Signed hashing: the sign bit makes collisions cancel on average
            # instead of always adding, which reduces hash-collision bias.
            sign = 1.0 if (digest >> 63) & 1 else -1.0
            # Sub-linear term frequency: the tenth occurrence of a word says
            # much less than the second.
            weight = 1.0 + math.log(count)
            vector[index] += sign * weight

        return l2_normalise(vector)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vectorise(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vectorise(text)
