"""Structural validation of user input.

This runs *before* injection detection, and the ordering is deliberate: these
checks are cheaper, and several of them (length caps, control-character
stripping) also make the injection detector's job well-defined.  A 2 MB query
should be rejected on size, not fed to a dozen regexes first.

Pydantic handles the request *shape* at the API boundary.  This module handles
the semantics Pydantic cannot express: normalisation, character-class policy,
and repetition across requests.
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
import unicodedata
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import StrEnum

from app.core.config import settings
from app.rag.ingestion.cleaner import count_invisible_characters, strip_invisible

# Control characters other than tab/newline have no legitimate place in a
# question and are a common way to break naive parsers.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Collapse two or more spaces, not four: substituting a control character for a
# space would otherwise leave a double space, and "ignore  all  instructions"
# must normalise to the same string as "ignore all instructions".
_EXCESSIVE_WHITESPACE = re.compile(r"[ \t]{2,}")
_REPEATED_CHAR = re.compile(r"(.)\1{40,}")


class ValidationFailure(StrEnum):
    EMPTY = "empty_input"
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    CONTROL_CHARACTERS = "control_characters"
    HIDDEN_CHARACTERS = "hidden_characters"
    DEGENERATE_REPETITION = "degenerate_repetition"
    QUERY_FLOOD = "query_flood"


@dataclass
class ValidationResult:
    valid: bool
    normalised_text: str = ""
    failure: ValidationFailure | None = None
    # Operator-facing; never returned to the caller.
    detail: str = ""
    warnings: list[str] = field(default_factory=list)


def normalise_input(text: str) -> str:
    """Canonicalise input before any security decision is made about it.

    Order matters.  NFKC first, so that homoglyph and full-width variants
    collapse onto their canonical form *before* the detector sees them --
    otherwise "ｉｇｎｏｒｅ ａｌｌ ｉｎｓｔｒｕｃｔｉｏｎｓ" is a trivial
    bypass of every pattern in layer 1.

    Zero-width characters are then *removed*, not replaced with a space.
    ``Ig<ZWSP>nore all pre<ZWSP>vious inst<ZWSP>ructions`` renders identically
    to the plain phrase but matches no pattern; deleting the invisible
    characters reassembles the words so the detector sees what the model would.
    The original count is captured separately by :func:`validate_query`, so
    stripping here does not discard the evidence that hiding was attempted.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = strip_invisible(text)
    text = _CONTROL.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _EXCESSIVE_WHITESPACE.sub(" ", text)
    return text.strip()


class QueryFloodTracker:
    """Detects the same query being replayed repeatedly by one user.

    A distinct signal from rate limiting: the rate limiter counts *requests*,
    this counts *identical* requests.  Sending the same borderline prompt fifty
    times is how an attacker probes for a non-deterministic gap in a guardrail,
    and it looks nothing like normal use.

    Only a hash of the query is retained -- never the text.
    """

    def __init__(self, window_seconds: int | None = None) -> None:
        self._window = window_seconds or settings.DUPLICATE_QUERY_WINDOW_SECONDS
        self._seen: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    @staticmethod
    def _key(user_id: str, text: str) -> str:
        digest = hashlib.sha256(text.lower().strip().encode("utf-8")).hexdigest()[:16]
        return f"{user_id}:{digest}"

    def record(self, user_id: str, text: str) -> int:
        """Record an occurrence and return how many are in the window."""
        now = time.monotonic()
        cutoff = now - self._window
        key = self._key(user_id, text)

        with self._lock:
            bucket = self._seen[key]
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            bucket.append(now)

            if len(self._seen) > 20_000:
                for stale in [k for k, v in self._seen.items() if not v]:
                    del self._seen[stale]

            return len(bucket)

    def reset(self) -> None:
        with self._lock:
            self._seen.clear()


flood_tracker = QueryFloodTracker()


def validate_query(text: str, *, user_id: str | None = None) -> ValidationResult:
    """Validate and normalise a user query."""
    raw = text or ""

    if not raw.strip():
        return ValidationResult(
            valid=False, failure=ValidationFailure.EMPTY, detail="blank input"
        )

    # Length is checked on the raw string first: normalisation of a very large
    # input is itself work an attacker should not be able to compel.
    if len(raw) > settings.MAX_QUERY_LENGTH * 4:
        return ValidationResult(
            valid=False,
            failure=ValidationFailure.TOO_LONG,
            detail=f"raw length {len(raw)}",
        )

    warnings: list[str] = []
    control_count = len(_CONTROL.findall(raw))
    hidden_count = count_invisible_characters(raw)

    normalised = normalise_input(raw)

    if len(normalised) < settings.MIN_QUERY_LENGTH:
        return ValidationResult(
            valid=False,
            failure=ValidationFailure.TOO_SHORT,
            detail=f"normalised length {len(normalised)}",
        )

    if len(normalised) > settings.MAX_QUERY_LENGTH:
        return ValidationResult(
            valid=False,
            failure=ValidationFailure.TOO_LONG,
            detail=f"normalised length {len(normalised)}",
        )

    if control_count > 5:
        return ValidationResult(
            valid=False,
            failure=ValidationFailure.CONTROL_CHARACTERS,
            detail=f"{control_count} control characters",
        )
    if control_count:
        warnings.append("control_characters_stripped")

    # Hidden characters are not fatal on their own -- they can arrive from an
    # innocent copy-paste -- but a large number is a deliberate act, and the
    # count is passed to the injection detector as evidence either way.
    if hidden_count > 20:
        return ValidationResult(
            valid=False,
            failure=ValidationFailure.HIDDEN_CHARACTERS,
            detail=f"{hidden_count} invisible characters",
        )
    if hidden_count:
        warnings.append("hidden_characters_present")

    if _REPEATED_CHAR.search(normalised):
        return ValidationResult(
            valid=False,
            failure=ValidationFailure.DEGENERATE_REPETITION,
            detail="long single-character run",
        )

    if user_id:
        occurrences = flood_tracker.record(user_id, normalised)
        if occurrences > settings.DUPLICATE_QUERY_LIMIT:
            return ValidationResult(
                valid=False,
                normalised_text=normalised,
                failure=ValidationFailure.QUERY_FLOOD,
                detail=f"{occurrences} identical queries in window",
            )

    return ValidationResult(valid=True, normalised_text=normalised, warnings=warnings)
