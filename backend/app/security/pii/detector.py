"""PII detection and redaction.

Four modes, chosen by ``PII_DETECTION_MODE``:

``off``     no scanning
``warn``    detect, record a security event, return the text unchanged
``redact``  replace each match with ``[TYPE_REDACTED]``  (default)
``block``   refuse to return an answer that contains PII

``redact`` is the default because it preserves the useful part of an answer.
``block`` is the right choice when *any* leak is unacceptable and a refused
answer is cheaper than a redacted one.

An optional Presidio backend (``PII_ENGINE=presidio``) adds NER-based
detection of names and addresses, which the pattern engine cannot do.  It is
optional because it pulls in spaCy and a language model; the fallback is
declared, not silent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.config import settings
from app.core.logging import get_logger
from app.security.pii.patterns import PATTERNS, VALIDATORS, PIIType

logger = get_logger("app.security.pii")

# How far either side of a match to look for corroborating context words.
CONTEXT_WINDOW = 48
# Matches below this confidence are discarded entirely.
MIN_CONFIDENCE = 0.5


@dataclass
class PIIMatch:
    type: PIIType
    start: int
    end: int
    value: str
    confidence: float
    validated: bool = False

    @property
    def placeholder(self) -> str:
        return f"[{self.type.value}_REDACTED]"

    def masked(self) -> str:
        """Partially masked form, for operator-facing detail only."""
        if len(self.value) <= 4:
            return "*" * len(self.value)
        return f"{self.value[:2]}{'*' * (len(self.value) - 4)}{self.value[-2:]}"


@dataclass
class PIIReport:
    matches: list[PIIMatch] = field(default_factory=list)
    redacted_text: str | None = None
    engine: str = "regex"

    @property
    def found(self) -> bool:
        return bool(self.matches)

    @property
    def types(self) -> list[str]:
        return sorted({m.type.value for m in self.matches})

    def counts(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for match in self.matches:
            result[match.type.value] = result.get(match.type.value, 0) + 1
        return result

    def as_detail(self) -> dict:
        """Audit-safe summary: types and counts, never values."""
        return {
            "engine": self.engine,
            "types": self.types,
            "counts": self.counts(),
            "total": len(self.matches),
        }


def _context_bonus(
    text: str, match_start: int, match_end: int, words: frozenset
) -> float:
    if not words:
        return 0.0
    window = text[
        max(0, match_start - CONTEXT_WINDOW) : match_end + CONTEXT_WINDOW
    ].lower()
    return 0.35 if any(word in window for word in words) else 0.0


def _overlaps(match: PIIMatch, accepted: list[PIIMatch]) -> bool:
    return any(
        match.start < existing.end and existing.start < match.end for existing in accepted
    )


class RegexPIIDetector:
    """Pattern + checksum detector (the default engine)."""

    engine = "regex"

    def detect(self, text: str, *, enabled_types: set[str] | None = None) -> PIIReport:
        if not text:
            return PIIReport(engine=self.engine)

        allowed = enabled_types if enabled_types is not None else settings.PII_ENTITIES
        candidates: list[PIIMatch] = []

        for pattern in PATTERNS:
            if pattern.type.value not in allowed:
                continue
            for found in pattern.regex.finditer(text):
                value = found.group(0)
                confidence = pattern.base_confidence
                validated = False

                if pattern.validator:
                    validator = VALIDATORS[pattern.validator]
                    if validator(value):
                        # A passing checksum is strong evidence: random digits
                        # pass Luhn only 10% of the time and Verhoeff 10%, but
                        # combined with the format constraint it is decisive.
                        confidence = min(1.0, confidence + 0.45)
                        validated = True
                    else:
                        # Failing a checksum is near-proof this is not the
                        # identifier it resembles, so drop it rather than
                        # redacting an order number.
                        continue

                confidence += _context_bonus(
                    text, found.start(), found.end(), pattern.context_words
                )
                confidence = min(confidence, 1.0)

                if confidence >= MIN_CONFIDENCE:
                    candidates.append(
                        PIIMatch(
                            type=pattern.type,
                            start=found.start(),
                            end=found.end(),
                            value=value,
                            confidence=round(confidence, 3),
                            validated=validated,
                        )
                    )

        # Resolve overlaps: a card number also matches the phone pattern.
        # Highest confidence wins, then the longest span.
        candidates.sort(key=lambda m: (-m.confidence, -(m.end - m.start), m.start))
        accepted: list[PIIMatch] = []
        for candidate in candidates:
            if not _overlaps(candidate, accepted):
                accepted.append(candidate)

        accepted.sort(key=lambda m: m.start)
        return PIIReport(matches=accepted, engine=self.engine)


class PresidioPIIDetector:
    """Optional NER-backed detector (``PII_ENGINE=presidio``)."""

    engine = "presidio"

    def __init__(self) -> None:
        from presidio_analyzer import AnalyzerEngine

        self._analyzer = AnalyzerEngine()
        self._fallback = RegexPIIDetector()

    def detect(
        self, text: str, *, enabled_types: set[str] | None = None
    ) -> PIIReport:  # pragma: no cover - optional dependency
        report = self._fallback.detect(text, enabled_types=enabled_types)
        report.engine = self.engine

        try:
            results = self._analyzer.analyze(text=text, language="en")
        except Exception as exc:
            logger.warning("presidio_analyze_failed", extra={"error": type(exc).__name__})
            return report

        mapping = {
            "PERSON": PIIType.EMAIL,  # placeholder type for name spans
            "LOCATION": PIIType.EMAIL,
        }
        for result in results:
            if result.score < 0.6 or result.entity_type not in mapping:
                continue
            match = PIIMatch(
                type=mapping[result.entity_type],
                start=result.start,
                end=result.end,
                value=text[result.start : result.end],
                confidence=round(result.score, 3),
            )
            if not _overlaps(match, report.matches):
                report.matches.append(match)

        report.matches.sort(key=lambda m: m.start)
        return report


def redact(text: str, matches: list[PIIMatch]) -> str:
    """Replace matches with type placeholders, right to left.

    Right to left so that each replacement cannot invalidate the offsets of the
    matches still to be processed.
    """
    if not matches:
        return text
    result = text
    for match in sorted(matches, key=lambda m: m.start, reverse=True):
        result = result[: match.start] + match.placeholder + result[match.end :]
    return re.sub(r"[ \t]{2,}", " ", result)


_detector: RegexPIIDetector | PresidioPIIDetector | None = None


def get_pii_detector():
    global _detector
    if _detector is None:
        if settings.PII_ENGINE == "presidio":
            try:
                _detector = PresidioPIIDetector()
            except ImportError:
                # Degrade loudly. Silently falling back would mean an operator
                # who configured NER detection never learns they are not
                # getting it.
                logger.error(
                    "presidio_unavailable_falling_back_to_regex",
                    extra={"hint": "pip install -r requirements-optional.txt"},
                )
                _detector = RegexPIIDetector()
        else:
            _detector = RegexPIIDetector()
    return _detector


def scan_pii(text: str, *, enabled_types: set[str] | None = None) -> PIIReport:
    return get_pii_detector().detect(text, enabled_types=enabled_types)
