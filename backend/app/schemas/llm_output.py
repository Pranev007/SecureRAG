"""The contract the model must satisfy.

This is the first stage of the output guardrail: before anything is inspected
for grounding, citations or PII, the response must *parse* into this shape.
A model that has been successfully hijacked usually stops producing the
required object -- so schema validation is not merely hygiene, it is a
detection signal in its own right, and a failure here fails closed.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RawCitation(BaseModel):
    """A citation as emitted by the model, before server-side verification."""

    model_config = ConfigDict(extra="ignore")

    index: int = Field(
        ..., ge=1, le=100, description="1-based index of the supplied data block"
    )
    quote: str = Field(
        default="",
        max_length=1000,
        description="Short verbatim span the claim rests on",
    )

    @field_validator("quote")
    @classmethod
    def _trim(cls, value: str) -> str:
        return value.strip()


class LLMAnswer(BaseModel):
    """The structured answer the model is instructed to return."""

    # extra="ignore" rather than "forbid": a model that adds a stray key is
    # sloppy, not hostile, and rejecting the whole answer over it would trade a
    # useful response for nothing. Every field we *act* on is validated.
    model_config = ConfigDict(extra="ignore")

    answer: str = Field(..., max_length=20000)
    citations: list[RawCitation] = Field(default_factory=list, max_length=25)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    sufficient_evidence: bool = Field(default=True)
    observed_injection_attempt: bool = Field(default=False)

    @field_validator("answer")
    @classmethod
    def _answer_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("answer must not be empty")
        return stripped

    @field_validator("confidence", mode="before")
    @classmethod
    def _coerce_confidence(cls, value: object) -> object:
        """Accept the shapes models actually emit for confidence.

        Percentages ("85%", 85) and the words high/medium/low all show up in
        practice.  Coercing them is better than discarding an otherwise valid
        answer, and anything unparseable falls back to a neutral 0.5 rather
        than to a flattering 1.0.
        """
        if isinstance(value, str):
            text = value.strip().rstrip("%")
            words = {"high": 0.85, "medium": 0.6, "moderate": 0.6, "low": 0.3}
            if text.lower() in words:
                return words[text.lower()]
            try:
                value = float(text)
            except ValueError:
                return 0.5
        if isinstance(value, int | float):
            number = float(value)
            if number > 1.0:
                return min(number / 100.0, 1.0)
            if number < 0.0:
                return 0.0
            return number
        return 0.5


class InjectionClassification(BaseModel):
    """Output contract for the optional LLM-based injection classifier."""

    model_config = ConfigDict(extra="ignore")

    is_injection: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    category: str = Field(default="unknown", max_length=64)
