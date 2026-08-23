"""Citation resolution and verification.

A citation the model emitted is a *claim*, not a fact.  Three failure modes
occur in practice and each is checked here:

1. **Hallucinated index** -- the model cites ``[7]`` when only four blocks were
   supplied.  Trivially detectable and always a hard failure.
2. **Real index, wrong content** -- the block exists but does not contain the
   quoted span.  Caught by verifying the quote against the chunk text.
3. **Uncited claims** -- the answer contains no markers at all while asserting
   facts.  Handled by the grounding check, not here.

The index -> document mapping is held **server-side**: the model only ever sees
``[1]``, ``[2]``.  It cannot cite a document id it was never given, which means
a hijacked model cannot fabricate a citation pointing at a file the user is not
allowed to see.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.rag.retrieval.types import ScoredChunk
from app.schemas.llm_output import LLMAnswer

_MARKER = re.compile(r"\[(\d{1,3})\]")
_NON_WORD = re.compile(r"[^a-z0-9]+")

# Fraction of a quote's words that must appear in the cited chunk for the quote
# to count as verified. Not 1.0: models normalise whitespace, fix casing, and
# trim punctuation when quoting, and failing those would make the check useless.
QUOTE_MATCH_THRESHOLD = 0.75


@dataclass
class ResolvedCitation:
    index: int
    chunk_id: str
    document_id: str
    filename: str
    page_number: int | None
    section: str | None
    quote: str
    quote_verified: bool
    label: str


@dataclass
class CitationReport:
    citations: list[ResolvedCitation] = field(default_factory=list)
    invalid_indices: list[int] = field(default_factory=list)
    unverified_quotes: list[int] = field(default_factory=list)
    inline_markers: list[int] = field(default_factory=list)
    sanitised_answer: str = ""

    @property
    def has_valid_citations(self) -> bool:
        return bool(self.citations)

    @property
    def is_clean(self) -> bool:
        return not self.invalid_indices and not self.unverified_quotes

    @property
    def accuracy(self) -> float:
        """Share of emitted citations that resolved and verified."""
        total = len(self.citations) + len(self.invalid_indices)
        if total == 0:
            return 0.0
        verified = sum(1 for c in self.citations if c.quote_verified)
        return round(verified / total, 4)


def _normalise(text: str) -> str:
    return _NON_WORD.sub(" ", text.lower()).strip()


def verify_quote(quote: str, chunk_content: str) -> bool:
    """Check that ``quote`` is plausibly drawn from ``chunk_content``."""
    if not quote:
        # No quote supplied is not a *wrong* quote; the index check still ran.
        return True

    normalised_quote = _normalise(quote)
    normalised_chunk = _normalise(chunk_content)
    if not normalised_quote:
        return True

    # Fast path: exact (normalised) containment.
    if normalised_quote in normalised_chunk:
        return True

    # Fallback: word-level overlap, which tolerates the small edits models make
    # when quoting while still rejecting a quote from a different source.
    quote_words = normalised_quote.split()
    chunk_words = set(normalised_chunk.split())
    if not quote_words:
        return True
    matched = sum(1 for word in quote_words if word in chunk_words)
    return matched / len(quote_words) >= QUOTE_MATCH_THRESHOLD


def strip_invalid_markers(answer: str, valid_indices: set[int]) -> str:
    """Remove inline ``[n]`` markers that do not point at a supplied block.

    Rewriting the answer rather than rejecting it: an otherwise good answer
    with one bad marker is worth keeping, and leaving the marker in would show
    the user a citation that resolves to nothing.
    """

    def _replace(match: re.Match[str]) -> str:
        return match.group(0) if int(match.group(1)) in valid_indices else ""

    cleaned = _MARKER.sub(_replace, answer)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def resolve_citations(
    answer: LLMAnswer, index_to_chunk: dict[int, ScoredChunk]
) -> CitationReport:
    """Resolve and verify every citation the model produced."""
    report = CitationReport()
    seen: set[int] = set()

    for raw in answer.citations:
        chunk = index_to_chunk.get(raw.index)
        if chunk is None:
            report.invalid_indices.append(raw.index)
            continue
        if raw.index in seen:
            continue
        seen.add(raw.index)

        verified = verify_quote(raw.quote, chunk.content)
        if not verified:
            report.unverified_quotes.append(raw.index)

        report.citations.append(
            ResolvedCitation(
                index=raw.index,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                filename=chunk.source_filename,
                page_number=chunk.page_number,
                section=chunk.section,
                quote=raw.quote[:500],
                quote_verified=verified,
                label=chunk.citation_label(),
            )
        )

    # Inline markers are checked independently: a model often cites correctly
    # in prose while emitting a malformed citations array, or vice versa.
    inline = {int(m) for m in _MARKER.findall(answer.answer)}
    report.inline_markers = sorted(inline)
    for index in sorted(inline - set(index_to_chunk)):
        if index not in report.invalid_indices:
            report.invalid_indices.append(index)

    report.sanitised_answer = strip_invalid_markers(answer.answer, set(index_to_chunk))

    # A marker used inline but omitted from the citations array is still a real
    # citation; resolve it so the user gets the source link.
    for index in sorted(inline & set(index_to_chunk) - seen):
        chunk = index_to_chunk[index]
        report.citations.append(
            ResolvedCitation(
                index=index,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                filename=chunk.source_filename,
                page_number=chunk.page_number,
                section=chunk.section,
                quote="",
                quote_verified=True,
                label=chunk.citation_label(),
            )
        )

    report.citations.sort(key=lambda c: c.index)
    return report
