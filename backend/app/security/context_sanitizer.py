"""Runtime sanitisation of retrieved context.

Defence in depth, applied *after* retrieval and *before* prompt assembly.
Quarantine at ingest already removed the worst chunks, so this stage exists for
what quarantine cannot cover:

* chunks that scored in the grey band -- suspicious enough to defang, not
  suspicious enough to withhold the user's own content;
* chunks ingested before a scanner rule existed (the corpus is not re-scanned
  on every deploy);
* fence-marker imitations, which are cheap to strip and pointless to keep.

The design choice that matters here is **surgical, sentence-level removal**.
Dropping the whole chunk would hand the attacker a denial-of-service primitive:
plant one sentence in a page and the entire page stops being answerable.
Removing only the offending sentences preserves the legitimate content around
them, and leaves an explicit marker so the model, the user, and the audit log
all see that something was removed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.config import settings
from app.core.logging import get_logger
from app.rag.ingestion.chunker import split_sentences
from app.rag.prompts.templates import defang_fences
from app.rag.retrieval.types import ScoredChunk
from app.security.context_scanner import get_context_scanner

logger = get_logger("app.security.context_sanitizer")

REMOVAL_MARKER = "[content removed: instruction-like text detected by security policy]"


@dataclass
class SanitisationReport:
    chunks_scanned: int = 0
    chunks_modified: int = 0
    chunks_dropped: int = 0
    sentences_removed: int = 0
    max_risk: float = 0.0
    labels: list[str] = field(default_factory=list)
    modified_chunk_ids: list[str] = field(default_factory=list)
    dropped_chunk_ids: list[str] = field(default_factory=list)

    @property
    def any_action_taken(self) -> bool:
        return bool(self.chunks_modified or self.chunks_dropped)

    def as_dict(self) -> dict:
        return {
            "chunks_scanned": self.chunks_scanned,
            "chunks_modified": self.chunks_modified,
            "chunks_dropped": self.chunks_dropped,
            "sentences_removed": self.sentences_removed,
            "max_risk": round(self.max_risk, 4),
            "labels": self.labels[:10],
        }


def neutralise_text(text: str, suspicious_indices: list[int]) -> tuple[str, int]:
    """Replace the flagged sentences with an explicit marker."""
    if not suspicious_indices:
        return text, 0

    sentences = split_sentences(text)
    flagged = set(suspicious_indices)
    kept: list[str] = []
    removed = 0
    marker_emitted = False

    for index, sentence in enumerate(sentences):
        if index in flagged:
            removed += 1
            # One marker per run of removals: a chunk gutted into fifteen
            # identical markers is noise in the context window.
            if not marker_emitted:
                kept.append(REMOVAL_MARKER)
                marker_emitted = True
            continue
        marker_emitted = False
        kept.append(sentence)

    return " ".join(kept).strip(), removed


def sanitise_chunks(
    chunks: list[ScoredChunk],
) -> tuple[list[ScoredChunk], SanitisationReport]:
    """Sanitise retrieved chunks in place-safe fashion, returning a report."""
    report = SanitisationReport(chunks_scanned=len(chunks))

    if not settings.CONTEXT_SANITISATION_ENABLED:
        return chunks, report

    scanner = get_context_scanner()
    surviving: list[ScoredChunk] = []

    for chunk in chunks:
        # Defang *before* scanning, for two reasons. First, the scan should
        # judge the text that will actually reach the model, not a version we
        # have already fixed. Second -- and this was a real bug -- the scanner
        # returns *sentence indices*, and neutralisation applies them by
        # splitting the text again; scanning one string and neutralising a
        # different one silently misaligns those indices and removes the wrong
        # sentences.
        cleaned = defang_fences(chunk.content)
        result = scanner.scan_detailed(cleaned)
        report.max_risk = max(report.max_risk, result.risk_score)
        for label in result.labels:
            if label not in report.labels:
                report.labels.append(label)

        if result.quarantine:
            # Should be rare: quarantine normally happens at ingest and the
            # chunk never becomes a candidate. Reaching here means the chunk
            # predates the current rules, so drop it and say so.
            report.chunks_dropped += 1
            report.dropped_chunk_ids.append(chunk.chunk_id)
            logger.warning(
                "context_chunk_dropped_at_runtime",
                extra={
                    "chunk_id": chunk.chunk_id,
                    "risk_score": result.risk_score,
                    "labels": result.labels[:5],
                },
            )
            continue

        removed = 0
        if result.neutralise and result.suspicious_sentences:
            cleaned, removed = neutralise_text(cleaned, result.suspicious_sentences)

        if cleaned != chunk.content:
            report.chunks_modified += 1
            report.sentences_removed += removed
            report.modified_chunk_ids.append(chunk.chunk_id)
            # Replace rather than mutate, so a caller holding the original
            # (the citation resolver, the playground) still sees what was
            # actually stored.
            chunk = ScoredChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                content=cleaned,
                source_filename=chunk.source_filename,
                chunk_index=chunk.chunk_index,
                page_number=chunk.page_number,
                section=chunk.section,
                owner_id=chunk.owner_id,
                injection_risk=max(chunk.injection_risk, result.risk_score),
                score=chunk.score,
                vector_score=chunk.vector_score,
                keyword_score=chunk.keyword_score,
                rerank_score=chunk.rerank_score,
                rank_sources=dict(chunk.rank_sources),
                meta={**chunk.meta, "sanitised": True, "sentences_removed": removed},
            )

        surviving.append(chunk)

    return surviving, report
