"""Document ingestion pipeline.

    validate -> parse -> clean -> chunk -> scan -> embed -> persist

The stages are separated so each can be tested in isolation and so the
security scan sits *between* chunking and embedding: a chunk carrying an
embedded instruction is scored once, at ingest, rather than on every retrieval.
Doing the work here means the expensive path (per query) stays cheap, and a
malicious chunk is quarantined before it can ever be selected.
"""

from __future__ import annotations

import hashlib
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from app.core.config import settings
from app.core.exceptions import IngestionError
from app.core.logging import get_logger
from app.rag.embeddings.base import EmbeddingProvider
from app.rag.embeddings.factory import get_embedding_provider
from app.rag.ingestion.chunker import Chunk, chunk_blocks
from app.rag.ingestion.cleaner import (
    clean_text,
    count_invisible_characters,
    strip_repeated_lines,
)
from app.rag.ingestion.parsers import ParsedBlock, parse_document

logger = get_logger("app.rag.ingestion")


@dataclass
class ChunkScanResult:
    """Outcome of scanning one chunk for embedded instructions."""

    risk_score: float = 0.0
    labels: list[str] = field(default_factory=list)
    quarantine: bool = False


class ChunkScanner(Protocol):
    """Scores a chunk for indirect prompt injection.

    Injected rather than imported directly so ingestion can be tested without
    the security layer, and so the security layer can evolve independently.
    The real implementation is
    :class:`app.security.context_scanner.IndirectInjectionScanner`.
    """

    def scan(self, text: str) -> ChunkScanResult: ...


class NullChunkScanner:
    """No-op scanner. Used only in tests that isolate the ingestion stages."""

    def scan(self, text: str) -> ChunkScanResult:
        return ChunkScanResult()


@dataclass
class IngestionResult:
    chunks: list[Chunk]
    embeddings: list[list[float]]
    scans: list[ChunkScanResult]
    page_count: int
    char_count: int
    embedding_model: str
    invisible_char_count: int
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def max_injection_risk(self) -> float:
        return max((s.risk_score for s in self.scans), default=0.0)

    @property
    def quarantined_count(self) -> int:
        return sum(1 for s in self.scans if s.quarantine)


# ---------------------------------------------------------------------------
# Upload validation
# ---------------------------------------------------------------------------


def sanitise_filename(filename: str) -> str:
    """Reduce a client-supplied filename to a safe display name.

    The name is never used to build a filesystem path (stored files are named
    by UUID), but it *is* rendered in the UI and included in citations, so it
    still needs directory components and control characters removed.
    """
    name = unicodedata.normalize("NFKC", filename or "")
    name = name.replace("\\", "/").split("/")[-1]
    name = "".join(c for c in name if c.isprintable() and c not in '<>:"|?*')
    name = name.strip(". ")
    return name[:255] or "untitled"


def extract_extension(filename: str) -> str:
    suffix = Path(sanitise_filename(filename)).suffix.lower().lstrip(".")
    return suffix or ""


def validate_upload(filename: str, data: bytes) -> tuple[str, str]:
    """Validate an upload and return ``(safe_filename, extension)``.

    Order matters: cheap checks (size, extension) run before anything parses
    attacker-controlled bytes.
    """
    safe_name = sanitise_filename(filename)

    if not data:
        raise IngestionError("The uploaded file is empty.")

    if len(data) > settings.MAX_UPLOAD_SIZE_BYTES:
        raise IngestionError(
            f"File exceeds the {settings.MAX_UPLOAD_SIZE_MB} MB limit.",
            internal_detail=f"size={len(data)}",
        )

    extension = extract_extension(safe_name)
    if extension not in settings.ALLOWED_FILE_EXTENSIONS:
        allowed = ", ".join(sorted(settings.ALLOWED_FILE_EXTENSIONS))
        raise IngestionError(
            f"Unsupported file type. Allowed: {allowed}.",
            internal_detail=f"extension={extension!r}",
        )

    return safe_name, extension


def content_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _default_scanner() -> ChunkScanner:
    """The real indirect-injection scanner.

    Imported lazily to keep the dependency one-directional: the security layer
    knows about ingestion types, ingestion does not import the security layer
    at module scope.
    """
    from app.security.context_scanner import get_context_scanner

    return get_context_scanner()


class IngestionPipeline:
    def __init__(
        self,
        embedder: EmbeddingProvider | None = None,
        scanner: ChunkScanner | None = None,
    ) -> None:
        self._embedder = embedder
        self._scanner = scanner if scanner is not None else _default_scanner()

    @property
    def embedder(self) -> EmbeddingProvider:
        if self._embedder is None:
            self._embedder = get_embedding_provider()
        return self._embedder

    def run(self, data: bytes, filename: str, extension: str) -> IngestionResult:
        timings: dict[str, float] = {}

        started = time.perf_counter()
        parsed = parse_document(data, filename, extension)
        timings["parse_ms"] = round((time.perf_counter() - started) * 1000, 2)

        started = time.perf_counter()
        blocks = self._clean_blocks(parsed.blocks)
        invisible = sum(count_invisible_characters(b.text) for b in parsed.blocks)
        timings["clean_ms"] = round((time.perf_counter() - started) * 1000, 2)

        if not blocks:
            raise IngestionError(
                "The document contained no usable text after cleaning.",
                internal_detail="all blocks empty post-clean",
            )

        started = time.perf_counter()
        chunks = chunk_blocks(blocks)
        timings["chunk_ms"] = round((time.perf_counter() - started) * 1000, 2)

        if not chunks:
            raise IngestionError(
                "The document produced no chunks.",
                internal_detail="chunker returned empty list",
            )

        started = time.perf_counter()
        scans = [self._scanner.scan(chunk.content) for chunk in chunks]
        timings["scan_ms"] = round((time.perf_counter() - started) * 1000, 2)

        started = time.perf_counter()
        # Quarantined chunks are still embedded and stored: the owner can see
        # what was flagged and why, and the retriever filters them out with a
        # WHERE clause. Discarding them would silently lose the user's data.
        embeddings = self.embedder.embed_documents([c.content for c in chunks])
        timings["embed_ms"] = round((time.perf_counter() - started) * 1000, 2)

        if len(embeddings) != len(chunks):  # pragma: no cover - provider contract
            raise IngestionError(
                "Embedding generation failed.",
                internal_detail=f"{len(embeddings)} vectors for {len(chunks)} chunks",
            )

        logger.info(
            "ingestion_completed",
            extra={
                "chunks": len(chunks),
                "pages": parsed.page_count,
                "quarantined": sum(1 for s in scans if s.quarantine),
                "invisible_chars": invisible,
                **timings,
            },
        )

        return IngestionResult(
            chunks=chunks,
            embeddings=embeddings,
            scans=scans,
            page_count=parsed.page_count,
            char_count=sum(c.char_count for c in chunks),
            embedding_model=self.embedder.model,
            invisible_char_count=invisible,
            timings_ms=timings,
        )

    def _clean_blocks(self, blocks: list[ParsedBlock]) -> list[ParsedBlock]:
        """Clean block text and remove running headers/footers."""
        pages: dict[int | None, list[int]] = {}
        for index, block in enumerate(blocks):
            pages.setdefault(block.page_number, []).append(index)

        if len(pages) >= 3:
            page_texts = [
                "\n".join(blocks[i].text for i in indices) for indices in pages.values()
            ]
            stripped = strip_repeated_lines(page_texts)
            removed: set[str] = set()
            for before, after in zip(page_texts, stripped, strict=True):
                removed |= {
                    line.strip()
                    for line in before.split("\n")
                    if line.strip() and line.strip() not in after
                }
            if removed:
                blocks = [b for b in blocks if b.text.strip() not in removed]

        cleaned: list[ParsedBlock] = []
        for block in blocks:
            text = clean_text(block.text)
            if text:
                cleaned.append(
                    ParsedBlock(
                        text=text,
                        page_number=block.page_number,
                        section=block.section,
                        kind=block.kind,
                    )
                )
        return cleaned
