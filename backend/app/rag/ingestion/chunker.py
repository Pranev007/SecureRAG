"""Structure-aware chunking.

Why not fixed-length splitting
------------------------------
Cutting every N characters is the default in most RAG tutorials and it is
actively harmful:

* it splits mid-sentence, so a chunk can assert the opposite of the source
  ("Employees are not entitled to" | "carry leave forward");
* it merges unrelated sections, diluting the embedding of both;
* it destroys the page/heading provenance a citation needs.

Strategy implemented here
-------------------------
1. **Respect structure first.**  Blocks are grouped by section (heading path)
   and page.  A chunk never spans two sections, because two sections are two
   topics and one vector cannot represent both well.
2. **Pack to a token budget.**  Within a section, paragraphs accumulate until
   ``CHUNK_TARGET_TOKENS`` is reached -- keeping whole paragraphs together
   wherever they fit.
3. **Degrade gracefully.**  A paragraph larger than the budget is split on
   sentence boundaries; a single sentence larger than the budget is split on
   word boundaries.  Splitting mid-word never happens.
4. **Overlap by whole sentences.**  ``CHUNK_OVERLAP_TOKENS`` of trailing
   sentences are prepended to the next chunk so that a fact spanning a boundary
   is retrievable from either side.  Overlapping by sentence rather than by
   character keeps both copies readable.
5. **Absorb fragments.**  A trailing chunk below ``CHUNK_MIN_CHARS`` is merged
   back into its predecessor: a 40-character chunk is noise in the index.

Every chunk keeps ``page_number`` and ``section`` so a citation can name both.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.core.config import settings
from app.rag.ingestion.parsers import ParsedBlock

# Abbreviations that end in a period but do not end a sentence.
_ABBREVIATIONS = frozenset(
    """
    mr mrs ms dr prof sr jr st vs etc e.g i.e inc ltd co corp dept est fig no
    approx cf al ca pp vol ed min max sec art
    """.split()
)

_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+")
_WORD = re.compile(r"\S+")


def estimate_tokens(text: str) -> int:
    """Approximate the token count of ``text``.

    Deliberately dependency-free.  A real BPE tokenizer (tiktoken) downloads
    its vocabulary at first use, which would make ingestion require network
    access and make the test suite non-hermetic.  The 4-characters-per-token
    heuristic is accurate to roughly +/-15% on English prose, and the chunk
    budget is a soft target, so the extra precision buys nothing here.

    Swapping in a real tokenizer means replacing this one function.
    """
    if not text:
        return 0
    words = len(_WORD.findall(text))
    # Blend a word-based and character-based estimate: word count alone
    # underestimates code and identifiers, character count alone overestimates
    # ordinary prose.
    return max(1, int(round((words * 1.3 + len(text) / 4) / 2)))


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences, tolerating common abbreviations."""
    if not text.strip():
        return []

    raw = _SENTENCE_END.split(text)
    sentences: list[str] = []
    for piece in raw:
        piece = piece.strip()
        if not piece:
            continue
        if sentences:
            previous = sentences[-1]
            last_word = previous.rstrip(".").rsplit(maxsplit=1)
            tail = last_word[-1].lower().strip("(\"'") if last_word else ""
            # Re-join when the previous fragment ended on an abbreviation or a
            # single initial ("J. Smith").
            if tail in _ABBREVIATIONS or (len(tail) == 1 and previous.endswith(".")):
                sentences[-1] = f"{previous} {piece}"
                continue
        sentences.append(piece)
    return sentences or [text.strip()]


@dataclass
class Chunk:
    """A retrieval unit with the provenance needed to cite it."""

    content: str
    chunk_index: int
    page_number: int | None = None
    section: str | None = None
    token_count: int = 0
    char_count: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Group:
    """Consecutive blocks sharing a section and page."""

    section: str | None
    page_number: int | None
    texts: list[str] = field(default_factory=list)


def _group_blocks(blocks: list[ParsedBlock]) -> list[_Group]:
    groups: list[_Group] = []
    for block in blocks:
        text = block.text.strip()
        if not text:
            continue
        if (
            groups
            and groups[-1].section == block.section
            and groups[-1].page_number == block.page_number
        ):
            groups[-1].texts.append(text)
        else:
            groups.append(
                _Group(
                    section=block.section,
                    page_number=block.page_number,
                    texts=[text],
                )
            )
    return groups


def _split_oversized(sentence: str, budget: int) -> list[str]:
    """Split a single over-long sentence on word boundaries."""
    words = _WORD.findall(sentence)
    pieces: list[str] = []
    current: list[str] = []
    for word in words:
        current.append(word)
        if estimate_tokens(" ".join(current)) >= budget:
            pieces.append(" ".join(current))
            current = []
    if current:
        pieces.append(" ".join(current))
    return pieces or [sentence]


def _overlap_tail(sentences: list[str], overlap_tokens: int) -> list[str]:
    """Return the trailing whole sentences worth roughly ``overlap_tokens``."""
    if overlap_tokens <= 0:
        return []
    tail: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        cost = estimate_tokens(sentence)
        # Never let the overlap grow into most of the next chunk.
        if total + cost > overlap_tokens and tail:
            break
        tail.insert(0, sentence)
        total += cost
        if total >= overlap_tokens:
            break
    return tail


def chunk_blocks(
    blocks: list[ParsedBlock],
    *,
    target_tokens: int | None = None,
    overlap_tokens: int | None = None,
    min_chars: int | None = None,
    max_chunks: int | None = None,
) -> list[Chunk]:
    """Turn parsed blocks into retrieval chunks."""
    target = target_tokens or settings.CHUNK_TARGET_TOKENS
    overlap = (
        overlap_tokens if overlap_tokens is not None else settings.CHUNK_OVERLAP_TOKENS
    )
    minimum = min_chars if min_chars is not None else settings.CHUNK_MIN_CHARS
    limit = max_chunks or settings.MAX_CHUNKS_PER_DOCUMENT

    # An overlap at or above the target would make every chunk a copy of its
    # neighbour and could fail to terminate.
    overlap = max(0, min(overlap, target // 2))

    chunks: list[Chunk] = []
    index = 0

    for group in _group_blocks(blocks):
        sentences: list[str] = []
        for text in group.texts:
            sentences.extend(split_sentences(text))

        # Pre-split anything that cannot fit on its own.
        expanded: list[str] = []
        for sentence in sentences:
            if estimate_tokens(sentence) > target:
                expanded.extend(_split_oversized(sentence, target))
            else:
                expanded.append(sentence)

        current: list[str] = []
        current_tokens = 0

        for sentence in expanded:
            cost = estimate_tokens(sentence)
            if current and current_tokens + cost > target:
                content = " ".join(current).strip()
                chunks.append(
                    Chunk(
                        content=content,
                        chunk_index=index,
                        page_number=group.page_number,
                        section=group.section,
                        token_count=current_tokens,
                        char_count=len(content),
                    )
                )
                index += 1
                if index >= limit:
                    return _absorb_fragments(chunks, minimum)
                carry = _overlap_tail(current, overlap)
                current = [*carry, sentence]
                current_tokens = sum(estimate_tokens(s) for s in current)
            else:
                current.append(sentence)
                current_tokens += cost

        if current:
            content = " ".join(current).strip()
            chunks.append(
                Chunk(
                    content=content,
                    chunk_index=index,
                    page_number=group.page_number,
                    section=group.section,
                    token_count=current_tokens,
                    char_count=len(content),
                )
            )
            index += 1
            if index >= limit:
                return _absorb_fragments(chunks, minimum)

    return _absorb_fragments(chunks, minimum)


def _absorb_fragments(chunks: list[Chunk], min_chars: int) -> list[Chunk]:
    """Merge undersized chunks into the previous chunk of the same section."""
    if not chunks:
        return []

    merged: list[Chunk] = []
    for chunk in chunks:
        if (
            merged
            and chunk.char_count < min_chars
            and merged[-1].section == chunk.section
            and merged[-1].page_number == chunk.page_number
        ):
            previous = merged[-1]
            previous.content = f"{previous.content} {chunk.content}".strip()
            previous.char_count = len(previous.content)
            previous.token_count = estimate_tokens(previous.content)
            continue
        merged.append(chunk)

    # Re-number so chunk_index is dense and matches storage order.
    for position, chunk in enumerate(merged):
        chunk.chunk_index = position
    return merged
