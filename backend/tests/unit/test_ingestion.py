"""Parsing, cleaning, chunking and upload validation."""

from __future__ import annotations

import io

import pytest

from app.core.exceptions import IngestionError
from app.rag.ingestion.chunker import (
    chunk_blocks,
    estimate_tokens,
    split_sentences,
)
from app.rag.ingestion.cleaner import (
    clean_text,
    count_invisible_characters,
    strip_repeated_lines,
)
from app.rag.ingestion.parsers import ParsedBlock, parse_document, sniff_format
from app.rag.ingestion.pipeline import (
    IngestionPipeline,
    sanitise_filename,
    validate_upload,
)

pytestmark = pytest.mark.unit


# ----------------------------------------------------------------------
# Filename handling
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("../../etc/passwd", "passwd"),
        ("..\\..\\windows\\system32\\config", "config"),
        ("/absolute/path/report.pdf", "report.pdf"),
        ("normal name.pdf", "normal name.pdf"),
        ("", "untitled"),
        ("...", "untitled"),
    ],
)
def test_filename_sanitisation_strips_path_components(raw, expected):
    assert sanitise_filename(raw) == expected


def test_filename_sanitisation_removes_control_characters():
    assert "\x00" not in sanitise_filename("evil\x00name.txt")
    assert "\n" not in sanitise_filename("line\nbreak.txt")


# ----------------------------------------------------------------------
# Upload validation
# ----------------------------------------------------------------------


def test_empty_upload_is_rejected():
    with pytest.raises(IngestionError):
        validate_upload("a.txt", b"")


def test_oversized_upload_is_rejected():
    from app.core.config import settings

    payload = b"x" * (settings.MAX_UPLOAD_SIZE_BYTES + 1)
    with pytest.raises(IngestionError) as caught:
        validate_upload("a.txt", payload)
    assert "exceeds" in caught.value.public_message


def test_disallowed_extension_is_rejected():
    with pytest.raises(IngestionError) as caught:
        validate_upload("payload.exe", b"MZ\x90\x00")
    assert "Unsupported file type" in caught.value.public_message
    # The rejection must not echo the caller's filename back to them.
    assert "payload.exe" not in caught.value.public_message


def test_extension_check_happens_before_parsing():
    # A .exe must never reach a parser, regardless of its content.
    with pytest.raises(IngestionError):
        validate_upload("script.sh", b"#!/bin/sh\nrm -rf /\n")


# ----------------------------------------------------------------------
# Content sniffing
# ----------------------------------------------------------------------


def test_sniffing_accepts_matching_content():
    assert sniff_format(b"%PDF-1.7\n...", "pdf") == "pdf"
    assert sniff_format(b"PK\x03\x04rest", "docx") == "docx"
    assert sniff_format(b"plain words", "txt") == "txt"
    assert sniff_format(b"# Title", "markdown") == "md"


def test_sniffing_rejects_extension_mismatch():
    # A ZIP renamed to .txt would otherwise be handed to the text parser.
    with pytest.raises(IngestionError) as caught:
        sniff_format(b"PK\x03\x04payload", "txt")
    assert "does not match" in caught.value.public_message

    with pytest.raises(IngestionError) as caught:
        sniff_format(b"%PDF-1.4", "md")
    assert "does not match" in caught.value.public_message


# ----------------------------------------------------------------------
# Cleaning
# ----------------------------------------------------------------------


def test_cleaning_folds_ligatures_and_dehyphenates():
    assert clean_text("The conﬁdential employ-\nment policy") == (
        "The confidential employment policy"
    )


def test_cleaning_strips_invisible_and_bidi_characters():
    hostile = "Normal​text‮reversed﻿ end"
    cleaned = clean_text(hostile)
    assert "​" not in cleaned
    assert "‮" not in cleaned
    assert "﻿" not in cleaned
    assert "Normal" in cleaned and "reversed" in cleaned


def test_invisible_characters_are_counted_as_a_signal():
    assert count_invisible_characters("clean text") == 0
    assert count_invisible_characters("a​b‮c") == 2


def test_cleaning_removes_page_number_artefacts():
    assert clean_text("Page 3 of 10\nActual content.") == "Actual content."


def test_running_headers_are_removed():
    pages = [f"ACME CONFIDENTIAL\nContent {i}\nfooter line" for i in range(5)]
    cleaned = strip_repeated_lines(pages)
    assert all("ACME CONFIDENTIAL" not in page for page in cleaned)
    assert all(f"Content {i}" in cleaned[i] for i in range(5))


# ----------------------------------------------------------------------
# Sentence splitting and chunking
# ----------------------------------------------------------------------


def test_sentence_splitting_respects_abbreviations():
    sentences = split_sentences("Dr. Smith signed it. The policy applies to all.")
    assert len(sentences) == 2
    assert sentences[0].startswith("Dr. Smith")


def test_token_estimate_is_monotonic():
    assert estimate_tokens("") == 0
    assert estimate_tokens("one") < estimate_tokens("one two three four five")


def test_chunks_never_span_two_sections():
    blocks = [
        ParsedBlock(text="Leave accrues monthly. " * 30, section="Leave", page_number=1),
        ParsedBlock(text="MFA is required. " * 30, section="Security", page_number=2),
    ]
    chunks = chunk_blocks(blocks, target_tokens=80, overlap_tokens=10)
    assert len(chunks) > 2
    for chunk in chunks:
        assert not ("Leave accrues" in chunk.content and "MFA is" in chunk.content)


def test_chunks_carry_page_and_section_provenance():
    blocks = [ParsedBlock(text="Body text here. " * 40, section="Policy", page_number=7)]
    chunks = chunk_blocks(blocks, target_tokens=60, overlap_tokens=10)
    assert chunks
    assert all(c.page_number == 7 for c in chunks)
    assert all(c.section == "Policy" for c in chunks)


def test_chunk_indices_are_dense_and_ordered():
    blocks = [ParsedBlock(text="Sentence number one. " * 60, page_number=1)]
    chunks = chunk_blocks(blocks, target_tokens=50, overlap_tokens=10)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_overlap_repeats_content_between_neighbours():
    blocks = [
        ParsedBlock(
            text=" ".join(f"Distinct sentence number {i}." for i in range(40)),
            page_number=1,
        )
    ]
    chunks = chunk_blocks(blocks, target_tokens=60, overlap_tokens=20, min_chars=1)
    assert len(chunks) >= 2
    first_words = set(chunks[0].content.split())
    second_words = set(chunks[1].content.split())
    assert first_words & second_words, "expected overlap between adjacent chunks"


def test_oversized_sentence_is_split_without_breaking_words():
    giant = "word " * 4000
    chunks = chunk_blocks([ParsedBlock(text=giant)], target_tokens=100, overlap_tokens=0)
    assert len(chunks) > 1
    for chunk in chunks:
        for token in chunk.content.split():
            assert token == "word"


def test_tiny_trailing_fragments_are_absorbed():
    blocks = [
        ParsedBlock(text="A full sentence of reasonable length here. " * 10 + "Tiny.")
    ]
    chunks = chunk_blocks(blocks, target_tokens=200, overlap_tokens=0, min_chars=120)
    assert all(c.char_count >= 100 for c in chunks[:-1])


def test_chunk_count_is_capped():
    blocks = [ParsedBlock(text="Sentence here. " * 2000)]
    chunks = chunk_blocks(blocks, target_tokens=20, overlap_tokens=0, max_chunks=25)
    assert len(chunks) <= 25


# ----------------------------------------------------------------------
# Parsers
# ----------------------------------------------------------------------


def test_text_parser_splits_paragraphs():
    parsed = parse_document(b"First para.\n\nSecond para.", "a.txt", "txt")
    assert len(parsed.blocks) == 2


def test_markdown_parser_tracks_heading_path():
    source = b"# Handbook\n\n## Leave\n\nStaff get 24 days.\n\n## Security\n\nUse MFA.\n"
    parsed = parse_document(source, "h.md", "md")
    sections = {b.section for b in parsed.blocks if b.kind != "heading"}
    assert "Handbook > Leave" in sections
    assert "Handbook > Security" in sections


def test_markdown_parser_keeps_code_fences_intact():
    source = b"# T\n\n```python\ndef f():\n\n    return 1\n```\n"
    parsed = parse_document(source, "c.md", "md")
    code_blocks = [b for b in parsed.blocks if "def f()" in b.text]
    assert code_blocks and "return 1" in code_blocks[0].text


def test_document_with_no_text_is_rejected():
    with pytest.raises(IngestionError) as caught:
        parse_document(b"   \n\n   ", "empty.txt", "txt")
    assert "No readable text" in caught.value.public_message
    # Operators get the specific cause; the user does not.
    assert "zero blocks" in caught.value.internal_detail


def test_pdf_parser_attributes_text_to_pages():
    from tests.factories import build_pdf

    data = build_pdf(
        [
            "Employees accrue two days of leave per month.",
            "Multi-factor authentication is mandatory for all accounts.",
            "Expense claims must be submitted within 30 days.",
        ]
    )
    parsed = parse_document(data, "handbook.pdf", "pdf")

    assert parsed.page_count == 3
    by_page = {b.page_number: b.text for b in parsed.blocks}
    assert "two days of leave" in by_page[1]
    assert "Multi-factor" in by_page[2]
    assert "30 days" in by_page[3]


def test_pdf_page_numbers_survive_chunking():
    from tests.factories import build_pdf

    data = build_pdf([f"Page {i} discusses topic number {i}. " * 8 for i in range(1, 5)])
    pipeline = IngestionPipeline()
    result = pipeline.run(data, "multi.pdf", "pdf")

    pages = {c.page_number for c in result.chunks}
    assert pages == {1, 2, 3, 4}
    for chunk in result.chunks:
        assert f"Page {chunk.page_number} discusses" in chunk.content


def test_docx_parser_extracts_headings_and_tables():
    docx = pytest.importorskip("docx")

    document = docx.Document()
    document.add_heading("Employee Handbook", level=1)
    document.add_heading("Leave Policy", level=2)
    document.add_paragraph("Employees accrue two days of leave per month.")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Grade"
    table.cell(0, 1).text = "Days"
    table.cell(1, 0).text = "Senior"
    table.cell(1, 1).text = "30"
    buffer = io.BytesIO()
    document.save(buffer)

    parsed = parse_document(buffer.getvalue(), "handbook.docx", "docx")
    texts = [b.text for b in parsed.blocks]
    assert any("accrue two days" in t for t in texts)
    assert any("Grade | Days" in t for t in texts)
    sections = {b.section for b in parsed.blocks if b.section}
    assert "Employee Handbook > Leave Policy" in sections


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------


def test_pipeline_produces_aligned_chunks_and_embeddings():
    pipeline = IngestionPipeline()
    source = (
        b"# Handbook\n\n## Leave\n\n"
        + b"Employees accrue two days of paid leave every month. " * 20
        + b"\n\n## Security\n\n"
        + b"Multi-factor authentication is mandatory for all accounts. " * 20
    )
    result = pipeline.run(source, "handbook.md", "md")

    assert result.chunks
    assert len(result.chunks) == len(result.embeddings) == len(result.scans)
    assert all(len(v) == pipeline.embedder.dimensions for v in result.embeddings)
    assert result.embedding_model
    assert set(result.timings_ms) >= {"parse_ms", "clean_ms", "chunk_ms", "embed_ms"}


def test_pipeline_reports_invisible_characters():
    pipeline = IngestionPipeline()
    hostile = ("Normal policy text here. " * 20) + ("​" * 60)
    result = pipeline.run(hostile.encode("utf-8"), "doc.txt", "txt")
    assert result.invisible_char_count >= 60
    assert "​" not in "".join(c.content for c in result.chunks)
