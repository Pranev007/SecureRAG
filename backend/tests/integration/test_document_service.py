"""Document service: ingest, store, list, authorise, delete."""

from __future__ import annotations

import pytest
from sqlalchemy import select

from app.core.exceptions import IngestionError, NotFoundError
from app.models.document import Document, DocumentChunk, DocumentStatus
from app.models.security_event import SecurityEvent, SecurityEventType
from app.rag.embeddings.factory import get_embedding_provider
from app.rag.ingestion.pipeline import content_digest
from app.services.document_service import DocumentService
from tests.factories import (
    HANDBOOK_MARKDOWN,
    SECURITY_POLICY_MARKDOWN,
    build_pdf,
)

pytestmark = pytest.mark.integration


def _ingest(db, owner, content: bytes, name: str):
    return DocumentService(db).ingest_upload(
        owner=owner, filename=name, data=content, content_type="text/markdown"
    )


def test_ingest_persists_document_and_chunks(db, user):
    document = _ingest(db, user, HANDBOOK_MARKDOWN.encode(), "handbook.md")

    assert document.status == DocumentStatus.READY.value
    assert document.chunk_count > 0
    assert document.char_count > 0
    assert document.owner_id == user.id
    assert document.content_sha256

    chunks = (
        db.execute(select(DocumentChunk).where(DocumentChunk.document_id == document.id))
        .scalars()
        .all()
    )
    assert len(chunks) == document.chunk_count
    assert all(c.owner_id == user.id for c in chunks)
    assert all(c.source_filename == "handbook.md" for c in chunks)


def test_stored_embeddings_round_trip_with_correct_dimensions(db, user):
    document = _ingest(db, user, HANDBOOK_MARKDOWN.encode(), "handbook.md")
    expected_dimensions = get_embedding_provider().dimensions

    chunks = (
        db.execute(select(DocumentChunk).where(DocumentChunk.document_id == document.id))
        .scalars()
        .all()
    )

    for chunk in chunks:
        assert chunk.embedding is not None
        assert len(chunk.embedding) == expected_dimensions
        assert all(isinstance(component, float) for component in chunk.embedding)
        assert chunk.embedding_model


def test_chunk_metadata_carries_section_and_index(db, user):
    document = _ingest(db, user, HANDBOOK_MARKDOWN.encode(), "handbook.md")
    chunks = (
        db.execute(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document.id)
            .order_by(DocumentChunk.chunk_index)
        )
        .scalars()
        .all()
    )

    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert any(c.section and "Leave Policy" in c.section for c in chunks)


def test_pdf_ingest_records_page_numbers(db, user):
    data = build_pdf(
        [
            "Employees accrue two days of paid leave every single month here.",
            "Multi-factor authentication is mandatory for every company account.",
        ]
    )
    document = DocumentService(db).ingest_upload(
        owner=user, filename="policy.pdf", data=data, content_type="application/pdf"
    )

    assert document.page_count == 2
    chunks = (
        db.execute(select(DocumentChunk).where(DocumentChunk.document_id == document.id))
        .scalars()
        .all()
    )
    assert {c.page_number for c in chunks} == {1, 2}


def test_duplicate_upload_is_idempotent(db, user):
    first = _ingest(db, user, HANDBOOK_MARKDOWN.encode(), "handbook.md")
    second = _ingest(db, user, HANDBOOK_MARKDOWN.encode(), "handbook-copy.md")
    assert first.id == second.id


def test_a_failed_ingest_can_be_retried(db, user):
    """Regression: dedup returned the failed row, poisoning the hash forever.

    (owner_id, content_sha256) is unique, so returning a non-READY document
    meant one transient failure made that content permanently un-uploadable --
    every retry reported "Ingested 0 sections" and returned the same carcass.
    """
    service = DocumentService(db)
    good = HANDBOOK_MARKDOWN.encode()

    # Simulate an ingest that died after the row was created.
    stub = Document(
        owner_id=user.id,
        filename="handbook.md",
        extension="md",
        content_type="text/markdown",
        file_size_bytes=len(good),
        content_sha256=content_digest(good),
        status=DocumentStatus.PROCESSING.value,
    )
    db.add(stub)
    db.commit()
    stub_id = stub.id

    retried = service.ingest_upload(
        owner=user, filename="handbook.md", data=good, content_type="text/markdown"
    )

    assert retried.id != stub_id, "the incomplete row should have been replaced"
    assert retried.status == DocumentStatus.READY.value
    assert retried.chunk_count > 0


def test_a_previously_failed_document_is_replaced_not_returned(db, user):
    service = DocumentService(db)
    good = HANDBOOK_MARKDOWN.encode()

    failed = Document(
        owner_id=user.id,
        filename="handbook.md",
        extension="md",
        content_type="text/markdown",
        file_size_bytes=len(good),
        content_sha256=content_digest(good),
        status=DocumentStatus.FAILED.value,
        error_message="earlier failure",
    )
    db.add(failed)
    db.commit()

    retried = service.ingest_upload(
        owner=user, filename="handbook.md", data=good, content_type="text/markdown"
    )
    assert retried.status == DocumentStatus.READY.value
    assert retried.error_message is None

    # And only one document survives for that digest.
    rows = (
        db.execute(
            select(Document).where(Document.content_sha256 == content_digest(good))
        )
        .scalars()
        .all()
    )
    assert len(rows) == 1


def test_two_users_may_hold_the_same_file_independently(db, user, other_user):
    mine = _ingest(db, user, HANDBOOK_MARKDOWN.encode(), "handbook.md")
    theirs = _ingest(db, other_user, HANDBOOK_MARKDOWN.encode(), "handbook.md")

    assert mine.id != theirs.id
    assert mine.owner_id != theirs.owner_id


def test_failed_ingest_marks_document_and_raises(db, user):
    service = DocumentService(db)
    with pytest.raises(IngestionError):
        service.ingest_upload(owner=user, filename="bad.exe", data=b"MZ\x90\x00")

    events = db.execute(select(SecurityEvent)).scalars().all()
    assert all(e.event_type != SecurityEventType.DOCUMENT_UPLOADED.value for e in events)


def test_listing_returns_only_the_callers_documents(db, user, other_user):
    _ingest(db, user, HANDBOOK_MARKDOWN.encode(), "mine.md")
    _ingest(db, other_user, SECURITY_POLICY_MARKDOWN.encode(), "theirs.md")

    service = DocumentService(db)
    mine, total = service.list_documents(owner=user)
    assert total == 1
    assert [d.filename for d in mine] == ["mine.md"]


def test_fetching_another_users_document_raises_not_found(db, user, other_user):
    theirs = _ingest(db, other_user, HANDBOOK_MARKDOWN.encode(), "theirs.md")
    service = DocumentService(db)

    with pytest.raises(NotFoundError):
        service.get_document(document_id=theirs.id, requester=user)


def test_cross_user_access_attempt_is_audited(db, user, other_user):
    theirs = _ingest(db, other_user, HANDBOOK_MARKDOWN.encode(), "theirs.md")
    service = DocumentService(db)

    with pytest.raises(NotFoundError):
        service.get_document(document_id=theirs.id, requester=user)

    denials = (
        db.execute(
            select(SecurityEvent).where(
                SecurityEvent.event_type == SecurityEventType.AUTHORIZATION_DENIED.value
            )
        )
        .scalars()
        .all()
    )
    assert len(denials) == 1
    assert denials[0].user_id == user.id
    assert denials[0].resource_id == theirs.id


def test_admin_may_read_any_document(db, user, admin_user):
    document = _ingest(db, user, HANDBOOK_MARKDOWN.encode(), "mine.md")
    fetched = DocumentService(db).get_document(
        document_id=document.id, requester=admin_user
    )
    assert fetched.id == document.id


def test_delete_removes_document_and_its_chunks(db, user):
    document = _ingest(db, user, HANDBOOK_MARKDOWN.encode(), "handbook.md")
    service = DocumentService(db)
    service.delete_document(document_id=document.id, requester=user)

    remaining = (
        db.execute(select(DocumentChunk).where(DocumentChunk.document_id == document.id))
        .scalars()
        .all()
    )
    assert remaining == []
    with pytest.raises(NotFoundError):
        service.get_document(document_id=document.id, requester=user)


def test_delete_of_another_users_document_is_refused(db, user, other_user):
    theirs = _ingest(db, other_user, HANDBOOK_MARKDOWN.encode(), "theirs.md")
    service = DocumentService(db)

    with pytest.raises(NotFoundError):
        service.delete_document(document_id=theirs.id, requester=user)

    assert (
        db.execute(select(DocumentChunk).where(DocumentChunk.document_id == theirs.id))
        .scalars()
        .first()
        is not None
    )


def test_upload_emits_an_audit_event_without_content(db, user):
    document = _ingest(db, user, HANDBOOK_MARKDOWN.encode(), "handbook.md")

    event = db.execute(
        select(SecurityEvent).where(
            SecurityEvent.event_type == SecurityEventType.DOCUMENT_UPLOADED.value
        )
    ).scalar_one()

    assert event.resource_id == document.id
    assert event.user_id == user.id
    assert event.detail["chunks"] == document.chunk_count
    # The audit trail must not contain document text.
    serialised = str(event.detail)
    assert "accrue two days" not in serialised
    assert "Leave Policy" not in serialised
