"""Document lifecycle: upload, ingest, list, fetch, delete.

Ingestion runs synchronously inside the request.  That is a deliberate
simplification: a queue (Celery/RQ + Redis) would be the right answer for large
files, but it doubles the operational surface of the project for no gain at
portfolio scale.  The trade-off, and the exact change needed to move to a
worker, is documented in docs/architecture.md.
"""

from __future__ import annotations

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.core.exceptions import IngestionError, NotFoundError
from app.core.logging import get_logger, redact_for_log
from app.models.document import (
    Document,
    DocumentChunk,
    DocumentStatus,
    DocumentVisibility,
)
from app.models.security_event import (
    SecurityAction,
    SecurityEventType,
    SecurityLayer,
    SecuritySeverity,
)
from app.models.user import User
from app.rag.ingestion.pipeline import (
    IngestionPipeline,
    content_digest,
    validate_upload,
)
from app.services.security_event_service import record_event

logger = get_logger("app.services.documents")


class DocumentService:
    def __init__(self, db: Session, pipeline: IngestionPipeline | None = None) -> None:
        self.db = db
        self.pipeline = pipeline or IngestionPipeline()

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------
    def ingest_upload(
        self,
        *,
        owner: User,
        filename: str,
        data: bytes,
        content_type: str = "application/octet-stream",
        visibility: DocumentVisibility = DocumentVisibility.PRIVATE,
        client_ref: str | None = None,
    ) -> Document:
        safe_name, extension = validate_upload(filename, data)
        digest = content_digest(data)

        existing = self.db.execute(
            select(Document).where(
                Document.owner_id == owner.id, Document.content_sha256 == digest
            )
        ).scalar_one_or_none()
        if existing is not None:
            if existing.status == DocumentStatus.READY.value:
                # Idempotent re-upload. Not an error: users re-drag files.
                logger.info(
                    "duplicate_upload_ignored", extra={"document_id": existing.id}
                )
                return existing

            # A previous attempt failed or died mid-ingest. Returning that row
            # would report "Ingested 0 sections", and because (owner, digest) is
            # unique it would make this content permanently un-uploadable for
            # this user -- one transient failure poisoning the hash forever.
            # Discard the carcass and ingest again.
            logger.info(
                "reingesting_incomplete_document",
                extra={
                    "document_id": existing.id,
                    "previous_status": existing.status,
                },
            )
            self.db.execute(
                delete(DocumentChunk).where(DocumentChunk.document_id == existing.id)
            )
            self.db.delete(existing)
            self.db.commit()

        document = Document(
            owner_id=owner.id,
            filename=safe_name,
            extension=extension,
            content_type=content_type[:128],
            file_size_bytes=len(data),
            content_sha256=digest,
            status=DocumentStatus.PROCESSING.value,
            visibility=visibility.value,
        )
        self.db.add(document)
        self.db.commit()
        self.db.refresh(document)

        try:
            result = self.pipeline.run(data, safe_name, extension)
        except IngestionError as exc:
            document.status = DocumentStatus.FAILED.value
            document.error_message = exc.public_message[:1024]
            self.db.commit()
            record_event(
                self.db,
                event_type=SecurityEventType.INGESTION_FAILED,
                layer=SecurityLayer.RETRIEVAL,
                severity=SecuritySeverity.LOW,
                action=SecurityAction.BLOCK,
                user_id=owner.id,
                resource_type="document",
                resource_id=document.id,
                client_ref=client_ref,
                detail={"reason": exc.internal_detail[:300], "extension": extension},
            )
            raise
        except Exception as exc:
            document.status = DocumentStatus.FAILED.value
            document.error_message = "Internal processing error."
            self.db.commit()
            logger.exception("ingestion_crashed", extra={"document_id": document.id})
            raise IngestionError(
                "The document could not be processed.",
                internal_detail=f"{type(exc).__name__}: {exc}",
            ) from exc

        rows = [
            DocumentChunk(
                document_id=document.id,
                owner_id=owner.id,
                visibility=visibility.value,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                content_sha256=content_digest(chunk.content.encode("utf-8")),
                token_count=chunk.token_count,
                char_count=chunk.char_count,
                page_number=chunk.page_number,
                section=chunk.section,
                source_filename=safe_name,
                embedding=embedding,
                embedding_model=result.embedding_model,
                injection_risk=round(scan.risk_score, 4),
                is_quarantined=scan.quarantine,
                injection_labels=scan.labels,
                meta={},
            )
            for chunk, embedding, scan in zip(
                result.chunks, result.embeddings, result.scans, strict=True
            )
        ]
        self.db.add_all(rows)

        document.status = DocumentStatus.READY.value
        document.page_count = result.page_count
        document.chunk_count = len(rows)
        document.char_count = result.char_count
        document.max_injection_risk = round(result.max_injection_risk, 4)
        document.quarantined_chunk_count = result.quarantined_count
        document.meta = {
            "embedding_model": result.embedding_model,
            "invisible_characters": result.invisible_char_count,
            "timings_ms": result.timings_ms,
        }
        self.db.commit()
        self.db.refresh(document)

        record_event(
            self.db,
            event_type=SecurityEventType.DOCUMENT_UPLOADED,
            layer=SecurityLayer.RETRIEVAL,
            severity=SecuritySeverity.INFO,
            action=SecurityAction.ALLOW,
            user_id=owner.id,
            resource_type="document",
            resource_id=document.id,
            client_ref=client_ref,
            content_ref=redact_for_log(digest),
            detail={
                "extension": extension,
                "size_bytes": len(data),
                "chunks": len(rows),
                "pages": result.page_count,
            },
        )

        if result.quarantined_count:
            # A document carrying embedded instructions is a security-relevant
            # event in its own right, whether or not it is ever retrieved.
            record_event(
                self.db,
                event_type=SecurityEventType.INDIRECT_INJECTION_DETECTED,
                layer=SecurityLayer.CONTEXT,
                severity=SecuritySeverity.HIGH,
                action=SecurityAction.QUARANTINE,
                user_id=owner.id,
                risk_score=result.max_injection_risk,
                detector="ingest_scan",
                resource_type="document",
                resource_id=document.id,
                client_ref=client_ref,
                detail={
                    "quarantined_chunks": result.quarantined_count,
                    "total_chunks": len(rows),
                    "labels": sorted(
                        {label for scan in result.scans for label in scan.labels}
                    )[:10],
                },
            )

        if result.invisible_char_count > 50:
            record_event(
                self.db,
                event_type=SecurityEventType.CHUNK_QUARANTINED,
                layer=SecurityLayer.CONTEXT,
                severity=SecuritySeverity.MEDIUM,
                action=SecurityAction.SANITISE,
                user_id=owner.id,
                detector="invisible_characters",
                resource_type="document",
                resource_id=document.id,
                detail={"invisible_characters": result.invisible_char_count},
            )

        return document

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def list_documents(
        self, *, owner: User, limit: int = 50, offset: int = 0
    ) -> tuple[list[Document], int]:
        total = self.db.execute(
            select(func.count())
            .select_from(Document)
            .where(Document.owner_id == owner.id)
        ).scalar_one()
        rows = (
            self.db.execute(
                select(Document)
                .where(Document.owner_id == owner.id)
                .order_by(Document.created_at.desc())
                .limit(min(limit, 200))
                .offset(max(offset, 0))
            )
            .scalars()
            .all()
        )
        return list(rows), int(total)

    def get_document(
        self, *, document_id: str, requester: User, client_ref: str | None = None
    ) -> Document:
        """Fetch a document, enforcing ownership.

        An unauthorised access attempt and a genuinely missing id both return
        404.  Returning 403 for "exists but not yours" would turn the endpoint
        into an existence oracle for other users' documents.
        """
        document = self.db.get(Document, document_id)

        if document is None:
            raise NotFoundError("Document not found.")

        if document.owner_id != requester.id and not requester.is_admin:
            record_event(
                self.db,
                event_type=SecurityEventType.AUTHORIZATION_DENIED,
                layer=SecurityLayer.AUTH,
                severity=SecuritySeverity.HIGH,
                action=SecurityAction.BLOCK,
                user_id=requester.id,
                risk_score=0.9,
                detector="document_ownership",
                resource_type="document",
                resource_id=document_id,
                client_ref=client_ref,
                detail={"operation": "read"},
            )
            raise NotFoundError("Document not found.")

        return document

    def get_chunks(
        self, *, document_id: str, requester: User, limit: int = 200
    ) -> list[DocumentChunk]:
        self.get_document(document_id=document_id, requester=requester)
        return list(
            self.db.execute(
                select(DocumentChunk)
                .where(DocumentChunk.document_id == document_id)
                .order_by(DocumentChunk.chunk_index)
                .limit(limit)
            )
            .scalars()
            .all()
        )

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------
    def delete_document(
        self, *, document_id: str, requester: User, client_ref: str | None = None
    ) -> None:
        document = self.get_document(
            document_id=document_id, requester=requester, client_ref=client_ref
        )

        # Explicit chunk delete as well as the FK cascade: the cascade depends
        # on the database enforcing it, and SQLite only does so with a pragma.
        self.db.execute(
            delete(DocumentChunk).where(DocumentChunk.document_id == document.id)
        )

        owner_id = document.owner_id
        self.db.delete(document)
        self.db.commit()

        record_event(
            self.db,
            event_type=SecurityEventType.DOCUMENT_DELETED,
            layer=SecurityLayer.RETRIEVAL,
            severity=SecuritySeverity.INFO,
            action=SecurityAction.ALLOW,
            user_id=requester.id,
            resource_type="document",
            resource_id=document_id,
            client_ref=client_ref,
            detail={"owner_was_requester": owner_id == requester.id},
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def stats(self, *, owner: User | None = None) -> dict[str, int]:
        doc_filter = [Document.owner_id == owner.id] if owner else []
        chunk_filter = [DocumentChunk.owner_id == owner.id] if owner else []
        return {
            "documents": int(
                self.db.execute(
                    select(func.count()).select_from(Document).where(*doc_filter)
                ).scalar_one()
            ),
            "chunks": int(
                self.db.execute(
                    select(func.count()).select_from(DocumentChunk).where(*chunk_filter)
                ).scalar_one()
            ),
            "quarantined_chunks": int(
                self.db.execute(
                    select(func.count())
                    .select_from(DocumentChunk)
                    .where(DocumentChunk.is_quarantined.is_(True), *chunk_filter)
                ).scalar_one()
            ),
        }
