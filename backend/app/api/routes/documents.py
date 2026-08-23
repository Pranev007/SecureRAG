"""Document management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Query, UploadFile, status

from app.api.deps import ClientRef, CurrentUser, DbSession, upload_rate_limit
from app.core.config import settings
from app.core.exceptions import IngestionError
from app.core.logging import get_logger
from app.schemas.common import MessageResponse, Page
from app.schemas.document import (
    DocumentChunkResponse,
    DocumentDetailResponse,
    DocumentResponse,
    DocumentUploadResponse,
)
from app.services.document_service import DocumentService

logger = get_logger("app.api.documents")

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a document",
    dependencies=[Depends(upload_rate_limit)],
)
async def upload_document(
    db: DbSession,
    current_user: CurrentUser,
    file: UploadFile = File(..., description="PDF, DOCX, TXT or Markdown"),
    client_ref: ClientRef = None,
) -> DocumentUploadResponse:
    """Ingest a document into the caller's private corpus.

    The upload is read with a hard byte ceiling rather than via
    ``await file.read()``: reading first and checking the size afterwards means
    a multi-gigabyte upload is already in memory by the time it is rejected.
    """
    limit = settings.MAX_UPLOAD_SIZE_BYTES
    buffer = bytearray()
    while chunk := await file.read(1024 * 256):
        buffer.extend(chunk)
        if len(buffer) > limit:
            raise IngestionError(
                f"File exceeds the {settings.MAX_UPLOAD_SIZE_MB} MB limit.",
                internal_detail="upload aborted mid-stream at the size ceiling",
            )

    document = DocumentService(db).ingest_upload(
        owner=current_user,
        filename=file.filename or "untitled",
        data=bytes(buffer),
        content_type=file.content_type or "application/octet-stream",
        client_ref=client_ref,
    )

    warnings: list[str] = []
    if document.quarantined_chunk_count:
        warnings.append(
            f"{document.quarantined_chunk_count} of {document.chunk_count} sections "
            "contain text that looks like instructions aimed at an AI assistant. "
            "Those sections have been excluded from retrieval."
        )
    if document.meta.get("invisible_characters", 0) > 50:
        warnings.append(
            "This document contains a large number of invisible characters, which "
            "can be used to hide text from human readers. They have been removed."
        )

    return DocumentUploadResponse(
        document=DocumentResponse.model_validate(document),
        message=f"Ingested {document.chunk_count} sections from {document.filename}.",
        warnings=warnings,
    )


@router.get(
    "",
    response_model=Page[DocumentResponse],
    summary="List the caller's documents",
)
def list_documents(
    db: DbSession,
    current_user: CurrentUser,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> Page[DocumentResponse]:
    documents, total = DocumentService(db).list_documents(
        owner=current_user, limit=limit, offset=offset
    )
    return Page[DocumentResponse](
        items=[DocumentResponse.model_validate(d) for d in documents],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentDetailResponse,
    summary="Fetch one document",
)
def get_document(
    document_id: str,
    db: DbSession,
    current_user: CurrentUser,
    include_chunks: bool = Query(
        default=False, description="Include the document's chunks"
    ),
    client_ref: ClientRef = None,
) -> DocumentDetailResponse:
    """Fetch a document the caller owns.

    A document belonging to another user returns 404, not 403: a 403 would
    confirm that the id exists.
    """
    service = DocumentService(db)
    document = service.get_document(
        document_id=document_id, requester=current_user, client_ref=client_ref
    )
    response = DocumentDetailResponse.model_validate(document)
    if include_chunks:
        chunks = service.get_chunks(document_id=document_id, requester=current_user)
        response.chunks = [DocumentChunkResponse.model_validate(c) for c in chunks]
    return response


@router.delete(
    "/{document_id}",
    response_model=MessageResponse,
    summary="Delete a document and its chunks",
)
def delete_document(
    document_id: str,
    db: DbSession,
    current_user: CurrentUser,
    client_ref: ClientRef = None,
) -> MessageResponse:
    DocumentService(db).delete_document(
        document_id=document_id, requester=current_user, client_ref=client_ref
    )
    return MessageResponse(message="Document deleted.")
