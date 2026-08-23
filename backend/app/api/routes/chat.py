"""Chat endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, status

from app.api.deps import ClientRef, CurrentUser, DbSession, chat_rate_limit
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ChatSessionDetailResponse,
    ChatSessionResponse,
    CitationResponse,
    MessageResponse,
    SecurityStatus,
)
from app.schemas.common import MessageResponse as SimpleMessage
from app.schemas.common import Page
from app.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post(
    "",
    response_model=ChatResponse,
    summary="Ask a question about your documents",
    dependencies=[Depends(chat_rate_limit)],
)
def ask(
    payload: ChatRequest,
    db: DbSession,
    current_user: CurrentUser,
    client_ref: ClientRef = None,
) -> ChatResponse:
    """Run the full guarded RAG pipeline.

    A request rejected by the guardrails returns **200 with
    ``security.blocked = true``**, not an HTTP error.  The request was
    processed successfully; the *policy decision* was to refuse it, and the
    client needs the same response shape either way to render it in the
    transcript.  HTTP errors are reserved for the request itself being wrong.
    """
    session, message, response = ChatService(db).ask(
        user=current_user,
        question=payload.question,
        session_id=payload.session_id,
        document_ids=payload.document_ids,
        top_k=payload.top_k,
        client_ref=client_ref,
    )

    return ChatResponse(
        answer=response.answer,
        session_id=session.id,
        message_id=message.id,
        sources=[
            CitationResponse(
                index=s.index,
                document_id=s.document_id,
                chunk_id=s.chunk_id,
                filename=s.filename,
                page=s.page_number,
                section=s.section,
                quote=s.quote,
                verified=s.verified,
                label=s.label,
            )
            for s in response.sources
        ],
        security=SecurityStatus(
            blocked=response.blocked,
            refused=response.refused,
            reason=response.reason,
            risk_score=response.risk_score,
            grounding_score=response.grounding_score,
            confidence=response.confidence,
            pii_detected=response.pii_detected,
            pii_types=response.pii_types,
            warnings=response.warnings,
        ),
        retrieved_chunk_count=response.retrieved_chunk_count,
        latency_ms=response.total_latency_ms,
        timings_ms=response.timings_ms,
        request_id=response.request_id,
    )


@router.get(
    "/sessions",
    response_model=Page[ChatSessionResponse],
    summary="List your chat sessions",
)
def list_sessions(
    db: DbSession,
    current_user: CurrentUser,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> Page[ChatSessionResponse]:
    sessions, total = ChatService(db).list_sessions(
        user=current_user, limit=limit, offset=offset
    )
    return Page[ChatSessionResponse](
        items=[ChatSessionResponse.model_validate(s) for s in sessions],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{session_id}",
    response_model=ChatSessionDetailResponse,
    summary="Fetch one chat session with its messages",
)
def get_session(
    session_id: str, db: DbSession, current_user: CurrentUser
) -> ChatSessionDetailResponse:
    service = ChatService(db)
    session = service.get_or_create_session(user=current_user, session_id=session_id)
    messages = service.get_messages(session_id=session_id, user=current_user)

    detail = ChatSessionDetailResponse.model_validate(session)
    detail.messages = [MessageResponse.model_validate(m) for m in messages]
    return detail


@router.delete(
    "/{session_id}",
    response_model=SimpleMessage,
    status_code=status.HTTP_200_OK,
    summary="Delete a chat session",
)
def delete_session(
    session_id: str, db: DbSession, current_user: CurrentUser
) -> SimpleMessage:
    ChatService(db).delete_session(session_id=session_id, user=current_user)
    return SimpleMessage(message="Chat session deleted.")
