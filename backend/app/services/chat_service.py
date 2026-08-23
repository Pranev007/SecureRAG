"""Chat sessions and message persistence.

The service owns *persistence and ownership*; the RAG pipeline owns the
answer.  Keeping them apart means the guardrail sequence has no reason to know
about sessions, and the session layer has no way to produce an answer that
skipped the guardrails.

Every message is persisted, including blocked and refused ones.  A security
dashboard that only sees successful requests is measuring the wrong
population -- the blocked ones are the interesting ones.
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.exceptions import NotFoundError
from app.core.logging import get_logger
from app.models.chat import ChatSession, Message, MessageRole
from app.models.user import User
from app.rag.pipeline import RagPipeline, RagResponse, get_rag_pipeline

logger = get_logger("app.services.chat")

MAX_TITLE_LENGTH = 60


class ChatService:
    def __init__(self, db: Session, pipeline: RagPipeline | None = None) -> None:
        self.db = db
        self.pipeline = pipeline or get_rag_pipeline()

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------
    def get_or_create_session(self, *, user: User, session_id: str | None) -> ChatSession:
        if session_id:
            session = self.db.get(ChatSession, session_id)
            # Same 404-not-403 rule as documents: a session id belonging to
            # someone else must not be distinguishable from one that does not
            # exist.
            if session is None or session.user_id != user.id:
                raise NotFoundError("Chat session not found.")
            return session

        session = ChatSession(user_id=user.id, title="New chat")
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        return session

    def list_sessions(
        self, *, user: User, limit: int = 50, offset: int = 0
    ) -> tuple[list[ChatSession], int]:
        total = self.db.execute(
            select(func.count())
            .select_from(ChatSession)
            .where(ChatSession.user_id == user.id)
        ).scalar_one()
        rows = (
            self.db.execute(
                select(ChatSession)
                .where(ChatSession.user_id == user.id)
                .order_by(ChatSession.updated_at.desc())
                .limit(min(limit, 200))
                .offset(max(offset, 0))
            )
            .scalars()
            .all()
        )
        return list(rows), int(total)

    def get_messages(self, *, session_id: str, user: User) -> list[Message]:
        session = self.get_or_create_session(user=user, session_id=session_id)
        return (
            self.db.execute(
                select(Message)
                .where(Message.session_id == session.id)
                .order_by(Message.created_at)
            )
            .scalars()
            .all()
        )

    def delete_session(self, *, session_id: str, user: User) -> None:
        session = self.get_or_create_session(user=user, session_id=session_id)
        self.db.delete(session)
        self.db.commit()

    # ------------------------------------------------------------------
    # Ask
    # ------------------------------------------------------------------
    def ask(
        self,
        *,
        user: User,
        question: str,
        session_id: str | None = None,
        document_ids: list[str] | None = None,
        top_k: int | None = None,
        client_ref: str | None = None,
    ) -> tuple[ChatSession, Message, RagResponse]:
        session = self.get_or_create_session(user=user, session_id=session_id)

        response = self.pipeline.answer(
            self.db,
            question,
            user=user,
            document_ids=document_ids,
            top_k=top_k,
            client_ref=client_ref,
        )

        # The *stored* user message is the original text. It is the user's own
        # data and they can see it in their history; the audit trail is where
        # content is withheld, not the user's own transcript.
        user_message = Message(
            session_id=session.id,
            user_id=user.id,
            role=MessageRole.USER.value,
            content=question[:20000],
            was_blocked=response.blocked,
            block_reason=response.reason if response.blocked else None,
            risk_score=response.risk_score,
            request_id=response.request_id,
        )
        self.db.add(user_message)

        assistant_message = Message(
            session_id=session.id,
            user_id=user.id,
            role=MessageRole.ASSISTANT.value,
            content=response.answer,
            was_blocked=response.blocked,
            block_reason=response.reason,
            risk_score=response.risk_score,
            grounding_score=response.grounding_score,
            pii_detected=response.pii_detected,
            citations=[
                {
                    "index": s.index,
                    "document_id": s.document_id,
                    "chunk_id": s.chunk_id,
                    "filename": s.filename,
                    "page": s.page_number,
                    "section": s.section,
                    "quote": s.quote,
                    "verified": s.verified,
                    "label": s.label,
                }
                for s in response.sources
            ],
            retrieved_chunk_count=response.retrieved_chunk_count,
            latency_ms=response.total_latency_ms,
            request_id=response.request_id,
            meta={
                "warnings": response.warnings,
                "confidence": response.confidence,
                "timings_ms": response.timings_ms,
                "refused": response.refused,
            },
        )
        self.db.add(assistant_message)

        session.message_count += 2
        if session.title == "New chat" and not response.blocked:
            session.title = _derive_title(question)

        self.db.commit()
        self.db.refresh(assistant_message)
        self.db.refresh(session)

        return session, assistant_message, response


def _derive_title(question: str) -> str:
    cleaned = " ".join(question.split())
    if len(cleaned) <= MAX_TITLE_LENGTH:
        return cleaned or "New chat"
    return cleaned[: MAX_TITLE_LENGTH - 1].rstrip() + "…"
