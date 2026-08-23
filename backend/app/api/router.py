"""Aggregate API router."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import auth, chat, documents, evaluation, security

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(documents.router)
api_router.include_router(chat.router)
api_router.include_router(security.router)
api_router.include_router(evaluation.router)
