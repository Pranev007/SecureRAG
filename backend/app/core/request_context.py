"""Per-request ambient context.

A single ``request_id`` is generated at the edge and carried through every
layer -- guardrails, retrieval, LLM call, output validation, security events --
via :mod:`contextvars`.  This is what makes it possible to reconstruct the full
decision trail for one request from the logs without threading an extra
argument through fifteen function signatures.
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar, Token

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_user_id: ContextVar[str | None] = ContextVar("user_id", default=None)


def new_request_id() -> str:
    return uuid.uuid4().hex


def set_request_id(request_id: str) -> Token[str | None]:
    return _request_id.set(request_id)


def get_request_id() -> str | None:
    return _request_id.get()


def set_user_id(user_id: str | None) -> Token[str | None]:
    return _user_id.set(user_id)


def get_user_id() -> str | None:
    return _user_id.get()


def reset_request_id(token: Token[str | None]) -> None:
    _request_id.reset(token)


def reset_user_id(token: Token[str | None]) -> None:
    _user_id.reset(token)
