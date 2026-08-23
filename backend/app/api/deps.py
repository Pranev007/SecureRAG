"""FastAPI dependencies: authentication, authorisation and rate limiting.

Making these dependencies rather than in-handler calls is a security property,
not a style choice: a route that forgets ``Depends(get_current_user)`` is
visibly unauthenticated in its own signature and in the generated OpenAPI
document, which makes the omission easy to spot in review.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.auth.tokens import decode_access_token
from app.core.config import settings
from app.core.exceptions import AuthenticationError, AuthorizationError, RateLimitError
from app.core.logging import get_logger
from app.core.request_context import set_user_id
from app.db.session import get_db
from app.models.security_event import (
    SecurityAction,
    SecurityEventType,
    SecurityLayer,
    SecuritySeverity,
)
from app.models.user import User
from app.security.rate_limit import limits_for
from app.services.security_event_service import hash_client, record_event

logger = get_logger("app.api.deps")

# auto_error=False so a missing header raises our own AuthenticationError and
# is rendered in the standard error envelope like everything else.
bearer_scheme = HTTPBearer(auto_error=False, description="JWT access token")

DbSession = Annotated[Session, Depends(get_db)]


def client_reference(request: Request) -> str | None:
    """A stable, non-identifying reference for the calling client.

    ``X-Forwarded-For`` is only consulted for the left-most entry and is hashed
    immediately.  It is spoofable, so this is a diagnostic aid for spotting a
    single noisy source -- never an authorisation input.
    """
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        candidate = forwarded.split(",")[0].strip()
    elif request.client:
        candidate = request.client.host
    else:
        return None
    return hash_client(candidate)


ClientRef = Annotated[str | None, Depends(client_reference)]


def get_current_user(
    db: DbSession,
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(bearer_scheme)
    ] = None,
) -> User:
    """Resolve the caller from the bearer token.

    The user is re-loaded from the database on every request rather than
    trusted from the token body, so a deactivated or demoted account loses
    access immediately instead of at token expiry.
    """
    if credentials is None or not credentials.credentials:
        raise AuthenticationError("Not authenticated.")

    payload = decode_access_token(credentials.credentials)
    user_id = payload.get("sub")
    if not user_id:
        raise AuthenticationError("Invalid authentication credentials.")

    user = db.get(User, user_id)
    if user is None or not user.is_active:
        raise AuthenticationError(
            "Invalid authentication credentials.",
            internal_detail="token subject missing or inactive",
        )

    set_user_id(user.id)
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def require_admin(
    db: DbSession, current_user: CurrentUser, client_ref: ClientRef = None
) -> User:
    """Restrict an endpoint to administrators."""
    if not current_user.is_admin:
        record_event(
            db,
            event_type=SecurityEventType.AUTHORIZATION_DENIED,
            layer=SecurityLayer.AUTH,
            severity=SecuritySeverity.MEDIUM,
            action=SecurityAction.BLOCK,
            user_id=current_user.id,
            risk_score=0.6,
            detector="role_check",
            client_ref=client_ref,
            detail={"required_role": "admin", "actual_role": current_user.role},
        )
        raise AuthorizationError("Administrator access is required.")
    return current_user


AdminUser = Annotated[User, Depends(require_admin)]


def rate_limit(bucket: str) -> Callable[..., None]:
    """Build a dependency that applies one rate-limit bucket to an endpoint.

    Requests are keyed by user id when authenticated and by hashed client
    reference otherwise, so one user cannot exhaust another's budget while an
    unauthenticated flood is still bounded.

    Implemented as a closure rather than a callable class on purpose: FastAPI
    resolves a dependency's type hints through the callable's ``__globals__``,
    which a class *instance* does not have.  With ``from __future__ import
    annotations`` in effect, a callable-class dependency silently fails to
    resolve ``Request``/``Session`` and FastAPI reclassifies them as request
    body fields -- turning every guarded endpoint into a 422.
    """

    def _enforce(
        request: Request,
        db: DbSession,
        client_ref: ClientRef = None,
    ) -> None:
        if not settings.RATE_LIMIT_ENABLED:
            return

        limiter, limit = limits_for(bucket)

        subject = "anonymous"
        user_id: str | None = None
        credentials = request.headers.get("Authorization", "")
        if credentials.lower().startswith("bearer "):
            try:
                payload = decode_access_token(credentials.split(" ", 1)[1])
                user_id = payload.get("sub")
                subject = f"user:{user_id}"
            except AuthenticationError:
                # An invalid token still gets limited -- by client -- so token
                # guessing is not a way around the limiter.
                subject = "invalid-token"

        key = f"{bucket}:{subject}:{client_ref or 'unknown'}"
        decision = limiter.check(key, limit)
        if decision.allowed:
            return

        record_event(
            db,
            event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
            layer=SecurityLayer.INPUT,
            severity=SecuritySeverity.MEDIUM,
            action=SecurityAction.BLOCK,
            user_id=user_id,
            risk_score=0.5,
            detector=f"rate_limit:{bucket}",
            client_ref=client_ref,
            detail={
                "bucket": bucket,
                "limit_per_minute": limit,
                "path": request.url.path,
            },
        )
        raise RateLimitError(retry_after=decision.retry_after)

    _enforce.__name__ = f"rate_limit_{bucket}"
    return _enforce


chat_rate_limit = rate_limit("chat")
upload_rate_limit = rate_limit("upload")
auth_rate_limit = rate_limit("auth")
