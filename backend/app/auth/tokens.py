"""JWT issuance and verification.

Claims are deliberately minimal.  The token carries the subject, role and
expiry; everything else is looked up from the database on each request.  A
role baked into a token would keep working after an admin is demoted, so the
token is treated as *identity*, and authorisation is always re-derived from
current state.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt

from app.core.config import settings
from app.core.exceptions import AuthenticationError
from app.models.user import User

ALGORITHM_ALLOWLIST = ("HS256", "HS384", "HS512")
TOKEN_TYPE = "access"


def create_access_token(user: User, *, expires_minutes: int | None = None) -> str:
    now = datetime.now(UTC)
    expiry = now + timedelta(
        minutes=expires_minutes or settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )
    payload: dict[str, Any] = {
        "sub": str(user.id),
        "role": user.role,
        "type": TOKEN_TYPE,
        "iat": int(now.timestamp()),
        "nbf": int(now.timestamp()),
        "exp": int(expiry.timestamp()),
        "jti": uuid.uuid4().hex,
        "iss": settings.APP_NAME,
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode and validate a token, or raise :class:`AuthenticationError`.

    The algorithm allowlist is explicit.  Accepting whatever the token's own
    header declares is the classic JWT vulnerability -- it lets an attacker
    present ``alg: none``, or ask an RS256 verifier to treat the public key as
    an HMAC secret.
    """
    if not token:
        raise AuthenticationError("Not authenticated.")

    algorithms = [settings.JWT_ALGORITHM]
    if settings.JWT_ALGORITHM not in ALGORITHM_ALLOWLIST:
        raise AuthenticationError(
            internal_detail=f"unsupported JWT_ALGORITHM={settings.JWT_ALGORITHM}"
        )

    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=algorithms,
            issuer=settings.APP_NAME,
            options={"require": ["exp", "sub", "iat"], "verify_iss": True},
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthenticationError(
            "Your session has expired. Please sign in again.",
            internal_detail="token expired",
        ) from exc
    except jwt.InvalidTokenError as exc:
        # One generic message: distinguishing "bad signature" from "malformed"
        # tells an attacker how close their forgery is.
        raise AuthenticationError(
            "Invalid authentication credentials.",
            internal_detail=f"{type(exc).__name__}",
        ) from exc

    if payload.get("type") != TOKEN_TYPE:
        raise AuthenticationError(
            "Invalid authentication credentials.",
            internal_detail=f"unexpected token type {payload.get('type')!r}",
        )
    return payload
