"""Authentication endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, status

from app.api.deps import ClientRef, CurrentUser, DbSession, auth_rate_limit
from app.auth.service import AuthService
from app.auth.tokens import create_access_token
from app.core.config import settings
from app.schemas.auth import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)

router = APIRouter(prefix="/auth", tags=["auth"])


def _token_response(user) -> TokenResponse:
    return TokenResponse(
        access_token=create_access_token(user),
        expires_in_seconds=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=UserResponse.model_validate(user),
    )


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an account",
    dependencies=[Depends(auth_rate_limit)],
)
def register(
    payload: RegisterRequest, db: DbSession, client_ref: ClientRef = None
) -> TokenResponse:
    """Register a new user.

    The first account created becomes an administrator, which avoids shipping a
    default admin credential.  Set ``ALLOW_REGISTRATION=false`` once your
    accounts exist.
    """
    user = AuthService(db).register(
        email=payload.email,
        password=payload.password,
        full_name=payload.full_name,
        client_ref=client_ref,
    )
    return _token_response(user)


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Exchange credentials for an access token",
    dependencies=[Depends(auth_rate_limit)],
)
def login(
    payload: LoginRequest, db: DbSession, client_ref: ClientRef = None
) -> TokenResponse:
    user = AuthService(db).authenticate(
        email=payload.email, password=payload.password, client_ref=client_ref
    )
    return _token_response(user)


@router.get("/me", response_model=UserResponse, summary="Current user")
def me(current_user: CurrentUser) -> UserResponse:
    return UserResponse.model_validate(current_user)
