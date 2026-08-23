"""Authentication endpoints and token handling."""

from __future__ import annotations

import time

import jwt
import pytest
from sqlalchemy import select

from app.auth.password import hash_password, validate_password_strength, verify_password
from app.auth.tokens import create_access_token, decode_access_token
from app.core.config import settings
from app.core.exceptions import AuthenticationError
from app.models.security_event import SecurityEvent, SecurityEventType
from tests.conftest import DEFAULT_PASSWORD, make_user

pytestmark = pytest.mark.api

GOOD_PASSWORD = "Correct-Horse-9-Battery"


# ----------------------------------------------------------------------
# Password primitives
# ----------------------------------------------------------------------


@pytest.mark.unit
def test_password_hash_is_salted_and_verifiable():
    first = hash_password(GOOD_PASSWORD)
    second = hash_password(GOOD_PASSWORD)

    assert first != second, "each hash must use a fresh salt"
    assert GOOD_PASSWORD not in first
    assert verify_password(GOOD_PASSWORD, first)
    assert not verify_password("wrong password 1", first)


@pytest.mark.unit
def test_verify_never_raises_on_a_malformed_hash():
    assert verify_password("anything", "") is False
    assert verify_password("anything", "not-a-bcrypt-hash") is False


@pytest.mark.unit
def test_long_passwords_are_not_silently_truncated_at_72_bytes():
    # bcrypt ignores bytes past 72; without pre-hashing these two would verify
    # against each other.
    base = "A" * 72
    stored = hash_password(base + "-first-tail-9")
    assert not verify_password(base + "-second-tail-9", stored)
    assert verify_password(base + "-first-tail-9", stored)


@pytest.mark.unit
@pytest.mark.parametrize(
    "weak",
    ["short1", "alllettersnodigits", "1234567890123", "password123"],
)
def test_weak_passwords_are_rejected(weak):
    with pytest.raises(ValueError):
        validate_password_strength(weak)


@pytest.mark.unit
def test_password_may_not_contain_the_account_email():
    with pytest.raises(ValueError, match="email"):
        validate_password_strength("alice12345678", email="alice@example.com")


# ----------------------------------------------------------------------
# Tokens
# ----------------------------------------------------------------------


@pytest.mark.unit
def test_token_round_trips_with_expected_claims(db):
    user = make_user(db)
    payload = decode_access_token(create_access_token(user))

    assert payload["sub"] == user.id
    assert payload["type"] == "access"
    assert payload["iss"] == settings.APP_NAME
    assert payload["exp"] > payload["iat"]


@pytest.mark.unit
def test_expired_token_is_rejected(db):
    user = make_user(db)
    token = create_access_token(user, expires_minutes=-1)
    with pytest.raises(AuthenticationError) as caught:
        decode_access_token(token)
    assert "expired" in caught.value.public_message.lower()


@pytest.mark.unit
def test_token_signed_with_another_key_is_rejected(db):
    user = make_user(db)
    forged = jwt.encode(
        {
            "sub": user.id,
            "type": "access",
            "iss": settings.APP_NAME,
            "iat": int(time.time()),
            "exp": int(time.time()) + 3600,
        },
        "an-attacker-chosen-key",
        algorithm="HS256",
    )
    with pytest.raises(AuthenticationError):
        decode_access_token(forged)


@pytest.mark.unit
def test_alg_none_token_is_rejected(db):
    """The classic JWT bypass: a token asserting it needs no signature."""
    user = make_user(db)
    unsigned = jwt.encode(
        {
            "sub": user.id,
            "type": "access",
            "iss": settings.APP_NAME,
            "iat": int(time.time()),
            "exp": int(time.time()) + 3600,
        },
        key="",
        algorithm="none",
    )
    with pytest.raises(AuthenticationError):
        decode_access_token(unsigned)


@pytest.mark.unit
def test_token_of_the_wrong_type_is_rejected(db):
    user = make_user(db)
    token = jwt.encode(
        {
            "sub": user.id,
            "type": "refresh",
            "iss": settings.APP_NAME,
            "iat": int(time.time()),
            "exp": int(time.time()) + 3600,
        },
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )
    with pytest.raises(AuthenticationError):
        decode_access_token(token)


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------


def test_registration_returns_a_usable_token(client):
    response = client.post(
        "/api/v1/auth/register",
        json={"email": "new@example.com", "password": GOOD_PASSWORD, "full_name": "New"},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["user"]["email"] == "new@example.com"

    me = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {body['access_token']}"},
    )
    assert me.status_code == 200
    assert me.json()["email"] == "new@example.com"


def test_registration_never_returns_the_password_hash(client):
    response = client.post(
        "/api/v1/auth/register",
        json={"email": "hash@example.com", "password": GOOD_PASSWORD},
    )
    assert "hashed_password" not in response.text
    assert GOOD_PASSWORD not in response.text


def test_first_registered_user_becomes_admin(client):
    first = client.post(
        "/api/v1/auth/register",
        json={"email": "first@example.com", "password": GOOD_PASSWORD},
    )
    second = client.post(
        "/api/v1/auth/register",
        json={"email": "second@example.com", "password": GOOD_PASSWORD},
    )
    assert first.json()["user"]["role"] == "admin"
    assert second.json()["user"]["role"] == "user"


def test_duplicate_registration_does_not_confirm_the_address_exists(client):
    client.post(
        "/api/v1/auth/register",
        json={"email": "dup@example.com", "password": GOOD_PASSWORD},
    )
    response = client.post(
        "/api/v1/auth/register",
        json={"email": "dup@example.com", "password": GOOD_PASSWORD},
    )
    assert response.status_code == 409
    message = response.json()["error"]["message"].lower()
    assert "already" not in message and "exists" not in message


def test_registration_rejects_a_weak_password(client):
    response = client.post(
        "/api/v1/auth/register",
        json={"email": "weak@example.com", "password": "password123"},
    )
    assert response.status_code == 422
    assert "common" in response.json()["error"]["message"].lower()


def test_registration_rejects_a_malformed_email(client):
    response = client.post(
        "/api/v1/auth/register",
        json={"email": "not-an-email", "password": GOOD_PASSWORD},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_email_is_normalised_to_lowercase(client):
    client.post(
        "/api/v1/auth/register",
        json={"email": "MiXeD@Example.COM", "password": GOOD_PASSWORD},
    )
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "mixed@example.com", "password": GOOD_PASSWORD},
    )
    assert response.status_code == 200


# ----------------------------------------------------------------------
# Login
# ----------------------------------------------------------------------


def test_login_succeeds_with_correct_credentials(client, user):
    response = client.post(
        "/api/v1/auth/login",
        json={"email": user.email, "password": DEFAULT_PASSWORD},
    )
    assert response.status_code == 200
    assert response.json()["user"]["id"] == user.id


def test_wrong_password_and_unknown_account_are_indistinguishable(client, user):
    wrong_password = client.post(
        "/api/v1/auth/login",
        json={"email": user.email, "password": "Definitely-Wrong-1"},
    )
    unknown_account = client.post(
        "/api/v1/auth/login",
        json={"email": "nobody@example.com", "password": "Definitely-Wrong-1"},
    )

    assert wrong_password.status_code == unknown_account.status_code == 401
    assert (
        wrong_password.json()["error"]["message"]
        == unknown_account.json()["error"]["message"]
    )


def test_inactive_account_cannot_log_in(client, db, user):
    user.is_active = False
    db.commit()
    response = client.post(
        "/api/v1/auth/login", json={"email": user.email, "password": DEFAULT_PASSWORD}
    )
    assert response.status_code == 401


def test_failed_login_is_audited_without_storing_the_email(client, db, user):
    client.post(
        "/api/v1/auth/login",
        json={"email": user.email, "password": "Definitely-Wrong-1"},
    )
    events = (
        db.execute(
            select(SecurityEvent).where(
                SecurityEvent.event_type == SecurityEventType.LOGIN_FAILED.value
            )
        )
        .scalars()
        .all()
    )
    assert len(events) == 1
    assert events[0].detail["reason"] == "bad_password"
    assert user.email not in str(events[0].detail)
    assert user.email not in (events[0].content_ref or "")


# ----------------------------------------------------------------------
# Protected endpoints
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/api/v1/auth/me"),
        ("get", "/api/v1/documents"),
        ("post", "/api/v1/documents"),
        ("get", "/api/v1/documents/some-id"),
        ("delete", "/api/v1/documents/some-id"),
    ],
)
def test_protected_endpoints_reject_anonymous_callers(client, method, path):
    response = getattr(client, method)(path)
    assert response.status_code == 401


@pytest.mark.parametrize(
    "header",
    ["Bearer", "Bearer ", "Bearer not.a.token", "Basic dXNlcjpwYXNz", "garbage"],
)
def test_malformed_authorization_headers_are_rejected(client, header):
    response = client.get("/api/v1/auth/me", headers={"Authorization": header})
    assert response.status_code == 401


def test_token_for_a_deleted_user_stops_working(client, db, user):
    headers = {"Authorization": f"Bearer {create_access_token(user)}"}
    assert client.get("/api/v1/auth/me", headers=headers).status_code == 200

    db.delete(user)
    db.commit()
    assert client.get("/api/v1/auth/me", headers=headers).status_code == 401


def test_deactivating_a_user_revokes_an_already_issued_token(client, db, user):
    headers = {"Authorization": f"Bearer {create_access_token(user)}"}
    assert client.get("/api/v1/auth/me", headers=headers).status_code == 200

    user.is_active = False
    db.commit()
    assert client.get("/api/v1/auth/me", headers=headers).status_code == 401
