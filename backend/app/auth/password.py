"""Password hashing.

``bcrypt`` is used directly rather than through passlib.  passlib's bcrypt
backend broke against bcrypt >= 4.1 and the project is effectively unmaintained;
the direct API is about fifteen lines and removes a dependency that sits on the
authentication path.

Argon2id would be the stronger modern choice.  bcrypt is used here because it
needs no compiled extras beyond its own wheel, is universally understood in
review, and is more than adequate at a configurable work factor.  The trade-off
is recorded in docs/security.md.
"""

from __future__ import annotations

import re

import bcrypt

from app.core.config import settings

# bcrypt truncates its input at 72 bytes. Passing a longer password silently
# ignores the tail, so we pre-hash instead of truncating -- otherwise two
# distinct long passwords sharing a 72-byte prefix would be interchangeable.
_BCRYPT_MAX_BYTES = 72

_COMMON_PASSWORDS = frozenset(
    """
    password password1 password123 12345678 123456789 1234567890 qwerty123
    letmein123 welcome123 admin123 changeme iloveyou1 sunshine1 princess1
    football1 baseball1 dragon123 monkey123 abc12345 passw0rd p@ssw0rd
    """.split()
)


def _prepare(password: str) -> bytes:
    """Return password bytes safe for bcrypt.

    Long passwords are folded to a fixed-length digest first.  SHA-256 output
    is base64-encoded because raw digest bytes can contain NUL, which bcrypt
    treats as a string terminator.
    """
    raw = password.encode("utf-8")
    if len(raw) <= _BCRYPT_MAX_BYTES:
        return raw
    import base64
    import hashlib

    return base64.b64encode(hashlib.sha256(raw).digest())


def hash_password(password: str) -> str:
    salt = bcrypt.gensalt(rounds=settings.BCRYPT_ROUNDS)
    return bcrypt.hashpw(_prepare(password), salt).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password. Never raises -- a malformed hash is simply a failure."""
    try:
        return bcrypt.checkpw(_prepare(password), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def dummy_verify() -> None:
    """Burn a comparable amount of CPU for a non-existent user.

    Without this, "no such user" returns measurably faster than "wrong
    password", turning the login endpoint into a user-enumeration oracle.
    """
    bcrypt.checkpw(
        b"timing-equalisation",
        bcrypt.hashpw(b"timing-equalisation", bcrypt.gensalt(rounds=4)),
    )


_HAS_LETTER = re.compile(r"[A-Za-z]")
_HAS_DIGIT = re.compile(r"\d")


def validate_password_strength(password: str, *, email: str | None = None) -> None:
    """Reject weak passwords. Raises :class:`ValueError` with a usable message.

    Deliberately simple: a length floor, a character-class floor, a small
    common-password list, and a check that the password is not derived from the
    account's own email.  Composition rules beyond this mostly push users toward
    predictable substitutions rather than better passwords.
    """
    if len(password) < settings.PASSWORD_MIN_LENGTH:
        raise ValueError(
            f"Password must be at least {settings.PASSWORD_MIN_LENGTH} characters."
        )
    if len(password) > 256:
        raise ValueError("Password must be at most 256 characters.")
    if not _HAS_LETTER.search(password) or not _HAS_DIGIT.search(password):
        raise ValueError("Password must contain both letters and digits.")

    lowered = password.lower()
    if lowered in _COMMON_PASSWORDS:
        raise ValueError("That password is too common.")
    if email:
        local_part = email.split("@")[0].lower()
        if len(local_part) >= 3 and local_part in lowered:
            raise ValueError("Password must not contain your email address.")
