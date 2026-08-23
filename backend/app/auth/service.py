"""Registration and login."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.auth.password import (
    dummy_verify,
    hash_password,
    validate_password_strength,
    verify_password,
)
from app.core.config import settings
from app.core.exceptions import (
    AuthenticationError,
    ConflictError,
    ValidationError,
)
from app.core.logging import get_logger
from app.models.security_event import (
    SecurityAction,
    SecurityEventType,
    SecurityLayer,
    SecuritySeverity,
)
from app.models.user import User, UserRole
from app.services.security_event_service import record_event

logger = get_logger("app.auth")


def normalise_email(email: str) -> str:
    return email.strip().lower()


class AuthService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def register(
        self,
        *,
        email: str,
        password: str,
        full_name: str | None = None,
        client_ref: str | None = None,
    ) -> User:
        if not settings.ALLOW_REGISTRATION:
            raise ValidationError(
                "Registration is currently closed.",
                internal_detail="ALLOW_REGISTRATION=false",
            )

        email = normalise_email(email)

        try:
            validate_password_strength(password, email=email)
        except ValueError as exc:
            # Password rules are safe to state: the user needs to know what to
            # fix, and the rules are not a secret.
            raise ValidationError(str(exc), internal_detail="weak password") from exc

        existing = self.db.execute(
            select(User).where(func.lower(User.email) == email)
        ).scalar_one_or_none()
        if existing is not None:
            raise ConflictError(
                "That email address cannot be registered.",
                internal_detail="duplicate email",
            )

        # The first account to register becomes the admin. Simpler and safer
        # than shipping a default admin password, and easy to explain.
        is_first_user = (
            self.db.execute(select(func.count()).select_from(User)).scalar_one() == 0
        )

        user = User(
            email=email,
            hashed_password=hash_password(password),
            full_name=(full_name or "").strip()[:255] or None,
            role=UserRole.ADMIN.value if is_first_user else UserRole.USER.value,
        )
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)

        record_event(
            self.db,
            event_type=SecurityEventType.REGISTRATION,
            layer=SecurityLayer.AUTH,
            severity=SecuritySeverity.INFO,
            action=SecurityAction.ALLOW,
            user_id=user.id,
            client_ref=client_ref,
            detail={"role": user.role, "bootstrap_admin": is_first_user},
        )
        return user

    def authenticate(
        self, *, email: str, password: str, client_ref: str | None = None
    ) -> User:
        email = normalise_email(email)
        user = self.db.execute(
            select(User).where(func.lower(User.email) == email)
        ).scalar_one_or_none()

        if user is None:
            # Spend comparable CPU so response time does not reveal whether the
            # account exists.
            dummy_verify()
            self._record_failure(email, "unknown_account", client_ref)
            raise AuthenticationError("Incorrect email or password.")

        if not verify_password(password, user.hashed_password):
            self._record_failure(email, "bad_password", client_ref, user_id=user.id)
            raise AuthenticationError("Incorrect email or password.")

        if not user.is_active:
            self._record_failure(email, "inactive_account", client_ref, user_id=user.id)
            raise AuthenticationError("Incorrect email or password.")

        record_event(
            self.db,
            event_type=SecurityEventType.LOGIN_SUCCEEDED,
            layer=SecurityLayer.AUTH,
            severity=SecuritySeverity.INFO,
            action=SecurityAction.ALLOW,
            user_id=user.id,
            client_ref=client_ref,
            detail={"role": user.role},
        )
        return user

    def _record_failure(
        self,
        email: str,
        reason: str,
        client_ref: str | None,
        user_id: str | None = None,
    ) -> None:
        from app.core.logging import redact_for_log

        record_event(
            self.db,
            event_type=SecurityEventType.LOGIN_FAILED,
            layer=SecurityLayer.AUTH,
            severity=SecuritySeverity.LOW,
            action=SecurityAction.BLOCK,
            user_id=user_id,
            risk_score=0.3,
            detector="credentials",
            # The attempted address is an identifier, not a credential, but it
            # is still personal data -- store a hash, not the address.
            content_ref=redact_for_log(email),
            client_ref=client_ref,
            detail={"reason": reason},
        )

    def get_by_id(self, user_id: str) -> User | None:
        return self.db.get(User, user_id)
