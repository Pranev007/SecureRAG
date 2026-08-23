"""Application exception hierarchy.

Design rule (SECURITY PRINCIPLE 8): the message a *client* sees and the detail
an *operator* sees are different fields.  ``public_message`` is safe to return
over the API; ``internal_detail`` names the specific rule that fired and is
only ever written to logs and the security-event table.
"""

from __future__ import annotations

from typing import Any


class SecureRAGError(Exception):
    """Base class for all application errors."""

    status_code: int = 500
    error_code: str = "internal_error"
    public_message: str = "An unexpected error occurred."

    def __init__(
        self,
        public_message: str | None = None,
        *,
        internal_detail: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.public_message = public_message or self.public_message
        self.internal_detail = internal_detail or self.public_message
        self.context = context or {}
        super().__init__(self.internal_detail)


class ValidationError(SecureRAGError):
    status_code = 422
    error_code = "validation_error"
    public_message = "The request could not be processed."


class AuthenticationError(SecureRAGError):
    status_code = 401
    error_code = "authentication_error"
    public_message = "Invalid credentials."


class AuthorizationError(SecureRAGError):
    status_code = 403
    error_code = "authorization_error"
    public_message = "You do not have access to this resource."


class NotFoundError(SecureRAGError):
    status_code = 404
    error_code = "not_found"
    public_message = "Resource not found."


class ConflictError(SecureRAGError):
    status_code = 409
    error_code = "conflict"
    public_message = "Resource already exists."


class RateLimitError(SecureRAGError):
    status_code = 429
    error_code = "rate_limited"
    public_message = "Too many requests. Please slow down."

    def __init__(self, retry_after: int = 60, **kwargs: Any) -> None:
        self.retry_after = retry_after
        super().__init__(**kwargs)


class SecurityBlockError(SecureRAGError):
    """Raised when a guardrail refuses a request.

    The public message is intentionally uniform across every guardrail so that
    an attacker cannot use the response text as an oracle for which specific
    rule they tripped.
    """

    status_code = 400
    error_code = "security_block"
    public_message = "Request rejected by security policy."

    def __init__(
        self,
        *,
        reason: str,
        risk_score: float = 1.0,
        internal_detail: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.reason = reason
        self.risk_score = risk_score
        super().__init__(
            self.public_message, internal_detail=internal_detail, context=context
        )


class IngestionError(SecureRAGError):
    status_code = 400
    error_code = "ingestion_error"
    public_message = "The document could not be processed."


class ProviderError(SecureRAGError):
    """An upstream LLM or embedding provider failed."""

    status_code = 502
    error_code = "provider_error"
    public_message = "The language model service is currently unavailable."


class ConfigurationError(SecureRAGError):
    status_code = 500
    error_code = "configuration_error"
    public_message = "The service is misconfigured."
