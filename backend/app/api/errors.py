"""Exception handlers producing a single, uniform error envelope.

Every error the API emits has the same shape, so the frontend has one code path
and an attacker gains no signal from response *structure*:

    {"error": {"code": ..., "message": ..., "request_id": ...}}
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.config import settings
from app.core.exceptions import RateLimitError, SecureRAGError, SecurityBlockError
from app.core.logging import get_logger
from app.core.request_context import get_request_id

logger = get_logger("app.errors")


def _envelope(
    code: str,
    message: str,
    *,
    extra: dict | None = None,
) -> dict:
    body: dict = {
        "error": {
            "code": code,
            "message": message,
            "request_id": get_request_id(),
        }
    }
    if extra:
        body["error"].update(extra)
    return body


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(SecurityBlockError)
    async def _security_block(_request: Request, exc: SecurityBlockError) -> JSONResponse:
        # The operator sees which detector fired; the client sees only that a
        # policy rejected the request (SECURITY PRINCIPLE 8).
        logger.warning(
            "security_block",
            extra={
                "reason": exc.reason,
                "risk_score": exc.risk_score,
                "detail": exc.internal_detail,
            },
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(
                exc.error_code,
                exc.public_message,
                extra={"blocked": True, "reason": exc.reason},
            ),
        )

    @app.exception_handler(RateLimitError)
    async def _rate_limited(_request: Request, exc: RateLimitError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(exc.error_code, exc.public_message),
            headers={"Retry-After": str(exc.retry_after)},
        )

    @app.exception_handler(SecureRAGError)
    async def _app_error(_request: Request, exc: SecureRAGError) -> JSONResponse:
        log = logger.error if exc.status_code >= 500 else logger.info
        log(
            "application_error",
            extra={"code": exc.error_code, "detail": exc.internal_detail},
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(exc.error_code, exc.public_message),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Field-level errors are safe to return -- they describe the caller's own
        # payload -- but the raw input is stripped so it never echoes back.
        fields = [
            {
                "field": ".".join(str(p) for p in err.get("loc", ())[1:]),
                "message": err.get("msg", "invalid value"),
            }
            for err in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content=_envelope(
                "validation_error",
                "The request payload failed validation.",
                extra={"fields": fields},
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope("http_error", str(exc.detail)),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def _unhandled(_request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_exception", extra={"exc_type": type(exc).__name__})
        message = "An unexpected error occurred."
        if settings.DEBUG:
            message = f"{type(exc).__name__}: {exc}"
        return JSONResponse(status_code=500, content=_envelope("internal_error", message))
