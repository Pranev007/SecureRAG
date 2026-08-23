"""HTTP middleware: request identity, timing and access logging."""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import get_logger
from app.core.request_context import (
    new_request_id,
    reset_request_id,
    reset_user_id,
    set_request_id,
    set_user_id,
)

logger = get_logger("app.access")

REQUEST_ID_HEADER = "X-Request-ID"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a request id, expose it on the response, and time the request.

    An inbound ``X-Request-ID`` is honoured so a trace can span the frontend and
    the API, but it is length-capped and stripped of anything non-printable --
    the header is attacker-controlled and ends up in logs.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        incoming = request.headers.get(REQUEST_ID_HEADER, "")
        sanitised = "".join(c for c in incoming if c.isalnum() or c in "-_")[:64]
        request_id = sanitised or new_request_id()

        rid_token = set_request_id(request_id)
        uid_token = set_user_id(None)
        request.state.request_id = request_id
        started = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.exception(
                "request_failed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": round(elapsed_ms, 2),
                },
            )
            raise
        finally:
            reset_user_id(uid_token)

        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers[REQUEST_ID_HEADER] = request_id
        response.headers["X-Response-Time-ms"] = f"{elapsed_ms:.2f}"

        # Health checks would otherwise dominate the log volume.
        if request.url.path not in {"/health", "/api/v1/health", "/metrics"}:
            logger.info(
                "request_completed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": round(elapsed_ms, 2),
                },
            )

        reset_request_id(rid_token)
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Baseline response hardening headers.

    The API returns JSON only, so a restrictive CSP costs nothing here and
    protects the interactive ``/docs`` page from becoming an injection surface.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Permissions-Policy", "geolocation=(), microphone=(), camera=()"
        )
        if not request.url.path.startswith(("/docs", "/redoc", "/openapi.json")):
            response.headers.setdefault(
                "Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'"
            )
        return response
