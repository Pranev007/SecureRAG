"""SecureRAG application entry point."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.errors import register_exception_handlers
from app.api.middleware import RequestContextMiddleware, SecurityHeadersMiddleware
from app.api.router import api_router
from app.api.routes.health import router as health_router
from app.core.config import settings
from app.core.logging import configure_logging, get_logger

logger = get_logger("app.main")

DESCRIPTION = """
**SecureRAG** is a Retrieval-Augmented Generation service built around the
assumption that *both* the user and the corpus are untrusted.

Every request passes through an input guardrail stage, an access-controlled
retrieval stage, a context-sanitisation stage, and an output guardrail stage
that verifies grounding, citations and PII before a single token reaches the
client.

See `/docs` for the interactive API, and `docs/security.md` in the repository
for the threat model and the reasoning behind each control.
"""


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    logger.info(
        "startup",
        extra={
            "environment": settings.ENVIRONMENT,
            "llm_provider": settings.LLM_PROVIDER,
            "embedding_provider": settings.EMBEDDING_PROVIDER,
            "retrieval_mode": settings.RETRIEVAL_MODE,
            "vector_backend": "sqlite_fallback" if settings.is_sqlite else "pgvector",
        },
    )
    if settings.ENVIRONMENT == "production":
        _assert_production_safety()
    yield
    logger.info("shutdown")


def _assert_production_safety() -> None:
    """Refuse to serve production traffic with development stand-ins.

    The deterministic ``echo`` LLM and ``hashing`` embedder exist so the project
    can be run and tested without credentials.  Silently serving real users with
    them would be worse than failing to boot.
    """
    problems: list[str] = []
    if settings.LLM_PROVIDER == "echo":
        problems.append("LLM_PROVIDER=echo is a test stub")
    if settings.EMBEDDING_PROVIDER == "hashing":
        problems.append("EMBEDDING_PROVIDER=hashing is a lexical stand-in")
    if settings.DEBUG:
        problems.append("DEBUG must be false in production")
    if problems:
        raise RuntimeError("Unsafe production configuration: " + "; ".join(problems))


def create_app() -> FastAPI:
    configure_logging()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Order matters: the outermost middleware runs first, so the request id
    # exists before anything else can log.
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "PATCH", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID", "X-Response-Time-ms", "Retry-After"],
    )

    register_exception_handlers(app)

    # Health is served both unprefixed (for container probes) and under the
    # versioned prefix (for consistency with the rest of the API).
    app.include_router(health_router)
    app.include_router(health_router, prefix=settings.API_PREFIX)
    app.include_router(api_router, prefix=settings.API_PREFIX)

    @app.get("/", include_in_schema=False)
    def root() -> dict:
        return {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "docs": "/docs",
            "api": settings.API_PREFIX,
        }

    return app


app = create_app()
