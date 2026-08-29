"""Central application configuration.

Every tunable knob in SecureRAG is defined here and sourced from the
environment, so that no security threshold, model name, or credential is ever
hard-coded in application logic.  See ``.env.example`` for documentation of
each variable.
"""

from __future__ import annotations

import secrets
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_DIR.parent


def _split_csv(raw: str) -> list[str]:
    """Parse a comma-separated environment value into a clean list.

    These fields are declared as ``str`` in :class:`Settings` rather than
    ``list[str]`` because pydantic-settings attempts a JSON decode on complex
    types before any validator runs, which makes a plain ``a,b,c`` env value a
    hard startup error.
    """
    return [item.strip() for item in raw.split(",") if item.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        populate_by_name=True,
    )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    APP_NAME: str = "SecureRAG"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: Literal["local", "test", "staging", "production"] = "local"
    DEBUG: bool = False
    API_PREFIX: str = "/api/v1"
    CORS_ORIGINS_RAW: str = Field(
        default="http://localhost:5173,http://localhost:3000",
        alias="CORS_ORIGINS",
    )

    @property
    def CORS_ORIGINS(self) -> list[str]:
        return _split_csv(self.CORS_ORIGINS_RAW)

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------
    # postgresql+psycopg://... in production; sqlite:///... for offline dev and
    # the test suite.  The vector store adapts to the dialect at runtime.
    DATABASE_URL: str = (
        "postgresql+psycopg://securerag:securerag@localhost:5432/securerag"
    )
    DB_ECHO: bool = False
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------
    JWT_SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(48))
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    ALLOW_REGISTRATION: bool = True
    PASSWORD_MIN_LENGTH: int = 10
    BCRYPT_ROUNDS: int = 12

    # ------------------------------------------------------------------
    # LLM provider
    # ------------------------------------------------------------------
    # "openai" covers any OpenAI-compatible /chat/completions endpoint
    # (OpenAI, Groq, Together, vLLM, LM Studio, OpenRouter, ...).
    # "echo" is a deterministic offline provider used by the test suite and the
    # no-credentials demo mode.  It is NOT a language model.
    LLM_PROVIDER: Literal["openai", "ollama", "echo"] = "echo"
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_BASE_URL: str = "https://api.openai.com/v1"
    LLM_API_KEY: str = ""
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 900
    LLM_TIMEOUT_SECONDS: float = 60.0

    # ------------------------------------------------------------------
    # Embedding provider
    # ------------------------------------------------------------------
    # "hashing" is a dependency-free deterministic lexical embedder used for
    # offline development and tests.  It yields real (lexical-only) similarity;
    # its limitations are documented in docs/architecture.md.
    EMBEDDING_PROVIDER: Literal["openai", "ollama", "hashing"] = "hashing"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_BASE_URL: str = "https://api.openai.com/v1"
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_DIMENSIONS: int = 384
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_TIMEOUT_SECONDS: float = 60.0

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    MAX_UPLOAD_SIZE_MB: int = 20
    ALLOWED_FILE_EXTENSIONS_RAW: str = Field(
        default="pdf,txt,md,markdown,docx", alias="ALLOWED_FILE_EXTENSIONS"
    )
    CHUNK_TARGET_TOKENS: int = 320
    CHUNK_OVERLAP_TOKENS: int = 60
    CHUNK_MIN_CHARS: int = 120
    MAX_CHUNKS_PER_DOCUMENT: int = 5000

    @property
    def ALLOWED_FILE_EXTENSIONS(self) -> set[str]:
        return {
            e.lower().lstrip(".") for e in _split_csv(self.ALLOWED_FILE_EXTENSIONS_RAW)
        }

    @property
    def MAX_UPLOAD_SIZE_BYTES(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    RETRIEVAL_MODE: Literal["vector", "keyword", "hybrid"] = "hybrid"
    RETRIEVAL_TOP_K: int = 5
    RETRIEVAL_CANDIDATE_K: int = 20
    HYBRID_RRF_K: int = 60
    RERANKER: Literal["none", "heuristic", "cross_encoder"] = "heuristic"
    CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    MIN_SIMILARITY: float = 0.05
    MAX_CONTEXT_CHARS: int = 12000

    # ------------------------------------------------------------------
    # Input security
    # ------------------------------------------------------------------
    MIN_QUERY_LENGTH: int = 3
    MAX_QUERY_LENGTH: int = 2000
    INJECTION_BLOCK_THRESHOLD: float = 0.75
    INJECTION_FLAG_THRESHOLD: float = 0.45
    INJECTION_USE_LLM_CLASSIFIER: bool = False
    DUPLICATE_QUERY_WINDOW_SECONDS: int = 60
    DUPLICATE_QUERY_LIMIT: int = 5

    # ------------------------------------------------------------------
    # Context (indirect injection) security
    # ------------------------------------------------------------------
    CONTEXT_SANITISATION_ENABLED: bool = True
    CONTEXT_INJECTION_QUARANTINE_THRESHOLD: float = 0.60
    CONTEXT_INJECTION_NEUTRALISE_THRESHOLD: float = 0.35

    # ------------------------------------------------------------------
    # Output guardrails
    # ------------------------------------------------------------------
    GROUNDING_ENABLED: bool = True
    GROUNDING_MIN_SCORE: float = 0.45
    GROUNDING_MODE: Literal["warn", "block"] = "block"
    # How a claim is checked against the retrieved context.
    #   "lexical" -- overlap / numeric / n-gram scoring. No dependencies.
    #   "nli"     -- cross-encoder entailment, with the lexical numeric gate
    #                retained because NLI models do not judge numbers reliably.
    #   "hybrid"  -- both, combined so each vetoes only where it is strong.
    # "nli" and "hybrid" need sentence-transformers; when it is missing the
    # verifier degrades to lexical and says so in the report's `method` field.
    GROUNDING_METHOD: Literal["lexical", "nli", "hybrid"] = "lexical"
    NLI_MODEL: str = "cross-encoder/nli-deberta-v3-base"
    NLI_MAX_LENGTH: int = 384
    NLI_BATCH_SIZE: int = 16
    # Premises scored per claim after the lexical shortlist. Higher is more
    # thorough and linearly more expensive; 4 covers a claim supported by a
    # sentence pair without making a CPU run unaffordable.
    NLI_TOP_PREMISES: int = 4
    NLI_ENTAILMENT_FLOOR: float = 0.50
    NLI_CONTRADICTION_THRESHOLD: float = 0.55
    REQUIRE_CITATIONS: bool = True
    OUTPUT_SAFETY_ENABLED: bool = True
    INSUFFICIENT_EVIDENCE_MESSAGE: str = (
        "I could not find sufficient evidence in your documents to answer that "
        "reliably. Try rephrasing the question, or upload a document that covers it."
    )

    # ------------------------------------------------------------------
    # PII
    # ------------------------------------------------------------------
    PII_DETECTION_MODE: Literal["off", "warn", "redact", "block"] = "redact"
    PII_ENGINE: Literal["regex", "presidio"] = "regex"
    PII_ENTITIES_RAW: str = Field(
        default="EMAIL,PHONE,CREDIT_CARD,AADHAAR,PAN,SSN,IP_ADDRESS,IBAN,API_KEY",
        alias="PII_ENTITIES",
    )
    PII_SCAN_INPUT: bool = True

    @property
    def PII_ENTITIES(self) -> set[str]:
        return {e.upper() for e in _split_csv(self.PII_ENTITIES_RAW)}

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    # Answer relevance asks "does this answer address the question that was
    # asked?" -- independent of whether it is grounded or correct. It is an
    # evaluation metric, not a runtime guardrail: nothing is blocked on it.
    ANSWER_RELEVANCE_ENABLED: bool = True
    ANSWER_RELEVANCE_MIN_SCORE: float = 0.50

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_PER_MINUTE: int = 20
    RATE_LIMIT_UPLOAD_PER_MINUTE: int = 5
    RATE_LIMIT_AUTH_PER_MINUTE: int = 10

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: Literal["json", "console"] = "json"
    SECURITY_EVENT_RETENTION_DAYS: int = 90

    # ------------------------------------------------------------------
    # Global posture
    # ------------------------------------------------------------------
    # When a guardrail raises an unexpected error, block rather than letting the
    # request through unchecked.  See SECURITY PRINCIPLE 7 in docs/security.md.
    FAIL_CLOSED: bool = True

    @field_validator("DATABASE_URL")
    @classmethod
    def _normalise_database_url(cls, v: str) -> str:
        """Accept the ``postgres://`` URLs managed hosts hand out.

        Render, Railway, Heroku and Fly all inject ``postgres://...``.
        SQLAlchemy 2.x removed that alias, so the app would die at startup with
        an opaque "Can't load plugin: sqlalchemy.dialects:postgres" -- a
        deployment failure with no obvious cause, on the one code path that is
        hardest to debug remotely. Normalising here costs nothing and keeps the
        driver choice (psycopg 3) explicit in one place.
        """
        for prefix in ("postgres://", "postgresql://"):
            if v.startswith(prefix):
                return "postgresql+psycopg://" + v[len(prefix) :]
        return v

    @field_validator(
        "INJECTION_BLOCK_THRESHOLD",
        "INJECTION_FLAG_THRESHOLD",
        "GROUNDING_MIN_SCORE",
        "CONTEXT_INJECTION_QUARANTINE_THRESHOLD",
        "CONTEXT_INJECTION_NEUTRALISE_THRESHOLD",
        "NLI_ENTAILMENT_FLOOR",
        "NLI_CONTRADICTION_THRESHOLD",
        "ANSWER_RELEVANCE_MIN_SCORE",
    )
    @classmethod
    def _validate_unit_interval(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("threshold must be within [0.0, 1.0]")
        return v

    @property
    def is_sqlite(self) -> bool:
        return self.DATABASE_URL.startswith("sqlite")


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()


settings = get_settings()
