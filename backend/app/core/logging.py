"""Structured logging.

Two formatters are provided: a JSON formatter for anything that ships logs to a
collector, and a human-readable console formatter for local development.  Both
automatically attach the ambient ``request_id`` / ``user_id`` so that a single
request can be traced end to end.

SECURITY: log records are deliberately built from explicit keyword fields.
Document text, chunk content, passwords and raw user queries must never be
passed to the logger -- see :func:`redact_for_log`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from app.core.config import settings
from app.core.request_context import get_request_id, get_user_id

# Attributes present on every LogRecord; anything else was supplied by the
# caller via `extra=` and therefore belongs in the structured payload.
_RESERVED = frozenset(
    """
    args asctime created exc_info exc_text filename funcName levelname levelno
    lineno module msecs message msg name pathname process processName
    relativeCreated stack_info thread threadName taskName
    """.split()
)


def redact_for_log(text: str | None, keep: int = 0) -> str:
    """Return a non-reversible reference to ``text`` that is safe to log.

    We log a short SHA-256 prefix rather than the content itself.  That is
    enough to correlate repeated identical inputs (useful for spotting attack
    campaigns) without ever writing user or document text to disk.
    """
    if not text:
        return ""
    digest = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:12]
    if keep > 0:
        prefix = text[:keep].replace("\n", " ")
        return f"{prefix}...#{digest}"
    return f"#{digest}"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = get_request_id()
        if request_id:
            payload["request_id"] = request_id
        user_id = get_user_id()
        if user_id:
            payload["user_id"] = user_id
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    _COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        colour = self._COLORS.get(record.levelname, "")
        stamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        request_id = get_request_id()
        rid = f" [{request_id[:8]}]" if request_id else ""
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in _RESERVED and not k.startswith("_")
        }
        suffix = ""
        if extras:
            suffix = " " + " ".join(f"{k}={v}" for k, v in extras.items())
        base = (
            f"{stamp}{rid} {colour}{record.levelname:<8}{self._RESET} "
            f"{record.name}: {record.getMessage()}{suffix}"
        )
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging() -> None:
    """Install the configured formatter on the root logger (idempotent)."""
    root = logging.getLogger()
    root.setLevel(settings.LOG_LEVEL.upper())
    for existing in list(root.handlers):
        root.removeHandler(existing)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter() if settings.LOG_FORMAT == "json" else ConsoleFormatter()
    )
    root.addHandler(handler)

    # Uvicorn installs its own handlers; route them through ours instead.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True

    logging.getLogger("sqlalchemy.engine").setLevel(
        logging.INFO if settings.DB_ECHO else logging.WARNING
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
