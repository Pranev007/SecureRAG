"""Rate limiting.

Implementation: a sliding-window log per key.  Each key holds the timestamps of
recent requests; a request is allowed if fewer than ``limit`` timestamps fall
inside the trailing window.

Why sliding window and not a fixed window
-----------------------------------------
A fixed window resets on a clock boundary, which lets a caller send ``limit``
requests at 11:59:59 and ``limit`` more at 12:00:00 -- double the intended rate
at exactly the moment a burst hurts most.  The sliding log has no boundary to
exploit and, at these limits, its memory cost is trivial.

KNOWN LIMITATION (stated plainly, also in docs/security.md)
-----------------------------------------------------------
State lives in the process.  With N uvicorn workers the effective limit is
N x ``limit``, and a restart clears every counter.  This is the correct
trade-off for a single-container deployment and an honest one to declare;
the fix is a Redis-backed store behind the same :class:`RateLimiter`
interface, which is why the interface exists at all.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass

from app.core.config import settings


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    limit: int
    remaining: int
    retry_after: int


class RateLimiter:
    """Thread-safe in-process sliding-window limiter."""

    def __init__(self, window_seconds: int = 60, max_keys: int = 50_000) -> None:
        self._window = window_seconds
        self._max_keys = max_keys
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, limit: int) -> RateLimitDecision:
        """Record a request against ``key`` and decide whether to allow it."""
        if limit <= 0:
            return RateLimitDecision(True, limit, 0, 0)

        now = time.monotonic()
        cutoff = now - self._window

        with self._lock:
            bucket = self._hits[key]
            while bucket and bucket[0] < cutoff:
                bucket.popleft()

            if len(bucket) >= limit:
                # Retry when the oldest hit leaves the window.
                retry_after = max(1, int(bucket[0] + self._window - now) + 1)
                return RateLimitDecision(False, limit, 0, retry_after)

            bucket.append(now)
            remaining = limit - len(bucket)

            # Opportunistic eviction. Without it an attacker rotating keys
            # (one per forged IP) would grow the dict without bound -- the
            # limiter itself becoming the memory-exhaustion vector.
            if len(self._hits) > self._max_keys:
                self._evict_idle(cutoff)

            return RateLimitDecision(True, limit, remaining, 0)

    def _evict_idle(self, cutoff: float) -> None:
        stale = [key for key, hits in self._hits.items() if not hits or hits[-1] < cutoff]
        for key in stale:
            del self._hits[key]

    def reset(self, key: str | None = None) -> None:
        with self._lock:
            if key is None:
                self._hits.clear()
            else:
                self._hits.pop(key, None)


# One limiter per bucket so a burst of uploads cannot exhaust a user's chat
# budget, and so each can be tuned independently.
chat_limiter = RateLimiter()
upload_limiter = RateLimiter()
auth_limiter = RateLimiter()


def limits_for(bucket: str) -> tuple[RateLimiter, int]:
    if bucket == "upload":
        return upload_limiter, settings.RATE_LIMIT_UPLOAD_PER_MINUTE
    if bucket == "auth":
        return auth_limiter, settings.RATE_LIMIT_AUTH_PER_MINUTE
    return chat_limiter, settings.RATE_LIMIT_PER_MINUTE


def reset_all_limiters() -> None:
    """Clear every limiter (used by tests)."""
    for limiter in (chat_limiter, upload_limiter, auth_limiter):
        limiter.reset()
