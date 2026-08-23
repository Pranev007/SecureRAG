"""Rate limiting: the limiter itself and its enforcement at the API edge."""

from __future__ import annotations

import pytest

from app.security.rate_limit import RateLimiter, reset_all_limiters

pytestmark = pytest.mark.security


@pytest.fixture
def rate_limited(monkeypatch):
    """Enable rate limiting with small, fast limits for the duration of a test."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(settings, "RATE_LIMIT_PER_MINUTE", 3)
    monkeypatch.setattr(settings, "RATE_LIMIT_UPLOAD_PER_MINUTE", 2)
    monkeypatch.setattr(settings, "RATE_LIMIT_AUTH_PER_MINUTE", 3)
    reset_all_limiters()
    yield
    reset_all_limiters()


# ----------------------------------------------------------------------
# The limiter in isolation
# ----------------------------------------------------------------------


def test_limiter_allows_up_to_the_limit_then_blocks():
    limiter = RateLimiter(window_seconds=60)
    decisions = [limiter.check("k", 3) for _ in range(4)]

    assert [d.allowed for d in decisions] == [True, True, True, False]
    assert decisions[0].remaining == 2
    assert decisions[-1].retry_after > 0


def test_limiter_keys_are_independent():
    limiter = RateLimiter(window_seconds=60)
    for _ in range(3):
        limiter.check("alice", 3)

    assert not limiter.check("alice", 3).allowed
    assert limiter.check("bob", 3).allowed


def test_limiter_window_slides_rather_than_resetting_on_a_boundary():
    """A fixed window would allow 2x the limit across a boundary."""
    limiter = RateLimiter(window_seconds=1)
    for _ in range(3):
        limiter.check("k", 3)
    assert not limiter.check("k", 3).allowed

    import time

    time.sleep(1.05)
    assert limiter.check("k", 3).allowed


def test_limiter_evicts_idle_keys_so_it_cannot_be_a_memory_sink():
    limiter = RateLimiter(window_seconds=0, max_keys=10)
    for index in range(50):
        limiter.check(f"key-{index}", 5)
    assert len(limiter._hits) <= 20


def test_a_zero_limit_disables_the_bucket():
    assert RateLimiter().check("k", 0).allowed


# ----------------------------------------------------------------------
# Enforcement at the API edge
# ----------------------------------------------------------------------


def test_login_attempts_are_rate_limited(client, rate_limited):
    payload = {"email": "nobody@example.com", "password": "Wrong-Password-1"}
    statuses = [
        client.post("/api/v1/auth/login", json=payload).status_code for _ in range(5)
    ]

    assert 429 in statuses
    assert statuses.index(429) == 3  # first three attempts were allowed


def test_a_rate_limited_response_carries_retry_after(client, rate_limited):
    payload = {"email": "nobody@example.com", "password": "Wrong-Password-1"}
    response = None
    for _ in range(6):
        response = client.post("/api/v1/auth/login", json=payload)
        if response.status_code == 429:
            break

    assert response is not None and response.status_code == 429
    assert int(response.headers["Retry-After"]) > 0
    assert response.json()["error"]["code"] == "rate_limited"


def test_uploads_have_their_own_budget(client, user_headers, rate_limited):
    from tests.factories import HANDBOOK_MARKDOWN

    statuses = []
    for index in range(4):
        # Distinct content so deduplication does not mask the limiter.
        body = f"# Doc {index}\n\n" + HANDBOOK_MARKDOWN
        statuses.append(
            client.post(
                "/api/v1/documents",
                headers=user_headers,
                files={"file": (f"d{index}.md", body.encode(), "text/markdown")},
            ).status_code
        )

    assert 429 in statuses
    assert statuses[:2] == [201, 201]


def test_exhausting_the_upload_budget_leaves_reads_working(
    client, user_headers, rate_limited
):
    """Buckets are separate, so an upload flood must not lock out listing."""
    from tests.factories import HANDBOOK_MARKDOWN

    for index in range(4):
        client.post(
            "/api/v1/documents",
            headers=user_headers,
            files={
                "file": (
                    f"d{index}.md",
                    f"# {index}\n\n{HANDBOOK_MARKDOWN}".encode(),
                    "text/markdown",
                )
            },
        )

    assert client.get("/api/v1/documents", headers=user_headers).status_code == 200


def test_one_users_limit_does_not_affect_another(
    client, user_headers, other_user_headers, rate_limited
):
    from tests.factories import HANDBOOK_MARKDOWN

    for index in range(4):
        client.post(
            "/api/v1/documents",
            headers=user_headers,
            files={
                "file": (
                    f"a{index}.md",
                    f"# A{index}\n\n{HANDBOOK_MARKDOWN}".encode(),
                    "text/markdown",
                )
            },
        )

    response = client.post(
        "/api/v1/documents",
        headers=other_user_headers,
        files={"file": ("b.md", HANDBOOK_MARKDOWN.encode(), "text/markdown")},
    )
    assert response.status_code == 201


def test_rate_limit_violations_are_audited(client, db, rate_limited):
    from sqlalchemy import select

    from app.models.security_event import SecurityEvent, SecurityEventType

    payload = {"email": "nobody@example.com", "password": "Wrong-Password-1"}
    for _ in range(5):
        client.post("/api/v1/auth/login", json=payload)

    events = (
        db.execute(
            select(SecurityEvent).where(
                SecurityEvent.event_type == SecurityEventType.RATE_LIMIT_EXCEEDED.value
            )
        )
        .scalars()
        .all()
    )
    assert events
    assert events[0].detail["bucket"] == "auth"
    assert events[0].detail["limit_per_minute"] == 3
