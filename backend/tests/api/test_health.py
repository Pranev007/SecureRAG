"""Health, error-envelope and middleware behaviour."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.api


def test_liveness_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["app"] == "SecureRAG"


def test_readiness_reports_dependencies(client):
    response = client.get("/health/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["checks"]["database"] == "ok"
    assert body["vector_backend"] in {"pgvector", "python_fallback"}


def test_health_is_also_served_under_api_prefix(client):
    assert client.get("/api/v1/health").status_code == 200


def test_every_response_carries_a_request_id(client):
    response = client.get("/health")
    assert response.headers["X-Request-ID"]
    assert float(response.headers["X-Response-Time-ms"]) >= 0


def test_inbound_request_id_is_honoured_but_sanitised(client):
    response = client.get(
        "/health", headers={"X-Request-ID": "trace-123<script>alert(1)</script>"}
    )
    returned = response.headers["X-Request-ID"]
    assert "<" not in returned and ">" not in returned
    assert returned.startswith("trace-123")


def test_security_headers_are_present(client):
    headers = client.get("/health").headers
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"


def test_unknown_route_uses_the_uniform_error_envelope(client):
    response = client.get("/api/v1/does-not-exist")
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "http_error"
    assert error["request_id"]


def test_openapi_schema_is_generated(client):
    schema = client.get("/openapi.json").json()
    assert schema["info"]["title"] == "SecureRAG"
    assert "/health" in schema["paths"]
