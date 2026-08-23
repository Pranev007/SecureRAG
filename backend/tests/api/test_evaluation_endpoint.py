"""The evaluation endpoint: admin gating, isolation, and the report shape."""

from __future__ import annotations

import pytest
from sqlalchemy import func, select

from app.models import Document, User

pytestmark = pytest.mark.api


def test_running_the_suite_requires_an_administrator(client, user_headers):
    response = client.post("/api/v1/evaluation/run", headers=user_headers, json={})
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "authorization_error"


def test_running_the_suite_requires_authentication(client):
    assert client.post("/api/v1/evaluation/run", json={}).status_code == 401


def test_an_admin_can_run_a_slice_of_the_suite(client, admin_headers):
    response = client.post(
        "/api/v1/evaluation/run",
        headers=admin_headers,
        json={"kinds": ["direct_injection", "benign_control"]},
    )
    assert response.status_code == 200

    body = response.json()
    assert body["totals"]["cases"] > 0
    assert body["security"]["attack_cases"] > 0
    assert body["security"]["benign_cases"] > 0
    # The claim the whole project rests on, asserted over HTTP.
    assert body["security"]["detection_rate"] >= 0.9
    assert body["security"]["false_positive_rate"] <= 0.1
    assert body["configuration"]["llm_provider"] == "echo"
    assert body["duration_seconds"] > 0


def test_the_run_does_not_touch_the_live_database(client, admin_headers, db):
    """The suite creates users and ingests a poisoned corpus of its own.

    If it ran against the application's database those fixtures would land in
    real data, so the endpoint gives it a throwaway one. This asserts the live
    database is untouched.
    """
    users_before = db.execute(select(func.count()).select_from(User)).scalar_one()
    documents_before = db.execute(select(func.count()).select_from(Document)).scalar_one()

    client.post(
        "/api/v1/evaluation/run",
        headers=admin_headers,
        json={"kinds": ["direct_injection"]},
    )

    assert db.execute(select(func.count()).select_from(User)).scalar_one() == users_before
    assert (
        db.execute(select(func.count()).select_from(Document)).scalar_one()
        == documents_before
    )
    # And specifically none of the evaluation's own fixtures leaked in.
    assert (
        db.execute(
            select(func.count()).select_from(User).where(User.email.like("eval-%"))
        ).scalar_one()
        == 0
    )


def test_per_case_detail_is_opt_in(client, admin_headers):
    lean = client.post(
        "/api/v1/evaluation/run",
        headers=admin_headers,
        json={"kinds": ["direct_injection"]},
    ).json()
    assert lean["cases"] == []

    detailed = client.post(
        "/api/v1/evaluation/run",
        headers=admin_headers,
        json={"kinds": ["direct_injection"], "include_cases": True},
    ).json()
    assert detailed["cases"]
    assert "expected_behaviour" in detailed["cases"][0]


def test_an_unknown_case_kind_yields_an_empty_run_not_an_error(client, admin_headers):
    response = client.post(
        "/api/v1/evaluation/run",
        headers=admin_headers,
        json={"kinds": ["not_a_real_kind"]},
    )
    assert response.status_code == 200
    assert response.json()["totals"]["cases"] == 0


def test_the_dataset_can_be_described_without_running_it(client, admin_headers):
    response = client.get("/api/v1/evaluation/dataset", headers=admin_headers)
    assert response.status_code == 200

    body = response.json()
    assert body["total"] > 0
    assert body["by_kind"]["benign_control"] > 0
    assert body["by_kind"]["direct_injection"] > 0


def test_the_dataset_endpoint_is_admin_only(client, user_headers):
    assert (
        client.get("/api/v1/evaluation/dataset", headers=user_headers).status_code == 403
    )
