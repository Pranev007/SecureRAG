"""Document endpoints: upload, listing, ownership and deletion."""

from __future__ import annotations

import pytest

from tests.factories import HANDBOOK_MARKDOWN, SECURITY_POLICY_MARKDOWN, build_pdf

pytestmark = pytest.mark.api


def _upload(client, headers, name="handbook.md", data=None, content_type="text/markdown"):
    return client.post(
        "/api/v1/documents",
        headers=headers,
        files={
            "file": (
                name,
                HANDBOOK_MARKDOWN.encode() if data is None else data,
                content_type,
            )
        },
    )


# ----------------------------------------------------------------------
# Upload
# ----------------------------------------------------------------------


def test_upload_ingests_and_reports_chunk_count(client, user_headers):
    response = _upload(client, user_headers)
    assert response.status_code == 201

    body = response.json()
    assert body["document"]["status"] == "ready"
    assert body["document"]["chunk_count"] > 0
    assert body["document"]["filename"] == "handbook.md"
    assert "Ingested" in body["message"]
    assert body["warnings"] == []


def test_upload_of_a_pdf_records_pages(client, user_headers):
    data = build_pdf(
        [
            "Employees accrue two days of paid annual leave every month.",
            "Multi-factor authentication is mandatory for all accounts.",
        ]
    )
    response = _upload(client, user_headers, "policy.pdf", data, "application/pdf")
    assert response.status_code == 201
    assert response.json()["document"]["page_count"] == 2


def test_upload_rejects_an_unsupported_extension(client, user_headers):
    response = _upload(
        client, user_headers, "malware.exe", b"MZ\x90\x00", "application/octet-stream"
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "ingestion_error"


def test_upload_rejects_content_that_contradicts_its_extension(client, user_headers):
    # A ZIP container renamed to .txt must not reach the text parser.
    response = _upload(
        client, user_headers, "notes.txt", b"PK\x03\x04payload", "text/plain"
    )
    assert response.status_code == 400


def test_upload_rejects_an_empty_file(client, user_headers):
    response = _upload(client, user_headers, "empty.txt", b"", "text/plain")
    assert response.status_code == 400
    assert "empty" in response.json()["error"]["message"].lower()


@pytest.mark.parametrize(
    "document_id",
    ["does-not-exist", "../../etc/passwd", "1 OR 1=1", "%00", "x" * 300],
)
def test_malformed_document_ids_return_404_not_500(client, user_headers, document_id):
    """Identifiers come from the URL, so they are attacker-controlled input.

    A traversal-shaped id is normalised by the router and never matches a
    route, so it produces a routing 404; everything else reaches the handler
    and produces an application 404. Either way: no 500, no stack trace.
    """
    response = client.get(f"/api/v1/documents/{document_id}", headers=user_headers)
    assert response.status_code == 404
    assert response.json()["error"]["code"] in {"not_found", "http_error"}
    assert "Traceback" not in response.text


def test_upload_path_traversal_filename_is_neutralised(client, user_headers):
    response = _upload(client, user_headers, "../../etc/passwd.md")
    assert response.status_code == 201
    assert response.json()["document"]["filename"] == "passwd.md"


def test_reuploading_the_same_bytes_is_idempotent(client, user_headers):
    first = _upload(client, user_headers)
    second = _upload(client, user_headers, name="handbook-copy.md")
    assert first.json()["document"]["id"] == second.json()["document"]["id"]

    listing = client.get("/api/v1/documents", headers=user_headers).json()
    assert listing["total"] == 1


# ----------------------------------------------------------------------
# Listing and fetching
# ----------------------------------------------------------------------


def test_listing_is_scoped_to_the_caller(client, user_headers, other_user_headers):
    _upload(client, user_headers, "mine.md")
    _upload(
        client,
        other_user_headers,
        "theirs.md",
        SECURITY_POLICY_MARKDOWN.encode(),
    )

    mine = client.get("/api/v1/documents", headers=user_headers).json()
    theirs = client.get("/api/v1/documents", headers=other_user_headers).json()

    assert [d["filename"] for d in mine["items"]] == ["mine.md"]
    assert [d["filename"] for d in theirs["items"]] == ["theirs.md"]


def test_listing_pagination(client, user_headers):
    _upload(client, user_headers, "a.md", b"# A\n\n" + b"Alpha content here. " * 20)
    _upload(client, user_headers, "b.md", b"# B\n\n" + b"Bravo content here. " * 20)

    page = client.get("/api/v1/documents?limit=1&offset=0", headers=user_headers).json()
    assert page["total"] == 2
    assert len(page["items"]) == 1
    assert page["limit"] == 1


def test_fetching_own_document_with_chunks(client, user_headers):
    document_id = _upload(client, user_headers).json()["document"]["id"]
    response = client.get(
        f"/api/v1/documents/{document_id}?include_chunks=true", headers=user_headers
    )
    assert response.status_code == 200

    body = response.json()
    assert body["id"] == document_id
    assert body["chunks"]
    assert body["chunks"][0]["chunk_index"] == 0
    assert "content" in body["chunks"][0]


def test_fetching_another_users_document_returns_404_not_403(
    client, user_headers, other_user_headers
):
    """403 would confirm the id exists. 404 reveals nothing."""
    document_id = _upload(client, other_user_headers).json()["document"]["id"]
    response = client.get(f"/api/v1/documents/{document_id}", headers=user_headers)

    assert response.status_code == 404
    missing = client.get("/api/v1/documents/does-not-exist", headers=user_headers)
    assert response.json()["error"]["message"] == missing.json()["error"]["message"]


def test_chunks_of_another_users_document_are_not_reachable(
    client, user_headers, other_user_headers
):
    document_id = _upload(client, other_user_headers).json()["document"]["id"]
    response = client.get(
        f"/api/v1/documents/{document_id}?include_chunks=true", headers=user_headers
    )
    assert response.status_code == 404
    assert "accrue" not in response.text


def test_admin_can_fetch_another_users_document(client, user_headers, admin_headers):
    document_id = _upload(client, user_headers).json()["document"]["id"]
    response = client.get(f"/api/v1/documents/{document_id}", headers=admin_headers)
    assert response.status_code == 200


# ----------------------------------------------------------------------
# Deletion
# ----------------------------------------------------------------------


def test_owner_can_delete_their_document(client, user_headers):
    document_id = _upload(client, user_headers).json()["document"]["id"]

    assert (
        client.delete(
            f"/api/v1/documents/{document_id}", headers=user_headers
        ).status_code
        == 200
    )
    assert (
        client.get(f"/api/v1/documents/{document_id}", headers=user_headers).status_code
        == 404
    )


def test_deleting_another_users_document_is_refused(
    client, user_headers, other_user_headers
):
    document_id = _upload(client, other_user_headers).json()["document"]["id"]

    assert (
        client.delete(
            f"/api/v1/documents/{document_id}", headers=user_headers
        ).status_code
        == 404
    )
    # The real owner still has it.
    assert (
        client.get(
            f"/api/v1/documents/{document_id}", headers=other_user_headers
        ).status_code
        == 200
    )
