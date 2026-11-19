"""
tests/test_collections.py
─────────────────────────────────────────────────────────────
Comprehensive tests for Knowledge Collections API endpoints.
─────────────────────────────────────────────────────────────
"""

import io
from pathlib import Path
from unittest.mock import patch

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession


@pytest.mark.asyncio
async def test_create_collection_success(client: AsyncClient):
    """Verify creating a collection returns 201 and valid schema."""
    resp = await client.post(
        "/api/collections",
        json={"name": "Academic Regulations", "description": "University policy docs"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "Academic Regulations"
    assert data["description"] == "University policy docs"
    assert "id" in data
    assert data["document_count"] == 0


@pytest.mark.asyncio
async def test_create_collection_duplicate_name_rejected(client: AsyncClient):
    """Verify duplicate collection names return HTTP 409 Conflict."""
    await client.post(
        "/api/collections",
        json={"name": "Syllabus Collection", "description": "First instance"},
    )
    dup_resp = await client.post(
        "/api/collections",
        json={"name": "Syllabus Collection", "description": "Duplicate attempt"},
    )
    assert dup_resp.status_code == 409
    assert "already exists" in dup_resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_create_collection_empty_name_rejected(client: AsyncClient):
    """Verify empty or whitespace-only collection names return 422 Unprocessable Entity."""
    resp = await client.post(
        "/api/collections",
        json={"name": "   ", "description": "Whitespace only"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_list_collections_empty(client: AsyncClient):
    """Verify listing when no collections exist returns empty list."""
    resp = await client.get("/api/collections")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["collections"] == []


@pytest.mark.asyncio
async def test_list_collections_with_counts(client: AsyncClient):
    """Verify listing collections returns all items with document counts."""
    await client.post("/api/collections", json={"name": "Col A"})
    await client.post("/api/collections", json={"name": "Col B"})

    resp = await client.get("/api/collections")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    names = {c["name"] for c in data["collections"]}
    assert "Col A" in names
    assert "Col B" in names


@pytest.mark.asyncio
async def test_get_collection_by_id(client: AsyncClient):
    """Verify getting single collection by ID."""
    created = (await client.post("/api/collections", json={"name": "Col Target"})).json()
    col_id = created["id"]

    resp = await client.get(f"/api/collections/{col_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == col_id
    assert resp.json()["name"] == "Col Target"


@pytest.mark.asyncio
async def test_get_collection_not_found(client: AsyncClient):
    """Verify 404 when collection does not exist."""
    resp = await client.get("/api/collections/non-existent-uuid")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_collection_success(client: AsyncClient):
    """Verify deleting collection returns 200 and removes record."""
    created = (await client.post("/api/collections", json={"name": "To Delete"})).json()
    col_id = created["id"]

    del_resp = await client.delete(f"/api/collections/{col_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "deleted"

    # Verify subsequent GET returns 404
    get_resp = await client.get(f"/api/collections/{col_id}")
    assert get_resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_collection_not_found(client: AsyncClient):
    """Verify deleting non-existent collection returns 404."""
    resp = await client.delete("/api/collections/non-existent-uuid")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_collection_cleans_multiple_documents_and_disk_files(
    client: AsyncClient, test_upload_dir: Path
):
    """
    Hardening: Verify deleting a collection cascades to child documents
    and deletes all disk files and the collection directory.
    """
    # 1. Create collection
    col_resp = await client.post("/api/collections", json={"name": "Multi Doc Cascade Col"})
    assert col_resp.status_code == 201
    col_id = col_resp.json()["id"]

    # 2. Upload doc 1
    f1 = {"file": ("cascade_doc1.txt", io.BytesIO(b"Document 1 text"), "text/plain")}
    r1 = await client.post(f"/api/collections/{col_id}/documents", files=f1)
    assert r1.status_code == 201
    doc1_id = r1.json()["id"]

    # 3. Upload doc 2
    f2 = {"file": ("cascade_doc2.txt", io.BytesIO(b"Document 2 text"), "text/plain")}
    r2 = await client.post(f"/api/collections/{col_id}/documents", files=f2)
    assert r2.status_code == 201
    doc2_id = r2.json()["id"]

    # Check files exist on disk before deletion
    col_storage_dir = test_upload_dir / col_id
    assert col_storage_dir.exists()
    assert len(list(col_storage_dir.glob("*"))) == 2

    # 4. Delete collection
    del_resp = await client.delete(f"/api/collections/{col_id}")
    assert del_resp.status_code == 200

    # 5. Verify storage directory and files are completely removed from disk
    assert not col_storage_dir.exists()

    # 6. Verify collection and child documents are deleted from DB
    get_col = await client.get(f"/api/collections/{col_id}")
    assert get_col.status_code == 404
    get_d1 = await client.get(f"/api/documents/{doc1_id}")
    assert get_d1.status_code == 404
    get_d2 = await client.get(f"/api/documents/{doc2_id}")
    assert get_d2.status_code == 404


@pytest.mark.asyncio
async def test_collection_deletion_rollback_preserves_files(
    client: AsyncClient, test_upload_dir: Path
):
    """
    Hardening: Verify that if DB commit fails during collection deletion,
    the rollback preserves both the DB records and all files on disk.
    """
    # 1. Create collection
    col_resp = await client.post("/api/collections", json={"name": "Rollback Safety Col"})
    assert col_resp.status_code == 201
    col_id = col_resp.json()["id"]

    # 2. Upload document
    f = {"file": ("safe_doc.txt", io.BytesIO(b"Preserved on failure"), "text/plain")}
    r = await client.post(f"/api/collections/{col_id}/documents", files=f)
    assert r.status_code == 201

    col_storage_dir = test_upload_dir / col_id
    assert col_storage_dir.exists()
    assert len(list(col_storage_dir.glob("*"))) == 1

    # 3. Simulate DB commit failure on delete collection
    with patch.object(AsyncSession, "commit", side_effect=RuntimeError("Simulated DB commit error")):
        del_resp = await client.delete(f"/api/collections/{col_id}")
        assert del_resp.status_code == 500

    # 4. Verify directory and files are STILL on disk
    assert col_storage_dir.exists()
    assert len(list(col_storage_dir.glob("*"))) == 1

    # 5. Verify collection is STILL intact in database
    get_col = await client.get(f"/api/collections/{col_id}")
    assert get_col.status_code == 200
    assert get_col.json()["name"] == "Rollback Safety Col"
