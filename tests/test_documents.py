"""
tests/test_documents.py
─────────────────────────────────────────────────────────────
Comprehensive tests for Document Upload, Ingestion Lifecycle,
Validation, Deduplication, and Deletion.
─────────────────────────────────────────────────────────────
"""

import io
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest
from httpx import AsyncClient
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models import Collection, Document

# Valid sample content with appropriate headers
SAMPLE_PDF_BYTES = b"%PDF-1.5\nSample academic handbook content for testing.\n%%EOF"
SAMPLE_DOCX_BYTES = b"PK\x03\x04\x14\x00\x00\x00\x08\x00" + b"A" * 50
SAMPLE_TXT_BYTES = "Dayananda Sagar University Regulations 2026.\nSection 1: Guidelines.".encode("utf-8")


@pytest.fixture
async def sample_collection_id(client: AsyncClient) -> str:
    """Helper fixture to create a valid parent collection for document tests."""
    resp = await client.post(
        "/api/collections",
        json={"name": "Regulations Collection", "description": "Docs for testing"},
    )
    return resp.json()["id"]


@pytest.mark.asyncio
async def test_upload_pdf_success(client: AsyncClient, sample_collection_id: str):
    """Verify uploading a valid PDF succeeds and returns status 'pending'."""
    files = {"file": ("handbook.pdf", io.BytesIO(SAMPLE_PDF_BYTES), "application/pdf")}
    resp = await client.post(f"/api/collections/{sample_collection_id}/documents", files=files)
    assert resp.status_code == 201
    data = resp.json()
    assert data["filename"] == "handbook.pdf"
    assert data["file_type"] == "pdf"
    assert data["status"] == "pending"
    assert data["size_bytes"] == len(SAMPLE_PDF_BYTES)
    assert "sha256" in data
    assert data["version"] == 1
    assert data["collection_id"] == sample_collection_id


@pytest.mark.asyncio
async def test_upload_docx_success(client: AsyncClient, sample_collection_id: str):
    """Verify uploading a valid DOCX succeeds."""
    files = {"file": ("syllabus.docx", io.BytesIO(SAMPLE_DOCX_BYTES), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
    resp = await client.post(f"/api/collections/{sample_collection_id}/documents", files=files)
    assert resp.status_code == 201
    data = resp.json()
    assert data["filename"] == "syllabus.docx"
    assert data["file_type"] == "docx"
    assert data["status"] == "pending"


@pytest.mark.asyncio
async def test_upload_txt_success(client: AsyncClient, sample_collection_id: str):
    """Verify uploading a valid TXT document succeeds."""
    files = {"file": ("notes.txt", io.BytesIO(SAMPLE_TXT_BYTES), "text/plain")}
    resp = await client.post(f"/api/collections/{sample_collection_id}/documents", files=files)
    assert resp.status_code == 201
    data = resp.json()
    assert data["filename"] == "notes.txt"
    assert data["file_type"] == "txt"
    assert data["status"] == "pending"


@pytest.mark.asyncio
async def test_upload_to_nonexistent_collection(client: AsyncClient):
    """Verify uploading to a non-existent collection returns 404."""
    files = {"file": ("doc.txt", io.BytesIO(SAMPLE_TXT_BYTES), "text/plain")}
    resp = await client.post("/api/collections/missing-uuid/documents", files=files)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_upload_empty_file_rejected(client: AsyncClient, sample_collection_id: str):
    """Verify uploading a 0-byte file returns 400 Bad Request."""
    files = {"file": ("empty.txt", io.BytesIO(b""), "text/plain")}
    resp = await client.post(f"/api/collections/{sample_collection_id}/documents", files=files)
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_upload_unsupported_extension_rejected(client: AsyncClient, sample_collection_id: str):
    """Verify uploading an unsupported file type returns 400."""
    files = {"file": ("script.py", io.BytesIO(b"print('hello')"), "text/x-python")}
    resp = await client.post(f"/api/collections/{sample_collection_id}/documents", files=files)
    assert resp.status_code == 400
    assert "unsupported" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_upload_magic_bytes_mismatch_rejected(client: AsyncClient, sample_collection_id: str):
    """Verify file with .pdf extension but invalid header bytes returns 400."""
    fake_pdf = b"NOT_A_REAL_PDF_HEADER_JUST_RANDOM_TEXT"
    files = {"file": ("spoofed.pdf", io.BytesIO(fake_pdf), "application/pdf")}
    resp = await client.post(f"/api/collections/{sample_collection_id}/documents", files=files)
    assert resp.status_code == 400
    assert "signature mismatch" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_upload_txt_with_null_bytes_rejected(client: AsyncClient, sample_collection_id: str):
    """Verify TXT file containing binary null bytes returns 400."""
    binary_txt = b"Text with null \x00\x00 byte injection"
    files = {"file": ("malicious.txt", io.BytesIO(binary_txt), "text/plain")}
    resp = await client.post(f"/api/collections/{sample_collection_id}/documents", files=files)
    assert resp.status_code == 400
    assert "null bytes" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_upload_duplicate_content_rejected(client: AsyncClient, sample_collection_id: str):
    """Verify uploading exact same content with different name returns 409 Conflict."""
    files1 = {"file": ("doc_a.pdf", io.BytesIO(SAMPLE_PDF_BYTES), "application/pdf")}
    resp1 = await client.post(f"/api/collections/{sample_collection_id}/documents", files=files1)
    assert resp1.status_code == 201

    # Upload same content as doc_b.pdf
    files2 = {"file": ("doc_b.pdf", io.BytesIO(SAMPLE_PDF_BYTES), "application/pdf")}
    resp2 = await client.post(f"/api/collections/{sample_collection_id}/documents", files=files2)
    assert resp2.status_code == 409
    assert "identical content already exists" in resp2.json()["detail"].lower()


@pytest.mark.asyncio
async def test_upload_same_filename_without_replace_rejected(client: AsyncClient, sample_collection_id: str):
    """Verify uploading same filename with different content without replace=true returns 409."""
    content1 = b"%PDF-1.4\nFirst version\n%%EOF"
    content2 = b"%PDF-1.4\nSecond modified version\n%%EOF"

    files1 = {"file": ("report.pdf", io.BytesIO(content1), "application/pdf")}
    resp1 = await client.post(f"/api/collections/{sample_collection_id}/documents", files=files1)
    assert resp1.status_code == 201

    files2 = {"file": ("report.pdf", io.BytesIO(content2), "application/pdf")}
    resp2 = await client.post(f"/api/collections/{sample_collection_id}/documents", files=files2)
    assert resp2.status_code == 409
    assert "replace=true" in resp2.json()["detail"].lower()


@pytest.mark.asyncio
async def test_upload_same_filename_with_replace_succeeds(client: AsyncClient, sample_collection_id: str):
    """Verify uploading same filename with replace=true increments version and updates doc."""
    content1 = b"%PDF-1.4\nOriginal content\n%%EOF"
    content2 = b"%PDF-1.4\nUpdated content with edits\n%%EOF"

    files1 = {"file": ("handbook_v1.pdf", io.BytesIO(content1), "application/pdf")}
    resp1 = await client.post(f"/api/collections/{sample_collection_id}/documents", files=files1)
    assert resp1.status_code == 201
    doc_id = resp1.json()["id"]
    assert resp1.json()["version"] == 1

    files2 = {"file": ("handbook_v1.pdf", io.BytesIO(content2), "application/pdf")}
    resp2 = await client.post(
        f"/api/collections/{sample_collection_id}/documents?replace=true",
        files=files2,
    )
    assert resp2.status_code == 200
    data = resp2.json()
    assert data["id"] == doc_id
    assert data["version"] == 2
    assert data["size_bytes"] == len(content2)


@pytest.mark.asyncio
async def test_list_documents(client: AsyncClient, sample_collection_id: str):
    """Verify listing documents returns all items in collection."""
    files1 = {"file": ("d1.txt", io.BytesIO(b"Document 1 text"), "text/plain")}
    files2 = {"file": ("d2.txt", io.BytesIO(b"Document 2 text"), "text/plain")}

    await client.post(f"/api/collections/{sample_collection_id}/documents", files=files1)
    await client.post(f"/api/collections/{sample_collection_id}/documents", files=files2)

    resp = await client.get(f"/api/collections/{sample_collection_id}/documents")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    filenames = {d["filename"] for d in data["documents"]}
    assert "d1.txt" in filenames
    assert "d2.txt" in filenames


@pytest.mark.asyncio
async def test_get_document_by_id(client: AsyncClient, sample_collection_id: str):
    """Verify fetching single document metadata by ID."""
    files = {"file": ("target.txt", io.BytesIO(SAMPLE_TXT_BYTES), "text/plain")}
    created = (await client.post(f"/api/collections/{sample_collection_id}/documents", files=files)).json()
    doc_id = created["id"]

    resp = await client.get(f"/api/documents/{doc_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == doc_id
    assert resp.json()["filename"] == "target.txt"


@pytest.mark.asyncio
async def test_get_document_not_found(client: AsyncClient):
    """Verify 404 for non-existent document ID."""
    resp = await client.get("/api/documents/non-existent-doc-id")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_document_success(client: AsyncClient, sample_collection_id: str):
    """Verify deleting a document removes the database row."""
    files = {"file": ("to_del.txt", io.BytesIO(b"Delete me"), "text/plain")}
    created = (await client.post(f"/api/collections/{sample_collection_id}/documents", files=files)).json()
    doc_id = created["id"]

    del_resp = await client.delete(f"/api/documents/{doc_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "deleted"

    # Subsequent GET must be 404
    get_resp = await client.get(f"/api/documents/{doc_id}")
    assert get_resp.status_code == 404


@pytest.mark.asyncio
async def test_path_traversal_filename_sanitized(client: AsyncClient, sample_collection_id: str):
    """Verify malicious filenames with path traversal characters are sanitized."""
    files = {"file": ("../../etc/passwd.txt", io.BytesIO(SAMPLE_TXT_BYTES), "text/plain")}
    resp = await client.post(f"/api/collections/{sample_collection_id}/documents", files=files)
    assert resp.status_code == 201
    saved_name = resp.json()["filename"]
    assert ".." not in saved_name
    assert "/" not in saved_name
    assert "\\" not in saved_name


# ═════════════════════════════════════════════════════════════
# Hardening Tests: Streaming 50MB+, DB Constraints & Rollbacks
# ═════════════════════════════════════════════════════════════

class PseudoStreamingLargeFile(io.RawIOBase):
    """
    Simulates streaming an oversized 52 MB PDF file chunk-by-chunk
    without allocating 52 MB of memory in RAM.
    """
    def __init__(self, total_size: int = 52 * 1024 * 1024):
        self.total_size = total_size
        self.bytes_read = 0
        self.header = b"%PDF-1.5\n"

    def readinto(self, b):
        if self.bytes_read >= self.total_size:
            return 0  # EOF
        chunk_len = min(len(b), self.total_size - self.bytes_read)
        if self.bytes_read < len(self.header):
            hdr_part = self.header[self.bytes_read:self.bytes_read + chunk_len]
            b[:len(hdr_part)] = hdr_part
            fill_len = chunk_len - len(hdr_part)
            if fill_len > 0:
                b[len(hdr_part):chunk_len] = b"A" * fill_len
        else:
            b[:chunk_len] = b"A" * chunk_len
        self.bytes_read += chunk_len
        return chunk_len

    def readable(self):
        return True


@pytest.mark.asyncio
async def test_upload_exceeding_50mb_streaming_rejected(
    client: AsyncClient, sample_collection_id: str, test_upload_dir: Path
):
    """
    Hardening: Verify uploading a file > 50 MB is aborted midway with HTTP 413,
    and any partial file written to disk is immediately cleaned up.
    """
    stream = PseudoStreamingLargeFile()
    files = {"file": ("oversized.pdf", stream, "application/pdf")}
    resp = await client.post(
        f"/api/collections/{sample_collection_id}/documents",
        files=files,
    )
    assert resp.status_code == 413
    assert "exceeds maximum" in resp.json()["detail"].lower()

    # Verify no partial files lingered on disk
    col_dir = test_upload_dir / sample_collection_id
    lingering = list(col_dir.glob("*"))
    assert len(lingering) == 0


@pytest.mark.asyncio
async def test_foreign_key_enforcement_direct_insert(db_session: AsyncSession):
    """
    Hardening: Verify PRAGMA foreign_keys=ON is active and rejects direct
    document insertions with a non-existent collection_id.
    """
    orphan_doc = Document(
        id=str(uuid.uuid4()),
        collection_id="invalid-collection-id-99999",
        filename="orphan.txt",
        stored_filename="orphan_stored.txt",
        file_type="txt",
        sha256="orphan_hash_12345",
        size_bytes=100,
        status="pending",
        version=1,
    )
    db_session.add(orphan_doc)
    with pytest.raises(IntegrityError):
        await db_session.commit()
    await db_session.rollback()


@pytest.mark.asyncio
async def test_database_level_duplicate_sha256_constraint(db_session: AsyncSession):
    """
    Hardening: Verify UniqueConstraint('collection_id', 'sha256') rejects
    duplicate content hashes within the same collection at the database level.
    """
    col = Collection(name="Constraint Test Col", description="Direct constraint test")
    db_session.add(col)
    await db_session.commit()
    await db_session.refresh(col)

    doc1 = Document(
        id=str(uuid.uuid4()),
        collection_id=col.id,
        filename="first.txt",
        stored_filename="first_stored.txt",
        file_type="txt",
        sha256="identical_sha256_hash",
        size_bytes=50,
        status="pending",
        version=1,
    )
    db_session.add(doc1)
    await db_session.commit()

    # Second insert with identical collection_id and sha256
    doc2 = Document(
        id=str(uuid.uuid4()),
        collection_id=col.id,
        filename="second.txt",
        stored_filename="second_stored.txt",
        file_type="txt",
        sha256="identical_sha256_hash",
        size_bytes=50,
        status="pending",
        version=1,
    )
    db_session.add(doc2)
    with pytest.raises(IntegrityError):
        await db_session.commit()
    await db_session.rollback()


@pytest.mark.asyncio
async def test_replacement_rollback_preserves_old_file_and_state(
    client: AsyncClient, sample_collection_id: str, test_upload_dir: Path
):
    """
    Hardening: Verify staged replacement workflow. If DB commit fails during
    replace=true, rollback preserves the old file and version 1 DB record,
    and removes the newly uploaded replacement file.
    """
    content_v1 = b"%PDF-1.4\nOriginal Version 1 Content\n%%EOF"
    f1 = {"file": ("manual.pdf", io.BytesIO(content_v1), "application/pdf")}
    r1 = await client.post(f"/api/collections/{sample_collection_id}/documents", files=f1)
    assert r1.status_code == 201
    doc_id = r1.json()["id"]

    col_dir = test_upload_dir / sample_collection_id
    stored_files_before = list(col_dir.glob("*"))
    assert len(stored_files_before) == 1
    old_file_path = stored_files_before[0]
    assert old_file_path.read_bytes() == content_v1

    # Attempt replacement with simulated commit failure
    content_v2 = b"%PDF-1.4\nUpdated Version 2 Content\n%%EOF"
    f2 = {"file": ("manual.pdf", io.BytesIO(content_v2), "application/pdf")}

    with patch.object(AsyncSession, "commit", side_effect=RuntimeError("Simulated replacement commit failure")):
        r2 = await client.post(
            f"/api/collections/{sample_collection_id}/documents?replace=true",
            files=f2,
        )
        assert r2.status_code == 500

    # Verify old file on disk is intact with original content
    assert old_file_path.exists()
    assert old_file_path.read_bytes() == content_v1

    # Verify new file was NOT leaked onto disk
    assert len(list(col_dir.glob("*"))) == 1

    # Verify database record still shows version 1
    doc_resp = await client.get(f"/api/documents/{doc_id}")
    assert doc_resp.status_code == 200
    assert doc_resp.json()["version"] == 1


@pytest.mark.asyncio
async def test_document_deletion_rollback_preserves_file(
    client: AsyncClient, sample_collection_id: str, test_upload_dir: Path
):
    """
    Hardening: Verify deletion ordering. If DB commit fails during document
    deletion, rollback preserves the file on disk and the DB record.
    """
    f = {"file": ("preserve_me.txt", io.BytesIO(b"Data that must not be deleted if DB fails"), "text/plain")}
    r = await client.post(f"/api/collections/{sample_collection_id}/documents", files=f)
    assert r.status_code == 201
    doc_id = r.json()["id"]

    col_dir = test_upload_dir / sample_collection_id
    stored_files = list(col_dir.glob("*"))
    assert len(stored_files) == 1
    stored_file_path = stored_files[0]

    # Attempt delete with simulated commit failure
    with patch.object(AsyncSession, "commit", side_effect=RuntimeError("Simulated delete commit failure")):
        del_resp = await client.delete(f"/api/documents/{doc_id}")
        assert del_resp.status_code == 500

    # Verify file is still present on disk
    assert stored_file_path.exists()

    # Verify document is still in database
    get_resp = await client.get(f"/api/documents/{doc_id}")
    assert get_resp.status_code == 200
