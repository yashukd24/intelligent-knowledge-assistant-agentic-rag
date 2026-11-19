"""
tests/test_document_deletion_router.py
─────────────────────────────────────────────────────────────
Comprehensive tests for recoverable document deletion and crash-safety.

Verifies:
  1. Normal deletion succeeds.
  2. Vector deletion failure retains DELETION_PENDING.
  3. Filesystem deletion failure retains DELETION_PENDING.
  4. Missing physical file is treated as already cleaned.
  5. Final SQLite commit failure can be retried.
  6. Crash simulation after pending-state commit.
  7. Crash simulation after vector cleanup.
  8. Crash simulation after file cleanup.
  9. Retry completes pending deletion.
  10. Two simultaneous deletion requests.
  11. Processing a pending-deletion document is rejected.
  12. Unrelated documents remain intact.
  13. Other collections remain intact.
  14. Existing 404 and API response behavior remain correct.
  15. Recovery processes multiple pending documents and continues after an individual failure.
─────────────────────────────────────────────────────────────
"""

import asyncio
import io
import math
from pathlib import Path
from typing import List, Optional
from unittest.mock import patch

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.collection import Collection
from backend.models.document import Document
from backend.schemas.chunk import ChunkMetadata
from backend.schemas.document import DocumentStatus
from backend.schemas.embedding import EmbeddedChunk
from backend.services.document_recovery import DocumentRecoveryService
from backend.services.file_storage import FileStorageService
from backend.services.vector_store import (
    VectorStorageError,
    VectorStoreService,
)


# ── Test Helpers & Deterministic Dummy Vectors ──────────────────────────────


def make_dummy_embedding(dim: int = 384, seed: int = 0) -> List[float]:
    """Generate a deterministic unit-normalized vector for testing."""
    raw = [float(((i + seed) % 17) + 1) for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


def make_embedded_chunk(
    chunk_index: int = 0,
    document_id: str = "doc_001",
    collection_id: str = "col_001",
    filename: str = "sample.txt",
    version: int = 1,
    dimension: int = 384,
) -> EmbeddedChunk:
    """Helper to construct an EmbeddedChunk with consistent metadata."""
    text = f"Sample text for document {document_id} chunk {chunk_index} version {version}."
    emb = make_dummy_embedding(dim=dimension, seed=chunk_index + version * 10)
    meta = ChunkMetadata(
        document_id=document_id,
        collection_id=collection_id,
        filename=filename,
        file_type="txt",
        chunk_index=chunk_index,
        char_count=len(text),
        page_number=None,
        page_range=None,
        start_char=chunk_index * 100,
        end_char=chunk_index * 100 + len(text),
        version=version,
    )
    return EmbeddedChunk(
        chunk_id=f"{document_id}_v{version}_c{chunk_index:04d}",
        document_id=document_id,
        collection_id=collection_id,
        chunk_text=text,
        metadata=meta,
        embedding=emb,
        embedding_dimension=dimension,
        model_name="test-model-v1",
    )


async def upload_and_index_document(
    client: AsyncClient,
    test_vector_store: VectorStoreService,
    collection_id: str,
    filename: str = "regulations.txt",
    content: bytes = b"Regulations paragraph one. Academic integrity guidelines.\n\nParagraph two.",
    chunk_count: int = 2,
    version: int = 1,
) -> tuple[str, str, int]:
    """
    Helper to upload a document via the REST client and index its vectors
    into the isolated test vector store.
    Returns (collection_id, document_id, chunk_count).
    """
    files = {"file": (filename, io.BytesIO(content), "text/plain")}
    resp = await client.post(f"/api/collections/{collection_id}/documents", files=files)
    assert resp.status_code == 201
    doc_data = resp.json()
    doc_id = doc_data["id"]

    chunks = [
        make_embedded_chunk(
            chunk_index=i,
            document_id=doc_id,
            collection_id=collection_id,
            filename=filename,
            version=version,
        )
        for i in range(chunk_count)
    ]
    test_vector_store.store_chunks(chunks, collection_id=collection_id)
    assert test_vector_store.count(collection_id) >= chunk_count
    return collection_id, doc_id, chunk_count


# ── Test Suite: 15 Required Deletion & Recovery Tests ────────────────────────


@pytest.mark.asyncio
async def test_1_normal_deletion_succeeds(
    client: AsyncClient,
    test_upload_dir: Path,
    test_vector_store: VectorStoreService,
):
    """1. Normal deletion succeeds completely across SQLite, ChromaDB, and disk."""
    col_resp = await client.post("/api/collections", json={"name": "Del Col 1"})
    assert col_resp.status_code == 201
    col_id = col_resp.json()["id"]

    _, doc_id, cnt = await upload_and_index_document(client, test_vector_store, col_id)

    col_dir = test_upload_dir / col_id
    assert len(list(col_dir.glob("*"))) == 1
    assert test_vector_store.count(col_id) == cnt

    # Issue REST deletion
    del_resp = await client.delete(f"/api/documents/{doc_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "deleted"

    # SQLite record removed
    assert (await client.get(f"/api/documents/{doc_id}")).status_code == 404

    # ChromaDB vectors removed
    assert test_vector_store.count(col_id) == 0

    # Physical file removed
    assert len(list(col_dir.glob("*"))) == 0


@pytest.mark.asyncio
async def test_2_vector_deletion_failure_retains_deletion_pending(
    client: AsyncClient,
    test_upload_dir: Path,
    test_vector_store: VectorStoreService,
):
    """2. Vector deletion failure returns HTTP 500 and retains DELETION_PENDING in SQLite."""
    col_resp = await client.post("/api/collections", json={"name": "Del Col 2"})
    assert col_resp.status_code == 201
    col_id = col_resp.json()["id"]

    _, doc_id, cnt = await upload_and_index_document(
        client, test_vector_store, col_id, filename="fail_vec.txt",
        content=b"Content for vector fail test."
    )

    with patch.object(
        VectorStoreService,
        "delete_document_vectors",
        side_effect=VectorStorageError("Simulated ChromaDB locked catalog error"),
    ):
        del_resp = await client.delete(f"/api/documents/{doc_id}")
        assert del_resp.status_code == 500
        assert "vector store deletion failed" in del_resp.json()["detail"].lower()

    # Target document MUST STILL EXIST in SQLite with DELETION_PENDING
    get_resp = await client.get(f"/api/documents/{doc_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["status"] == DocumentStatus.DELETION_PENDING.value

    # Physical file still exists on disk
    col_dir = test_upload_dir / col_id
    assert len(list(col_dir.glob("*"))) == 1

    # Vectors still exist in ChromaDB
    assert test_vector_store.count(col_id) == cnt


@pytest.mark.asyncio
async def test_3_filesystem_deletion_failure_retains_deletion_pending(
    client: AsyncClient,
    test_upload_dir: Path,
    test_vector_store: VectorStoreService,
):
    """3. Filesystem deletion failure returns HTTP 500, retains DELETION_PENDING, and does not report false success."""
    col_resp = await client.post("/api/collections", json={"name": "Del Col 3"})
    assert col_resp.status_code == 201
    col_id = col_resp.json()["id"]

    _, doc_id, _ = await upload_and_index_document(
        client, test_vector_store, col_id, filename="fail_fs.txt",
        content=b"Content for filesystem fail test."
    )

    # Patch FileStorageService.delete_stored_file to simulate OSError (returns False)
    with patch.object(FileStorageService, "delete_stored_file", return_value=False):
        del_resp = await client.delete(f"/api/documents/{doc_id}")
        assert del_resp.status_code == 500
        assert "filesystem deletion failed" in del_resp.json()["detail"].lower()

    # SQLite document MUST retain DELETION_PENDING status (never reports false success)
    get_resp = await client.get(f"/api/documents/{doc_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["status"] == DocumentStatus.DELETION_PENDING.value

    # Physical file is still on disk
    col_dir = test_upload_dir / col_id
    assert len(list(col_dir.glob("*"))) == 1


@pytest.mark.asyncio
async def test_4_missing_physical_file_is_treated_as_already_cleaned(
    client: AsyncClient,
    test_upload_dir: Path,
    test_vector_store: VectorStoreService,
):
    """4. If physical file is already missing from disk, deletion treats it as cleaned and succeeds."""
    col_resp = await client.post("/api/collections", json={"name": "Del Col 4"})
    assert col_resp.status_code == 201
    col_id = col_resp.json()["id"]

    _, doc_id, _ = await upload_and_index_document(
        client, test_vector_store, col_id, filename="missing_file.txt",
        content=b"Content for missing file test."
    )

    # Manually delete the file before calling DELETE
    col_dir = test_upload_dir / col_id
    files = list(col_dir.glob("*"))
    assert len(files) == 1
    files[0].unlink()
    assert len(list(col_dir.glob("*"))) == 0

    # Issue DELETE request
    del_resp = await client.delete(f"/api/documents/{doc_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "deleted"

    # SQLite record removed
    assert (await client.get(f"/api/documents/{doc_id}")).status_code == 404
    # ChromaDB vectors removed
    assert test_vector_store.count(col_id) == 0


@pytest.mark.asyncio
async def test_5_final_sqlite_commit_failure_can_be_retried(
    client: AsyncClient,
    test_upload_dir: Path,
    test_vector_store: VectorStoreService,
):
    """5. Final SQLite commit failure retains DELETION_PENDING and subsequent retry succeeds."""
    col_resp = await client.post("/api/collections", json={"name": "Del Col 5"})
    assert col_resp.status_code == 201
    col_id = col_resp.json()["id"]

    _, doc_id, _ = await upload_and_index_document(
        client, test_vector_store, col_id, filename="commit_fail.txt",
        content=b"Content for commit fail test."
    )

    commit_count = 0
    orig_commit = AsyncSession.commit

    async def mock_commit(self, *args, **kwargs):
        nonlocal commit_count
        commit_count += 1
        # Commit 1: setting DELETION_PENDING (allowed)
        # Commit 2: final delete commit (fails)
        if commit_count == 2:
            raise RuntimeError("Simulated final commit failure")
        return await orig_commit(self, *args, **kwargs)

    with patch.object(AsyncSession, "commit", mock_commit):
        del_resp = await client.delete(f"/api/documents/{doc_id}")
        assert del_resp.status_code == 500

    # Document remains in SQLite as DELETION_PENDING
    get_resp = await client.get(f"/api/documents/{doc_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["status"] == DocumentStatus.DELETION_PENDING.value

    # Subsequent retry without failure succeeds completely
    retry_resp = await client.delete(f"/api/documents/{doc_id}")
    assert retry_resp.status_code == 200
    assert retry_resp.json()["status"] == "deleted"

    # Fully purged
    assert (await client.get(f"/api/documents/{doc_id}")).status_code == 404
    assert test_vector_store.count(col_id) == 0


@pytest.mark.asyncio
async def test_6_crash_simulation_after_pending_state_commit(
    client: AsyncClient,
    test_upload_dir: Path,
    test_vector_store: VectorStoreService,
):
    """6. Crash simulation after DELETION_PENDING commit: retry safely resumes and deletes all."""
    col_resp = await client.post("/api/collections", json={"name": "Del Col 6"})
    assert col_resp.status_code == 201
    col_id = col_resp.json()["id"]

    _, doc_id, _ = await upload_and_index_document(
        client, test_vector_store, col_id, filename="crash_phase1.txt",
        content=b"Content for crash phase 1 test."
    )

    # Simulate crash right after DELETION_PENDING is committed by raising in vector store
    with patch.object(
        VectorStoreService,
        "delete_document_vectors",
        side_effect=RuntimeError("Process terminated after phase 1 commit"),
    ):
        fail_resp = await client.delete(f"/api/documents/{doc_id}")
        assert fail_resp.status_code == 500

    # Verify status is DELETION_PENDING
    doc_state = (await client.get(f"/api/documents/{doc_id}")).json()
    assert doc_state["status"] == DocumentStatus.DELETION_PENDING.value

    # Resume on retry (recovery call)
    retry_resp = await client.delete(f"/api/documents/{doc_id}")
    assert retry_resp.status_code == 200

    # All components clean
    assert (await client.get(f"/api/documents/{doc_id}")).status_code == 404
    assert test_vector_store.count(col_id) == 0


@pytest.mark.asyncio
async def test_7_crash_simulation_after_vector_cleanup(
    client: AsyncClient,
    test_upload_dir: Path,
    test_vector_store: VectorStoreService,
):
    """7. Crash simulation after vectors deleted but before file unlinked: retry resumes cleanly."""
    col_resp = await client.post("/api/collections", json={"name": "Del Col 7"})
    assert col_resp.status_code == 201
    col_id = col_resp.json()["id"]

    _, doc_id, _ = await upload_and_index_document(
        client, test_vector_store, col_id, filename="crash_phase2.txt",
        content=b"Content for crash phase 2 test."
    )

    # Vectors delete, but crash occurs before file deletion finishes
    with patch.object(
        FileStorageService,
        "delete_stored_file",
        side_effect=RuntimeError("Process terminated after vectors deleted"),
    ):
        try:
            await client.delete(f"/api/documents/{doc_id}")
        except Exception:
            pass

    # Vectors are gone from ChromaDB
    assert test_vector_store.count(col_id) == 0

    # Document still in SQLite as DELETION_PENDING
    get_resp = await client.get(f"/api/documents/{doc_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["status"] == DocumentStatus.DELETION_PENDING.value

    # Retry cleans file and completes deletion
    retry_resp = await client.delete(f"/api/documents/{doc_id}")
    assert retry_resp.status_code == 200
    assert (await client.get(f"/api/documents/{doc_id}")).status_code == 404


@pytest.mark.asyncio
async def test_8_crash_simulation_after_file_cleanup(
    client: AsyncClient,
    test_upload_dir: Path,
    test_vector_store: VectorStoreService,
):
    """8. Crash simulation after file cleanup: retry removes final SQLite record."""
    col_resp = await client.post("/api/collections", json={"name": "Del Col 8"})
    assert col_resp.status_code == 201
    col_id = col_resp.json()["id"]

    _, doc_id, _ = await upload_and_index_document(
        client, test_vector_store, col_id, filename="crash_phase3.txt",
        content=b"Content for crash phase 3 test."
    )

    commit_count = 0
    orig_commit = AsyncSession.commit

    async def mock_commit(self, *args, **kwargs):
        nonlocal commit_count
        commit_count += 1
        if commit_count == 2:
            raise RuntimeError("Crash after file unlinked")
        return await orig_commit(self, *args, **kwargs)

    with patch.object(AsyncSession, "commit", mock_commit):
        await client.delete(f"/api/documents/{doc_id}")

    # File and vectors are already gone, but doc row remains as DELETION_PENDING
    col_dir = test_upload_dir / col_id
    assert len(list(col_dir.glob("*"))) == 0
    assert test_vector_store.count(col_id) == 0

    # Retry succeeds and removes SQLite row
    retry_resp = await client.delete(f"/api/documents/{doc_id}")
    assert retry_resp.status_code == 200
    assert (await client.get(f"/api/documents/{doc_id}")).status_code == 404


@pytest.mark.asyncio
async def test_9_retry_completes_pending_deletion(
    client: AsyncClient,
    test_upload_dir: Path,
    test_vector_store: VectorStoreService,
):
    """9. Verify retry completes multi-version pending deletion after transient vector store error."""
    col_resp = await client.post("/api/collections", json={"name": "Del Col 9"})
    assert col_resp.status_code == 201
    col_id = col_resp.json()["id"]

    _, doc_id, _ = await upload_and_index_document(
        client, test_vector_store, col_id, filename="target.txt",
        content=b"Target document content for test 9.", chunk_count=2, version=1
    )
    # Add version 2 vectors for target document
    v2_chunks = [
        make_embedded_chunk(
            chunk_index=i,
            document_id=doc_id,
            collection_id=col_id,
            filename="target.txt",
            version=2,
        )
        for i in range(2)
    ]
    test_vector_store.store_chunks(v2_chunks, collection_id=col_id)

    # Unrelated document
    _, other_doc_id, cnt_other = await upload_and_index_document(
        client, test_vector_store, col_id, filename="unrelated.txt",
        content=b"Unrelated document content for test 9.", chunk_count=3
    )

    # Step 1: Simulate transient failure on first delete attempt
    with patch.object(
        VectorStoreService,
        "delete_document_vectors",
        side_effect=VectorStorageError("Transient ChromaDB lock"),
    ):
        fail_resp = await client.delete(f"/api/documents/{doc_id}")
        assert fail_resp.status_code == 500

    # Document remains in DELETION_PENDING
    get_state = (await client.get(f"/api/documents/{doc_id}")).json()
    assert get_state["status"] == DocumentStatus.DELETION_PENDING.value

    # Step 2: Retry completes cleanly
    retry_resp = await client.delete(f"/api/documents/{doc_id}")
    assert retry_resp.status_code == 200
    assert retry_resp.json()["status"] == "deleted"

    # Target doc is gone; unrelated doc is intact
    assert (await client.get(f"/api/documents/{doc_id}")).status_code == 404
    assert (await client.get(f"/api/documents/{other_doc_id}")).status_code == 200
    assert test_vector_store.count(col_id) == cnt_other


@pytest.mark.asyncio
async def test_10_simultaneous_deletion_requests(
    client: AsyncClient,
    test_upload_dir: Path,
    test_vector_store: VectorStoreService,
):
    """10. Two simultaneous deletion requests complete safely without corrupting data or throwing 500."""
    col_resp = await client.post("/api/collections", json={"name": "Del Col 10"})
    assert col_resp.status_code == 201
    col_id = col_resp.json()["id"]

    _, doc_id, _ = await upload_and_index_document(
        client, test_vector_store, col_id, filename="concurrent.txt",
        content=b"Concurrent deletion test content."
    )

    # Execute concurrent delete calls simultaneously
    res1, res2 = await asyncio.gather(
        client.delete(f"/api/documents/{doc_id}"),
        client.delete(f"/api/documents/{doc_id}"),
    )

    # Both must complete with non-500 status codes (200 or 404)
    assert res1.status_code in (200, 404)
    assert res2.status_code in (200, 404)
    # At least one succeeded
    assert 200 in (res1.status_code, res2.status_code)

    # Complete cleanup verified
    assert (await client.get(f"/api/documents/{doc_id}")).status_code == 404
    assert test_vector_store.count(col_id) == 0
    col_dir = test_upload_dir / col_id
    assert len(list(col_dir.glob("*"))) == 0


@pytest.mark.asyncio
async def test_11_processing_pending_deletion_document_is_rejected(
    client: AsyncClient,
    test_vector_store: VectorStoreService,
):
    """11. Attempting to process a document in DELETION_PENDING is rejected with 422 and preserves state."""
    col_resp = await client.post("/api/collections", json={"name": "Del Col 11"})
    assert col_resp.status_code == 201
    col_id = col_resp.json()["id"]

    _, doc_id, _ = await upload_and_index_document(
        client, test_vector_store, col_id, filename="reject_proc.txt",
        content=b"Content for reject processing test."
    )

    # Set document to DELETION_PENDING via simulated failure
    with patch.object(
        VectorStoreService,
        "delete_document_vectors",
        side_effect=VectorStorageError("Simulated lock"),
    ):
        await client.delete(f"/api/documents/{doc_id}")

    # Confirm it is in DELETION_PENDING
    assert (await client.get(f"/api/documents/{doc_id}")).json()["status"] == DocumentStatus.DELETION_PENDING.value

    # Attempt to process the document
    proc_resp = await client.post(f"/api/documents/{doc_id}/process")
    assert proc_resp.status_code == 422
    assert "pending deletion" in proc_resp.json()["detail"].lower()

    # Verify status was NOT changed back to processing, indexed, or failed
    doc_after = (await client.get(f"/api/documents/{doc_id}")).json()
    assert doc_after["status"] == DocumentStatus.DELETION_PENDING.value


@pytest.mark.asyncio
async def test_12_unrelated_documents_remain_intact(
    client: AsyncClient,
    test_upload_dir: Path,
    test_vector_store: VectorStoreService,
):
    """12. Deleting Doc 1 leaves Doc 2 vectors, file, and SQLite record in the same collection 100% intact."""
    col_resp = await client.post("/api/collections", json={"name": "Del Col 12"})
    assert col_resp.status_code == 201
    col_id = col_resp.json()["id"]

    _, doc1_id, cnt1 = await upload_and_index_document(
        client, test_vector_store,
        col_id, filename="doc1.txt", content=b"Content for document one.", chunk_count=2
    )
    _, doc2_id, cnt2 = await upload_and_index_document(
        client, test_vector_store,
        col_id, filename="doc2.txt", content=b"Content for document two.", chunk_count=3
    )

    assert test_vector_store.count(col_id) == cnt1 + cnt2

    # Delete Doc 1 via REST
    del_resp = await client.delete(f"/api/documents/{doc1_id}")
    assert del_resp.status_code == 200

    # Doc 1 is gone
    assert (await client.get(f"/api/documents/{doc1_id}")).status_code == 404

    # Doc 2 is INTACT in SQLite
    doc2_resp = await client.get(f"/api/documents/{doc2_id}")
    assert doc2_resp.status_code == 200
    assert doc2_resp.json()["filename"] == "doc2.txt"

    # Doc 2 file is INTACT on disk
    col_dir = test_upload_dir / col_id
    assert len(list(col_dir.glob("*"))) == 1

    # Doc 2 vectors in ChromaDB are INTACT
    assert test_vector_store.count(col_id) == cnt2


@pytest.mark.asyncio
async def test_13_other_collections_remain_intact(
    client: AsyncClient,
    test_vector_store: VectorStoreService,
):
    """13. Deleting a document in Collection A does not affect Collection B vectors or documents."""
    col_a = (await client.post("/api/collections", json={"name": "Col A"})).json()["id"]
    col_b = (await client.post("/api/collections", json={"name": "Col B"})).json()["id"]

    _, doc_a, cnt_a = await upload_and_index_document(
        client, test_vector_store,
        col_a, filename="docA.txt", content=b"Alpha content.", chunk_count=2
    )
    _, doc_b, cnt_b = await upload_and_index_document(
        client, test_vector_store,
        col_b, filename="docB.txt", content=b"Beta content.", chunk_count=3
    )

    assert test_vector_store.count(col_a) == cnt_a
    assert test_vector_store.count(col_b) == cnt_b

    # Delete Doc A via REST
    del_resp = await client.delete(f"/api/documents/{doc_a}")
    assert del_resp.status_code == 200

    # Collection A is empty
    assert test_vector_store.count(col_a) == 0

    # Collection B is completely unaffected
    assert test_vector_store.count(col_b) == cnt_b
    assert (await client.get(f"/api/documents/{doc_b}")).status_code == 200


@pytest.mark.asyncio
async def test_14_existing_404_and_api_response_behavior(
    client: AsyncClient,
    test_vector_store: VectorStoreService,
):
    """14. Existing API response structure and 404 behaviour are preserved."""
    col_resp = await client.post("/api/collections", json={"name": "Del Col 14"})
    assert col_resp.status_code == 201
    col_id = col_resp.json()["id"]

    _, doc_id, _ = await upload_and_index_document(
        client, test_vector_store,
        col_id, filename="handbook.txt", chunk_count=1
    )

    # Delete existing document
    del_resp = await client.delete(f"/api/documents/{doc_id}")
    assert del_resp.status_code == 200
    body = del_resp.json()
    assert body["status"] == "deleted"
    assert body["id"] == doc_id
    assert "handbook.txt" in body["message"]

    # Delete non-existent document
    missing_resp = await client.delete("/api/documents/non-existent-uuid-99999")
    assert missing_resp.status_code == 404
    assert "not found" in missing_resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_15_recovery_processes_multiple_pending_documents_and_continues(
    db_session: AsyncSession,
    test_upload_dir: Path,
    test_vector_store: VectorStoreService,
):
    """15. Recovery service processes multiple pending documents and continues when an individual one fails."""
    # Create test collection directly in db_session
    col = Collection(name="Recovery Collection", description="Testing recovery service")
    db_session.add(col)
    await db_session.commit()
    await db_session.refresh(col)

    storage_svc = FileStorageService(upload_dir=test_upload_dir)
    col_dir = test_upload_dir / col.id
    col_dir.mkdir(parents=True, exist_ok=True)

    # Create 3 documents in DELETION_PENDING state
    docs: List[Document] = []
    for i in range(1, 4):
        fn = f"file_{i}.txt"
        stored_fn = f"stored_uuid_{i}_{fn}"
        (col_dir / stored_fn).write_bytes(f"Content for file {i}".encode())
        d = Document(
            collection_id=col.id,
            filename=fn,
            stored_filename=stored_fn,
            file_type="txt",
            sha256=f"dummy_sha256_{i}_" + "0" * 48,
            size_bytes=100,
            status=DocumentStatus.DELETION_PENDING.value,
        )
        db_session.add(d)
        docs.append(d)

    await db_session.commit()
    for d in docs:
        await db_session.refresh(d)

    # Store chunks in vector store for each document
    for d in docs:
        chunks = [
            make_embedded_chunk(chunk_index=0, document_id=d.id, collection_id=col.id, filename=d.filename)
        ]
        test_vector_store.store_chunks(chunks, collection_id=col.id)

    assert test_vector_store.count(col.id) == 3
    assert len(list(col_dir.glob("*"))) == 3

    # Doc 1 will fail during vector cleanup; Doc 2 and Doc 3 will succeed
    doc1_id = docs[0].id
    doc2_id = docs[1].id
    doc3_id = docs[2].id

    orig_del_vectors = test_vector_store.delete_document_vectors

    def selective_delete_vectors(collection_id: str, document_id: str, **kwargs):
        if document_id == doc1_id:
            raise VectorStorageError(f"Simulated storage failure for {document_id}")
        return orig_del_vectors(collection_id, document_id, **kwargs)

    recovery_svc = DocumentRecoveryService(storage_service=storage_svc, vector_store=test_vector_store)

    with patch.object(test_vector_store, "delete_document_vectors", side_effect=selective_delete_vectors):
        report = await recovery_svc.recover_pending_deletions(db_session)

    # Verify report metrics
    assert report.total_found == 3
    assert doc1_id in report.failed
    assert doc2_id in report.succeeded
    assert doc3_id in report.succeeded

    # Doc 1 still exists in SQLite with DELETION_PENDING
    d1_check = (await db_session.execute(select(Document).where(Document.id == doc1_id))).scalar_one_or_none()
    assert d1_check is not None
    await db_session.refresh(d1_check)
    assert d1_check.status == DocumentStatus.DELETION_PENDING.value
    assert "Recovery attempt failed" in (d1_check.error or "")

    # Doc 2 and Doc 3 removed from SQLite
    d2_check = (await db_session.execute(select(Document).where(Document.id == doc2_id))).scalar_one_or_none()
    d3_check = (await db_session.execute(select(Document).where(Document.id == doc3_id))).scalar_one_or_none()
    assert d2_check is None
    assert d3_check is None

    # Files for Doc 2 and Doc 3 removed; Doc 1 file remains
    remaining_files = list(col_dir.glob("*"))
    assert len(remaining_files) == 1
    assert remaining_files[0].name == docs[0].stored_filename

    # Vectors for Doc 2 and Doc 3 removed; Doc 1 vectors remain
    assert test_vector_store.count(col.id) == 1
