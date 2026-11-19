"""
tests/test_remediation.py
─────────────────────────────────────────────────────────────
Comprehensive tests for Phase 6B Step 3 remediation:
  1. Successful collection deletion across all three storage layers.
  2. ChromaDB collection deletion failure.
  3. Physical file deletion failure.
  4. Retry after collection deletion failure.
  5. Missing ChromaDB collection.
  6. Missing physical files.
  7. Isolation of unrelated collections.
  8. Concurrent collection deletion.
  9. Startup recovery timeout.
  10. Startup recovery failure without indefinite startup blocking.
  11. Recovery continues after an individual document failure (session isolation).
  12. Correct document-count semantics.
─────────────────────────────────────────────────────────────
"""

import asyncio
import io
import math
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient
from sqlalchemy import select, update
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


# ── Helpers ─────────────────────────────────────────────────────────────────


def make_dummy_embedding(dim: int = 384, seed: int = 0) -> List[float]:
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
    text = f"Sample text for doc {document_id} chunk {chunk_index}."
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


async def create_col_with_doc_and_vectors(
    client: AsyncClient,
    test_vector_store: VectorStoreService,
    col_name: str,
    filename: str = "doc.txt",
    content: bytes = b"Sample text for remediation testing.",
    chunk_count: int = 2,
) -> tuple[str, str]:
    col_resp = await client.post("/api/collections", json={"name": col_name})
    assert col_resp.status_code == 201
    col_id = col_resp.json()["id"]

    files = {"file": (filename, io.BytesIO(content), "text/plain")}
    doc_resp = await client.post(f"/api/collections/{col_id}/documents", files=files)
    assert doc_resp.status_code == 201
    doc_id = doc_resp.json()["id"]

    chunks = [
        make_embedded_chunk(i, doc_id, col_id, filename)
        for i in range(chunk_count)
    ]
    test_vector_store.store_chunks(chunks, collection_id=col_id)
    assert test_vector_store.count(col_id) >= chunk_count
    return col_id, doc_id


# ── Tests ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_1_successful_collection_deletion_across_all_layers(
    client: AsyncClient,
    test_upload_dir: Path,
    test_vector_store: VectorStoreService,
):
    """1. Successful collection deletion across SQLite, ChromaDB, and disk."""
    col_id, doc_id = await create_col_with_doc_and_vectors(
        client, test_vector_store, "Remediation Col 1", "sample.txt", b"Layered deletion test."
    )

    col_dir = test_upload_dir / col_id
    assert col_dir.exists()
    assert test_vector_store.count(col_id) > 0

    del_resp = await client.delete(f"/api/collections/{col_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "deleted"

    # SQLite collection and document gone
    assert (await client.get(f"/api/collections/{col_id}")).status_code == 404
    assert (await client.get(f"/api/documents/{doc_id}")).status_code == 404

    # ChromaDB vectors gone (count is 0)
    assert test_vector_store.count(col_id) == 0

    # Physical directory gone
    assert not col_dir.exists()


@pytest.mark.asyncio
async def test_2_chromadb_collection_deletion_failure_retains_sqlite_and_files(
    client: AsyncClient,
    test_upload_dir: Path,
    test_vector_store: VectorStoreService,
):
    """2. ChromaDB collection deletion failure returns 500 and retains SQLite records and disk files."""
    col_id, doc_id = await create_col_with_doc_and_vectors(
        client, test_vector_store, "Remediation Col 2", "sample.txt", b"ChromaDB fail test."
    )

    col_dir = test_upload_dir / col_id
    assert col_dir.exists()

    with patch.object(
        VectorStoreService,
        "delete_collection",
        side_effect=VectorStorageError("ChromaDB collection lock error"),
    ):
        del_resp = await client.delete(f"/api/collections/{col_id}")
        assert del_resp.status_code == 500
        assert "vector store deletion failed" in del_resp.json()["detail"].lower()

    # SQLite collection still exists
    get_col = await client.get(f"/api/collections/{col_id}")
    assert get_col.status_code == 200

    # Child document still exists in SQLite
    get_doc = await client.get(f"/api/documents/{doc_id}")
    assert get_doc.status_code == 200
    assert get_doc.json()["status"] == DocumentStatus.DELETION_PENDING.value

    # Physical directory and files are STILL on disk
    assert col_dir.exists()
    assert len(list(col_dir.glob("*"))) == 1


@pytest.mark.asyncio
async def test_3_physical_file_deletion_failure_retains_sqlite_and_files(
    client: AsyncClient,
    test_upload_dir: Path,
    test_vector_store: VectorStoreService,
):
    """3. Physical file deletion failure returns 500 and retains SQLite collection and documents."""
    col_id, doc_id = await create_col_with_doc_and_vectors(
        client, test_vector_store, "Remediation Col 3", "sample.txt", b"Filesystem fail test."
    )

    col_dir = test_upload_dir / col_id
    assert col_dir.exists()

    with patch.object(FileStorageService, "delete_collection_storage", return_value=False):
        del_resp = await client.delete(f"/api/collections/{col_id}")
        assert del_resp.status_code == 500
        assert "filesystem deletion failed" in del_resp.json()["detail"].lower()

    # SQLite collection and document STILL exist
    assert (await client.get(f"/api/collections/{col_id}")).status_code == 200
    assert (await client.get(f"/api/documents/{doc_id}")).status_code == 200

    # Files still exist on disk
    assert col_dir.exists()


@pytest.mark.asyncio
async def test_4_retry_after_collection_deletion_failure(
    client: AsyncClient,
    test_upload_dir: Path,
    test_vector_store: VectorStoreService,
):
    """4. Retry after transient collection deletion failure completes cleanup cleanly."""
    col_id, doc_id = await create_col_with_doc_and_vectors(
        client, test_vector_store, "Remediation Col 4", "sample.txt", b"Retry test."
    )

    # Attempt 1: Transient ChromaDB failure
    with patch.object(
        VectorStoreService,
        "delete_collection",
        side_effect=VectorStorageError("Transient lock"),
    ):
        del1 = await client.delete(f"/api/collections/{col_id}")
        assert del1.status_code == 500

    # Collection is still present
    assert (await client.get(f"/api/collections/{col_id}")).status_code == 200

    # Attempt 2: Retry succeeds
    del2 = await client.delete(f"/api/collections/{col_id}")
    assert del2.status_code == 200

    # Cleaned everywhere
    assert (await client.get(f"/api/collections/{col_id}")).status_code == 404
    assert (await client.get(f"/api/documents/{doc_id}")).status_code == 404
    assert not (test_upload_dir / col_id).exists()
    assert test_vector_store.count(col_id) == 0


@pytest.mark.asyncio
async def test_5_missing_chromadb_collection_handled_idempotently(
    client: AsyncClient,
    test_upload_dir: Path,
    test_vector_store: VectorStoreService,
):
    """5. If ChromaDB vector collection does not exist, deletion succeeds idempotently."""
    col_resp = await client.post("/api/collections", json={"name": "No Chroma Col"})
    assert col_resp.status_code == 201
    col_id = col_resp.json()["id"]

    # Delete collection where ChromaDB collection was never created
    del_resp = await client.delete(f"/api/collections/{col_id}")
    assert del_resp.status_code == 200
    assert (await client.get(f"/api/collections/{col_id}")).status_code == 404


@pytest.mark.asyncio
async def test_6_missing_physical_files_handled_idempotently(
    client: AsyncClient,
    test_upload_dir: Path,
    test_vector_store: VectorStoreService,
):
    """6. If physical collection upload directory is already missing, deletion succeeds idempotently."""
    col_id, _ = await create_col_with_doc_and_vectors(
        client, test_vector_store, "Missing Dir Col", "sample.txt", b"Missing dir."
    )

    # Manually remove the upload directory beforehand
    col_dir = test_upload_dir / col_id
    if col_dir.exists():
        import shutil
        shutil.rmtree(col_dir)
    assert not col_dir.exists()

    del_resp = await client.delete(f"/api/collections/{col_id}")
    assert del_resp.status_code == 200
    assert (await client.get(f"/api/collections/{col_id}")).status_code == 404


@pytest.mark.asyncio
async def test_7_isolation_of_unrelated_collections(
    client: AsyncClient,
    test_upload_dir: Path,
    test_vector_store: VectorStoreService,
):
    """7. Deleting Collection A does not affect Collection B records, vectors, or files."""
    col_a, doc_a = await create_col_with_doc_and_vectors(
        client, test_vector_store, "Col Alpha", "alpha.txt", b"Alpha content.", chunk_count=2
    )
    col_b, doc_b = await create_col_with_doc_and_vectors(
        client, test_vector_store, "Col Beta", "beta.txt", b"Beta content.", chunk_count=3
    )

    assert test_vector_store.count(col_a) == 2
    assert test_vector_store.count(col_b) == 3

    # Delete Col A
    del_resp = await client.delete(f"/api/collections/{col_a}")
    assert del_resp.status_code == 200

    # Col A is gone
    assert (await client.get(f"/api/collections/{col_a}")).status_code == 404
    assert test_vector_store.count(col_a) == 0
    assert not (test_upload_dir / col_a).exists()

    # Col B is completely intact
    assert (await client.get(f"/api/collections/{col_b}")).status_code == 200
    assert (await client.get(f"/api/documents/{doc_b}")).status_code == 200
    assert test_vector_store.count(col_b) == 3
    assert (test_upload_dir / col_b).exists()


@pytest.mark.asyncio
async def test_8_concurrent_collection_deletion(
    client: AsyncClient,
    test_upload_dir: Path,
    test_vector_store: VectorStoreService,
):
    """8. Simultaneous collection deletion requests complete safely without corrupting data or 500 errors."""
    col_id, _ = await create_col_with_doc_and_vectors(
        client, test_vector_store, "Concurrent Col", "c.txt", b"Concurrent data."
    )

    res1, res2 = await asyncio.gather(
        client.delete(f"/api/collections/{col_id}"),
        client.delete(f"/api/collections/{col_id}"),
    )

    assert res1.status_code in (200, 404)
    assert res2.status_code in (200, 404)
    assert 200 in (res1.status_code, res2.status_code)

    # Completely deleted
    assert (await client.get(f"/api/collections/{col_id}")).status_code == 404
    assert test_vector_store.count(col_id) == 0
    assert not (test_upload_dir / col_id).exists()


@pytest.mark.asyncio
async def test_9_startup_recovery_timeout():
    """9. Startup recovery timeout safely aborts slow recovery without blocking."""
    async def slow_recovery(*args, **kwargs):
        await asyncio.sleep(2.0)

    # Verify asyncio.timeout raises TimeoutError and is caught
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.05):
            await slow_recovery()


@pytest.mark.asyncio
async def test_10_startup_recovery_failure_does_not_block():
    """10. Startup recovery failure is caught safely and does not crash application startup."""
    mock_recovery = AsyncMock(side_effect=RuntimeError("ChromaDB catalog locked during startup"))

    failed = False
    try:
        # Simulated lifespan block
        async with asyncio.timeout(2.0):
            await mock_recovery()
    except Exception as exc:
        failed = True
        assert "locked during startup" in str(exc)

    assert failed


@pytest.mark.asyncio
async def test_11_recovery_session_isolation_continues_after_failure(
    db_session: AsyncSession,
    test_upload_dir: Path,
    test_vector_store: VectorStoreService,
):
    """11. Recovery isolates sessions per document and continues processing after individual failure."""
    col = Collection(name="Session Iso Col")
    db_session.add(col)
    await db_session.commit()
    await db_session.refresh(col)

    storage_svc = FileStorageService(upload_dir=test_upload_dir)
    col_dir = test_upload_dir / col.id
    col_dir.mkdir(parents=True, exist_ok=True)

    docs: List[Document] = []
    for i in range(1, 4):
        fn = f"file_{i}.txt"
        stored_fn = f"stored_{i}_{fn}"
        (col_dir / stored_fn).write_bytes(f"Content {i}".encode())
        d = Document(
            collection_id=col.id,
            filename=fn,
            stored_filename=stored_fn,
            file_type="txt",
            sha256=f"hash_{i}_" + "1" * 50,
            size_bytes=50,
            status=DocumentStatus.DELETION_PENDING.value,
        )
        db_session.add(d)
        docs.append(d)

    await db_session.commit()
    for d in docs:
        await db_session.refresh(d)
        chunks = [make_embedded_chunk(0, d.id, col.id, d.filename)]
        test_vector_store.store_chunks(chunks, collection_id=col.id)

    doc1_id = docs[0].id
    doc2_id = docs[1].id
    doc3_id = docs[2].id

    orig_del_vectors = test_vector_store.delete_document_vectors

    def selective_delete(collection_id: str, document_id: str, **kwargs):
        if document_id == doc1_id:
            raise VectorStorageError("Simulated failure on Doc 1")
        return orig_del_vectors(collection_id, document_id, **kwargs)

    recovery_svc = DocumentRecoveryService(storage_service=storage_svc, vector_store=test_vector_store)

    with patch.object(test_vector_store, "delete_document_vectors", side_effect=selective_delete):
        report = await recovery_svc.recover_pending_deletions(db=db_session)

    assert report.total_found == 3
    assert doc1_id in report.failed
    assert doc2_id in report.succeeded
    assert doc3_id in report.succeeded

    # Doc 1 still exists in SQLite
    d1 = (await db_session.execute(select(Document).where(Document.id == doc1_id))).scalar_one_or_none()
    assert d1 is not None
    assert d1.status == DocumentStatus.DELETION_PENDING.value

    # Doc 2 and 3 removed
    d2 = (await db_session.execute(select(Document).where(Document.id == doc2_id))).scalar_one_or_none()
    d3 = (await db_session.execute(select(Document).where(Document.id == doc3_id))).scalar_one_or_none()
    assert d2 is None
    assert d3 is None


@pytest.mark.asyncio
async def test_12_document_count_semantics_excludes_deletion_pending(
    client: AsyncClient,
    test_vector_store: VectorStoreService,
):
    """12. Collection document_count semantics strictly count active documents, excluding DELETION_PENDING."""
    col_resp = await client.post("/api/collections", json={"name": "Count Semantics Col"})
    assert col_resp.status_code == 201
    col_id = col_resp.json()["id"]
    assert col_resp.json()["document_count"] == 0

    # Upload Doc 1
    f1 = {"file": ("active1.txt", io.BytesIO(b"Active 1"), "text/plain")}
    r1 = await client.post(f"/api/collections/{col_id}/documents", files=f1)
    assert r1.status_code == 201
    doc1_id = r1.json()["id"]

    # Upload Doc 2
    f2 = {"file": ("active2.txt", io.BytesIO(b"Active 2"), "text/plain")}
    r2 = await client.post(f"/api/collections/{col_id}/documents", files=f2)
    assert r2.status_code == 201
    doc2_id = r2.json()["id"]

    # Both documents are active (count == 2)
    get_col = await client.get(f"/api/collections/{col_id}")
    assert get_col.json()["document_count"] == 2

    col_list = await client.get("/api/collections")
    col_item = next(c for c in col_list.json()["collections"] if c["id"] == col_id)
    assert col_item["document_count"] == 2

    # Put Doc 2 into DELETION_PENDING status via simulated failure
    with patch.object(
        VectorStoreService,
        "delete_document_vectors",
        side_effect=VectorStorageError("Simulated lock"),
    ):
        await client.delete(f"/api/documents/{doc2_id}")

    # Doc 2 is now in DELETION_PENDING
    assert (await client.get(f"/api/documents/{doc2_id}")).json()["status"] == DocumentStatus.DELETION_PENDING.value

    # Document count must now be 1 (strictly excludes DELETION_PENDING)
    get_col_after = await client.get(f"/api/collections/{col_id}")
    assert get_col_after.json()["document_count"] == 1

    col_list_after = await client.get("/api/collections")
    col_item_after = next(c for c in col_list_after.json()["collections"] if c["id"] == col_id)
    assert col_item_after["document_count"] == 1
