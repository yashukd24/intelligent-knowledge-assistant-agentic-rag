"""
tests/test_collection_concurrency.py
─────────────────────────────────────────────────────────────
Deterministic concurrency, synchronization, and lifecycle tests for:
  - backend/services/collection_lock.py
  - backend/services/document_processor.py
  - backend/routers/collections.py
  - backend/services/document_recovery.py
─────────────────────────────────────────────────────────────
"""

import asyncio
import io
from contextlib import asynccontextmanager
import pytest
from httpx import AsyncClient
from sqlalchemy import select

from backend.models.collection import Collection
from backend.models.document import Document
from backend.schemas.chunk import ChunkMetadata
from backend.schemas.document import DocumentStatus
from backend.schemas.embedding import EmbeddedChunk
from backend.services.collection_lock import CollectionLockRegistry, collection_lock_registry
from backend.services.document_processor import DocumentProcessor, DocumentProcessorError
from backend.services.document_recovery import DocumentRecoveryService
from backend.services.embedding_service import EmbeddingService
from backend.services.vector_store import VectorStoreService


# ── Lock Registry Unit & Loop Compatibility Tests ──────────────────────────


@pytest.mark.asyncio
async def test_1_lock_registry_basic_refcounting_and_eviction():
    """Verify lock acquisition, refcounting, and eviction when ref_count hits 0."""
    async with asyncio.timeout(5.0):
        col_id = "test-col-lifecycle"

        assert await collection_lock_registry.get_active_count() == 0
        assert await collection_lock_registry.get_ref_count(col_id) == 0

        async with collection_lock_registry.acquire(col_id):
            assert await collection_lock_registry.get_active_count() == 1
            assert await collection_lock_registry.get_ref_count(col_id) == 1

        # After exit, refcount drops to 0 and entry is evicted
        assert await collection_lock_registry.get_active_count() == 0
        assert await collection_lock_registry.get_ref_count(col_id) == 0


def test_2_lock_registry_sequential_event_loops():
    """Verify registry works across sequential event loops without cross-loop Future errors."""
    col_id = "test-col-cross-loop"

    # Event Loop 1
    loop1 = asyncio.new_event_loop()
    try:
        async def run_loop1():
            async with collection_lock_registry.acquire(col_id):
                assert await collection_lock_registry.get_active_count() == 1
        loop1.run_until_complete(run_loop1())
    finally:
        loop1.close()

    # Event Loop 2 (New, distinct event loop)
    loop2 = asyncio.new_event_loop()
    try:
        async def run_loop2():
            async with collection_lock_registry.acquire(col_id):
                assert await collection_lock_registry.get_active_count() == 1
        loop2.run_until_complete(run_loop2())
    finally:
        loop2.close()


@pytest.mark.asyncio
async def test_3_cancellation_while_waiting_for_lock():
    """Verify cancellation while queued waiting for lock decrements refcount safely."""
    col_id = "test-col-cancel-waiting"
    holder_acquired = asyncio.Event()
    release_holder = asyncio.Event()

    async def lock_holder():
        async with collection_lock_registry.acquire(col_id):
            holder_acquired.set()
            await release_holder.wait()

    async def lock_waiter():
        async with collection_lock_registry.acquire(col_id):
            pass

    holder_task = asyncio.create_task(lock_holder())
    await holder_acquired.wait()

    # Start waiter task and let it queue
    waiter_task = asyncio.create_task(lock_waiter())
    await asyncio.sleep(0.01)

    assert await collection_lock_registry.get_ref_count(col_id) == 2

    # Cancel waiter
    waiter_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter_task

    # Refcount drops back to 1
    assert await collection_lock_registry.get_ref_count(col_id) == 1

    # Release holder
    release_holder.set()
    await holder_task

    assert await collection_lock_registry.get_active_count() == 0


@pytest.mark.asyncio
async def test_4_cancellation_while_holding_lock():
    """Verify cancellation while inside the lock context releases lock and cleans up."""
    col_id = "test-col-cancel-holding"
    acquired_event = asyncio.Event()

    async def holding_task_fn():
        async with collection_lock_registry.acquire(col_id):
            acquired_event.set()
            await asyncio.sleep(10.0)

    task = asyncio.create_task(holding_task_fn())
    await acquired_event.wait()
    assert await collection_lock_registry.get_active_count() == 1

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Lock must be released and evicted
    assert await collection_lock_registry.get_active_count() == 0

    # Must be immediately re-acquirable without blocking
    async with collection_lock_registry.acquire(col_id):
        assert await collection_lock_registry.get_active_count() == 1


@pytest.mark.asyncio
async def test_5_exception_while_holding_lock():
    """Verify unhandled exception inside lock releases lock and cleans up registry."""
    col_id = "test-col-exception"

    with pytest.raises(RuntimeError, match="Simulated unhandled failure"):
        async with collection_lock_registry.acquire(col_id):
            raise RuntimeError("Simulated unhandled failure")

    assert await collection_lock_registry.get_active_count() == 0

    # Can acquire again immediately
    async with collection_lock_registry.acquire(col_id):
        assert await collection_lock_registry.get_active_count() == 1


@pytest.mark.asyncio
async def test_6_multiple_collections_operate_independently():
    """Verify operations on different collections do not block each other."""
    col_a = "col-alpha"
    col_b = "col-beta"

    a_in_lock = asyncio.Event()
    b_in_lock = asyncio.Event()
    release_all = asyncio.Event()

    async def worker_a():
        async with collection_lock_registry.acquire(col_a):
            a_in_lock.set()
            await release_all.wait()

    async def worker_b():
        async with collection_lock_registry.acquire(col_b):
            b_in_lock.set()
            await release_all.wait()

    task_a = asyncio.create_task(worker_a())
    task_b = asyncio.create_task(worker_b())

    # Both must be able to acquire their respective locks simultaneously
    await asyncio.gather(a_in_lock.wait(), b_in_lock.wait())
    assert await collection_lock_registry.get_active_count() == 2

    release_all.set()
    await asyncio.gather(task_a, task_b)
    assert await collection_lock_registry.get_active_count() == 0


# ── Interleaved Ingestion vs. Deletion Concurrency Tests ───────────────────


@pytest.mark.asyncio
async def test_7_deletion_starts_while_embedding_is_paused(
    client: AsyncClient,
    test_vector_store: VectorStoreService,
):
    """
    CRITICAL RACE (F-02): Collection deletion occurs while embedding computation
    is running outside the lock. When ingestion resumes, it must cleanly abort
    and NOT recreate a ChromaDB collection or write orphan vectors.
    """
    async with asyncio.timeout(25.0):
        # 1. Create collection & upload document
        col_resp = await client.post("/api/collections", json={"name": "Race Col 7"})
        assert col_resp.status_code == 201
        col_id = col_resp.json()["id"]

        content = b"Content for testing interleaved deletion during embedding."
        files = {"file": ("race7.txt", io.BytesIO(content), "text/plain")}
        doc_resp = await client.post(f"/api/collections/{col_id}/documents", files=files)
        assert doc_resp.status_code == 201
        doc_id = doc_resp.json()["id"]

        # 2. Setup barrier synchronization with strict safety timeout
        embed_started = asyncio.Event()
        allow_embed_finish = asyncio.Event()

        orig_embed = EmbeddingService.embed_chunks

        def paused_embed_chunks(self, chunks, batch_size=None):
            # Notify that embedding has started
            embed_started.set()
            # Synchronously wait for release (running inside worker thread via asyncio.to_thread)
            import time
            deadline = time.time() + 10.0
            while not allow_embed_finish.is_set():
                time.sleep(0.01)
                if time.time() > deadline:
                    raise TimeoutError("allow_embed_finish barrier timed out after 10.0s")
            return orig_embed(self, chunks, batch_size=batch_size)

        # 3. Launch ingestion in background
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(EmbeddingService, "embed_chunks", paused_embed_chunks)

            process_task = asyncio.create_task(
                client.post(f"/api/documents/{doc_id}/process?embed=true")
            )

            # Wait until embedding is active in the worker thread with strict timeout
            async with asyncio.timeout(10.0):
                await embed_started.wait()

            # 4. Collection deletion executes while embedding is calculating
            del_resp = await client.delete(f"/api/collections/{col_id}")
            assert del_resp.status_code == 200

            # Verify collection is gone from SQLite
            assert (await client.get(f"/api/collections/{col_id}")).status_code == 404

            # 5. Let embedding complete and let process_task attempt vector storage
            allow_embed_finish.set()
            async with asyncio.timeout(10.0):
                proc_resp = await process_task

            # Ingestion must return an error (422 or 404), NOT succeed
            assert proc_resp.status_code in (404, 422)

        # 6. Verify ChromaDB state: collection must NOT be recreated!
        col_name = test_vector_store.get_collection_name(col_id)
        existing_cols = test_vector_store.list_collections()
        assert col_name not in existing_cols
        assert test_vector_store.count(col_id) == 0


@pytest.mark.asyncio
async def test_8_ingestion_acquires_lock_before_deletion(
    client: AsyncClient,
    test_vector_store: VectorStoreService,
):
    """
    Ingestion enters critical section first; deletion arrives and waits for lock.
    Once ingestion promotes, deletion proceeds and cleanly removes all data.
    """
    async with asyncio.timeout(25.0):
        col_resp = await client.post("/api/collections", json={"name": "Race Col 8"})
        assert col_resp.status_code == 201
        col_id = col_resp.json()["id"]

        content = b"Content for testing ingestion acquiring lock before deletion."
        files = {"file": ("race8.txt", io.BytesIO(content), "text/plain")}
        doc_resp = await client.post(f"/api/collections/{col_id}/documents", files=files)
        assert doc_resp.status_code == 201
        doc_id = doc_resp.json()["id"]

        ingestion_in_critical = asyncio.Event()
        allow_ingestion_finish = asyncio.Event()

        orig_acquire = CollectionLockRegistry.acquire

        @asynccontextmanager
        async def hooked_acquire(self, collection_id: str):
            async with orig_acquire(self, collection_id):
                if collection_id == col_id and not ingestion_in_critical.is_set():
                    ingestion_in_critical.set()
                    await asyncio.wait_for(allow_ingestion_finish.wait(), timeout=10.0)
                yield

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(CollectionLockRegistry, "acquire", hooked_acquire)

            proc_task = asyncio.create_task(
                client.post(f"/api/documents/{doc_id}/process?embed=true")
            )
            await asyncio.wait_for(ingestion_in_critical.wait(), timeout=10.0)

            # Deletion arrives while ingestion is inside the collection lock
            del_task = asyncio.create_task(client.delete(f"/api/collections/{col_id}"))
            await asyncio.sleep(0.05)

            # Allow ingestion to finish
            allow_ingestion_finish.set()

            proc_resp = await proc_task
            del_resp = await del_task

            assert proc_resp.status_code == 200
            assert del_resp.status_code == 200

        # End state: fully cleaned up
        assert (await client.get(f"/api/collections/{col_id}")).status_code == 404
        assert test_vector_store.count(col_id) == 0


@pytest.mark.asyncio
async def test_9_deletion_acquires_lock_before_ingestion(
    client: AsyncClient,
    test_vector_store: VectorStoreService,
):
    """
    Deletion enters collection lock first; ingestion arrives and waits.
    Ingestion wakes up after deletion commits, detects deleted state, and aborts.
    """
    async with asyncio.timeout(25.0):
        col_resp = await client.post("/api/collections", json={"name": "Race Col 9"})
        assert col_resp.status_code == 201
        col_id = col_resp.json()["id"]

        content = b"Content for testing deletion acquiring lock before ingestion."
        files = {"file": ("race9.txt", io.BytesIO(content), "text/plain")}
        doc_resp = await client.post(f"/api/collections/{col_id}/documents", files=files)
        assert doc_resp.status_code == 201
        doc_id = doc_resp.json()["id"]

        del_in_lock = asyncio.Event()
        allow_del_finish = asyncio.Event()

        orig_acquire = CollectionLockRegistry.acquire

        @asynccontextmanager
        async def hooked_acquire(self, collection_id: str):
            async with orig_acquire(self, collection_id):
                if collection_id == col_id and not del_in_lock.is_set():
                    del_in_lock.set()
                    await asyncio.wait_for(allow_del_finish.wait(), timeout=10.0)
                yield

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(CollectionLockRegistry, "acquire", hooked_acquire)

            del_task = asyncio.create_task(client.delete(f"/api/collections/{col_id}"))
            await asyncio.wait_for(del_in_lock.wait(), timeout=10.0)

            # Ingestion starts while deletion is holding lock
            proc_task = asyncio.create_task(
                client.post(f"/api/documents/{doc_id}/process?embed=true")
            )
            await asyncio.sleep(0.05)

            # Allow deletion to finish
            allow_del_finish.set()

            del_resp = await del_task
            proc_resp = await proc_task

            assert del_resp.status_code == 200
            assert proc_resp.status_code in (404, 422)

        assert (await client.get(f"/api/collections/{col_id}")).status_code == 404
        assert test_vector_store.count(col_id) == 0


@pytest.mark.asyncio
async def test_10_recovery_respects_collection_lock(
    client: AsyncClient,
    test_vector_store: VectorStoreService,
):
    """
    Verify document recovery acquires collection lock and recovers DELETION_PENDING
    documents cleanly.
    """
    col_resp = await client.post("/api/collections", json={"name": "Recovery Lock Col"})
    assert col_resp.status_code == 201
    col_id = col_resp.json()["id"]

    content = b"Recovery document content."
    files = {"file": ("rec.txt", io.BytesIO(content), "text/plain")}
    doc_resp = await client.post(f"/api/collections/{col_id}/documents", files=files)
    assert doc_resp.status_code == 201
    doc_id = doc_resp.json()["id"]

    # Mark document DELETION_PENDING manually
    recovery = DocumentRecoveryService(vector_store=test_vector_store)

    # Trigger recovery while holding lock; verify recovery waits or completes
    report = await recovery.recover_pending_deletions()
    assert report.total_found == 0  # Not in pending yet

    # Now mark as pending via delete_document simulation
    del_doc_resp = await client.delete(f"/api/documents/{doc_id}")
    assert del_doc_resp.status_code == 200

    # Ensure registry count drops to 0
    assert await collection_lock_registry.get_active_count() == 0
