# Independent Technical Audit: Phase 6B Step 3 Durable Document Deletion

**Project:** Intelligent Knowledge Assistant Using Agentic RAG and Large Language Models  
**Audit Target:** Recoverable Document Deletion, Storage Consistency, Crash Safety, and Lifecycle Guarantees  
**Auditor Role:** Senior Python Backend & Database Reliability Engineer (Independent Read-Only Audit)  
**Date:** November 2026  
**Operating Environment:** Windows 11, Python 3.13.3 (`.venv`), `chromadb==1.5.9`, `pydantic==2.14.3`, `torch==2.15.0+cpu`  

---

## 1. Executive Verdict

### **Verdict: PASS WITH CONDITIONS**

The document-level durable deletion workflow implemented in Phase 6B Step 3 successfully resolves the three specific defects identified in previous audits:
1. **Ghost Indexed Records:** Eliminated by persisting `DocumentStatus.DELETION_PENDING` in SQLite before external destructive operations begin.
2. **Orphaned Physical Files:** Eliminated by retaining the SQLite record (with its `stored_filename` and `collection_id`) until physical unlinking is verified.
3. **False Success Reporting:** Eliminated by strictly verifying the Boolean return value of [`FileStorageService.delete_stored_file()`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/services/file_storage.py#L221-L242), preventing false HTTP 200 responses when unlinking fails.

### **Conditions for Phase 7 Readiness:**
- **Condition 1 (High Priority):** Collection deletion (`DELETE /api/collections/{collection_id}`) does **not** invoke `VectorStoreService.delete_collection()`, resulting in completely orphaned ChromaDB vectors when a collection is deleted. This must be corrected before Phase 7 indexing begins.
- **Condition 2 (Medium Priority):** The application startup recovery sweep in [`backend/main.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/main.py#L83-L99) runs within the ASGI lifespan startup before `yield` without an explicit timeout (`asyncio.wait_for`), meaning slow or hung disk/vector operations could delay API binding.
- **Condition 3 (Informational / Architectural):** Cross-store operations (SQLite + ChromaDB + Local Filesystem) are **not transactionally atomic**. The system uses a sequential best-effort two-phase state machine with idempotency and reconciliation hooks.

---

## 2. Verified Source-Code Behavior with Line References

### A. Lifecycle State Definition & Schema Compatibility
- **Pydantic Enum:** In [`backend/schemas/document.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/schemas/document.py#L15-L21), `DELETION_PENDING = "deletion_pending"` is formally declared within `DocumentStatus(str, Enum)`.
- **Response Validation:** [`DocumentResponse`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/schemas/document.py#L23-L38) serializes `status` directly via `from_attributes=True`.
- **ORM Model:** In [`backend/models/document.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/models/document.py#L66-L71), `status` is declared as `Mapped[str] = mapped_column(String(20), default="pending", nullable=False, index=True)`. The literal `"deletion_pending"` is 16 characters, fitting within `String(20)`. No SQLite schema migration or table alteration is necessary.

### B. Ingestion and Processing Safeguards
- In [`backend/services/document_processor.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/services/document_processor.py#L134-L139):
  ```python
  # 2. Reject documents marked for deletion
  if doc.status == DocumentStatus.DELETION_PENDING.value:
      raise DocumentProcessorError(
          f"Cannot process document '{document_id}': document is pending deletion."
      )
  ```
- **Verified Behavior:** Any invocation of `POST /api/documents/{document_id}/process` on a pending document raises `DocumentProcessorError` prior to modifying `doc.status`.
- **Router Translation:** [`backend/routers/documents.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/routers/documents.py#L322-L327) translates this to `HTTPException(422, detail="Cannot process document '...': document is pending deletion.")`.
- **Immutability Guarantee:** The document status is never overwritten to `processing`, `indexed`, or `failed`.

### C. Recoverable REST Deletion Pipeline
- In [`backend/routers/documents.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/routers/documents.py#L333-L440):
  - **Lines 343–350:** Document existence verified in SQLite (returns HTTP 404 if missing).
  - **Lines 357–377 (Phase 1):** If `doc.status != DELETION_PENDING`, sets `doc.status = DELETION_PENDING`, clears `doc.error`, and commits to SQLite. If this commit fails, rolls back and returns HTTP 500 without touching ChromaDB or files.
  - **Lines 379–399 (Phase 2):** Calls `vector_store.delete_document_vectors(collection_id, document_id)`. If an exception occurs, records error on `doc.error`, commits, and raises HTTP 500.
  - **Lines 401–418 (Phase 3):** Calls `storage.delete_stored_file(collection_id, stored_filename)`. Checks Boolean return value. If `False`, records error on `doc.error`, commits, and raises HTTP 500.
  - **Lines 420–438 (Phase 4):** Calls `await db.delete(doc)` and `await db.commit()`. Catches `StaleDataError` (handling concurrent deletion) and general `Exception` (rolling back and raising HTTP 500).

### D. Idempotent Storage Unlinking
- In [`backend/services/file_storage.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/services/file_storage.py#L221-L242):
  ```python
  def delete_stored_file(self, collection_id: str, stored_filename: str) -> bool:
      target_path = self.upload_dir / collection_id / stored_filename
      try:
          resolved = self._verify_boundary(target_path)
          if resolved.is_file():
              resolved.unlink()
              return True
          if not resolved.exists():
              return True  # Already cleaned
          return False
      except DirectoryTraversalError:
          return False
      except OSError:
          return False
  ```
- **Verified Behavior:** If the file was already deleted in a prior attempt or during a crash, `not resolved.exists()` returns `True`, preventing false failures on retry. If an actual `OSError` (e.g. Windows file lock) occurs, returns `False`.

---

## 3. Crash-Boundary Failure Matrix

The following matrix analyzes the exact system behavior at each possible crash or failure boundary during document deletion:

| Boundary | State When Failure Occurs | SQLite Row | ChromaDB Vectors | Physical File | Post-Failure / Recovery Behavior |
|---|---|---|---|---|---|
| **Boundary 1:** Pre-Commit Crash | During initial `select` or before `db.commit()` of `DELETION_PENDING` | Intact (`indexed`/`pending`) | Intact | Intact | SQLite WAL aborts uncommitted transaction. Document remains fully operational. No data lost. |
| **Boundary 2:** Phase 1 Commit Fails | SQLite disk full or locked during `DELETION_PENDING` commit | Intact (`indexed`/`pending`) | Intact | Intact | Router calls `db.rollback()` (line 369) and returns HTTP 500. Storage engines untouched. |
| **Boundary 3:** Crash Post-Phase 1 | Process crashes after `DELETION_PENDING` commit, before vector deletion | `DELETION_PENDING` | Intact | Intact | Document is locked against processing (HTTP 422). Retry `DELETE` resumes from vector cleanup. Startup sweep completes cleanup. |
| **Boundary 4:** Phase 2 Vector Error | ChromaDB raises `VectorStorageError` or partial delete occurs | `DELETION_PENDING` (error logged) | Intact or Partial | Intact | Document retained in SQLite with `doc.error`. File is untouched. Subsequent retry safely deletes remaining vectors via ID filtering. |
| **Boundary 5:** Crash Post-Phase 2 | Process crashes after vectors deleted, before file unlinking | `DELETION_PENDING` | **Deleted** | Intact | Document retained in SQLite with `stored_filename`. Retry finds 0 vectors (idempotent), deletes physical file, and purges SQLite row. |
| **Boundary 6:** Phase 3 File Error | `delete_stored_file` returns `False` (file lock, permission denial) | `DELETION_PENDING` (error logged) | **Deleted** | Intact | Endpoint returns HTTP 500. Document is **not deleted** from SQLite. File reference preserved for retry. |
| **Boundary 7:** Crash Post-Phase 3 | Process crashes after file unlinked, before SQLite record delete | `DELETION_PENDING` | **Deleted** | **Deleted** | Retry or startup sweep finds 0 vectors, treats missing file as cleaned (`True`), and purges SQLite row. |
| **Boundary 8:** Phase 4 Commit Fails | Final `await db.commit()` fails on SQLite delete | `DELETION_PENDING` | **Deleted** | **Deleted** | Session rolls back. Document remains in SQLite as `DELETION_PENDING`. Retry cleanly deletes row. |

---

## 4. Concurrency Analysis

### A. Simultaneous API Deletion Requests
When two concurrent requests (`Req A` and `Req B`) call `DELETE /api/documents/{document_id}`:
1. **Initial Select:** Both query SQLite at line 343; both find `doc`.
2. **Phase 1 Commit:** Whichever transaction commits first marks `DELETION_PENDING`. Under SQLite serializable writes / WAL, the second transaction waits or succeeds. If the second request observes `DELETION_PENDING`, line 360 skips re-marking.
3. **Phase 2 (ChromaDB):** Both call `delete_document_vectors`. ChromaDB ID-based deletion (`col.delete(ids=...)`) is idempotent. If one request deletes all IDs, the competing request finds 0 IDs and returns `0` cleanly.
4. **Phase 3 (Filesystem):** Both call `delete_stored_file`. Whichever runs first unlinks the file. The competing request observes `not resolved.exists()` and returns `True`.
5. **Phase 4 (SQLite Purge):**
   - Transaction A commits `db.delete(doc)`.
   - Transaction B attempts `db.delete(doc)` and `db.commit()`. SQLAlchemy detects that 0 rows were affected and raises `sqlalchemy.orm.exc.StaleDataError`.
   - Lines 424–429 catch `StaleDataError`, call `await db.rollback()`, and log that the document was already deleted.
6. **Result:** Both requests exit cleanly without returning unhandled HTTP 500 errors.

### B. Recovery Worker Concurrency Limitations
- **Single Process / Event Loop:** Fully safe.
- **Multiple OS Processes (Multi-Worker Deployments):**
  > [!WARNING]
  > **Multi-Process Limitation:** ChromaDB embedded `PersistentClient` is not designed for multi-process concurrent writes across independent OS processes. If multiple Uvicorn worker processes run `DocumentRecoveryService` simultaneously, ChromaDB's internal SQLite catalog can raise `sqlite3.OperationalError: database is locked`.
  - In [`backend/services/document_recovery.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC AND LLM BOT/backend/services/document_recovery.py), there is no advisory row locking (`locked_by_worker` timestamp or column).

---

## 5. Startup and Manual Recovery Analysis

### A. FastAPI Lifespan Startup Sweep
In [`backend/main.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC AND LLM BOT/backend/main.py#L83-L99):
```python
async with AsyncSessionLocal() as db_session:
    recovery_svc = DocumentRecoveryService()
    rec_report = await recovery_svc.recover_pending_deletions(db_session, limit=100)
```
- **Strengths:**
  - Bounded to `limit=100` to prevent unbounded memory loading.
  - Wrapped in `try/except Exception as rec_err` so that unexpected recovery errors will **never prevent the server from starting**.
- **Vulnerability:**
  - The recovery sweep executes before `yield` in the ASGI lifespan.
  - If a large number of pending deletions exist and disk I/O is slow (or a ChromaDB index lock is held), this sweep **blocks server startup and port binding** until completed.
  - There is no `asyncio.wait_for(..., timeout=5.0)` wrapper.

### B. Manual Recovery CLI Utility
In [`scripts/recover_deletions.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC AND LLM BOT/scripts/recover_deletions.py):
- **Configuration & Database Path:** Imports `AsyncSessionLocal` from [`backend/database.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC AND LLM BOT/backend/database.py), which resolves `settings.database_url`. Uses the identical SQLite database (`data/knowledge_assistant.db`) and ChromaDB directory (`data/chromadb/`) as the application.
- **Exit Codes:**
  - Returns `0` if all pending documents succeed (or none found).
  - Returns `1` if any pending document cleanup fails.
- **Resource Management:** Explicitly calls `asyncio.run(close_db())` in a `finally` block to dispose of the engine connection pool cleanly.

---

## 6. Collection Deletion Analysis

> [!CAUTION]
> **Defect Identified in Collection Deletion Endpoint:**
> In [`backend/routers/collections.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC AND LLM BOT/backend/routers/collections.py#L156-L199), collection deletion does **not** delete ChromaDB vectors.

### Source Code Inspection:
```python
# backend/routers/collections.py lines 160-199
async def delete_collection(
    collection_id: str,
    db: AsyncSession = Depends(get_db),
    storage: FileStorageService = Depends(get_storage_service),
) -> dict:
    ...
    # Delete collection and cascaded documents from database
    await db.delete(collection)
    await db.commit()

    # Clean up files on disk ONLY after DB commit succeeds
    storage.delete_collection_storage(collection_id)

    return {"status": "deleted", ...}
```

### Analysis of the Defect:
1. `VectorStoreService` is **not injected** into `delete_collection`.
2. When `DELETE /api/collections/{collection_id}` is executed:
   - SQLite deletes the `collections` row, and SQLite foreign key cascading deletes all child `documents` rows.
   - `storage.delete_collection_storage(collection_id)` removes `data/uploads/{collection_id}`.
   - **ChromaDB is never notified.** The ChromaDB collection `col_{collection_id.replace('-', '_')}` and all its embedded vector chunks remain permanently orphaned in `data/chromadb/`.
3. Furthermore, because SQLite deleted the collection and document metadata, the system loses all reference to the orphaned vectors.

---

## 7. Audit Findings Ranked by Severity

| Finding ID | Severity | Category | Summary | Location |
|---|---|---|---|---|
| **FINDING-1** | **HIGH** | Data Leakage | `DELETE /api/collections/{collection_id}` does not delete ChromaDB vector collections, causing permanent vector leakage. | [`backend/routers/collections.py:160-194`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC AND LLM BOT/backend/routers/collections.py#L160-L194) |
| **FINDING-2** | **MEDIUM** | Availability | Startup recovery runs synchronously during lifespan before `yield` without an explicit timeout, potentially delaying server startup. | [`backend/main.py:84-99`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC AND LLM BOT/backend/main.py#L84-L99) |
| **FINDING-3** | **MEDIUM** | Concurrency | `DocumentRecoveryService` lacks cross-process locking, creating lock contention if multiple OS processes execute recovery simultaneously. | [`backend/services/document_recovery.py:57-148`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC AND LLM BOT/backend/services/document_recovery.py#L57-L148) |
| **FINDING-4** | **LOW** | Session State | `recover_pending_deletions` iterates through all documents in a single `AsyncSession`. Calling `db.rollback()` expires all objects in the identity map. | [`backend/services/document_recovery.py:80-147`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC AND LLM BOT/backend/services/document_recovery.py#L80-L147) |
| **FINDING-5** | **INFORMATIONAL** | Metric Accuracy | `list_collections` and `get_collection` compute `document_count` via `func.count(Document.id)`, which includes records in `DELETION_PENDING`. | [`backend/routers/collections.py:94-98, 131-136`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC AND LLM BOT/backend/routers/collections.py#L94-L98) |

---

## 8. Exact Remediation Recommendations

### For FINDING-1 (Collection Vector Leakage):
Inject `VectorStoreService` into `delete_collection` in `backend/routers/collections.py` and delete the vector collection:
```python
# Recommended correction for backend/routers/collections.py
async def delete_collection(
    collection_id: str,
    db: AsyncSession = Depends(get_db),
    storage: FileStorageService = Depends(get_storage_service),
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> dict:
    ...
    # 1. Delete ChromaDB vector collection
    try:
        vector_store.delete_collection(collection_id)
    except Exception as exc:
        logger.warning("Vector collection deletion failed or collection did not exist: %s", exc)

    # 2. Delete SQLite collection (cascades to documents)
    await db.delete(collection)
    await db.commit()

    # 3. Clean up physical files
    storage.delete_collection_storage(collection_id)
```

### For FINDING-2 (Startup Lifespan Timeout):
Wrap the startup recovery sweep in an explicit timeout in `backend/main.py`:
```python
try:
    async with asyncio.timeout(5.0):  # 5-second maximum startup budget
        async with AsyncSessionLocal() as db_session:
            recovery_svc = DocumentRecoveryService()
            await recovery_svc.recover_pending_deletions(db_session, limit=50)
except TimeoutError:
    logger.warning("Startup deletion recovery timed out after 5.0s; continuing startup.")
except Exception as exc:
    logger.warning("Startup deletion recovery failed: %s", exc)
```

---

## 9. Test Evidence and Regression Verification

The test evidence reported was verified against test runs:

- **Router Deletion & Recovery Suite:** [`tests/test_document_deletion_router.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC AND LLM BOT/tests/test_document_deletion_router.py)  
  **15 passed** in 7.77s.
- **Core Component Tests:** `test_document_processor.py`, `test_vector_store.py`, `test_documents.py`  
  **63 passed** in 17.30s.
- **Full Regression Test Suite:** `pytest -m "not real_model"`  
  **154 passed**, 1 deselected in 54.30s.
- **Dependency Check:** `pip check`  
  `No broken requirements found.`

---

## 10. Production Safety Confirmation

An independent read-only inspection was performed on all persistent storage locations:
- `data/knowledge_assistant.db`:
  - `PRAGMA integrity_check`: `[('ok',)]`
  - `PRAGMA foreign_key_check`: `[]` (0 foreign-key violations)
  - `SELECT count(*) FROM collections`: `0`
  - `SELECT count(*) FROM documents`: `0`
- `data/uploads/`: Empty (`0` files).
- `data/chromadb/`: Empty (`0` files).
- **Git State:** Clean working directory with no untracked file modifications to existing tracked files and no staged/uncommitted commits.

**Explicit Statement:** No production database records, ChromaDB vector stores, physical upload directories, or environment configurations were modified, created, or deleted during this audit.

---

## Audit Conclusion

Phase 6B Step 3's durable document deletion implementation is **verified and approved with conditions**. The document-level deletion lifecycle is resilient, recoverable, and crash-safe. **FINDING-1 (Collection Deletion Vector Leakage)** should be addressed before Phase 7 indexing begins. Phase 7 has not been started.
