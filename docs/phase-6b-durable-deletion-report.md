# Phase 6B Step 3 — Durable Document Deletion Implementation Report

**Project:** Intelligent Knowledge Assistant Using Agentic RAG and Large Language Models  
**Component:** REST Document Deletion, Storage Consistency, and Crash-Safety Recovery  
**Status:** Completed and Verified  
**Date:** November 2026  

---

## 1. Executive Summary

During Phase 6B Step 3, an independent technical audit identified three fundamental data consistency vulnerabilities in the REST document deletion pipeline (`DELETE /api/documents/{document_id}`):
1. **Ghost Indexed Records:** ChromaDB vector deletion could succeed while the subsequent SQLite transaction failed or crashed, leaving an indexed SQLite record without vector embeddings.
2. **Orphaned Physical Files:** SQLite document records could be committed as deleted while physical filesystem unlinking failed or crashed, leaving unreferenced physical files in `data/uploads/` that were impossible to clean via future API calls (which returned 404).
3. **False Success Reporting:** The endpoint ignored the Boolean return value of `FileStorageService.delete_stored_file()`, reporting HTTP 200 success even when file unlinking failed.

To resolve these defects without introducing external distributed transaction coordinators or heavyweight message brokers, a **recoverable two-phase deletion workflow** centered on a persistent `DocumentStatus.DELETION_PENDING` lifecycle state was implemented.

All 15 targeted crash-safety, concurrency, and recovery tests passed in **7.77s**, the 63 core component tests passed in **17.30s**, and the complete 154-test regression suite passed in **54.30s** with 0 regressions.

---

## 2. New Deletion Lifecycle Architecture

The deletion flow replaces the prior best-effort sequence with a persistent, staged state machine:

```
                      ┌─────────────────────────────────────────┐
                      │   Client: DELETE /api/documents/{id}    │
                      └────────────────────┬────────────────────┘
                                           │
                                           ▼
                      ┌─────────────────────────────────────────┐
                      │ 1. Lookup Document in SQLite            │
                      │    (Return HTTP 404 if not found)       │
                      └────────────────────┬────────────────────┘
                                           │
                                           ▼
                      ┌─────────────────────────────────────────┐
                      │ 2. Stage DELETION_PENDING State         │
                      │    status = 'deletion_pending'          │
                      │    Committed to SQLite before cleanup   │
                      └────────────────────┬────────────────────┘
                                           │
                        ┌──────────────────┴──────────────────┐
                        │                                     │
                        ▼                                     ▼
     ┌──────────────────────────────────────┐  ┌─────────────────────────────────────┐
     │ 3. Delete ChromaDB Vectors           │  │ Failure: Document remains in        │
     │    All versions for this document ID │─►│ SQLite as 'deletion_pending'.       │
     └──────────────────┬───────────────────┘  │ Error recorded; HTTP 500 returned.  │
                        │                      └─────────────────────────────────────┘
                        ▼                                     ▲
     ┌──────────────────────────────────────┐                 │
     │ 4. Delete Physical File from Disk    │                 │
     │    Verify Boolean return value.      │─────────────────┘
     │    Missing files treated as cleaned. │  (No false success reported)
     └──────────────────┬───────────────────┘
                        │
                        ▼
     ┌──────────────────────────────────────┐
     │ 5. Purge SQLite Document Record      │
     │    await db.delete(doc)              │
     │    await db.commit()                 │
     └──────────────────┬───────────────────┘
                        │
                        ▼
     ┌──────────────────────────────────────┐
     │ 6. Return HTTP 200 Success           │
     │    {"status": "deleted", ...}        │
     └──────────────────────────────────────┘
```

---

## 3. Failure and Recovery Behavior

The staged architecture handles every potential crash or failure point predictably:

| Failure Point | SQLite State | ChromaDB State | Physical File State | Recovery Mechanism |
|---|---|---|---|---|
| **Phase 1 Commit Fails** | Intact (`indexed`/`pending`) | Intact | Intact | Request aborted via rollback; document remains functional. |
| **Phase 2: Vector Deletion Fails** | `DELETION_PENDING` | Intact / Partial | Intact | Retrying `DELETE` or invoking recovery resumes vector cleanup. Processing rejected with HTTP 422. |
| **Crash after Phase 2 (Vectors Deleted)** | `DELETION_PENDING` | Deleted | Intact | Next `DELETE` finds 0 vectors (idempotent), removes physical file, and purges SQLite row. |
| **Phase 3: File Deletion Fails** | `DELETION_PENDING` | Deleted | Intact on disk | HTTP 500 returned. Document remains in SQLite with `stored_filename`, allowing retries to clean the file. |
| **Crash after Phase 3 (File Deleted)** | `DELETION_PENDING` | Deleted | Cleaned | Next `DELETE` treats missing file as cleaned and commits final SQLite deletion. |
| **Phase 4: Final SQLite Commit Fails** | `DELETION_PENDING` | Deleted | Cleaned | Retrying `DELETE` commits the deletion. No ghost or orphan records. |

---

## 4. Concurrency Safety Strategy

When two simultaneous `DELETE /api/documents/{document_id}` requests arrive:
1. Both requests retrieve the document row.
2. Whichever request commits first sets `doc.status = DELETION_PENDING`. If the second request observes `DELETION_PENDING`, it skips redundant re-marking and participates in cleanup.
3. Both invoke ChromaDB vector deletion. ChromaDB deletion by ID is idempotent: deleting IDs that are already removed returns 0 without raising errors.
4. Both invoke filesystem deletion. `FileStorageService.delete_stored_file()` returns `True` if unlinked or if already absent (`not resolved.exists()`).
5. Both attempt `await db.delete(doc)` and `await db.commit()`:
   - Transaction 1 commits and removes the row from SQLite.
   - Transaction 2 encounters `sqlalchemy.orm.exc.StaleDataError` because the row no longer exists.
   - The endpoint intercepts `StaleDataError`, rolls back Session 2, and logs that the document was already deleted by a concurrent transaction.
6. **Result:** Neither transaction raises HTTP 500. At least one returns HTTP 200 (`status: "deleted"`), and data integrity across all storage engines is preserved.

---

## 5. Recovery Mechanism & Invocation

A dedicated recovery service was implemented in [`backend/services/document_recovery.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/services/document_recovery.py):

### A. Automated Bounded Startup Sweep
During FastAPI startup in [`backend/main.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/main.py), the application lifespan executes a non-blocking recovery check:
```python
async with AsyncSessionLocal() as db_session:
    recovery_svc = DocumentRecoveryService()
    report = await recovery_svc.recover_pending_deletions(db_session, limit=100)
```
- **Bounded:** Limits processing to at most 100 pending deletions at startup to prevent indefinite startup delays.
- **Resilient:** Individual document failures are logged and recorded on `doc.error`; the loop continues to process all remaining pending items.
- **Fail-safe:** Wrapped in an outer `try/except` block to ensure startup never crashes if recovery encounters an unexpected error.

### B. Manual Operator CLI Utility
Operators can trigger recovery on demand using [`scripts/recover_deletions.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/scripts/recover_deletions.py):
```bash
# Sweep all pending deletions
python scripts/recover_deletions.py

# Sweep with batch limit
python scripts/recover_deletions.py --limit 50
```

---

## 6. Migration and Schema Assessment

An audit of the SQLite schema and SQLAlchemy ORM models confirmed:
- In [`backend/models/document.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/models/document.py), `Document.status` is declared as:
  ```python
  status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False, index=True)
  ```
- The string value `"deletion_pending"` has a length of 16 characters, well within the `String(20)` allocation.
- In SQLite, the column type affinity is `VARCHAR(20)` with no `CHECK` constraint.
- The Pydantic schema [`backend/schemas/document.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/schemas/document.py) was updated to include `DELETION_PENDING = "deletion_pending"`.
- **Conclusion:** No database schema alteration, table migration, or Alembic script is required.

---

## 7. Exact Files Modified and Created

| File Path | Status | Summary of Changes |
|---|---|---|
| [`backend/schemas/document.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/schemas/document.py) | Modified | Added `DocumentStatus.DELETION_PENDING = "deletion_pending"`. |
| [`backend/services/file_storage.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/services/file_storage.py) | Modified | Updated `delete_stored_file` to return `True` if file is missing (idempotent cleanup) and `False` on filesystem error. |
| [`backend/services/document_processor.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/services/document_processor.py) | Modified | Added check rejecting `DELETION_PENDING` documents with `DocumentProcessorError` (HTTP 422). |
| [`backend/routers/documents.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/routers/documents.py) | Modified | Implemented two-phase recoverable deletion, `StaleDataError` handling, and Boolean check on file cleanup. |
| [`backend/services/document_recovery.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/services/document_recovery.py) | **Created** | Implemented `DocumentRecoveryService` and `DeletionRecoveryReport`. |
| [`backend/services/__init__.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/services/__init__.py) | Modified | Exported `DocumentRecoveryService` and `DeletionRecoveryReport`. |
| [`backend/main.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/main.py) | Modified | Integrated non-blocking startup recovery hook in `lifespan`. |
| [`scripts/recover_deletions.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/scripts/recover_deletions.py) | **Created** | Created operator CLI utility for manual deletion recovery. |
| [`tests/test_document_deletion_router.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/tests/test_document_deletion_router.py) | Modified | Implemented comprehensive 15-test deletion and recovery test suite. |

---

## 8. Test Verification Summary

### A. Dedicated Deletion & Recovery Suite
```
pytest tests/test_document_deletion_router.py -v
======================== 15 passed, 1 warning in 7.77s ========================
```
- Test 1: `test_1_normal_deletion_succeeds` — **PASSED**
- Test 2: `test_2_vector_deletion_failure_retains_deletion_pending` — **PASSED**
- Test 3: `test_3_filesystem_deletion_failure_retains_deletion_pending` — **PASSED**
- Test 4: `test_4_missing_physical_file_is_treated_as_already_cleaned` — **PASSED**
- Test 5: `test_5_final_sqlite_commit_failure_can_be_retried` — **PASSED**
- Test 6: `test_6_crash_simulation_after_pending_state_commit` — **PASSED**
- Test 7: `test_7_crash_simulation_after_vector_cleanup` — **PASSED**
- Test 8: `test_8_crash_simulation_after_file_cleanup` — **PASSED**
- Test 9: `test_9_retry_completes_pending_deletion` — **PASSED**
- Test 10: `test_10_simultaneous_deletion_requests` — **PASSED**
- Test 11: `test_11_processing_pending_deletion_document_is_rejected` — **PASSED**
- Test 12: `test_12_unrelated_documents_remain_intact` — **PASSED**
- Test 13: `test_13_other_collections_remain_intact` — **PASSED**
- Test 14: `test_14_existing_404_and_api_response_behavior` — **PASSED**
- Test 15: `test_15_recovery_processes_multiple_pending_documents_and_continues` — **PASSED**

### B. Core Component Test Suite
```
pytest tests/test_document_processor.py tests/test_vector_store.py tests/test_documents.py -v
============================= 63 passed in 17.30s =============================
```

### C. Full Regression Suite
```
pytest -m "not real_model" -v
================ 154 passed, 1 deselected, 1 warning in 54.30s ================
```

### D. Dependency Check
```
pip check
No broken requirements found.
```

### E. Production Data Audit
- `data/knowledge_assistant.db`: `PRAGMA integrity_check` returned `ok`. Row count: 0 across all tables.
- `data/uploads/`: Empty.
- `data/chromadb/`: Empty.

---

## 9. Remaining Architectural Considerations

1. **Collection Cascading Deletion:** While document deletion is now durable, `DELETE /api/collections/{collection_id}` currently cascades deletions in SQLite and removes the collection upload directory, but relies on SQLite cascade foreign keys rather than explicit ChromaDB collection deletion. Collection vector purge will be finalized in Phase 7.
2. **Periodic Daemon Scheduler:** The recovery service is executed at application startup and on demand via CLI. If long-running deployments require real-time background sweeps without waiting for server restart, a lightweight cron/timer job can be attached to call `DocumentRecoveryService.recover_pending_deletions()`.
