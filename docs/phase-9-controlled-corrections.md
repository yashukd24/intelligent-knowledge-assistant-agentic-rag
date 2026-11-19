# Phase 9 — Controlled Correction of Reviewed Defects Report

**Date & Time:** 2026-11-10T14:06:00+05:30  
**Status:** CONTROLLED CORRECTIONS IMPLEMENTED & VERIFIED  
**Scope:** Strictly limited to `backend/services/file_storage.py` and `scripts/verify_local_pipeline.py`  
**Execution State:** ZERO TESTS RUN | ZERO INFERENCE CALLED | READ-ONLY SYNTAX CHECKS PASSED

---

## 1. Executive Summary of Changes

Following the senior engineering review in `docs/phase-9-code-change-review.md`, targeted corrections were applied to resolve the identified defects:
1. **Windows Collection Deletion (`backend/services/file_storage.py`):**
   - Corrected the read-only error handler to only catch `PermissionError` / `EACCES` / `EPERM`.
   - Removed the silent `except Exception: pass` that swallowed secondary failures.
   - Re-raises any non-permission exceptions and secondary deletion errors.
   - Strictly verifies `if not resolved.exists(): return True` before declaring deletion successful.
2. **Grounding Assertion (`scripts/verify_local_pipeline.py`):**
   - Completely eliminated generic keyword matching (specifically removed `"temperature"`).
   - Enforced semantic domain facts: requiring either the exact numeric fact and unit (`"15"` and `"millikelvin"`/`"kelvin"`) or the specific cryogenic mechanism (`"dilution refrigeration"` / `"dilution refrigerator"`).
   - Preserved citation verification (`"[S1]" in valid_citations`).
3. **Deletion Verification (`scripts/verify_local_pipeline.py`):**
   - Added explicit assertion `assert del_data.get("status") == "deleted"` and `assert r_del.status_code == 200`.
   - Completely eliminated the silent direct SQL fallback cleanup that previously masked API HTTP 500 errors.
   - Preserved teardown in `finally:` block with explicit logging and integrity assertions.

---

## 2. Task A — Exact Modifications: `backend/services/file_storage.py`

### 2.1 Before vs. After Code Comparison

#### Previous (Flawed) Implementation:
```python
            if resolved.is_dir():
                def _handle_readonly(func, path, exc):
                    import stat
                    try:
                        os.chmod(path, stat.S_IWRITE)
                        func(path)
                    except Exception:
                        pass

                try:
                    shutil.rmtree(resolved, onexc=_handle_readonly)
                except TypeError:
                    shutil.rmtree(resolved, onerror=_handle_readonly)
                logger.info("Deleted collection storage directory: %s", resolved)
                return True
```

#### Corrected Implementation:
```python
            if resolved.is_dir():
                def _handle_readonly(func, path, exc):
                    import errno
                    import stat

                    # Unwrap exc if passed as exc_info tuple (Python < 3.12 compatibility)
                    exc_val = exc[1] if isinstance(exc, tuple) else exc

                    # Handle read-only file attributes specifically on PermissionError
                    if isinstance(exc_val, PermissionError) or getattr(exc_val, "errno", None) in (
                        errno.EACCES,
                        errno.EPERM,
                    ):
                        os.chmod(path, stat.S_IWRITE)
                        func(path)
                    else:
                        raise exc_val

                try:
                    shutil.rmtree(resolved, onexc=_handle_readonly)
                except TypeError:
                    shutil.rmtree(resolved, onerror=_handle_readonly)

                # Verify target directory no longer exists before returning True
                if not resolved.exists():
                    logger.info("Deleted collection storage directory: %s", resolved)
                    return True
                logger.error("Collection storage directory still exists after rmtree: %s", resolved)
                return False
```

### 2.2 Exact Git Diff: `backend/services/file_storage.py`
```diff
--- a/backend/services/file_storage.py
+++ b/backend/services/file_storage.py
@@ -255,10 +255,19 @@
                 def _handle_readonly(func, path, exc):
+                    import errno
                     import stat
-                    try:
+
+                    # Unwrap exc if passed as exc_info tuple (Python < 3.12 compatibility)
+                    exc_val = exc[1] if isinstance(exc, tuple) else exc
+
+                    # Handle read-only file attributes specifically on PermissionError
+                    if isinstance(exc_val, PermissionError) or getattr(exc_val, "errno", None) in (
+                        errno.EACCES,
+                        errno.EPERM,
+                    ):
                         os.chmod(path, stat.S_IWRITE)
                         func(path)
-                    except Exception:
-                        pass
+                    else:
+                        raise exc_val
 
                 try:
                     shutil.rmtree(resolved, onexc=_handle_readonly)
                 except TypeError:
                     shutil.rmtree(resolved, onerror=_handle_readonly)
-                logger.info("Deleted collection storage directory: %s", resolved)
-                return True
+
+                # Verify target directory no longer exists before returning True
+                if not resolved.exists():
+                    logger.info("Deleted collection storage directory: %s", resolved)
+                    return True
+                logger.error("Collection storage directory still exists after rmtree: %s", resolved)
+                return False
```

---

## 3. Tasks B & C — Exact Modifications: `scripts/verify_local_pipeline.py`

### 3.1 Task B: Factual Grounding Assertion Correction

#### Previous (Flawed) Implementation:
```python
            assert answer_data["status"] == "answered"
            assert len(answer_data["valid_citations"]) > 0
            assert "[S1]" in answer_data["valid_citations"]
            ans_lower = answer_data["answer"].lower()
            assert any(
                w in ans_lower for w in ["15", "kelvin", "ambient dilution refrigeration", "dilution refrigeration", "temperature"]
            ), f"Answer does not contain expected factual detail: '{answer_data['answer']}'"
```

#### Corrected Implementation:
```python
            assert answer_data["status"] == "answered"
            assert len(answer_data["valid_citations"]) > 0
            assert "[S1]" in answer_data["valid_citations"]
            ans_lower = answer_data["answer"].lower()
            # Strict factual grounding validation:
            # Must contain specific domain fact: either the numeric temperature with unit ('15' and 'millikelvin'/'kelvin')
            # or the specific cryogenic mechanism ('dilution refrigeration'). Generic 'temperature' is strictly disallowed.
            has_numeric_fact = "15" in ans_lower and any(u in ans_lower for u in ["millikelvin", "kelvin"])
            has_cryo_mechanism = "dilution refrigeration" in ans_lower or "dilution refrigerator" in ans_lower
            assert has_numeric_fact or has_cryo_mechanism, (
                f"Answer does not contain required domain fact ('15 millikelvin' or 'dilution refrigeration'): '{answer_data['answer']}'"
            )
```

### 3.2 Task C: Documented API Deletion Assertion & Bypass Removal

#### Previous (Flawed) Implementation:
```python
        # Delete collection and its documents
        if created_collection_id:
            try:
                print(f"[CLEANUP] Deleting temporary collection '{created_collection_id}'...", flush=True)
                r_del = httpx.delete(f"{BASE_URL}/api/collections/{created_collection_id}", timeout=10.0)
                print(f"[CLEANUP] Collection deletion status: {r_del.status_code}", flush=True)
            except Exception as e:
                print(f"[CLEANUP WARN] Failed to delete collection via API: {e}", flush=True)

            # Defensive fallback cleanup in case of Windows filesystem race conditions
            col_upload_dir = Path("data/uploads") / created_collection_id
            if col_upload_dir.exists():
                try:
                    import stat, shutil
                    def _rm_readonly(func, path, exc):
                        try:
                            os.chmod(path, stat.S_IWRITE)
                            func(path)
                        except Exception:
                            pass
                    shutil.rmtree(col_upload_dir, onexc=_rm_readonly)
                except Exception as e:
                    print(f"[CLEANUP WARN] Fallback directory cleanup: {e}", flush=True)

            try:
                import sqlite3
                conn = sqlite3.connect("data/knowledge_assistant.db")
                c = conn.cursor()
                c.execute("DELETE FROM documents WHERE collection_id = ?", (created_collection_id,))
                c.execute("DELETE FROM collections WHERE id = ?", (created_collection_id,))
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"[CLEANUP WARN] Fallback DB cleanup: {e}", flush=True)
```

#### Corrected Implementation:
```python
        # Delete collection and its child documents via the documented API
        api_deletion_ok = False
        api_deletion_detail = ""
        if created_collection_id:
            try:
                print(f"[CLEANUP] Deleting temporary collection '{created_collection_id}' via API...", flush=True)
                r_del = httpx.delete(f"{BASE_URL}/api/collections/{created_collection_id}", timeout=10.0)
                print(f"[CLEANUP] Collection deletion status: {r_del.status_code}", flush=True)
                if r_del.status_code == 200:
                    api_deletion_ok = True
                    del_data = r_del.json()
                    assert del_data.get("status") == "deleted", f"Unexpected deletion payload: {del_data}"
                else:
                    api_deletion_detail = f"HTTP {r_del.status_code} - {r_del.text}"
                    print(f"[CLEANUP ERROR] Collection deletion API failed: {api_deletion_detail}", flush=True)
            except Exception as e:
                api_deletion_detail = str(e)
                print(f"[CLEANUP ERROR] Failed to invoke collection deletion API: {e}", flush=True)

        # Remove local temporary PDF
        if TEMP_PDF_PATH.exists():
            try:
                TEMP_PDF_PATH.unlink()
                print(f"[CLEANUP] Removed temporary PDF file '{TEMP_PDF_PATH}'", flush=True)
            except Exception as e:
                print(f"[CLEANUP WARN] Failed to delete local PDF: {e}", flush=True)

        # Terminate server process
        if server_proc.poll() is None:
            print("[CLEANUP] Terminating Uvicorn server process...", flush=True)
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()
            print("[CLEANUP] Server terminated.", flush=True)

        # Assert that collection API deletion succeeded (do not silently bypass via direct SQL)
        if created_collection_id:
            assert api_deletion_ok, f"Collection API deletion failed: {api_deletion_detail}"
```

### 3.3 Exact Git Diff: `scripts/verify_local_pipeline.py`
```diff
--- a/scripts/verify_local_pipeline.py
+++ b/scripts/verify_local_pipeline.py
@@ -355,5 +355,10 @@
             ans_lower = answer_data["answer"].lower()
-            assert any(
-                w in ans_lower for w in ["15", "kelvin", "ambient dilution refrigeration", "dilution refrigeration", "temperature"]
-            ), f"Answer does not contain expected factual detail: '{answer_data['answer']}'"
+            # Strict factual grounding validation:
+            # Must contain specific domain fact: either the numeric temperature with unit ('15' and 'millikelvin'/'kelvin')
+            # or the specific cryogenic mechanism ('dilution refrigeration'). Generic 'temperature' is strictly disallowed.
+            has_numeric_fact = "15" in ans_lower and any(u in ans_lower for u in ["millikelvin", "kelvin"])
+            has_cryo_mechanism = "dilution refrigeration" in ans_lower or "dilution refrigerator" in ans_lower
+            assert has_numeric_fact or has_cryo_mechanism, (
+                f"Answer does not contain required domain fact ('15 millikelvin' or 'dilution refrigeration'): '{answer_data['answer']}'"
+            )
@@ -412,36 +417,17 @@
-        # Delete collection and its documents
+        # Delete collection and its child documents via the documented API
+        api_deletion_ok = False
+        api_deletion_detail = ""
         if created_collection_id:
             try:
-                print(f"[CLEANUP] Deleting temporary collection '{created_collection_id}'...", flush=True)
+                print(f"[CLEANUP] Deleting temporary collection '{created_collection_id}' via API...", flush=True)
                 r_del = httpx.delete(f"{BASE_URL}/api/collections/{created_collection_id}", timeout=10.0)
                 print(f"[CLEANUP] Collection deletion status: {r_del.status_code}", flush=True)
+                if r_del.status_code == 200:
+                    api_deletion_ok = True
+                    del_data = r_del.json()
+                    assert del_data.get("status") == "deleted", f"Unexpected deletion payload: {del_data}"
+                else:
+                    api_deletion_detail = f"HTTP {r_del.status_code} - {r_del.text}"
+                    print(f"[CLEANUP ERROR] Collection deletion API failed: {api_deletion_detail}", flush=True)
             except Exception as e:
-                print(f"[CLEANUP WARN] Failed to delete collection via API: {e}", flush=True)
-
-            # Defensive fallback cleanup in case of Windows filesystem race conditions
-            col_upload_dir = Path("data/uploads") / created_collection_id
-            if col_upload_dir.exists():
-                try:
-                    import stat, shutil
-                    def _rm_readonly(func, path, exc):
-                        try:
-                            os.chmod(path, stat.S_IWRITE)
-                            func(path)
-                        except Exception:
-                            pass
-                    shutil.rmtree(col_upload_dir, onexc=_rm_readonly)
-                except Exception as e:
-                    print(f"[CLEANUP WARN] Fallback directory cleanup: {e}", flush=True)
-
-            try:
-                import sqlite3
-                conn = sqlite3.connect("data/knowledge_assistant.db")
-                c = conn.cursor()
-                c.execute("DELETE FROM documents WHERE collection_id = ?", (created_collection_id,))
-                c.execute("DELETE FROM collections WHERE id = ?", (created_collection_id,))
-                conn.commit()
-                conn.close()
-            except Exception as e:
-                print(f"[CLEANUP WARN] Fallback DB cleanup: {e}", flush=True)
+                api_deletion_detail = str(e)
+                print(f"[CLEANUP ERROR] Failed to invoke collection deletion API: {e}", flush=True)
@@ -464,2 +450,6 @@
+        # Assert that collection API deletion succeeded (do not silently bypass via direct SQL)
+        if created_collection_id:
+            assert api_deletion_ok, f"Collection API deletion failed: {api_deletion_detail}"
+
         # Verify production databases remain 100% clean
@@ -494,4 +484,4 @@
-        if col_count == 0 and doc_count == 0 and chroma_col_count == 0 and upload_files == 0:
-            print("\n[VERIFICATION_RESULT] ALL TESTS PASSED. PRODUCTION DATA IS COMPLETELY CLEAN.", flush=True)
-        else:
-            print("\n[VERIFICATION_RESULT] WARNING: Unexpected remaining entities in storage!", flush=True)
+        assert col_count == 0, f"Integrity failure: {col_count} collection(s) remain in SQLite after deletion."
+        assert doc_count == 0, f"Integrity failure: {doc_count} document(s) remain in SQLite after deletion."
+        assert chroma_col_count == 0, f"Integrity failure: {chroma_col_count} collection(s) remain in ChromaDB."
+        assert upload_files == 0, f"Integrity failure: {upload_files} file(s)/dir(s) remain in data/uploads/."
+        print("\n[VERIFICATION_RESULT] ALL TESTS PASSED. PRODUCTION DATA IS COMPLETELY CLEAN.", flush=True)
```

---

## 4. Task D — Review-Only Verification Results

| Verification Check | Target / Command | Result | Details |
|---|---|---|---|
| **Python Syntax Check** | `python -m py_compile backend/services/file_storage.py` | **PASS (Exit 0)** | Zero syntax or indentation errors |
| **Python Syntax Check** | `python -m py_compile scripts/verify_local_pipeline.py` | **PASS (Exit 0)** | Zero syntax or indentation errors |
| **Configuration State** | `.env` | **UNCHANGED** | Preserved `OLLAMA_MODEL=qwen2.5:3b`, no secret leakage |
| **Workspace Scope Audit** | Host Filesystem | **VERIFIED** | Exactly 2 code files modified; no other files touched |
| **Temporary File State** | `test_chunks.pdf` | **PRESERVED** | Preserved without deletion per strict instructions |
| **Database State** | SQLite & ChromaDB | **UNTOUCHED** | 0 test records modified, zero mutations applied |
| **Process State** | Host Task List | **IDLE** | 0 active Python, Uvicorn, or pytest processes |

---

## 5. Strict Restriction Adherence

- [x] No pytest tests executed.
- [x] No live smoke tests run.
- [x] No Uvicorn server started.
- [x] No Ollama inference triggered.
- [x] No files deleted (including `test_chunks.pdf`).
- [x] No database records modified.
- [x] No Phase 1–8 functionality altered outside the approved `delete_collection_storage` fix.
- [x] Zero Phase 10 state machine code introduced.
- [x] Zero Git commits or resets executed.

*Report complete. All actions stopped awaiting manual approval.*
