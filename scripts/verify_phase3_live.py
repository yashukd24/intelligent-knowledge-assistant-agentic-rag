"""
scripts/verify_phase3_live.py
─────────────────────────────────────────────────────────────
Phase 3 Live Uvicorn Verification Script.
Starts Uvicorn server, runs end-to-end integration requests
against all Phase 3 endpoints, verifies responses, and shuts down.
─────────────────────────────────────────────────────────────
"""

import io
import sys
import time
import subprocess
import httpx

HOST = "127.0.0.1"
PORT = 8000
BASE_URL = f"http://{HOST}:{PORT}"

print(f"[PHASE 3 LIVE] Starting Uvicorn on {BASE_URL}...", flush=True)
proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", HOST, "--port", str(PORT)],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)

try:
    # 1. Wait for server startup
    started = False
    for attempt in range(20):
        time.sleep(0.5)
        try:
            r = httpx.get(f"{BASE_URL}/health", timeout=1.0)
            if r.status_code == 200:
                started = True
                print(f"[PHASE 3 LIVE] Server alive after {(attempt+1)*0.5:.1f}s!", flush=True)
                break
        except Exception:
            pass

    if not started:
        print("[PHASE 3 LIVE ERROR] Uvicorn failed to start within timeout.", flush=True)
        stdout, stderr = proc.communicate(timeout=2)
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
        sys.exit(1)

    # 2. Test GET /health
    print("\n--- 1. Checking GET /health ---", flush=True)
    r_health = httpx.get(f"{BASE_URL}/health", timeout=5.0)
    assert r_health.status_code == 200
    print(f"Health Response: {r_health.json()}")

    # 3. Test POST /api/collections (Create collection)
    print("\n--- 2. Testing POST /api/collections ---", flush=True)
    col_payload = {"name": "Phase3 Test Collection", "description": "Created during live verification"}
    r_create_col = httpx.post(f"{BASE_URL}/api/collections", json=col_payload, timeout=5.0)
    print(f"Create Collection Status: {r_create_col.status_code}")
    print(f"Collection Data: {r_create_col.json()}")
    assert r_create_col.status_code == 201
    col_id = r_create_col.json()["id"]

    # 4. Test GET /api/collections (List collections)
    print("\n--- 3. Testing GET /api/collections ---", flush=True)
    r_list_col = httpx.get(f"{BASE_URL}/api/collections", timeout=5.0)
    assert r_list_col.status_code == 200
    print(f"Listed {r_list_col.json()['total']} collections.")

    # 5. Test POST /api/collections/{id}/documents (Upload valid PDF)
    print("\n--- 4. Testing POST /api/collections/{id}/documents (PDF Upload) ---", flush=True)
    pdf_bytes = b"%PDF-1.4\nAcademic Handbook 2026 content for live test.\n%%EOF"
    files = {"file": ("student_handbook.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    r_upload_pdf = httpx.post(f"{BASE_URL}/api/collections/{col_id}/documents", files=files, timeout=5.0)
    print(f"Upload PDF Status: {r_upload_pdf.status_code}")
    print(f"Uploaded Doc: {r_upload_pdf.json()}")
    assert r_upload_pdf.status_code == 201
    doc1_id = r_upload_pdf.json()["id"]
    assert r_upload_pdf.json()["status"] == "pending"
    assert r_upload_pdf.json()["filename"] == "student_handbook.pdf"

    # 6. Test duplicate upload detection (same SHA-256) -> 409
    print("\n--- 5. Testing Duplicate Content Detection (HTTP 409) ---", flush=True)
    files_dup = {"file": ("duplicate_name.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    r_dup = httpx.post(f"{BASE_URL}/api/collections/{col_id}/documents", files=files_dup, timeout=5.0)
    print(f"Duplicate Upload Status: {r_dup.status_code}")
    print(f"Duplicate Error: {r_dup.json()}")
    assert r_dup.status_code == 409

    # 7. Test TXT upload
    print("\n--- 6. Testing TXT Document Upload ---", flush=True)
    txt_bytes = b"University Regulations Chapter 1: Academic Honesty."
    files_txt = {"file": ("regulations.txt", io.BytesIO(txt_bytes), "text/plain")}
    r_upload_txt = httpx.post(f"{BASE_URL}/api/collections/{col_id}/documents", files=files_txt, timeout=5.0)
    print(f"Upload TXT Status: {r_upload_txt.status_code}")
    assert r_upload_txt.status_code == 201
    doc2_id = r_upload_txt.json()["id"]

    # 7b. Test replace=true workflow
    print("\n--- 6b. Testing Staged Document Replacement (?replace=true) ---", flush=True)
    txt_v2_bytes = b"University Regulations Chapter 1: Academic Honesty (Updated Revision 2026)."
    files_txt_v2 = {"file": ("regulations.txt", io.BytesIO(txt_v2_bytes), "text/plain")}
    # First attempt without replace=true -> 409
    r_no_rep = httpx.post(f"{BASE_URL}/api/collections/{col_id}/documents", files=files_txt_v2, timeout=5.0)
    print(f"Without replace=true Status: {r_no_rep.status_code}")
    assert r_no_rep.status_code == 409

    # Second attempt with replace=true -> 200
    files_txt_v2_retry = {"file": ("regulations.txt", io.BytesIO(txt_v2_bytes), "text/plain")}
    r_rep = httpx.post(f"{BASE_URL}/api/collections/{col_id}/documents?replace=true", files=files_txt_v2_retry, timeout=5.0)
    print(f"With replace=true Status: {r_rep.status_code}, Version: {r_rep.json()['version']}")
    assert r_rep.status_code == 200
    assert r_rep.json()["id"] == doc2_id
    assert r_rep.json()["version"] == 2

    # 8. Test list documents in collection
    print("\n--- 7. Testing GET /api/collections/{id}/documents ---", flush=True)
    r_docs = httpx.get(f"{BASE_URL}/api/collections/{col_id}/documents", timeout=5.0)
    assert r_docs.status_code == 200
    print(f"Documents in collection: {r_docs.json()['total']}")
    assert r_docs.json()["total"] == 2

    # 9. Test GET /api/documents/{id}
    print("\n--- 8. Testing GET /api/documents/{id} ---", flush=True)
    r_get_doc = httpx.get(f"{BASE_URL}/api/documents/{doc1_id}", timeout=5.0)
    assert r_get_doc.status_code == 200
    print(f"Fetched Doc Metadata: {r_get_doc.json()['filename']} (status: {r_get_doc.json()['status']})")

    # 10. Test DELETE /api/documents/{id}
    print("\n--- 9. Testing DELETE /api/documents/{id} ---", flush=True)
    r_del_doc = httpx.delete(f"{BASE_URL}/api/documents/{doc1_id}", timeout=5.0)
    assert r_del_doc.status_code == 200
    print(f"Deleted Doc: {r_del_doc.json()}")
    # Verify 404
    assert httpx.get(f"{BASE_URL}/api/documents/{doc1_id}").status_code == 404

    # 11. Test DELETE /api/collections/{id}
    print("\n--- 10. Testing DELETE /api/collections/{id} (Cascade) ---", flush=True)
    r_del_col = httpx.delete(f"{BASE_URL}/api/collections/{col_id}", timeout=5.0)
    assert r_del_col.status_code == 200
    print(f"Deleted Collection: {r_del_col.json()}")
    assert httpx.get(f"{BASE_URL}/api/collections/{col_id}").status_code == 404

    # 12. Test OpenAPI Docs
    print("\n--- 11. Checking /docs & /openapi.json ---", flush=True)
    r_openapi = httpx.get(f"{BASE_URL}/openapi.json", timeout=5.0)
    assert r_openapi.status_code == 200
    paths = list(r_openapi.json()["paths"].keys())
    print("Registered OpenAPI Paths:")
    for p in paths:
        print(f"  - {p}")
    assert "/api/collections" in paths
    assert "/api/collections/{collection_id}/documents" in paths
    assert "/api/documents/{document_id}" in paths
    assert "/health" in paths

    print("\n" + "=" * 60, flush=True)
    print("ALL PHASE 3 LIVE ENDPOINT VERIFICATIONS PASSED!", flush=True)
    print("=" * 60, flush=True)

finally:
    print("\n[PHASE 3 LIVE] Terminating Uvicorn process...", flush=True)
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    print("[PHASE 3 LIVE] Server stopped cleanly.", flush=True)
