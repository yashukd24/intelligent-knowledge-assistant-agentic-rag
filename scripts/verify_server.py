"""
scripts/verify_server.py
─────────────────────────────────────────────────────────────
Rule 5: Verify live application startup with Uvicorn.
1. Starts `python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000`
2. Verifies GET /health
3. Verifies GET /docs
4. Verifies GET /openapi.json
5. Shuts down Uvicorn cleanly.
─────────────────────────────────────────────────────────────
"""

import sys
import time
import subprocess
import httpx

HOST = "127.0.0.1"
PORT = 8000
BASE_URL = f"http://{HOST}:{PORT}"

print(f"[SERVER VERIFY] Launching Uvicorn on {BASE_URL}...", flush=True)
proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", HOST, "--port", str(PORT)],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)

try:
    # Wait for server to come up (up to 10 seconds)
    started = False
    for attempt in range(20):
        time.sleep(0.5)
        try:
            r = httpx.get(f"{BASE_URL}/", timeout=1.0)
            if r.status_code == 200:
                started = True
                print(f"[SERVER VERIFY] Server responded after {(attempt+1)*0.5:.1f}s!", flush=True)
                break
        except Exception:
            pass

    if not started:
        print("[SERVER VERIFY ERROR] Server failed to start within timeout.", flush=True)
        stdout, stderr = proc.communicate(timeout=2)
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
        sys.exit(1)

    # ── Test GET /health ──────────────────────────────────────
    print("\n[SERVER VERIFY] Testing GET /health...", flush=True)
    res_health = httpx.get(f"{BASE_URL}/health", timeout=5.0)
    print(f"  HTTP Status Code: {res_health.status_code}")
    print(f"  Response JSON: {res_health.json()}")
    assert res_health.status_code == 200, f"Expected 200, got {res_health.status_code}"
    health_data = res_health.json()
    assert health_data["status"] == "ok", f"Expected 'ok', got {health_data['status']}"
    assert health_data["components"]["api"]["status"] == "ok"
    assert health_data["components"]["database"]["status"] == "ok"
    print("  [PASS] GET /health verified successfully!")

    # ── Test GET /docs ────────────────────────────────────────
    print("\n[SERVER VERIFY] Testing GET /docs...", flush=True)
    res_docs = httpx.get(f"{BASE_URL}/docs", timeout=5.0)
    print(f"  HTTP Status Code: {res_docs.status_code}")
    print(f"  Content Type: {res_docs.headers.get('content-type')}")
    assert res_docs.status_code == 200, f"Expected 200, got {res_docs.status_code}"
    assert "swagger-ui" in res_docs.text.lower(), "Swagger UI HTML not found in /docs"
    print("  [PASS] GET /docs Swagger UI verified successfully!")

    # ── Test GET /openapi.json ────────────────────────────────
    print("\n[SERVER VERIFY] Testing GET /openapi.json...", flush=True)
    res_openapi = httpx.get(f"{BASE_URL}/openapi.json", timeout=5.0)
    assert res_openapi.status_code == 200
    openapi_data = res_openapi.json()
    print(f"  API Title: {openapi_data.get('info', {}).get('title')}")
    print(f"  API Version: {openapi_data.get('info', {}).get('version')}")
    print(f"  Paths in OpenAPI schema: {list(openapi_data.get('paths', {}).keys())}")
    print("  [PASS] GET /openapi.json verified successfully!")

    print("\n" + "=" * 60, flush=True)
    print("LIVE UVICORN SERVER VERIFICATION COMPLETED: ALL PASS", flush=True)
    print("=" * 60, flush=True)

finally:
    print("\n[SERVER VERIFY] Stopping live Uvicorn server...", flush=True)
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    print("[SERVER VERIFY] Uvicorn server stopped cleanly.", flush=True)
