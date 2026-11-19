"""
scripts/rule3_diagnostics.py
─────────────────────────────────────────────────────────────
Rule 3 Diagnostic Script for Phase 2 Root Cause Analysis.
Runs diagnostics 1 through 7 step-by-step with checkpoints.
─────────────────────────────────────────────────────────────
"""

import sys
import time
import asyncio
from datetime import datetime
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 60, flush=True)
print("PHASE 2 — RULE 3 DIAGNOSTIC SUITE", flush=True)
print(f"Timestamp: {datetime.now().isoformat()}", flush=True)
print(f"Python: {sys.version}", flush=True)
print("=" * 60, flush=True)

# ── 1. Import backend.config ────────────────────────────────
t0 = time.time()
print("\n[DIAG 1] Importing backend.config...", flush=True)
from backend.config import get_settings
settings = get_settings()
print(f"[DIAG 1 OK] backend.config imported in {time.time()-t0:.3f}s. App: {settings.app_name}", flush=True)

# ── 2. Import backend.database ──────────────────────────────
t0 = time.time()
print("\n[DIAG 2] Importing backend.database...", flush=True)
from backend.database import engine, AsyncSessionLocal, init_db, close_db, get_db
print(f"[DIAG 2 OK] backend.database imported in {time.time()-t0:.3f}s.", flush=True)

# ── 3. Import backend.main ──────────────────────────────────
t0 = time.time()
print("\n[DIAG 3] Importing backend.main...", flush=True)
from backend.main import create_app, app
print(f"[DIAG 3 OK] backend.main imported in {time.time()-t0:.3f}s.", flush=True)

# ── 4. Instantiate the FastAPI app ──────────────────────────
t0 = time.time()
print("\n[DIAG 4] Instantiating FastAPI app...", flush=True)
fresh_app = create_app()
print(f"[DIAG 4 OK] FastAPI app instantiated in {time.time()-t0:.3f}s. Routes: {[r.path for r in fresh_app.routes]}", flush=True)

# ── 5. Initialize the database independently ────────────────
async def diag_5():
    t0 = time.time()
    print("\n[DIAG 5] Initializing database independently...", flush=True)
    await init_db()
    print(f"[DIAG 5 OK] init_db() completed in {time.time()-t0:.3f}s.", flush=True)
    await close_db()
    print("[DIAG 5 OK] close_db() completed.", flush=True)

asyncio.run(diag_5())

# ── 6. Execute the health endpoint without lifespan ─────────
async def diag_6():
    t0 = time.time()
    print("\n[DIAG 6] Executing health endpoint without lifespan...", flush=True)
    from httpx import ASGITransport, AsyncClient
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from backend.models.base import Base

    test_engine = create_async_engine("sqlite+aiosqlite:///:memory:", connect_args={"check_same_thread": False}, future=True)
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(bind=test_engine, expire_on_commit=False, autocommit=False, autoflush=False)

    app_instance = create_app()

    async def override_db():
        async with factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    app_instance.dependency_overrides[get_db] = override_db

    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        print(f"[DIAG 6 OK] Status: {resp.status_code}, Payload: {resp.json()} (took {time.time()-t0:.3f}s)", flush=True)

    await test_engine.dispose()

asyncio.run(diag_6())

# ── 7. Execute health endpoint with correctly managed lifespan
async def diag_7():
    t0 = time.time()
    print("\n[DIAG 7] Executing health endpoint with correctly managed lifespan...", flush=True)
    from httpx import ASGITransport, AsyncClient
    from contextlib import asynccontextmanager

    app_instance = create_app()
    # ASGITransport with httpx AsyncClient manages lifespan when using LifespanManager or async with app.router.lifespan_context(app)
    async with app_instance.router.lifespan_context(app_instance):
        print("[DIAG 7] Lifespan started.", flush=True)
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
            print(f"[DIAG 7 OK] Status: {resp.status_code}, Body: {resp.json()} (took {time.time()-t0:.3f}s)", flush=True)
    print("[DIAG 7] Lifespan shutdown completed.", flush=True)

asyncio.run(diag_7())

print("\n" + "=" * 60, flush=True)
print("ALL DIAGNOSTICS 1 THROUGH 7 PASSED SUCCESSFULLY!", flush=True)
print("=" * 60, flush=True)
