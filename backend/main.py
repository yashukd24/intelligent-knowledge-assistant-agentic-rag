"""
backend/main.py
─────────────────────────────────────────────────────────────
FastAPI application entrypoint.

Start the server with:
    uvicorn backend.main:app --host 127.0.0.1 --port 8000 --workers 1

Process Concurrency Boundary:
    The application runs in single-process mode (workers=1). Embedded ChromaDB
    (PersistentClient) requires a single process per data directory.
    Concurrent async requests within the process are fully supported.

API documentation is auto-generated at:
    http://127.0.0.1:8000/docs   (Swagger UI)
    http://127.0.0.1:8000/redoc  (ReDoc)
─────────────────────────────────────────────────────────────
"""

import asyncio
import logging
import logging.config
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import get_settings
from backend.database import close_db, init_db
from backend.routers.agent import router as agent_router
from backend.routers.categories import router as categories_router
from backend.routers.chat_history import router as chat_history_router
from backend.routers.evaluation import router as evaluation_router
from backend.routers.collections import router as collections_router
from backend.routers.documents import collection_docs_router, docs_router
from backend.routers.health import router as health_router
from backend.routers.rag import router as rag_router
from backend.routers.rag_answer import router as rag_answer_router
from backend.routers.retrieval import router as retrieval_router


def _configure_logging(settings) -> None:
    """
    Set up structured logging to console and rotating file.
    Called once at startup before any logger is used.
    """
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "backend.log", encoding="utf-8"),
        ],
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Startup:
      1. Configure logging.
      2. Ensure required directories exist.
      3. Initialize SQLite tables.

    Shutdown:
      1. Close database engine.
    """
    settings = get_settings()

    # ── Startup ──────────────────────────────────────────────
    _configure_logging(settings)
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)
    logger.info("Claude model : %s", settings.claude_model)
    logger.info("Database     : %s", settings.database_url)
    logger.info("=" * 60)

    # Ensure runtime directories exist
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.log_dir).mkdir(parents=True, exist_ok=True)

    # Initialize database tables
    await init_db()

    # Bounded startup recovery for any pending document deletions
    STARTUP_RECOVERY_TIMEOUT_SECONDS = 5.0
    STARTUP_RECOVERY_MAX_DOCUMENTS = 50

    try:
        from backend.database import AsyncSessionLocal
        from backend.services.document_recovery import DocumentRecoveryService

        async with asyncio.timeout(STARTUP_RECOVERY_TIMEOUT_SECONDS):
            recovery_svc = DocumentRecoveryService(session_factory=AsyncSessionLocal)
            rec_report = await recovery_svc.recover_pending_deletions(limit=STARTUP_RECOVERY_MAX_DOCUMENTS)

        skipped_count = max(0, rec_report.total_found - len(rec_report.succeeded) - len(rec_report.failed))
        if rec_report.total_found > 0:
            logger.info(
                "Startup deletion recovery summary: %d found, %d recovered, %d failed, %d skipped.",
                rec_report.total_found,
                len(rec_report.succeeded),
                len(rec_report.failed),
                skipped_count,
            )
            if rec_report.failed:
                logger.warning("Startup recovery failed for documents: %s", rec_report.failed)
    except TimeoutError:
        logger.warning(
            "Startup deletion recovery exceeded timeout limit (%.1fs). Server startup proceeded without blocking.",
            STARTUP_RECOVERY_TIMEOUT_SECONDS,
        )
    except Exception as rec_err:
        logger.warning("Startup deletion recovery encountered an error: %s", rec_err)

    # Pre-warm sentence-transformers embedding model to eliminate first-request latency
    try:
        from backend.services.embedding_service import EmbeddingService
        emb_svc = EmbeddingService()
        await asyncio.to_thread(emb_svc._get_model)
        logger.info("EmbeddingService model 'BAAI/bge-small-en-v1.5' pre-warmed successfully.")
    except Exception as emb_err:
        logger.warning("EmbeddingService pre-warm encountered: %s", emb_err)

    yield  # ← Application runs here

    # ── Shutdown ─────────────────────────────────────────────
    logger.info("Shutting down %s...", settings.app_name)
    await close_db()
    logger.info("Shutdown complete.")


# ── Build FastAPI application ────────────────────────────────
def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=settings.app_description,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ── CORS ─────────────────────────────────────────────────
    # Allows the Streamlit frontend (port 8501) to call the backend (port 8000)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:8501",
            "http://127.0.0.1:8501",
            "http://localhost:3000",   # kept for future use
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Global exception handler ─────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logging.getLogger(__name__).error(
            "Unhandled exception on %s %s: %s",
            request.method,
            request.url.path,
            exc,
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "type": type(exc).__name__},
        )

    # ── Routers ──────────────────────────────────────────────
    app.include_router(health_router)
    app.include_router(collections_router)
    app.include_router(collection_docs_router)
    app.include_router(docs_router)
    app.include_router(retrieval_router)
    app.include_router(rag_router)
    app.include_router(rag_answer_router)
    app.include_router(agent_router)
    app.include_router(categories_router)
    app.include_router(chat_history_router)
    app.include_router(evaluation_router)

    return app


app = create_app()
