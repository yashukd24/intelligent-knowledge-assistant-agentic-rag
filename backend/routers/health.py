"""
backend/routers/health.py
─────────────────────────────────────────────────────────────
Health check router.

Endpoints:
  GET /health   — full component status (API + database)
  GET /         — root redirect message
─────────────────────────────────────────────────────────────
"""

import logging
from datetime import UTC, datetime

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import get_settings
from backend.database import get_db
from backend.schemas.health import ComponentHealth, HealthResponse, ServiceStatus

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Health"])


@router.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    """Root endpoint — confirms the API is alive."""
    settings = get_settings()
    return {
        "message": f"Welcome to {settings.app_name} API",
        "version": settings.app_version,
        "docs": "/docs",
    }


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Application health check",
    description="Returns the operational status of all backend components.",
)
async def health_check(db: AsyncSession = Depends(get_db)) -> HealthResponse:
    """
    Checks:
    1. API layer — always OK if this endpoint responds.
    2. Database  — executes SELECT 1 against SQLite.

    Returns HTTP 200 with status='ok' when all components are healthy.
    Returns HTTP 200 with status='degraded' when some components have issues.
    (HTTP 503 is handled by the global exception handler in main.py for crashes.)
    """
    settings = get_settings()
    components: dict[str, ComponentHealth] = {}

    # ── 1. API component ─────────────────────────────────────
    components["api"] = ComponentHealth(
        status=ServiceStatus.OK,
        message="FastAPI is running",
    )

    # ── 2. Database component ────────────────────────────────
    try:
        await db.execute(text("SELECT 1"))
        components["database"] = ComponentHealth(
            status=ServiceStatus.OK,
            message="SQLite reachable",
        )
    except Exception as exc:
        logger.error("Database health check failed: %s", exc)
        components["database"] = ComponentHealth(
            status=ServiceStatus.DOWN,
            message=f"Database error: {exc}",
        )

    # ── Overall status ───────────────────────────────────────
    if any(c.status == ServiceStatus.DOWN for c in components.values()):
        overall = ServiceStatus.DOWN
    elif any(c.status == ServiceStatus.DEGRADED for c in components.values()):
        overall = ServiceStatus.DEGRADED
    else:
        overall = ServiceStatus.OK

    return HealthResponse(
        status=overall,
        app_name=settings.app_name,
        version=settings.app_version,
        timestamp=datetime.now(UTC),
        components=components,
    )
