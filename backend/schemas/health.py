"""
backend/schemas/health.py
─────────────────────────────────────────────────────────────
Pydantic v2 response schemas for the health check endpoint.
─────────────────────────────────────────────────────────────
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class ServiceStatus(str, Enum):
    """Status of an individual backend service component."""
    OK = "ok"
    DEGRADED = "degraded"
    DOWN = "down"


class ComponentHealth(BaseModel):
    """Health status of a single system component."""
    status: ServiceStatus
    message: str = ""


class HealthResponse(BaseModel):
    """
    Response schema for GET /health.
    Returns overall application status plus per-component checks.
    """
    status: ServiceStatus = Field(description="Overall system status")
    app_name: str = Field(description="Application name")
    version: str = Field(description="Application version")
    timestamp: datetime = Field(description="UTC timestamp of the health check")
    components: dict[str, ComponentHealth] = Field(
        description="Per-component health details"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "ok",
                "app_name": "Intelligent Knowledge Assistant",
                "version": "0.1.0",
                "timestamp": "2026-11-05T07:00:00Z",
                "components": {
                    "api": {"status": "ok", "message": "FastAPI running"},
                    "database": {"status": "ok", "message": "SQLite reachable"},
                },
            }
        }
    }
