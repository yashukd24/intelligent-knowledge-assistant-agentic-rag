"""
tests/test_health.py
─────────────────────────────────────────────────────────────
Tests for the health check endpoint.

Covers:
  T1  Root endpoint returns 200 and welcome message
  T2  Health endpoint returns 200
  T3  Health response contains 'status' field
  T4  Health status is 'ok'
  T5  Health response contains 'components' dict
  T6  API component is 'ok'
  T7  Database component is 'ok'
  T8  Health response contains app_name and version
  T9  Timestamp field is present in response
  T10 Response is valid JSON (implicit — httpx raises if not)
─────────────────────────────────────────────────────────────
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_root_returns_200(client: AsyncClient):
    """T1: Root endpoint responds with 200."""
    response = await client.get("/")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_root_contains_welcome(client: AsyncClient):
    """T1b: Root response contains welcome message."""
    response = await client.get("/")
    data = response.json()
    assert "message" in data
    assert "Knowledge Assistant" in data["message"]


@pytest.mark.asyncio
async def test_health_returns_200(client: AsyncClient):
    """T2: /health returns HTTP 200."""
    response = await client.get("/health")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_has_status_field(client: AsyncClient):
    """T3: /health response contains 'status' key."""
    response = await client.get("/health")
    assert "status" in response.json()


@pytest.mark.asyncio
async def test_health_status_is_ok(client: AsyncClient):
    """T4: Overall status is 'ok' when all components are healthy."""
    response = await client.get("/health")
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_health_has_components(client: AsyncClient):
    """T5: /health response contains 'components' dict."""
    response = await client.get("/health")
    data = response.json()
    assert "components" in data
    assert isinstance(data["components"], dict)


@pytest.mark.asyncio
async def test_health_api_component_ok(client: AsyncClient):
    """T6: 'api' component status is 'ok'."""
    response = await client.get("/health")
    components = response.json()["components"]
    assert "api" in components
    assert components["api"]["status"] == "ok"


@pytest.mark.asyncio
async def test_health_database_component_ok(client: AsyncClient):
    """T7: 'database' component status is 'ok' (in-memory SQLite)."""
    response = await client.get("/health")
    components = response.json()["components"]
    assert "database" in components
    assert components["database"]["status"] == "ok"


@pytest.mark.asyncio
async def test_health_contains_app_name_and_version(client: AsyncClient):
    """T8: /health response contains app_name and version."""
    response = await client.get("/health")
    data = response.json()
    assert "app_name" in data
    assert "version" in data
    assert data["app_name"] == "Intelligent Knowledge Assistant"
    assert data["version"] == "0.1.0"


@pytest.mark.asyncio
async def test_health_contains_timestamp(client: AsyncClient):
    """T9: /health response contains a timestamp field."""
    response = await client.get("/health")
    data = response.json()
    assert "timestamp" in data
    assert data["timestamp"] is not None


@pytest.mark.asyncio
async def test_docs_endpoint_accessible(client: AsyncClient):
    """T10: Swagger /docs is accessible (HTML response)."""
    response = await client.get("/docs")
    assert response.status_code == 200
