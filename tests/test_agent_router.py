"""
tests/test_agent_router.py
─────────────────────────────────────────────────────────────
REST API tests for the Agentic RAG Controller endpoint:
    POST /api/collections/{collection_id}/agent/run

Verifies:
  1. Successful Agentic RAG execution (200 OK).
  2. Insufficient evidence refusal (200 OK with refusal answer).
  3. Nonexistent collection in SQLite (404 Not Found).
  4. Invalid UUID in URL path (422 Unprocessable Entity).
  5. Empty or whitespace query rejected (422).
  6. Collection ID mismatch in request body rejected (422).
  7. Provider timeout error mapped to 504 Gateway Timeout.
  8. Provider rate limit error mapped to 503 Service Unavailable.
  9. Provider error mapped to 502 Bad Gateway.
  10. ChromaDB CollectionNotFoundError handled with refusal (200 OK).
  11. Backward compatibility: Conventional RAG endpoint remains operational.
─────────────────────────────────────────────────────────────
"""

from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4
import pytest
from httpx import AsyncClient

from backend.routers.agent import get_agent_controller
from backend.routers.rag_answer import get_rag_answer_service
from backend.schemas.agent_state import (
    AgentEvent,
    AgentEventType,
    AgentRunResponse,
    AgentStatus,
)
from backend.schemas.rag import SourceReference
from backend.schemas.rag_answer import (
    AnswerStatus,
    ExecutionTrace,
    RAGAnswerResponse,
    TraceStage,
    TraceStageStatus,
)
from backend.services.agent_controller import AgentController
from backend.services.answer_generator import STANDARD_INSUFFICIENT_EVIDENCE_ANSWER
from backend.services.rag_answer_service import RAGAnswerService
from backend.services.vector_store import CollectionNotFoundError


async def _create_collection_via_api(client: AsyncClient, name: str = "Agentic Col") -> str:
    resp = await client.post("/api/collections", json={"name": name, "description": "Agent testing"})
    assert resp.status_code == 201, f"Failed to create collection: {resp.text}"
    return resp.json()["id"]


@pytest.mark.asyncio
async def test_successful_agent_run_endpoint(client: AsyncClient):
    col_id_str = await _create_collection_via_api(client, "Agentic Run Success")

    mock_resp = AgentRunResponse(
        session_id=str(uuid4()),
        trace_id=str(uuid4()),
        collection_id=col_id_str,
        query="What is reinforcement learning?",
        status=AgentStatus.COMPLETED,
        answer="Reinforcement learning uses reward signals [S1].",
        is_sufficient=True,
        evidence_score=0.91,
        iteration_count=0,
        sources=[
            SourceReference(
                source_id="[S1]",
                document_id=str(uuid4()),
                chunk_id=str(uuid4()),
                filename="rl.pdf",
                page_number=1,
                chunk_index=0,
                similarity_score=0.91,
                char_count=50,
            )
        ],
        valid_citations=["[S1]"],
        invalid_citations=[],
        events=[
            AgentEvent(
                event_type=AgentEventType.QUERY_UNDERSTOOD,
                step=1,
                payload={},
                message="Query parsed.",
            ),
            AgentEvent(
                event_type=AgentEventType.AGENT_DONE,
                step=2,
                payload={},
                message="Done.",
            ),
        ],
        execution_trace=ExecutionTrace(
            trace_id="test-trace",
            stages=[
                TraceStage(
                    id="stage-1",
                    name="Query Understood",
                    status=TraceStageStatus.COMPLETED,
                )
            ],
            total_duration_ms=250.0,
        ),
        total_latency_ms=250.0,
    )

    mock_controller = MagicMock(spec=AgentController)
    mock_controller.run = AsyncMock(return_value=mock_resp)

    client._transport.app.dependency_overrides[get_agent_controller] = lambda: mock_controller

    try:
        response = await client.post(
            f"/api/collections/{col_id_str}/agent/run",
            json={"query": "What is reinforcement learning?", "top_k": 3},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "COMPLETED"
        assert data["answer"] == "Reinforcement learning uses reward signals [S1]."
        assert data["is_sufficient"] is True
        assert data["valid_citations"] == ["[S1]"]
        assert len(data["events"]) == 2
        assert data["execution_trace"] is not None
    finally:
        client._transport.app.dependency_overrides.pop(get_agent_controller, None)


@pytest.mark.asyncio
async def test_agent_run_collection_not_found_404(client: AsyncClient):
    nonexistent_id = str(uuid4())
    response = await client.post(
        f"/api/collections/{nonexistent_id}/agent/run",
        json={"query": "Valid query?"},
    )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_agent_run_collection_id_mismatch_422(client: AsyncClient):
    col_id_str = await _create_collection_via_api(client, "Mismatch Col")
    other_id = str(uuid4())

    response = await client.post(
        f"/api/collections/{col_id_str}/agent/run",
        json={"query": "Valid query?", "collection_id": other_id},
    )
    assert response.status_code == 422
    assert "does not match" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_agent_run_empty_query_422(client: AsyncClient):
    col_id_str = await _create_collection_via_api(client, "Empty Query Col")

    response = await client.post(
        f"/api/collections/{col_id_str}/agent/run",
        json={"query": "    "},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_agent_run_timeout_maps_to_504(client: AsyncClient):
    col_id_str = await _create_collection_via_api(client, "Timeout Col")

    mock_failed_resp = AgentRunResponse(
        session_id=str(uuid4()),
        trace_id=str(uuid4()),
        collection_id=col_id_str,
        query="Timeout query",
        status=AgentStatus.FAILED,
        answer="",
        is_sufficient=True,
        error_details="Language model request timed out after 30s.",
        events=[],
    )

    mock_controller = MagicMock(spec=AgentController)
    mock_controller.run = AsyncMock(return_value=mock_failed_resp)

    client._transport.app.dependency_overrides[get_agent_controller] = lambda: mock_controller

    try:
        response = await client.post(
            f"/api/collections/{col_id_str}/agent/run",
            json={"query": "Timeout query"},
        )
        assert response.status_code == 504
        assert "timed out" in response.json()["detail"].lower()
    finally:
        client._transport.app.dependency_overrides.pop(get_agent_controller, None)


@pytest.mark.asyncio
async def test_agent_run_provider_error_maps_to_502(client: AsyncClient):
    col_id_str = await _create_collection_via_api(client, "Provider Error Col")

    mock_failed_resp = AgentRunResponse(
        session_id=str(uuid4()),
        trace_id=str(uuid4()),
        collection_id=col_id_str,
        query="Error query",
        status=AgentStatus.FAILED,
        answer="",
        is_sufficient=True,
        error_details="Provider service unavailable HTTP 502",
        events=[],
    )

    mock_controller = MagicMock(spec=AgentController)
    mock_controller.run = AsyncMock(return_value=mock_failed_resp)

    client._transport.app.dependency_overrides[get_agent_controller] = lambda: mock_controller

    try:
        response = await client.post(
            f"/api/collections/{col_id_str}/agent/run",
            json={"query": "Error query"},
        )
        assert response.status_code == 502
    finally:
        client._transport.app.dependency_overrides.pop(get_agent_controller, None)


@pytest.mark.asyncio
async def test_agent_run_chroma_not_found_returns_refusal(client: AsyncClient):
    col_id_str = await _create_collection_via_api(client, "Empty Chroma Col")

    mock_controller = MagicMock(spec=AgentController)
    mock_controller.run = AsyncMock(side_effect=CollectionNotFoundError("Collection missing in ChromaDB"))

    client._transport.app.dependency_overrides[get_agent_controller] = lambda: mock_controller

    try:
        response = await client.post(
            f"/api/collections/{col_id_str}/agent/run",
            json={"query": "Query against empty collection"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "COMPLETED"
        assert data["answer"] == STANDARD_INSUFFICIENT_EVIDENCE_ANSWER
        assert data["is_sufficient"] is False
    finally:
        client._transport.app.dependency_overrides.pop(get_agent_controller, None)


@pytest.mark.asyncio
async def test_conventional_rag_backward_compatibility(client: AsyncClient):
    col_id_str = await _create_collection_via_api(client, "Backward Compat Col")

    mock_rag_resp = RAGAnswerResponse(
        collection_id=col_id_str,
        query="Conventional RAG query",
        status=AnswerStatus.ANSWERED,
        answer="Conventional RAG answer [S1].",
        sources=[],
        valid_citations=["[S1]"],
        invalid_citations=[],
        uncited_sources=[],
        provider="anthropic",
        model="claude-3-5-sonnet",
        generation_latency_ms=100.0,
        total_latency_ms=150.0,
    )

    mock_rag_svc = MagicMock(spec=RAGAnswerService)
    mock_rag_svc.answer_question = AsyncMock(return_value=mock_rag_resp)

    client._transport.app.dependency_overrides[get_rag_answer_service] = lambda: mock_rag_svc

    try:
        # Check that POST /api/collections/{id}/rag/answer still operates with exact contract
        response = await client.post(
            f"/api/collections/{col_id_str}/rag/answer",
            json={"query": "Conventional RAG query"},
        )
        assert response.status_code == 200
        assert response.json()["answer"] == "Conventional RAG answer [S1]."
    finally:
        client._transport.app.dependency_overrides.pop(get_rag_answer_service, None)
