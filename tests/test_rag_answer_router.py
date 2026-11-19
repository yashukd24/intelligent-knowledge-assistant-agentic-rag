"""
tests/test_rag_answer_router.py
─────────────────────────────────────────────────────────────
Comprehensive REST API tests for grounded RAG answer endpoint:
    POST /api/collections/{collection_id}/rag/answer

Verifies:
  1. Successful grounded answer generation (200 OK).
  2. Insufficient evidence handling (200 OK, status="insufficient_evidence").
  3. Nonexistent collection in SQLite (404 Not Found).
  4. Invalid UUID in path (422 Unprocessable Entity).
  5. Empty or whitespace query rejected (422).
  6. Query exceeding 2048 characters rejected (422).
  7. Collection ID mismatch in request body rejected (422).
  8. Parameter boundary violations rejected (422).
  9. Timeout mapped to 504 Gateway Timeout.
  10. Rate limit mapped to 503 Service Unavailable.
  11. Provider error mapped to 502 Bad Gateway.
  12. Internal execution error mapped to 500.
─────────────────────────────────────────────────────────────
"""

from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4
import pytest
from httpx import AsyncClient

from backend.routers.rag_answer import get_rag_answer_service
from backend.schemas.rag import SourceReference
from backend.schemas.rag_answer import (
    AnswerStatus,
    RAGAnswerRequest,
    RAGAnswerResponse,
)
from backend.services.llm_provider import TokenUsage
from backend.services.rag_answer_service import RAGAnswerService
from backend.services.rag_service import RAGExecutionError
from backend.services.vector_store import CollectionNotFoundError


async def _create_collection_via_api(client: AsyncClient, name: str = "RAG Answer Col") -> str:
    resp = await client.post("/api/collections", json={"name": name, "description": "Testing answers"})
    assert resp.status_code == 201, f"Failed to create collection: {resp.text}"
    return resp.json()["id"]


@pytest.mark.asyncio
async def test_successful_rag_answer_endpoint(client: AsyncClient):
    col_id_str = await _create_collection_via_api(client, "Answer Success Col")

    mock_resp = RAGAnswerResponse(
        collection_id=col_id_str,
        query="What is the admission policy?",
        status=AnswerStatus.ANSWERED,
        answer="The admission policy requires a minimum GPA of 3.0 [S1].",
        sources=[
            SourceReference(
                source_id="[S1]",
                document_id=str(uuid4()),
                chunk_id=str(uuid4()),
                filename="admissions.pdf",
                page_number=2,
                chunk_index=0,
                similarity_score=0.92,
                char_count=120,
            )
        ],
        valid_citations=["[S1]"],
        invalid_citations=[],
        uncited_sources=[],
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        usage=TokenUsage(input_tokens=150, output_tokens=25, total_tokens=175),
        generation_latency_ms=450.0,
        total_latency_ms=620.0,
        retrieval_metadata={"candidate_count": 1},
        error_message=None,
    )

    mock_svc = MagicMock(spec=RAGAnswerService)
    mock_svc.answer_question = AsyncMock(return_value=mock_resp)

    client._transport.app.dependency_overrides[get_rag_answer_service] = lambda: mock_svc

    payload = {
        "query": "What is the admission policy?",
        "top_k": 3,
        "max_context_chars": 8000,
    }
    resp = await client.post(f"/api/collections/{col_id_str}/rag/answer", json=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "answered"
    assert "The admission policy requires" in data["answer"]
    assert len(data["sources"]) == 1
    assert data["valid_citations"] == ["[S1]"]
    assert data["model"] == "claude-3-5-sonnet-20241022"
    assert data["usage"]["total_tokens"] == 175


@pytest.mark.asyncio
async def test_insufficient_evidence_endpoint(client: AsyncClient):
    col_id_str = await _create_collection_via_api(client, "Answer Empty Col")

    mock_resp = RAGAnswerResponse(
        collection_id=col_id_str,
        query="What is the weather on Mars?",
        status=AnswerStatus.INSUFFICIENT_EVIDENCE,
        answer="The provided documents do not contain sufficient information to answer this question.",
        sources=[],
        valid_citations=[],
        invalid_citations=[],
        uncited_sources=[],
        provider="system",
        model="none",
        usage=None,
        generation_latency_ms=0.0,
        total_latency_ms=12.0,
        retrieval_metadata={"candidate_count": 0},
        error_message=None,
    )

    mock_svc = MagicMock(spec=RAGAnswerService)
    mock_svc.answer_question = AsyncMock(return_value=mock_resp)

    client._transport.app.dependency_overrides[get_rag_answer_service] = lambda: mock_svc

    payload = {"query": "What is the weather on Mars?"}
    resp = await client.post(f"/api/collections/{col_id_str}/rag/answer", json=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "insufficient_evidence"
    assert "sufficient information" in data["answer"]
    assert data["sources"] == []
    assert data["valid_citations"] == []


@pytest.mark.asyncio
async def test_collection_not_found_404(client: AsyncClient):
    missing_id = uuid4()
    payload = {"query": "What are the rules?"}

    # No mock override needed here; default service will look up in SQLite and raise 404
    resp = await client.post(f"/api/collections/{missing_id}/rag/answer", json=payload)

    assert resp.status_code == 404
    data = resp.json()
    assert f"Collection '{missing_id}' not found" in data["detail"]


@pytest.mark.asyncio
async def test_invalid_uuid_path_422(client: AsyncClient):
    resp = await client.post("/api/collections/not-a-valid-uuid/rag/answer", json={"query": "test"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_empty_query_422(client: AsyncClient):
    col_id = await _create_collection_via_api(client, "Validation Col 1")
    resp = await client.post(f"/api/collections/{col_id}/rag/answer", json={"query": ""})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_whitespace_query_422(client: AsyncClient):
    col_id = await _create_collection_via_api(client, "Validation Col 2")
    resp = await client.post(f"/api/collections/{col_id}/rag/answer", json={"query": "   \n\t  "})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_query_too_long_422(client: AsyncClient):
    col_id = await _create_collection_via_api(client, "Validation Col 3")
    resp = await client.post(f"/api/collections/{col_id}/rag/answer", json={"query": "A" * 2049})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_collection_id_mismatch_422(client: AsyncClient):
    col_id = await _create_collection_via_api(client, "Mismatch Col")
    other_id = str(uuid4())

    payload = {
        "collection_id": other_id,
        "query": "Valid query here",
    }
    resp = await client.post(f"/api/collections/{col_id}/rag/answer", json=payload)
    assert resp.status_code == 422
    assert "does not match" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_invalid_top_k_422(client: AsyncClient):
    col_id = await _create_collection_via_api(client, "Invalid TopK Col")
    resp = await client.post(f"/api/collections/{col_id}/rag/answer", json={"query": "test", "top_k": 0})
    assert resp.status_code == 422

    resp2 = await client.post(f"/api/collections/{col_id}/rag/answer", json={"query": "test", "top_k": 101})
    assert resp2.status_code == 422


@pytest.mark.asyncio
async def test_invalid_similarity_threshold_422(client: AsyncClient):
    col_id = await _create_collection_via_api(client, "Invalid Threshold Col")
    resp = await client.post(
        f"/api/collections/{col_id}/rag/answer",
        json={"query": "test", "similarity_threshold": 1.5},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_invalid_max_context_chars_422(client: AsyncClient):
    col_id = await _create_collection_via_api(client, "Invalid Chars Col")
    resp = await client.post(
        f"/api/collections/{col_id}/rag/answer",
        json={"query": "test", "max_context_chars": 100},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_timeout_error_maps_to_504(client: AsyncClient):
    col_id = await _create_collection_via_api(client, "Timeout Col")

    mock_resp = RAGAnswerResponse(
        collection_id=col_id,
        query="Timeout question",
        status=AnswerStatus.TIMEOUT,
        answer="Request to LLM provider timed out.",
        sources=[],
        valid_citations=[],
        invalid_citations=[],
        uncited_sources=[],
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        usage=None,
        generation_latency_ms=30000.0,
        total_latency_ms=30050.0,
        retrieval_metadata={},
        error_message="Operation timed out after 30.0s",
    )

    mock_svc = MagicMock(spec=RAGAnswerService)
    mock_svc.answer_question = AsyncMock(return_value=mock_resp)

    client._transport.app.dependency_overrides[get_rag_answer_service] = lambda: mock_svc

    resp = await client.post(f"/api/collections/{col_id}/rag/answer", json={"query": "Timeout question"})
    assert resp.status_code == 504
    assert "timed out" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_rate_limit_error_maps_to_503(client: AsyncClient):
    col_id = await _create_collection_via_api(client, "RateLimit Col")

    mock_resp = RAGAnswerResponse(
        collection_id=col_id,
        query="Rate limit question",
        status=AnswerStatus.RATE_LIMITED,
        answer="Rate limit exceeded.",
        sources=[],
        valid_citations=[],
        invalid_citations=[],
        uncited_sources=[],
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        usage=None,
        generation_latency_ms=100.0,
        total_latency_ms=120.0,
        retrieval_metadata={},
        error_message="Anthropic 429 rate limit",
    )

    mock_svc = MagicMock(spec=RAGAnswerService)
    mock_svc.answer_question = AsyncMock(return_value=mock_resp)

    client._transport.app.dependency_overrides[get_rag_answer_service] = lambda: mock_svc

    resp = await client.post(f"/api/collections/{col_id}/rag/answer", json={"query": "Rate limit question"})
    assert resp.status_code == 503
    assert "rate limit" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_provider_error_maps_to_502(client: AsyncClient):
    col_id = await _create_collection_via_api(client, "ProviderError Col")

    mock_resp = RAGAnswerResponse(
        collection_id=col_id,
        query="Provider error question",
        status=AnswerStatus.PROVIDER_ERROR,
        answer="LLM provider error occurred.",
        sources=[],
        valid_citations=[],
        invalid_citations=[],
        uncited_sources=[],
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        usage=None,
        generation_latency_ms=200.0,
        total_latency_ms=220.0,
        retrieval_metadata={},
        error_message="Anthropic 500 error",
    )

    mock_svc = MagicMock(spec=RAGAnswerService)
    mock_svc.answer_question = AsyncMock(return_value=mock_resp)

    client._transport.app.dependency_overrides[get_rag_answer_service] = lambda: mock_svc

    resp = await client.post(f"/api/collections/{col_id}/rag/answer", json={"query": "Provider error question"})
    assert resp.status_code == 502
    assert "provider service error" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_rag_execution_error_maps_to_500(client: AsyncClient):
    col_id = await _create_collection_via_api(client, "InternalError Col")

    mock_svc = MagicMock(spec=RAGAnswerService)
    mock_svc.answer_question = AsyncMock(side_effect=RAGExecutionError("Corrupted index"))

    client._transport.app.dependency_overrides[get_rag_answer_service] = lambda: mock_svc

    resp = await client.post(f"/api/collections/{col_id}/rag/answer", json={"query": "Crash question"})
    assert resp.status_code == 500
    assert "internal execution error" in resp.json()["detail"].lower()
