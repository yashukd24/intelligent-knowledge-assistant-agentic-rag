"""
tests/test_rag_router.py
─────────────────────────────────────────────────────────────
Comprehensive REST API tests for conventional RAG endpoint:
    POST /api/collections/{collection_id}/rag/context

Verifies:
  1. Successful RAG context and prompt assembly (200 OK).
  2. Custom top_k and max_context_chars parameter handling.
  3. Similarity threshold filtering.
  4. Empty retrieval producing 200 OK with clean no-evidence state.
  5. Invalid collection UUID in URL path (422).
  6. Nonexistent collection in SQLite (404).
  7. Empty query rejected (422).
  8. Whitespace-only query rejected (422).
  9. Query exceeding 2048 characters rejected (422).
  10. top_k below 1 or above 100 rejected (422).
  11. Similarity threshold out of [-1.0, 1.0] rejected (422).
  12. max_context_chars out of [500, 60000] rejected (422).
  13. Collection ID mismatch attempt in request body rejected (422).
  14. Service exception handling and error sanitization (500).
  15. Response schema correctness and source integrity.
  16. Dependency override behavior.
  17. Missing ChromaDB collection with valid SQLite collection (200 OK).
─────────────────────────────────────────────────────────────
"""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from httpx import AsyncClient
import pytest

from backend.routers.rag import get_rag_service
from backend.schemas.rag import RAGResponse, SourceReference
from backend.services.prompt_builder import PromptBuilder
from backend.services.rag_service import (
    RAGExecutionError,
    RAGService,
)
from backend.services.vector_store import CollectionNotFoundError


async def create_collection_via_api(client: AsyncClient, name: str = "Test RAG Collection") -> str:
    """Helper to create a collection via REST API and return its UUID string."""
    resp = await client.post("/api/collections", json={"name": name, "description": "RAG test collection"})
    assert resp.status_code == 201, f"Failed to create collection: {resp.text}"
    return resp.json()["id"]


@pytest.mark.asyncio
async def test_1_successful_rag_context_endpoint(client: AsyncClient):
    col_id = await create_collection_via_api(client, "RAG Success Col")

    payload = {
        "query": "What are the degree requirements?",
        "top_k": 3,
        "max_context_chars": 8000,
    }
    resp = await client.post(f"/api/collections/{col_id}/rag/context", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["query"] == "What are the degree requirements?"
    assert data["collection_id"] == col_id
    assert "assembled_context" in data
    assert "sources" in data
    assert "prompt_text" in data
    assert "=== SYSTEM INSTRUCTIONS ===" in data["prompt_text"]
    assert "=== USER QUESTION ===" in data["prompt_text"]


@pytest.mark.asyncio
async def test_2_rag_with_custom_top_k_and_context_budget(client: AsyncClient):
    col_id = await create_collection_via_api(client, "Budget Col")

    payload = {
        "query": "Summary of leave of absence",
        "top_k": 2,
        "max_context_chars": 2000,
    }
    resp = await client.post(f"/api/collections/{col_id}/rag/context", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["context_character_count"] <= 2000


@pytest.mark.asyncio
async def test_3_rag_with_similarity_threshold(client: AsyncClient):
    col_id = await create_collection_via_api(client, "Threshold Col")

    payload = {
        "query": "Academic integrity",
        "similarity_threshold": 0.50,
    }
    resp = await client.post(f"/api/collections/{col_id}/rag/context", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["retrieval_metadata"]["similarity_threshold"] == 0.50


@pytest.mark.asyncio
async def test_4_empty_retrieval_returns_200_with_empty_context(client: AsyncClient):
    col_id = await create_collection_via_api(client, "Empty Col")

    payload = {"query": "Nonexistent topic query"}
    resp = await client.post(f"/api/collections/{col_id}/rag/context", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["retrieved_chunk_count"] == 0
    assert data["assembled_context"] == ""
    assert data["sources"] == []
    assert "No relevant evidence was retrieved" in data["prompt_text"]


@pytest.mark.asyncio
async def test_5_invalid_collection_uuid_format(client: AsyncClient):
    resp = await client.post("/api/collections/not-a-valid-uuid/rag/context", json={"query": "Test query"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_6_nonexistent_collection_uuid(client: AsyncClient):
    missing_id = str(uuid4())
    resp = await client.post(f"/api/collections/{missing_id}/rag/context", json={"query": "Test query"})
    assert resp.status_code == 404
    assert f"Collection '{missing_id}' not found" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_7_empty_query_rejected(client: AsyncClient):
    col_id = await create_collection_via_api(client)
    resp = await client.post(f"/api/collections/{col_id}/rag/context", json={"query": ""})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_8_whitespace_only_query_rejected(client: AsyncClient):
    col_id = await create_collection_via_api(client)
    resp = await client.post(f"/api/collections/{col_id}/rag/context", json={"query": "   \n\t  "})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_9_query_exceeding_2048_characters_rejected(client: AsyncClient):
    col_id = await create_collection_via_api(client)
    long_query = "a" * 2049
    resp = await client.post(f"/api/collections/{col_id}/rag/context", json={"query": long_query})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_10_top_k_out_of_bounds_rejected(client: AsyncClient):
    col_id = await create_collection_via_api(client)
    # top_k < 1
    resp1 = await client.post(f"/api/collections/{col_id}/rag/context", json={"query": "Q", "top_k": 0})
    assert resp1.status_code == 422

    # top_k > 100
    resp2 = await client.post(f"/api/collections/{col_id}/rag/context", json={"query": "Q", "top_k": 101})
    assert resp2.status_code == 422


@pytest.mark.asyncio
async def test_11_similarity_threshold_out_of_bounds_rejected(client: AsyncClient):
    col_id = await create_collection_via_api(client)
    # Below -1.0
    resp1 = await client.post(
        f"/api/collections/{col_id}/rag/context",
        json={"query": "Q", "similarity_threshold": -1.5},
    )
    assert resp1.status_code == 422

    # Above 1.0
    resp2 = await client.post(
        f"/api/collections/{col_id}/rag/context",
        json={"query": "Q", "similarity_threshold": 1.5},
    )
    assert resp2.status_code == 422


@pytest.mark.asyncio
async def test_12_max_context_chars_out_of_bounds_rejected(client: AsyncClient):
    col_id = await create_collection_via_api(client)
    # Below 500
    resp1 = await client.post(
        f"/api/collections/{col_id}/rag/context",
        json={"query": "Q", "max_context_chars": 400},
    )
    assert resp1.status_code == 422

    # Above 60000
    resp2 = await client.post(
        f"/api/collections/{col_id}/rag/context",
        json={"query": "Q", "max_context_chars": 70000},
    )
    assert resp2.status_code == 422


@pytest.mark.asyncio
async def test_13_collection_id_mismatch_in_body_rejected(client: AsyncClient):
    col_id_1 = await create_collection_via_api(client, "Col 1")
    other_col_id = str(uuid4())

    resp = await client.post(
        f"/api/collections/{col_id_1}/rag/context",
        json={"query": "Valid query", "collection_id": other_col_id},
    )
    assert resp.status_code == 422
    assert "does not match authoritative collection ID in URL path" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_14_rag_service_execution_error_sanitized(client: AsyncClient):
    col_id = await create_collection_via_api(client, "Error Col")

    mock_rag_svc = MagicMock(spec=RAGService)
    mock_rag_svc.build_rag_context = AsyncMock(
        side_effect=RAGExecutionError("Internal database lock at /var/secret/path.db")
    )

    app = client._transport.app
    app.dependency_overrides[get_rag_service] = lambda: mock_rag_svc
    try:
        resp = await client.post(f"/api/collections/{col_id}/rag/context", json={"query": "Safe query"})
        assert resp.status_code == 500
        detail = resp.json()["detail"]
        assert "Internal database lock" not in detail
        assert "secret" not in detail
        assert "internal execution error" in detail
    finally:
        app.dependency_overrides.pop(get_rag_service, None)


@pytest.mark.asyncio
async def test_15_response_schema_correctness_and_source_integrity(client: AsyncClient):
    col_id = await create_collection_via_api(client, "Schema Col")

    mock_rag_svc = MagicMock(spec=RAGService)
    src = SourceReference(
        source_id="[S1]",
        document_id="doc_xyz",
        chunk_id="chunk_123",
        filename="course_catalog.pdf",
        page_number=2,
        page_range="2-3",
        chunk_index=0,
        similarity_score=0.925,
        char_count=100,
    )
    mock_resp = RAGResponse(
        query="What are prerequisites?",
        collection_id=col_id,
        retrieved_chunk_count=1,
        assembled_context="[S1] Prerequisite text",
        context_character_count=22,
        sources=[src],
        source_metadata={"[S1]": {"filename": "course_catalog.pdf", "similarity_score": 0.925}},
        prompt_text="=== SYSTEM INSTRUCTIONS ===\n[S1] Prerequisite text\n=== USER QUESTION ===\nWhat are prerequisites?",
        excluded_chunks=[],
        retrieval_metadata={"top_k": 5},
    )
    mock_rag_svc.build_rag_context = AsyncMock(return_value=mock_resp)

    app = client._transport.app
    app.dependency_overrides[get_rag_service] = lambda: mock_rag_svc
    try:
        resp = await client.post(
            f"/api/collections/{col_id}/rag/context",
            json={"query": "What are prerequisites?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["sources"][0]["source_id"] == "[S1]"
        assert data["sources"][0]["filename"] == "course_catalog.pdf"
        assert data["sources"][0]["page_range"] == "2-3"
        assert "[S1]" in data["source_metadata"]
    finally:
        app.dependency_overrides.pop(get_rag_service, None)


@pytest.mark.asyncio
async def test_16_dependency_override_behavior(client: AsyncClient):
    col_id = await create_collection_via_api(client, "Override Col")

    mock_rag_svc = MagicMock(spec=RAGService)
    mock_resp = RAGResponse(
        query="Override test",
        collection_id=col_id,
        retrieved_chunk_count=0,
        assembled_context="Overridden context",
        context_character_count=18,
        sources=[],
        source_metadata={},
        prompt_text="Overridden prompt",
        excluded_chunks=[],
        retrieval_metadata={},
    )
    mock_rag_svc.build_rag_context = AsyncMock(return_value=mock_resp)

    app = client._transport.app
    app.dependency_overrides[get_rag_service] = lambda: mock_rag_svc
    try:
        resp = await client.post(
            f"/api/collections/{col_id}/rag/context",
            json={"query": "Override test"},
        )
        assert resp.status_code == 200
        assert resp.json()["assembled_context"] == "Overridden context"
    finally:
        app.dependency_overrides.pop(get_rag_service, None)


@pytest.mark.asyncio
async def test_17_missing_chromadb_collection_with_valid_sqlite_collection(client: AsyncClient):
    col_id = await create_collection_via_api(client, "Chroma Missing Col")

    mock_rag_svc = MagicMock(spec=RAGService)
    mock_rag_svc.build_rag_context = AsyncMock(
        side_effect=CollectionNotFoundError(f"Collection {col_id} does not exist in ChromaDB")
    )
    mock_rag_svc.prompt_builder = PromptBuilder()

    app = client._transport.app
    app.dependency_overrides[get_rag_service] = lambda: mock_rag_svc
    try:
        resp = await client.post(
            f"/api/collections/{col_id}/rag/context",
            json={"query": "Query for empty collection"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["retrieved_chunk_count"] == 0
        assert data["assembled_context"] == ""
        assert data["retrieval_metadata"]["collection_empty"] is True
    finally:
        app.dependency_overrides.pop(get_rag_service, None)
