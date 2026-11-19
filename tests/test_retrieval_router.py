"""
tests/test_retrieval_router.py
─────────────────────────────────────────────────────────────
Comprehensive REST API tests for semantic retrieval endpoint:
    POST /api/collections/{collection_id}/query

Verifies:
  1.  Successful semantic retrieval.
  2.  Successful retrieval with custom top_k.
  3.  Successful retrieval with a similarity threshold.
  4.  Empty results for empty collections.
  5.  Invalid collection UUID in URL path.
  6.  Nonexistent collection in SQLite.
  7.  Empty query string rejected.
  8.  Whitespace-only query string rejected.
  9.  Query exceeding 2048 characters rejected.
  10. top_k below minimum (< 1) rejected.
  11. top_k above maximum (> 100) rejected.
  12. Invalid similarity threshold rejected.
  13. Collection ID mismatch attempt in request body rejected.
  14. RetrievalService exception handling and error sanitization.
  15. Response schema correctness and provenance verification.
  16. Dependency override behavior.
  17. No accidental cross-collection retrieval.
  18. Candidate oversampling behavior (F-04).
─────────────────────────────────────────────────────────────
"""

import io
import math
import uuid
from typing import List, Optional
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient

from backend.config import Settings
from backend.models.collection import Collection
from backend.models.document import Document
from backend.routers.retrieval import get_retrieval_service
from backend.schemas.chunk import ChunkMetadata, DocumentChunk
from backend.schemas.document import DocumentStatus
from backend.schemas.embedding import EmbeddedChunk
from backend.schemas.retrieval import RetrievalQuery, RetrievalResponse, RetrievedChunk
from backend.services.retrieval_service import (
    RetrievalExecutionError,
    RetrievalService,
)
from backend.services.vector_store import VectorStoreService


# ── Test Helpers & Fast Deterministic Embedder ──────────────────────────────


def make_dummy_embedding(dim: int = 384, seed: int = 0) -> List[float]:
    """Generate a deterministic unit-normalized vector for testing."""
    raw = [float(((i + seed) % 17) + 1) for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


class MockEmbeddingService:
    """Fast deterministic mock of EmbeddingService to avoid loading HuggingFace weights."""

    def __init__(self, dimension: int = 384, default_seed: int = 42):
        self.dimension = dimension
        self.default_seed = default_seed
        self.vector_override: Optional[List[float]] = None

    def embed_text(self, text: str) -> List[float]:
        if self.vector_override is not None:
            return self.vector_override
        seed = abs(hash(text)) % 1000
        return make_dummy_embedding(dim=self.dimension, seed=seed)


def make_embedded_chunk(
    chunk_index: int = 0,
    document_id: str = "doc_001",
    collection_id: str = "col_001",
    filename: str = "regulations.pdf",
    file_type: str = "pdf",
    chunk_text: Optional[str] = None,
    page_number: Optional[int] = 1,
    page_range: Optional[str] = None,
    dimension: int = 384,
    version: int = 1,
    seed: Optional[int] = None,
) -> EmbeddedChunk:
    """Helper to construct an EmbeddedChunk with consistent metadata."""
    text = chunk_text or f"Sample chunk text content at index {chunk_index}."
    effective_seed = seed if seed is not None else chunk_index
    emb = make_dummy_embedding(dim=dimension, seed=effective_seed)

    meta = ChunkMetadata(
        document_id=document_id,
        collection_id=collection_id,
        filename=filename,
        file_type=file_type,
        chunk_index=chunk_index,
        char_count=len(text),
        page_number=page_number,
        page_range=page_range,
        start_char=chunk_index * 100,
        end_char=chunk_index * 100 + len(text),
        version=version,
    )

    doc_chunk = DocumentChunk(
        chunk_id=f"{document_id}_v{version}_c{chunk_index:04d}",
        document_id=document_id,
        collection_id=collection_id,
        filename=filename,
        file_type=file_type,
        chunk_index=chunk_index,
        chunk_text=text,
        char_count=len(text),
        page_number=page_number,
        page_range=page_range,
        metadata=meta,
        version=version,
    )

    return EmbeddedChunk.from_document_chunk(
        chunk=doc_chunk,
        embedding=emb,
        embedding_dimension=dimension,
        model_name="test-model-v1",
    )


async def create_collection_via_api(client: AsyncClient, name: str = "Test Collection") -> str:
    """Helper to create a collection via the REST API and return its UUID string."""
    resp = await client.post("/api/collections", json={"name": name, "description": "Test description"})
    assert resp.status_code == 201, f"Failed to create collection: {resp.text}"
    return resp.json()["id"]


async def upload_document_via_api(
    client: AsyncClient,
    collection_id: str,
    filename: str = "handbook.txt",
    content: Optional[bytes] = None,
) -> str:
    """Helper to upload a document via the REST API, set status=INDEXED, and return document UUID."""
    if content is None:
        if filename.endswith(".pdf"):
            content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n>>\nendobj\ntrailer\n<<\n>>\n%%EOF"
            mime = "application/pdf"
        else:
            content = b"Important university policies and guidelines.\n\nGrading criteria and academic standards."
            mime = "text/plain"
    else:
        mime = "application/pdf" if filename.endswith(".pdf") else "text/plain"

    files = {"file": (filename, io.BytesIO(content), mime)}
    resp = await client.post(f"/api/collections/{collection_id}/documents", files=files)
    assert resp.status_code == 201, f"Failed to upload document: {resp.text}"
    doc_id = resp.json()["id"]

    # Mark document as INDEXED in SQLite so RetrievalService validation gate accepts it
    app = client._transport.app
    async with app.state.session_factory() as session:
        doc = await session.get(Document, doc_id)
        assert doc is not None
        doc.status = DocumentStatus.INDEXED.value
        await session.commit()

    return doc_id


@pytest.fixture
def mock_retrieval_setup(client: AsyncClient, test_vector_store: VectorStoreService, test_settings: Settings):
    """Wires a fast, deterministic RetrievalService into client's FastAPI app."""
    mock_embed = MockEmbeddingService(dimension=384)
    service = RetrievalService(
        embedding_service=mock_embed,
        vector_store=test_vector_store,
        settings=test_settings,
    )
    app = client._transport.app
    app.dependency_overrides[get_retrieval_service] = lambda: service
    yield mock_embed, service
    app.dependency_overrides.pop(get_retrieval_service, None)


# ── Test Suite ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_1_successful_semantic_retrieval(
    client: AsyncClient, test_vector_store: VectorStoreService, mock_retrieval_setup
):
    """1. Successful semantic retrieval returns HTTP 200 with matching chunks and accurate schema."""
    mock_embed, _ = mock_retrieval_setup
    col_id = await create_collection_via_api(client, name="Col 1")
    doc_id = await upload_document_via_api(client, col_id, filename="regulations.txt")

    chunk = make_embedded_chunk(chunk_index=0, document_id=doc_id, collection_id=col_id, version=1, seed=10)
    test_vector_store.store_chunks([chunk], collection_id=col_id)

    # Set mock embedder to return chunk's embedding
    mock_embed.vector_override = chunk.embedding

    resp = await client.post(
        f"/api/collections/{col_id}/query",
        json={"query": "university regulations", "top_k": 5},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["query"] == "university regulations"
    assert data["collection_id"] == col_id
    assert data["returned_results"] == 1
    assert data["total_candidates"] == 1
    assert len(data["results"]) == 1

    first = data["results"][0]
    assert first["chunk_id"] == chunk.chunk_id
    assert first["document_id"] == doc_id
    assert first["collection_id"] == col_id
    assert first["similarity_score"] > 0.99
    assert first["filename"] is not None
    assert first["file_type"] is not None


@pytest.mark.asyncio
async def test_2_successful_retrieval_with_custom_top_k(
    client: AsyncClient, test_vector_store: VectorStoreService, mock_retrieval_setup
):
    """2. top_k parameter bounds the number of returned chunks."""
    mock_embed, _ = mock_retrieval_setup
    col_id = await create_collection_via_api(client, name="Col 2")
    doc_id = await upload_document_via_api(client, col_id, filename="multi_chunk.txt")

    chunks = [
        make_embedded_chunk(chunk_index=i, document_id=doc_id, collection_id=col_id, version=1, seed=i)
        for i in range(5)
    ]
    test_vector_store.store_chunks(chunks, collection_id=col_id)
    mock_embed.vector_override = chunks[0].embedding

    resp = await client.post(
        f"/api/collections/{col_id}/query",
        json={"query": "multi chunk query", "top_k": 2},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["returned_results"] == 2
    assert len(data["results"]) == 2
    assert data["retrieval_metadata"]["top_k"] == 2


@pytest.mark.asyncio
async def test_3_successful_retrieval_with_similarity_threshold(
    client: AsyncClient, test_vector_store: VectorStoreService, mock_retrieval_setup
):
    """3. similarity_threshold filters out candidates with similarity score below threshold."""
    mock_embed, _ = mock_retrieval_setup
    col_id = await create_collection_via_api(client, name="Col 3")
    doc_id = await upload_document_via_api(client, col_id, filename="thresh.txt")

    # High match seed=0, low match seed=100
    c_match = make_embedded_chunk(chunk_index=0, document_id=doc_id, collection_id=col_id, version=1, seed=0)
    c_low = make_embedded_chunk(chunk_index=1, document_id=doc_id, collection_id=col_id, version=1, seed=100)
    test_vector_store.store_chunks([c_match, c_low], collection_id=col_id)

    mock_embed.vector_override = c_match.embedding

    resp = await client.post(
        f"/api/collections/{col_id}/query",
        json={"query": "exact match query", "similarity_threshold": 0.95},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["returned_results"] == 1
    assert data["results"][0]["chunk_id"] == c_match.chunk_id


@pytest.mark.asyncio
async def test_4_empty_results(client: AsyncClient, mock_retrieval_setup):
    """4. Querying an empty collection returns HTTP 200 with an empty results list."""
    col_id = await create_collection_via_api(client, name="Col 4")

    resp = await client.post(
        f"/api/collections/{col_id}/query",
        json={"query": "find anything"},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["results"] == []
    assert data["returned_results"] == 0
    assert data["total_candidates"] == 0


@pytest.mark.asyncio
async def test_5_invalid_collection_uuid(client: AsyncClient):
    """5. Non-UUID collection_id in URL path returns HTTP 422 Unprocessable Entity."""
    resp = await client.post(
        "/api/collections/not-a-valid-uuid/query",
        json={"query": "hello"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_6_nonexistent_collection(client: AsyncClient):
    """6. Nonexistent collection UUID in SQLite returns HTTP 404 Not Found."""
    random_col_id = str(uuid.uuid4())
    resp = await client.post(
        f"/api/collections/{random_col_id}/query",
        json={"query": "hello"},
    )
    assert resp.status_code == 404
    assert f"Collection '{random_col_id}' not found." in resp.json()["detail"]


@pytest.mark.asyncio
async def test_7_empty_query(client: AsyncClient):
    """7. Empty string query returns HTTP 422 Unprocessable Entity."""
    col_id = await create_collection_via_api(client, name="Col 7")
    resp = await client.post(
        f"/api/collections/{col_id}/query",
        json={"query": ""},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_8_whitespace_only_query(client: AsyncClient):
    """8. Whitespace-only query string returns HTTP 422 Unprocessable Entity."""
    col_id = await create_collection_via_api(client, name="Col 8")
    resp = await client.post(
        f"/api/collections/{col_id}/query",
        json={"query": "   \t\n  "},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_9_query_exceeding_2048_characters(client: AsyncClient):
    """9. Query exceeding 2048 characters returns HTTP 422 Unprocessable Entity."""
    col_id = await create_collection_via_api(client, name="Col 9")
    long_query = "a" * 2049
    resp = await client.post(
        f"/api/collections/{col_id}/query",
        json={"query": long_query},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_10_top_k_below_minimum(client: AsyncClient):
    """10. top_k < 1 returns HTTP 422 Unprocessable Entity."""
    col_id = await create_collection_via_api(client, name="Col 10")
    resp = await client.post(
        f"/api/collections/{col_id}/query",
        json={"query": "valid query", "top_k": 0},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_11_top_k_above_maximum(client: AsyncClient):
    """11. top_k > 100 returns HTTP 422 Unprocessable Entity."""
    col_id = await create_collection_via_api(client, name="Col 11")
    resp = await client.post(
        f"/api/collections/{col_id}/query",
        json={"query": "valid query", "top_k": 101},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_12_invalid_similarity_threshold(client: AsyncClient):
    """12. similarity_threshold outside [-1.0, 1.0] returns HTTP 422 Unprocessable Entity."""
    col_id = await create_collection_via_api(client, name="Col 12")

    # > 1.0
    resp_high = await client.post(
        f"/api/collections/{col_id}/query",
        json={"query": "valid query", "similarity_threshold": 1.5},
    )
    assert resp_high.status_code == 422

    # < -1.0
    resp_low = await client.post(
        f"/api/collections/{col_id}/query",
        json={"query": "valid query", "similarity_threshold": -1.5},
    )
    assert resp_low.status_code == 422


@pytest.mark.asyncio
async def test_13_collection_id_mismatch_attempt_in_request_body(client: AsyncClient):
    """13. Attempting to supply a conflicting collection_id in the body returns HTTP 422."""
    col_id_path = await create_collection_via_api(client, name="Col 13")
    col_id_body = str(uuid.uuid4())

    resp = await client.post(
        f"/api/collections/{col_id_path}/query",
        json={"query": "valid query", "collection_id": col_id_body},
    )
    assert resp.status_code == 422
    assert "does not match authoritative collection ID in URL path" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_14_retrieval_service_exception_handling(client: AsyncClient, mock_retrieval_setup):
    """14. RetrievalService execution failure returns HTTP 500 without leaking stack traces or internal paths."""
    _, service = mock_retrieval_setup
    col_id = await create_collection_via_api(client, name="Col 14")

    with patch.object(
        service,
        "retrieve",
        side_effect=RetrievalExecutionError("Internal vector database lock failure on /internal/data/path"),
    ):
        resp = await client.post(
            f"/api/collections/{col_id}/query",
            json={"query": "valid query"},
        )

    assert resp.status_code == 500
    data = resp.json()
    assert data["detail"] == "Retrieval service encountered an internal execution error."
    assert "/internal/data/path" not in str(data)


@pytest.mark.asyncio
async def test_15_response_schema_correctness(
    client: AsyncClient, test_vector_store: VectorStoreService, mock_retrieval_setup
):
    """15. Response adheres strictly to RetrievalResponse schema and contains complete provenance."""
    mock_embed, _ = mock_retrieval_setup
    col_id = await create_collection_via_api(client, name="Col 15")
    doc_id = await upload_document_via_api(client, col_id, filename="provenance.pdf")

    chunk = make_embedded_chunk(
        chunk_index=0,
        document_id=doc_id,
        collection_id=col_id,
        filename="provenance.pdf",
        page_number=3,
        page_range="3-4",
        version=1,
    )
    test_vector_store.store_chunks([chunk], collection_id=col_id)
    mock_embed.vector_override = chunk.embedding

    resp = await client.post(
        f"/api/collections/{col_id}/query",
        json={"query": "provenance query"},
    )

    assert resp.status_code == 200
    data = resp.json()

    # Validate against Pydantic schema
    validated_response = RetrievalResponse.model_validate(data)
    assert validated_response.query == "provenance query"
    assert validated_response.collection_id == col_id
    assert validated_response.returned_results == 1
    assert validated_response.total_candidates == 1

    r0 = validated_response.results[0]
    assert r0.chunk_id == chunk.chunk_id
    assert r0.document_id == doc_id
    assert r0.page_number == 3
    assert r0.page_range == "3-4"
    assert r0.version == 1
    assert r0.filename == "provenance.pdf"


@pytest.mark.asyncio
async def test_16_dependency_override_behavior(client: AsyncClient):
    """16. Endpoint supports overriding get_retrieval_service via FastAPI dependency_overrides."""
    col_id = await create_collection_via_api(client, name="Col 16")

    mock_service = AsyncMock(spec=RetrievalService)
    mock_service.retrieve.return_value = RetrievalResponse(
        query="mock query",
        collection_id=col_id,
        results=[],
        total_candidates=99,
        returned_results=0,
        rejected_candidates=99,
        accepted_candidates=0,
        retrieval_metadata={"mocked": True},
    )

    app = client._transport.app
    app.dependency_overrides[get_retrieval_service] = lambda: mock_service
    try:
        resp = await client.post(
            f"/api/collections/{col_id}/query",
            json={"query": "mock query"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_candidates"] == 99
        assert data["retrieval_metadata"]["mocked"] is True
        mock_service.retrieve.assert_awaited_once()
    finally:
        app.dependency_overrides.pop(get_retrieval_service, None)


@pytest.mark.asyncio
async def test_17_no_accidental_cross_collection_retrieval(
    client: AsyncClient, test_vector_store: VectorStoreService, mock_retrieval_setup
):
    """17. Chunks belonging to Collection B are never returned when querying Collection A."""
    mock_embed, _ = mock_retrieval_setup
    col_a = await create_collection_via_api(client, name="Col 17A")
    col_b = await create_collection_via_api(client, name="Col 17B")

    doc_b = await upload_document_via_api(client, col_b, filename="secret_b.txt")
    chunk_b = make_embedded_chunk(chunk_index=0, document_id=doc_b, collection_id=col_b, version=1, seed=42)
    test_vector_store.store_chunks([chunk_b], collection_id=col_b)

    mock_embed.vector_override = chunk_b.embedding

    # Query Collection A using chunk B's embedding
    resp = await client.post(
        f"/api/collections/{col_a}/query",
        json={"query": "give me col B secret"},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["results"] == []
    assert data["returned_results"] == 0


@pytest.mark.asyncio
async def test_18_candidate_oversampling_behavior(
    client: AsyncClient, test_vector_store: VectorStoreService, mock_retrieval_setup
):
    """18. Verifies candidate oversampling queries 3x top_k from vector store to satisfy top_k after filtering (F-04)."""
    mock_embed, service = mock_retrieval_setup
    col_id = await create_collection_via_api(client, name="Col 18")
    doc_id = await upload_document_via_api(client, col_id, filename="oversample.txt")

    # Store 3 chunks
    chunks = [
        make_embedded_chunk(chunk_index=i, document_id=doc_id, collection_id=col_id, version=1, seed=i)
        for i in range(3)
    ]
    test_vector_store.store_chunks(chunks, collection_id=col_id)
    mock_embed.vector_override = chunks[0].embedding

    query_spy_calls = []
    orig_query = test_vector_store.query_vectors

    def query_vectors_spy(*args, **kwargs):
        query_spy_calls.append(kwargs)
        return orig_query(*args, **kwargs)

    with patch.object(test_vector_store, "query_vectors", side_effect=query_vectors_spy):
        resp = await client.post(
            f"/api/collections/{col_id}/query",
            json={"query": "oversampling test", "top_k": 2},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["returned_results"] == 2

    # Verify candidate_k was computed as min(max(2 * 3, 2), 100) = 6
    assert len(query_spy_calls) == 1
    assert query_spy_calls[0]["top_k"] == 6
