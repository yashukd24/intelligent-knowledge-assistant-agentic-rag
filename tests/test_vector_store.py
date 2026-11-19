"""
tests/test_vector_store.py
─────────────────────────────────────────────────────────────
Comprehensive unit & integration tests for VectorStoreService.

Covers all 18 mandatory requirements:
  1.  Persistent client initialization
  2.  Collection creation
  3.  Collection retrieval
  4.  Correct cosine-distance configuration
  5.  384-dimensional vector insertion
  6.  Text and metadata preservation
  7.  Retrieval by chunk ID
  8.  Idempotent upsert
  9.  Document-specific vector deletion
  10. Collection count accuracy
  11. Invalid vector dimension rejection
  12. NaN and infinity rejection
  13. Empty text rejection
  14. Missing or invalid metadata handling (no None in Chroma)
  15. Partial insertion / error handling
  16. Persistence after client/service restart
  17. Collection isolation
  18. Repeated deletion behavior
  + Additional tests:
  19. Collection name generation and sanitization
  20. Batching with >50 chunks (e.g. 65 chunks)
  21. Entire collection deletion
  22. Collection listing
─────────────────────────────────────────────────────────────
"""

import math
from pathlib import Path
from typing import List, Optional
import uuid

import pytest

from backend.schemas.chunk import ChunkMetadata, DocumentChunk
from backend.schemas.embedding import EmbeddedChunk
from backend.schemas.retrieval import RetrievalQuery, RetrievalResponse, RetrievedChunk
from backend.services.vector_store import (
    CollectionNotFoundError,
    InvalidChunkDataError,
    InvalidVectorError,
    VectorDimensionMismatchError,
    VectorSearchResults,
    VectorStorageError,
    VectorStoreService,
)


# ── Test Helpers & Fixtures ──────────────────────────────────────────────────


def make_dummy_embedding(dim: int = 384, seed: int = 0) -> List[float]:
    """Generate a deterministic unit-normalized vector for testing."""
    raw = [float(((i + seed) % 17) + 1) for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


def make_embedded_chunk(
    chunk_index: int = 0,
    document_id: str = "doc_test_001",
    collection_id: str = "col_test_001",
    filename: str = "sample.pdf",
    file_type: str = "pdf",
    chunk_text: Optional[str] = None,
    page_number: Optional[int] = 1,
    page_range: Optional[str] = None,
    dimension: int = 384,
    embedding: Optional[List[float]] = None,
    model_name: str = "BAAI/bge-small-en-v1.5",
    seed: Optional[int] = None,
) -> EmbeddedChunk:
    """Helper to construct an EmbeddedChunk with consistent metadata."""
    text = chunk_text if chunk_text is not None else f"Sample chunk text content at index {chunk_index}."
    effective_seed = seed if seed is not None else chunk_index
    emb = embedding if embedding is not None else make_dummy_embedding(dim=dimension, seed=effective_seed)

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
    )

    doc_chunk = DocumentChunk(
        chunk_id=f"{document_id}_c{chunk_index:04d}",
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
    )

    return EmbeddedChunk.from_document_chunk(
        chunk=doc_chunk,
        embedding=emb,
        embedding_dimension=dimension,
        model_name=model_name,
    )


@pytest.fixture
def temp_chroma_dir(tmp_path: Path) -> Path:
    """Provides an isolated temporary directory for ChromaDB tests."""
    chroma_dir = tmp_path / "chromadb_test"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return chroma_dir


@pytest.fixture
def vector_service(temp_chroma_dir: Path) -> VectorStoreService:
    """Provides an initialized VectorStoreService on an isolated temp directory."""
    return VectorStoreService(
        persist_dir=str(temp_chroma_dir),
        batch_size=50,
        expected_dimension=384,
    )


# ── Test Suite ───────────────────────────────────────────────────────────────


def test_1_persistent_client_initialization(temp_chroma_dir: Path):
    """1. Verify PersistentClient initializes correctly with local persistence."""
    service = VectorStoreService(persist_dir=str(temp_chroma_dir))
    assert service.persist_dir == temp_chroma_dir
    assert service.expected_dimension == 384
    assert service.batch_size == 50
    assert service.client is not None
    assert temp_chroma_dir.exists()


def test_2_collection_creation(vector_service: VectorStoreService):
    """2. Verify get_or_create_collection creates a Chroma collection with approved name mapping."""
    col_id = "123e4567-e89b-12d3-a456-426614174000"
    col = vector_service.get_or_create_collection(col_id)
    assert col is not None
    expected_name = "col_123e4567_e89b_12d3_a456_426614174000"
    assert col.name == expected_name


def test_3_collection_retrieval(vector_service: VectorStoreService):
    """3. Verify get_collection returns existing collection or raises CollectionNotFoundError."""
    col_id = "test-retrieval-col"
    vector_service.get_or_create_collection(col_id)

    # Retrieval of existing collection
    col = vector_service.get_collection(col_id)
    assert col.name == "col_test_retrieval_col"

    # Retrieval of non-existent collection raises CollectionNotFoundError
    with pytest.raises(CollectionNotFoundError, match="does not exist"):
        vector_service.get_collection("non-existent-collection-id")


def test_4_cosine_distance_configuration(vector_service: VectorStoreService):
    """4. Verify collection is created with cosine distance configuration."""
    col_id = "cosine-metric-col"
    col = vector_service.get_or_create_collection(col_id)
    assert col.metadata is not None
    assert col.metadata.get("hnsw:space") == "cosine"


def test_5_vector_insertion_384_dimensions(vector_service: VectorStoreService):
    """5. Verify 384-dimensional vector insertion succeeds."""
    col_id = "insert-384-col"
    chunks = [
        make_embedded_chunk(chunk_index=0, collection_id=col_id),
        make_embedded_chunk(chunk_index=1, collection_id=col_id),
    ]

    stored_count = vector_service.store_chunks(chunks, collection_id=col_id)
    assert stored_count == 2
    assert vector_service.count(col_id) == 2


def test_6_text_and_metadata_preservation(vector_service: VectorStoreService):
    """6. Verify text and metadata are preserved without distortion."""
    col_id = "metadata-preservation-col"
    chunk = make_embedded_chunk(
        chunk_index=0,
        document_id="doc_preservation_001",
        collection_id=col_id,
        filename="report.pdf",
        file_type="pdf",
        chunk_text="Critical academic research finding text.",
        page_number=3,
        page_range="3-4",
        model_name="BAAI/bge-small-en-v1.5",
    )

    vector_service.store_chunks([chunk], collection_id=col_id)

    records = vector_service.get_chunks_by_ids(col_id, [chunk.chunk_id])
    assert len(records["ids"]) == 1
    assert records["ids"][0] == chunk.chunk_id
    assert records["documents"][0] == "Critical academic research finding text."

    meta = records["metadatas"][0]
    assert meta["document_id"] == "doc_preservation_001"
    assert meta["collection_id"] == col_id
    assert meta["chunk_index"] == 0
    assert meta["filename"] == "report.pdf"
    assert meta["file_type"] == "pdf"
    assert meta["page_number"] == 3
    assert meta["page_range"] == "3-4"
    assert meta["model_name"] == "BAAI/bge-small-en-v1.5"
    assert meta["char_count"] == len("Critical academic research finding text.")


def test_7_retrieval_by_chunk_ids(vector_service: VectorStoreService):
    """7. Verify retrieval by chunk ID returns embeddings, documents, and metadatas."""
    col_id = "retrieval-by-id-col"
    chunks = [
        make_embedded_chunk(chunk_index=i, collection_id=col_id)
        for i in range(3)
    ]
    vector_service.store_chunks(chunks, collection_id=col_id)

    target_ids = [chunks[0].chunk_id, chunks[2].chunk_id]
    result = vector_service.get_chunks_by_ids(col_id, target_ids)

    assert len(result["ids"]) == 2
    assert set(result["ids"]) == set(target_ids)
    for emb in result["embeddings"]:
        assert len(emb) == 384
    assert len(result["documents"]) == 2
    assert len(result["metadatas"]) == 2


def test_8_idempotent_upsert(vector_service: VectorStoreService):
    """8. Verify upserting the same chunk ID updates in place without duplicating."""
    col_id = "idempotent-upsert-col"
    chunk = make_embedded_chunk(
        chunk_index=0,
        collection_id=col_id,
        chunk_text="Initial text version.",
    )

    # Initial insert
    vector_service.store_chunks([chunk], collection_id=col_id)
    assert vector_service.count(col_id) == 1

    # Re-index with updated text using identical chunk_id
    updated_chunk = make_embedded_chunk(
        chunk_index=0,
        collection_id=col_id,
        chunk_text="Updated text version after revision.",
    )
    vector_service.store_chunks([updated_chunk], collection_id=col_id)

    # Count must remain 1
    assert vector_service.count(col_id) == 1

    # Text must be updated
    result = vector_service.get_chunks_by_ids(col_id, [chunk.chunk_id])
    assert result["documents"][0] == "Updated text version after revision."


def test_9_document_specific_vector_deletion(vector_service: VectorStoreService):
    """9. Verify delete_document_vectors deletes only the specified document's chunks."""
    col_id = "doc-deletion-col"
    doc_a_chunks = [
        make_embedded_chunk(chunk_index=i, document_id="doc_A", collection_id=col_id)
        for i in range(3)
    ]
    doc_b_chunks = [
        make_embedded_chunk(chunk_index=i, document_id="doc_B", collection_id=col_id)
        for i in range(2)
    ]

    vector_service.store_chunks(doc_a_chunks + doc_b_chunks, collection_id=col_id)
    assert vector_service.count(col_id) == 5

    # Delete doc_A
    deleted_count = vector_service.delete_document_vectors(col_id, "doc_A")
    assert deleted_count == 3
    assert vector_service.count(col_id) == 2

    # Verify doc_A chunks are gone
    doc_a_ids = [c.chunk_id for c in doc_a_chunks]
    res_a = vector_service.get_chunks_by_ids(col_id, doc_a_ids)
    assert len(res_a["ids"]) == 0

    # Verify doc_B chunks remain intact
    doc_b_ids = [c.chunk_id for c in doc_b_chunks]
    res_b = vector_service.get_chunks_by_ids(col_id, doc_b_ids)
    assert len(res_b["ids"]) == 2


def test_10_collection_count_accuracy(vector_service: VectorStoreService):
    """10. Verify count accurately reflects additions and deletions."""
    col_id = "count-accuracy-col"
    vector_service.get_or_create_collection(col_id)
    assert vector_service.count(col_id) == 0

    chunks = [
        make_embedded_chunk(chunk_index=i, document_id="doc_count", collection_id=col_id)
        for i in range(4)
    ]
    vector_service.store_chunks(chunks, collection_id=col_id)
    assert vector_service.count(col_id) == 4

    vector_service.delete_document_vectors(col_id, "doc_count")
    assert vector_service.count(col_id) == 0


def test_11_invalid_vector_dimension_rejection(vector_service: VectorStoreService):
    """11. Verify vectors with incorrect dimension (not 384) are rejected."""
    col_id = "dim-rejection-col"

    # 128 dimensions instead of 384
    bad_chunk = make_embedded_chunk(
        chunk_index=0,
        collection_id=col_id,
        dimension=128,
        embedding=make_dummy_embedding(dim=128),
    )

    with pytest.raises(VectorDimensionMismatchError, match="does not match expected dimension 384"):
        vector_service.store_chunks([bad_chunk], collection_id=col_id)


def test_12_nan_and_infinity_rejection(vector_service: VectorStoreService):
    """12. Verify vectors containing NaN or Inf are rejected."""
    col_id = "nan-inf-rejection-col"

    # NaN in vector
    nan_emb = make_dummy_embedding(384)
    nan_emb[5] = float("nan")
    # Bypass Pydantic validation on EmbeddedChunk to test service-level guardrail
    nan_chunk = make_embedded_chunk(chunk_index=0, collection_id=col_id)
    nan_chunk.embedding[5] = float("nan")

    with pytest.raises(InvalidVectorError, match="Non-finite value"):
        vector_service.store_chunks([nan_chunk], collection_id=col_id)

    # Inf in vector
    inf_chunk = make_embedded_chunk(chunk_index=1, collection_id=col_id)
    inf_chunk.embedding[10] = float("inf")

    with pytest.raises(InvalidVectorError, match="Non-finite value"):
        vector_service.store_chunks([inf_chunk], collection_id=col_id)


def test_13_empty_text_rejection(vector_service: VectorStoreService):
    """13. Verify chunks with empty or whitespace-only text are rejected."""
    col_id = "empty-text-col"
    chunk = make_embedded_chunk(chunk_index=0, collection_id=col_id)
    chunk.chunk_text = "   "

    with pytest.raises(InvalidChunkDataError, match="cannot be empty or whitespace only"):
        vector_service.store_chunks([chunk], collection_id=col_id)


def test_14_missing_or_invalid_metadata_handling(vector_service: VectorStoreService):
    """14. Verify metadata with None values (e.g. DOCX/TXT without page numbers) does not pass None to Chroma."""
    col_id = "none-metadata-col"
    chunk = make_embedded_chunk(
        chunk_index=0,
        collection_id=col_id,
        file_type="docx",
        page_number=None,
        page_range=None,
    )

    vector_service.store_chunks([chunk], collection_id=col_id)

    result = vector_service.get_chunks_by_ids(col_id, [chunk.chunk_id])
    meta = result["metadatas"][0]

    # page_number and page_range should either not exist in metadata or be absent, NEVER None
    assert "page_number" not in meta or meta["page_number"] is not None
    assert "page_range" not in meta or meta["page_range"] is not None
    # All present metadata values must be valid scalars
    for k, v in meta.items():
        assert isinstance(v, (str, int, float, bool)), f"Non-scalar metadata value for key {k}: {type(v)}"


def test_15_partial_insertion_error_handling(vector_service: VectorStoreService):
    """15. Verify pre-validation rejects batches containing an invalid chunk before storing any."""
    col_id = "partial-error-col"
    vector_service.get_or_create_collection(col_id)

    good_chunk_1 = make_embedded_chunk(chunk_index=0, collection_id=col_id)
    bad_chunk = make_embedded_chunk(chunk_index=1, collection_id=col_id)
    bad_chunk.chunk_text = ""  # invalid text
    good_chunk_2 = make_embedded_chunk(chunk_index=2, collection_id=col_id)

    with pytest.raises(InvalidChunkDataError):
        vector_service.store_chunks([good_chunk_1, bad_chunk, good_chunk_2], collection_id=col_id)

    # None of the chunks should have been stored
    assert vector_service.count(col_id) == 0


def test_16_persistence_after_client_restart(temp_chroma_dir: Path):
    """16. Verify data persists and is accessible after closing and reopening the client."""
    col_id = "restart-persist-col"
    chunk = make_embedded_chunk(
        chunk_index=0,
        collection_id=col_id,
        chunk_text="Data that must survive a restart.",
    )

    # First session: insert and close
    service1 = VectorStoreService(persist_dir=str(temp_chroma_dir))
    service1.store_chunks([chunk], collection_id=col_id)
    assert service1.count(col_id) == 1
    del service1

    # Second session: open against same directory
    service2 = VectorStoreService(persist_dir=str(temp_chroma_dir))
    assert service2.count(col_id) == 1

    records = service2.get_chunks_by_ids(col_id, [chunk.chunk_id])
    assert records["documents"][0] == "Data that must survive a restart."
    assert len(records["embeddings"][0]) == 384


def test_17_collection_isolation(vector_service: VectorStoreService):
    """17. Verify collections are strictly isolated from each other."""
    col_a = "collection-alpha"
    col_b = "collection-beta"

    chunk_a = make_embedded_chunk(chunk_index=0, document_id="doc_a", collection_id=col_a)
    chunk_b = make_embedded_chunk(chunk_index=0, document_id="doc_b", collection_id=col_b)

    vector_service.store_chunks([chunk_a], collection_id=col_a)
    vector_service.store_chunks([chunk_b], collection_id=col_b)

    assert vector_service.count(col_a) == 1
    assert vector_service.count(col_b) == 1

    # Query col_a for chunk_b -> must be empty
    res_a = vector_service.get_chunks_by_ids(col_a, [chunk_b.chunk_id])
    assert len(res_a["ids"]) == 0

    # Query col_b for chunk_a -> must be empty
    res_b = vector_service.get_chunks_by_ids(col_b, [chunk_a.chunk_id])
    assert len(res_b["ids"]) == 0


def test_18_repeated_deletion_behavior(vector_service: VectorStoreService):
    """18. Verify repeated deletion of the same document returns 0 and does not error."""
    col_id = "repeated-del-col"
    chunk = make_embedded_chunk(chunk_index=0, document_id="doc_repeat", collection_id=col_id)
    vector_service.store_chunks([chunk], collection_id=col_id)

    # First deletion: returns 1
    del_1 = vector_service.delete_document_vectors(col_id, "doc_repeat")
    assert del_1 == 1
    assert vector_service.count(col_id) == 0

    # Second deletion: returns 0 safely without error
    del_2 = vector_service.delete_document_vectors(col_id, "doc_repeat")
    assert del_2 == 0
    assert vector_service.count(col_id) == 0


# ── Additional Supporting Tests ──────────────────────────────────────────────


def test_19_collection_name_mapping():
    """19. Verify static collection name mapping logic and validation."""
    # Standard UUID
    assert (
        VectorStoreService.get_collection_name("123e4567-e89b-12d3-a456-426614174000")
        == "col_123e4567_e89b_12d3_a456_426614174000"
    )
    # Alphanumeric with hyphens
    assert VectorStoreService.get_collection_name("my-col-1") == "col_my_col_1"

    # Empty string raises error
    with pytest.raises(InvalidChunkDataError):
        VectorStoreService.get_collection_name("")

    with pytest.raises(InvalidChunkDataError):
        VectorStoreService.get_collection_name("   ")


def test_20_bounded_batching_large_chunk_set(vector_service: VectorStoreService):
    """20. Verify bounded batching with >50 chunks (e.g. 65 chunks across 2 batches)."""
    col_id = "large-batch-col"
    # Service has batch_size=50
    chunks = [
        make_embedded_chunk(chunk_index=i, document_id="doc_batch", collection_id=col_id)
        for i in range(65)
    ]

    stored = vector_service.store_chunks(chunks, collection_id=col_id)
    assert stored == 65
    assert vector_service.count(col_id) == 65


def test_21_delete_and_list_collections(vector_service: VectorStoreService):
    """21. Verify list_collections and delete_collection operations."""
    col_1 = "list-test-col-1"
    col_2 = "list-test-col-2"

    vector_service.get_or_create_collection(col_1)
    vector_service.get_or_create_collection(col_2)

    col_names = vector_service.list_collections()
    assert "col_list_test_col_1" in col_names
    assert "col_list_test_col_2" in col_names

    # Delete col_1
    deleted = vector_service.delete_collection(col_1)
    assert deleted is True

    col_names_after = vector_service.list_collections()
    assert "col_list_test_col_1" not in col_names_after
    assert "col_list_test_col_2" in col_names_after

    # Delete non-existent returns False
    assert vector_service.delete_collection("non-existent-col") is False


def test_22_empty_chunks_store_returns_zero(vector_service: VectorStoreService):
    """22. Verify store_chunks with empty sequence returns 0 without error."""
    assert vector_service.store_chunks([], collection_id="some-col") == 0
    assert vector_service.get_chunks_by_ids("some-col", []) == {
        "ids": [],
        "embeddings": [],
        "documents": [],
        "metadatas": [],
    }


# ── Phase 7.2 Vector Search & Retrieval Tests ────────────────────────────────


def test_23_query_successful_semantic_search(vector_service: VectorStoreService):
    """23. Successful vector similarity search returns matching chunks with metadata."""
    col_id = "test-query-col"
    vector_service.get_or_create_collection(col_id)

    # Store 3 chunks with deterministic embeddings
    chunks = [
        make_embedded_chunk(chunk_index=i, document_id="doc_q1", collection_id=col_id)
        for i in range(3)
    ]
    vector_service.store_chunks(chunks, collection_id=col_id)

    # Query with the exact embedding of chunk 0
    query_vec = chunks[0].embedding
    results = vector_service.query_vectors(
        collection_id=col_id,
        query_vector=query_vec,
        top_k=2,
    )

    assert isinstance(results, VectorSearchResults)
    assert len(results) == 2
    assert results.total_candidates == 2
    assert results.returned_results == 2

    # Top match should be chunk 0
    top = results[0]
    assert isinstance(top, RetrievedChunk)
    assert top.chunk_id == chunks[0].chunk_id
    assert top.document_id == "doc_q1"
    assert top.collection_id == col_id
    assert top.similarity_score > 0.99
    assert top.distance < 0.01
    assert top.filename == "sample.pdf"
    assert top.file_type == "pdf"
    assert top.page_number == 1


def test_24_cosine_distance_to_similarity_conversion(vector_service: VectorStoreService):
    """24. Verify exact mathematical relationship between cosine distance and similarity (s = 1 - d)."""
    col_id = "dist-sim-col"
    vector_service.get_or_create_collection(col_id)

    emb = make_dummy_embedding(dim=384, seed=42)
    chunk = make_embedded_chunk(chunk_index=0, document_id="doc_ds", collection_id=col_id, embedding=emb)
    vector_service.store_chunks([chunk], collection_id=col_id)

    results = vector_service.query_vectors(
        collection_id=col_id,
        query_vector=emb,
        top_k=1,
    )

    assert len(results) == 1
    res = results[0]
    # Check similarity is strictly 1 - distance within float precision
    expected_sim = max(-1.0, min(1.0, 1.0 - res.distance))
    assert abs(res.similarity_score - expected_sim) < 1e-7
    assert -1.0 <= res.similarity_score <= 1.0


def test_25_query_correct_ranking_order(vector_service: VectorStoreService):
    """25. Verify returned chunks are sorted descending by similarity (ascending by distance)."""
    col_id = "ranking-col"
    vector_service.get_or_create_collection(col_id)

    chunks = [
        make_embedded_chunk(chunk_index=i, document_id="doc_rank", collection_id=col_id, seed=i * 5)
        for i in range(5)
    ]
    vector_service.store_chunks(chunks, collection_id=col_id)

    query_vec = chunks[2].embedding
    results = vector_service.query_vectors(
        collection_id=col_id,
        query_vector=query_vec,
        top_k=5,
    )

    assert len(results) == 5
    for i in range(len(results) - 1):
        assert results[i].similarity_score >= results[i + 1].similarity_score
        assert results[i].distance <= results[i + 1].distance


def test_26_query_top_k_behavior(vector_service: VectorStoreService):
    """26. Verify top_k parameter bounds the number of returned chunks."""
    col_id = "topk-col"
    vector_service.get_or_create_collection(col_id)

    chunks = [
        make_embedded_chunk(chunk_index=i, document_id="doc_topk", collection_id=col_id)
        for i in range(10)
    ]
    vector_service.store_chunks(chunks, collection_id=col_id)

    query_vec = chunks[0].embedding

    # top_k = 3
    res_3 = vector_service.query_vectors(col_id, query_vec, top_k=3)
    assert len(res_3) == 3

    # top_k = 7
    res_7 = vector_service.query_vectors(col_id, query_vec, top_k=7)
    assert len(res_7) == 7

    # top_k exceeding total count returns total count
    res_20 = vector_service.query_vectors(col_id, query_vec, top_k=20)
    assert len(res_20) == 10


def test_27_query_empty_collection(vector_service: VectorStoreService):
    """27. Querying an empty collection returns empty VectorSearchResults without errors."""
    col_id = "empty-query-col"
    vector_service.get_or_create_collection(col_id)

    query_vec = make_dummy_embedding(dim=384, seed=1)
    results = vector_service.query_vectors(col_id, query_vec, top_k=5)

    assert isinstance(results, VectorSearchResults)
    assert len(results) == 0
    assert results.total_candidates == 0
    assert results.returned_results == 0
    assert results.results == []


def test_28_query_missing_collection_raises_error(vector_service: VectorStoreService):
    """28. Querying a non-existent collection raises CollectionNotFoundError."""
    query_vec = make_dummy_embedding(dim=384, seed=1)
    with pytest.raises(CollectionNotFoundError):
        vector_service.query_vectors("non-existent-col-uuid", query_vec, top_k=5)


def test_29_query_invalid_vector_dimension(vector_service: VectorStoreService):
    """29. Querying with wrong dimension raises VectorDimensionMismatchError."""
    col_id = "dim-err-col"
    vector_service.get_or_create_collection(col_id)

    # 100-dim instead of 384
    bad_vec = [0.1] * 100
    with pytest.raises(VectorDimensionMismatchError) as exc_info:
        vector_service.query_vectors(col_id, bad_vec, top_k=5)
    assert "100" in str(exc_info.value)
    assert "384" in str(exc_info.value)


def test_30_query_nan_vector_raises_error(vector_service: VectorStoreService):
    """30. Querying with NaN in vector raises InvalidVectorError."""
    col_id = "nan-err-col"
    vector_service.get_or_create_collection(col_id)

    bad_vec = make_dummy_embedding(dim=384, seed=1)
    bad_vec[10] = float("nan")

    with pytest.raises(InvalidVectorError):
        vector_service.query_vectors(col_id, bad_vec, top_k=5)


def test_31_query_infinite_vector_raises_error(vector_service: VectorStoreService):
    """31. Querying with Inf in vector raises InvalidVectorError."""
    col_id = "inf-err-col"
    vector_service.get_or_create_collection(col_id)

    bad_vec = make_dummy_embedding(dim=384, seed=1)
    bad_vec[20] = float("inf")

    with pytest.raises(InvalidVectorError):
        vector_service.query_vectors(col_id, bad_vec, top_k=5)


def test_32_query_invalid_top_k_raises_error(vector_service: VectorStoreService):
    """32. Querying with top_k < 1 or non-integer raises InvalidChunkDataError."""
    col_id = "topk-err-col"
    vector_service.get_or_create_collection(col_id)
    query_vec = make_dummy_embedding(dim=384, seed=1)

    with pytest.raises(InvalidChunkDataError):
        vector_service.query_vectors(col_id, query_vec, top_k=0)

    with pytest.raises(InvalidChunkDataError):
        vector_service.query_vectors(col_id, query_vec, top_k=-5)

    with pytest.raises(InvalidChunkDataError):
        vector_service.query_vectors(col_id, query_vec, top_k="five")  # type: ignore


def test_33_query_document_id_filtering(vector_service: VectorStoreService):
    """33. Optional document_ids filter restricts search strictly to specified documents."""
    col_id = "doc-filter-col"
    vector_service.get_or_create_collection(col_id)

    chunks_doc1 = [
        make_embedded_chunk(chunk_index=i, document_id="doc_alpha", collection_id=col_id)
        for i in range(3)
    ]
    chunks_doc2 = [
        make_embedded_chunk(chunk_index=i, document_id="doc_beta", collection_id=col_id)
        for i in range(3)
    ]
    vector_service.store_chunks(chunks_doc1 + chunks_doc2, collection_id=col_id)

    query_vec = chunks_doc1[0].embedding

    # Filter by single document (as string)
    res_alpha = vector_service.query_vectors(col_id, query_vec, top_k=5, document_ids="doc_alpha")
    assert len(res_alpha) == 3
    assert all(c.document_id == "doc_alpha" for c in res_alpha)

    # Filter by single document (as list)
    res_beta = vector_service.query_vectors(col_id, query_vec, top_k=5, document_ids=["doc_beta"])
    assert len(res_beta) == 3
    assert all(c.document_id == "doc_beta" for c in res_beta)

    # Filter by multiple documents
    res_both = vector_service.query_vectors(col_id, query_vec, top_k=10, document_ids=["doc_alpha", "doc_beta"])
    assert len(res_both) == 6

    # Duplicate document IDs rejected
    with pytest.raises(InvalidChunkDataError):
        vector_service.query_vectors(col_id, query_vec, top_k=5, document_ids=["doc_alpha", "doc_alpha"])


def test_34_query_metadata_filtering(vector_service: VectorStoreService):
    """34. Optional metadata filter correctly filters candidates by scalar metadata attributes."""
    col_id = "meta-filter-col"
    vector_service.get_or_create_collection(col_id)

    c_pdf = make_embedded_chunk(chunk_index=0, document_id="doc_pdf", collection_id=col_id, file_type="pdf")
    c_txt = make_embedded_chunk(chunk_index=1, document_id="doc_txt", collection_id=col_id, file_type="txt")
    vector_service.store_chunks([c_pdf, c_txt], collection_id=col_id)

    query_vec = c_pdf.embedding

    # Filter by file_type = 'pdf'
    res_pdf = vector_service.query_vectors(col_id, query_vec, top_k=5, where={"file_type": "pdf"})
    assert len(res_pdf) == 1
    assert res_pdf[0].file_type == "pdf"

    # Combined with document filter
    res_comb = vector_service.query_vectors(
        col_id,
        query_vec,
        top_k=5,
        document_ids="doc_pdf",
        where={"file_type": "pdf"},
    )
    assert len(res_comb) == 1
    assert res_comb[0].document_id == "doc_pdf"


def test_35_query_similarity_threshold_filtering(vector_service: VectorStoreService):
    """35. Similarity threshold filters out chunks below the threshold while preserving total_candidates."""
    col_id = "thresh-col"
    vector_service.get_or_create_collection(col_id)

    # Create 3 chunks with very different seeds
    c0 = make_embedded_chunk(chunk_index=0, document_id="doc_th", collection_id=col_id, seed=1)
    c1 = make_embedded_chunk(chunk_index=1, document_id="doc_th", collection_id=col_id, seed=50)
    c2 = make_embedded_chunk(chunk_index=2, document_id="doc_th", collection_id=col_id, seed=200)
    vector_service.store_chunks([c0, c1, c2], collection_id=col_id)

    # Query with c0 embedding without threshold
    unfiltered = vector_service.query_vectors(col_id, c0.embedding, top_k=3)
    assert len(unfiltered) == 3
    assert unfiltered.total_candidates == 3
    assert unfiltered.returned_results == 3

    # Query with a high threshold (e.g. 0.95) that only matches c0 itself
    filtered = vector_service.query_vectors(col_id, c0.embedding, top_k=3, similarity_threshold=0.95)
    assert len(filtered) == 1
    assert filtered.total_candidates == 3  # Raw candidates found
    assert filtered.returned_results == 1  # Candidates passing threshold
    assert filtered[0].chunk_id == c0.chunk_id


def test_36_query_no_matching_results(vector_service: VectorStoreService):
    """36. If no chunks exceed similarity threshold, returns empty results with returned_results=0."""
    col_id = "nomatch-col"
    vector_service.get_or_create_collection(col_id)

    chunk = make_embedded_chunk(chunk_index=0, document_id="doc_nm", collection_id=col_id, seed=5)
    vector_service.store_chunks([chunk], collection_id=col_id)

    # Query with an impossible threshold (e.g. 1.0)
    res = vector_service.query_vectors(col_id, [0.0] * 383 + [1.0], top_k=5, similarity_threshold=0.99999)
    assert len(res) == 0
    assert res.returned_results == 0


def test_37_query_multiple_documents_in_one_collection(vector_service: VectorStoreService):
    """37. Query can return ranked results spanning multiple documents in the same collection."""
    col_id = "multidoc-col"
    vector_service.get_or_create_collection(col_id)

    chunks = [
        make_embedded_chunk(chunk_index=i, document_id=f"doc_{i}", collection_id=col_id, seed=i)
        for i in range(4)
    ]
    vector_service.store_chunks(chunks, collection_id=col_id)

    res = vector_service.query_vectors(col_id, chunks[0].embedding, top_k=3)
    doc_ids = {c.document_id for c in res}
    assert len(res) == 3
    assert "doc_0" in doc_ids


def test_38_query_collection_isolation(vector_service: VectorStoreService):
    """38. Queries to Collection A never return vectors stored in Collection B."""
    col_a = "isolated-col-a"
    col_b = "isolated-col-b"

    vector_service.get_or_create_collection(col_a)
    vector_service.get_or_create_collection(col_b)

    chunk_a = make_embedded_chunk(chunk_index=0, document_id="doc_a", collection_id=col_a)
    chunk_b = make_embedded_chunk(chunk_index=0, document_id="doc_b", collection_id=col_b)

    vector_service.store_chunks([chunk_a], collection_id=col_a)
    vector_service.store_chunks([chunk_b], collection_id=col_b)

    res_a = vector_service.query_vectors(col_a, chunk_b.embedding, top_k=5)
    assert all(c.collection_id == col_a for c in res_a)
    assert not any(c.document_id == "doc_b" for c in res_a)


def test_39_existing_stored_vectors_remain_unchanged_after_queries(vector_service: VectorStoreService):
    """39. Verify vector queries are strictly read-only and do not alter stored vectors."""
    col_id = "readonly-query-col"
    vector_service.get_or_create_collection(col_id)

    chunks = [
        make_embedded_chunk(chunk_index=i, document_id="doc_ro", collection_id=col_id)
        for i in range(3)
    ]
    vector_service.store_chunks(chunks, collection_id=col_id)

    initial_count = vector_service.count(col_id)
    initial_chunks = vector_service.get_chunks_by_ids(col_id, [c.chunk_id for c in chunks])

    # Run multiple queries
    for _ in range(5):
        vector_service.query_vectors(col_id, chunks[0].embedding, top_k=2)

    after_count = vector_service.count(col_id)
    after_chunks = vector_service.get_chunks_by_ids(col_id, [c.chunk_id for c in chunks])

    assert initial_count == after_count == 3
    assert initial_chunks["ids"] == after_chunks["ids"]
    assert initial_chunks["documents"] == after_chunks["documents"]


def test_40_backward_compatibility_with_existing_methods(vector_service: VectorStoreService):
    """40. Existing VectorStoreService methods (store, get, delete, count) continue to function identically."""
    col_id = "compat-col"
    vector_service.get_or_create_collection(col_id)

    chunk = make_embedded_chunk(chunk_index=0, document_id="doc_compat", collection_id=col_id)
    stored = vector_service.store_chunks([chunk], collection_id=col_id)
    assert stored == 1
    assert vector_service.count(col_id) == 1

    fetched = vector_service.get_chunks_by_ids(col_id, [chunk.chunk_id])
    assert fetched["ids"] == [chunk.chunk_id]

    deleted = vector_service.delete_document_vectors(col_id, "doc_compat")
    assert deleted == 1
    assert vector_service.count(col_id) == 0


def test_41_retrieval_query_schema_validation():
    """41. RetrievalQuery Pydantic schema enforces all validation rules strictly."""
    valid_uuid = uuid.uuid4()
    doc_uuid1 = uuid.uuid4()
    doc_uuid2 = uuid.uuid4()

    # Valid query
    q = RetrievalQuery(
        query="What is agentic RAG?",
        collection_id=valid_uuid,
        top_k=5,
        document_ids=[doc_uuid1, doc_uuid2],
        similarity_threshold=0.35,
    )
    assert q.query == "What is agentic RAG?"
    assert q.top_k == 5
    assert len(q.document_ids) == 2
    assert q.similarity_threshold == 0.35

    # Rejects empty or whitespace query
    with pytest.raises(ValueError):
        RetrievalQuery(query="", collection_id=valid_uuid)
    with pytest.raises(ValueError):
        RetrievalQuery(query="   ", collection_id=valid_uuid)

    # Rejects over-length query (> 2048 chars)
    with pytest.raises(ValueError):
        RetrievalQuery(query="x" * 2049, collection_id=valid_uuid)

    # Rejects top_k < 1
    with pytest.raises(ValueError):
        RetrievalQuery(query="hello", collection_id=valid_uuid, top_k=0)

    # Rejects top_k > 100
    with pytest.raises(ValueError):
        RetrievalQuery(query="hello", collection_id=valid_uuid, top_k=101)

    # Rejects duplicate document IDs
    with pytest.raises(ValueError):
        RetrievalQuery(query="hello", collection_id=valid_uuid, document_ids=[doc_uuid1, doc_uuid1])

    # Rejects threshold outside [-1, 1]
    with pytest.raises(ValueError):
        RetrievalQuery(query="hello", collection_id=valid_uuid, similarity_threshold=1.1)
    with pytest.raises(ValueError):
        RetrievalQuery(query="hello", collection_id=valid_uuid, similarity_threshold=-1.5)


def test_42_retrieved_chunk_and_response_schema():
    """42. RetrievedChunk and RetrievalResponse schemas validate and expose correct properties."""
    col_id = str(uuid.uuid4())
    doc_id = str(uuid.uuid4())

    chunk = RetrievedChunk(
        chunk_id="chunk_001",
        document_id=doc_id,
        collection_id=col_id,
        text="Sample text content for retrieval.",
        similarity_score=0.85,
        distance=0.15,
        chunk_index=0,
        filename="test.pdf",
        file_type="pdf",
        page_number=1,
    )
    assert chunk.chunk_id == "chunk_001"
    assert chunk.chunk_text == "Sample text content for retrieval."  # Alias property
    assert chunk.similarity_score == 0.85
    assert chunk.distance == 0.15

    resp = RetrievalResponse(
        query="test query",
        collection_id=col_id,
        results=[chunk],
        total_candidates=1,
        returned_results=1,
        retrieval_metadata={"threshold": 0.5},
    )
    assert resp.total_candidates == 1
    assert resp.returned_results == 1
    assert len(resp.results) == 1
    assert resp.results[0].chunk_id == "chunk_001"
