"""
tests/test_retrieval_service.py
─────────────────────────────────────────────────────────────
Comprehensive unit and integration tests for RetrievalService
and authoritative SQLite validation gates.

Covers all 22 mandatory requirements:
  1.  Successful retrieval.
  2.  Correct reuse of existing embedding service.
  3.  Indexed document accepted.
  4.  Missing document rejected.
  5.  Deletion-pending document rejected.
  6.  Failed document rejected.
  7.  Processing document rejected.
  8.  Pending document rejected.
  9.  Wrong collection rejected.
  10. Outdated vector version rejected.
  11. Correct active version accepted.
  12. Missing version metadata rejected.
  13. Invalid document metadata rejected.
  14. Similarity threshold enforced.
  15. Empty candidate results.
  16. All candidates rejected.
  17. Multiple valid chunks from one document.
  18. Duplicate candidate chunk IDs.
  19. Batched SQLite validation rather than one query per chunk.
  20. Database error propagated appropriately.
  21. Missing ChromaDB collection handled correctly.
  22. No second embedding model instance created.
─────────────────────────────────────────────────────────────
"""

import math
import uuid
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.collection import Collection
from backend.models.document import Document
from backend.schemas.chunk import ChunkMetadata, DocumentChunk
from backend.schemas.document import DocumentStatus
from backend.schemas.embedding import EmbeddedChunk
from backend.schemas.retrieval import RetrievalQuery, RetrievalResponse, RetrievedChunk
from backend.services.embedding_service import EmbeddingService
from backend.services.retrieval_service import (
    InvalidRetrievalQueryError,
    RetrievalExecutionError,
    RetrievalService,
)
from backend.services.vector_store import (
    CollectionNotFoundError,
    VectorSearchResults,
    VectorStoreService,
)


# ── Test Helpers ─────────────────────────────────────────────────────────────


def make_dummy_embedding(dim: int = 384, seed: int = 0) -> List[float]:
    """Generate a deterministic unit-normalized vector for testing."""
    raw = [float(((i + seed) % 17) + 1) for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


class MockEmbeddingService:
    """Fast, deterministic mock of EmbeddingService for isolated retrieval tests."""

    def __init__(self, dimension: int = 384, default_seed: int = 42):
        self.dimension = dimension
        self.default_seed = default_seed
        self.embed_text_calls: List[str] = []

    def embed_text(self, text: str) -> List[float]:
        self.embed_text_calls.append(text)
        seed = abs(hash(text)) % 1000
        return make_dummy_embedding(dim=self.dimension, seed=seed)


def make_embedded_chunk(
    chunk_index: int = 0,
    document_id: str = "doc_001",
    collection_id: str = "col_001",
    filename: str = "sample.pdf",
    file_type: str = "pdf",
    chunk_text: Optional[str] = None,
    page_number: Optional[int] = 1,
    page_range: Optional[str] = None,
    dimension: int = 384,
    embedding: Optional[List[float]] = None,
    version: Optional[int] = 1,
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


async def create_db_document(
    db: AsyncSession,
    collection_id: str,
    status: DocumentStatus = DocumentStatus.INDEXED,
    version: int = 1,
    filename: str = "sample.pdf",
    doc_id: Optional[str] = None,
) -> Document:
    """Helper to insert an authoritative document record into SQLite."""
    actual_id = doc_id or str(uuid.uuid4())
    doc = Document(
        id=actual_id,
        collection_id=collection_id,
        filename=filename,
        stored_filename=f"stored_{actual_id}_{filename}",
        file_type="pdf",
        sha256=f"sha_{actual_id[:16]}",
        size_bytes=1024,
        status=status.value,
        version=version,
        page_count=2,
        chunk_count=2,
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)
    return doc


# ── Test Suite ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_1_successful_retrieval(db_session: AsyncSession, test_vector_store: VectorStoreService):
    """1. Successful retrieval: validates input, searches vectors, validates via SQLite, and returns response."""
    col_id = str(uuid.uuid4())
    col = Collection(id=col_id, name="Test Col 1")
    db_session.add(col)
    await db_session.commit()

    doc = await create_db_document(db_session, col_id, status=DocumentStatus.INDEXED, version=1)

    # Store 2 chunks in ChromaDB
    chunks = [
        make_embedded_chunk(chunk_index=0, document_id=doc.id, collection_id=col_id, version=1, seed=10),
        make_embedded_chunk(chunk_index=1, document_id=doc.id, collection_id=col_id, version=1, seed=20),
    ]
    test_vector_store.store_chunks(chunks, collection_id=col_id)

    # Use mock embedder whose embedding for "sample query" matches chunk 0
    mock_embed = MockEmbeddingService()
    mock_embed.embed_text = MagicMock(return_value=chunks[0].embedding)

    service = RetrievalService(embedding_service=mock_embed, vector_store=test_vector_store)

    query = RetrievalQuery(
        query="sample query",
        collection_id=uuid.UUID(col_id),
        top_k=5,
    )

    response = await service.retrieve(query, db=db_session)

    assert isinstance(response, RetrievalResponse)
    assert response.query == "sample query"
    assert response.collection_id == col_id
    assert response.total_candidates == 2
    assert response.accepted_candidates == 2
    assert response.rejected_candidates == 0
    assert response.returned_results == 2
    assert len(response.results) == 2

    # Top match should be chunk 0 with high similarity
    top = response.results[0]
    assert top.chunk_id == chunks[0].chunk_id
    assert top.document_id == doc.id
    assert top.similarity_score > 0.99
    assert top.filename == "sample.pdf"
    assert top.page_number == 1


def test_2_correct_reuse_of_existing_embedding_service():
    """2. RetrievalService correctly reuses the injected EmbeddingService instance."""
    mock_embed = MockEmbeddingService(dimension=384)
    service = RetrievalService(embedding_service=mock_embed)
    assert service.embedding_service is mock_embed


@pytest.mark.asyncio
async def test_3_indexed_document_accepted(db_session: AsyncSession, test_vector_store: VectorStoreService):
    """3. Chunks belonging to an INDEXED document are accepted."""
    col_id = str(uuid.uuid4())
    col = Collection(id=col_id, name="Test Col 3")
    db_session.add(col)
    await db_session.commit()

    doc = await create_db_document(db_session, col_id, status=DocumentStatus.INDEXED, version=1)

    chunk = make_embedded_chunk(chunk_index=0, document_id=doc.id, collection_id=col_id, version=1)
    test_vector_store.store_chunks([chunk], collection_id=col_id)

    mock_embed = MockEmbeddingService()
    mock_embed.embed_text = MagicMock(return_value=chunk.embedding)

    service = RetrievalService(embedding_service=mock_embed, vector_store=test_vector_store)
    resp = await service.retrieve({"query": "find me", "collection_id": col_id}, db=db_session)

    assert resp.accepted_candidates == 1
    assert resp.returned_results == 1
    assert resp.results[0].chunk_id == chunk.chunk_id


@pytest.mark.asyncio
async def test_4_missing_document_rejected(db_session: AsyncSession, test_vector_store: VectorStoreService):
    """4. Chunks whose document ID does not exist in SQLite are rejected."""
    col_id = str(uuid.uuid4())
    col = Collection(id=col_id, name="Test Col 4")
    db_session.add(col)
    await db_session.commit()

    missing_doc_id = str(uuid.uuid4())
    chunk = make_embedded_chunk(chunk_index=0, document_id=missing_doc_id, collection_id=col_id, version=1)
    test_vector_store.store_chunks([chunk], collection_id=col_id)

    mock_embed = MockEmbeddingService()
    mock_embed.embed_text = MagicMock(return_value=chunk.embedding)

    service = RetrievalService(embedding_service=mock_embed, vector_store=test_vector_store)
    resp = await service.retrieve({"query": "find me", "collection_id": col_id}, db=db_session)

    assert resp.total_candidates == 1
    assert resp.accepted_candidates == 0
    assert resp.rejected_candidates == 1
    assert resp.returned_results == 0
    assert len(resp.results) == 0
    assert resp.retrieval_metadata["rejection_reasons"]["document_not_found"] == 1


@pytest.mark.asyncio
async def test_5_deletion_pending_document_rejected(db_session: AsyncSession, test_vector_store: VectorStoreService):
    """5. Chunks belonging to a DELETION_PENDING document are rejected."""
    col_id = str(uuid.uuid4())
    col = Collection(id=col_id, name="Test Col 5")
    db_session.add(col)
    await db_session.commit()

    doc = await create_db_document(db_session, col_id, status=DocumentStatus.DELETION_PENDING, version=1)
    chunk = make_embedded_chunk(chunk_index=0, document_id=doc.id, collection_id=col_id, version=1)
    test_vector_store.store_chunks([chunk], collection_id=col_id)

    mock_embed = MockEmbeddingService()
    mock_embed.embed_text = MagicMock(return_value=chunk.embedding)

    service = RetrievalService(embedding_service=mock_embed, vector_store=test_vector_store)
    resp = await service.retrieve({"query": "find me", "collection_id": col_id}, db=db_session)

    assert resp.accepted_candidates == 0
    assert resp.rejected_candidates == 1
    assert resp.retrieval_metadata["rejection_reasons"]["not_indexed"] == 1


@pytest.mark.asyncio
async def test_6_failed_document_rejected(db_session: AsyncSession, test_vector_store: VectorStoreService):
    """6. Chunks belonging to a FAILED document are rejected."""
    col_id = str(uuid.uuid4())
    col = Collection(id=col_id, name="Test Col 6")
    db_session.add(col)
    await db_session.commit()

    doc = await create_db_document(db_session, col_id, status=DocumentStatus.FAILED, version=1)
    chunk = make_embedded_chunk(chunk_index=0, document_id=doc.id, collection_id=col_id, version=1)
    test_vector_store.store_chunks([chunk], collection_id=col_id)

    mock_embed = MockEmbeddingService()
    mock_embed.embed_text = MagicMock(return_value=chunk.embedding)

    service = RetrievalService(embedding_service=mock_embed, vector_store=test_vector_store)
    resp = await service.retrieve({"query": "find me", "collection_id": col_id}, db=db_session)

    assert resp.accepted_candidates == 0
    assert resp.rejected_candidates == 1
    assert resp.retrieval_metadata["rejection_reasons"]["not_indexed"] == 1


@pytest.mark.asyncio
async def test_7_processing_document_rejected(db_session: AsyncSession, test_vector_store: VectorStoreService):
    """7. Chunks belonging to a PROCESSING document are rejected."""
    col_id = str(uuid.uuid4())
    col = Collection(id=col_id, name="Test Col 7")
    db_session.add(col)
    await db_session.commit()

    doc = await create_db_document(db_session, col_id, status=DocumentStatus.PROCESSING, version=1)
    chunk = make_embedded_chunk(chunk_index=0, document_id=doc.id, collection_id=col_id, version=1)
    test_vector_store.store_chunks([chunk], collection_id=col_id)

    mock_embed = MockEmbeddingService()
    mock_embed.embed_text = MagicMock(return_value=chunk.embedding)

    service = RetrievalService(embedding_service=mock_embed, vector_store=test_vector_store)
    resp = await service.retrieve({"query": "find me", "collection_id": col_id}, db=db_session)

    assert resp.accepted_candidates == 0
    assert resp.rejected_candidates == 1
    assert resp.retrieval_metadata["rejection_reasons"]["not_indexed"] == 1


@pytest.mark.asyncio
async def test_8_pending_document_rejected(db_session: AsyncSession, test_vector_store: VectorStoreService):
    """8. Chunks belonging to a PENDING document are rejected."""
    col_id = str(uuid.uuid4())
    col = Collection(id=col_id, name="Test Col 8")
    db_session.add(col)
    await db_session.commit()

    doc = await create_db_document(db_session, col_id, status=DocumentStatus.PENDING, version=1)
    chunk = make_embedded_chunk(chunk_index=0, document_id=doc.id, collection_id=col_id, version=1)
    test_vector_store.store_chunks([chunk], collection_id=col_id)

    mock_embed = MockEmbeddingService()
    mock_embed.embed_text = MagicMock(return_value=chunk.embedding)

    service = RetrievalService(embedding_service=mock_embed, vector_store=test_vector_store)
    resp = await service.retrieve({"query": "find me", "collection_id": col_id}, db=db_session)

    assert resp.accepted_candidates == 0
    assert resp.rejected_candidates == 1
    assert resp.retrieval_metadata["rejection_reasons"]["not_indexed"] == 1


@pytest.mark.asyncio
async def test_9_wrong_collection_rejected(db_session: AsyncSession, test_vector_store: VectorStoreService):
    """9. Chunks whose SQLite document belongs to a different collection are rejected."""
    col1_id = str(uuid.uuid4())
    col2_id = str(uuid.uuid4())
    db_session.add(Collection(id=col1_id, name="Col 9A"))
    db_session.add(Collection(id=col2_id, name="Col 9B"))
    await db_session.commit()

    # Document belongs to col2 in SQLite
    doc_in_col2 = await create_db_document(db_session, col2_id, status=DocumentStatus.INDEXED, version=1)

    # But vector was stored under col1 in ChromaDB
    chunk = make_embedded_chunk(chunk_index=0, document_id=doc_in_col2.id, collection_id=col1_id, version=1)
    test_vector_store.store_chunks([chunk], collection_id=col1_id)

    mock_embed = MockEmbeddingService()
    mock_embed.embed_text = MagicMock(return_value=chunk.embedding)

    service = RetrievalService(embedding_service=mock_embed, vector_store=test_vector_store)
    resp = await service.retrieve({"query": "find me", "collection_id": col1_id}, db=db_session)

    assert resp.accepted_candidates == 0
    assert resp.rejected_candidates == 1
    assert resp.retrieval_metadata["rejection_reasons"]["wrong_collection"] == 1


@pytest.mark.asyncio
async def test_10_outdated_vector_version_rejected(db_session: AsyncSession, test_vector_store: VectorStoreService):
    """10. Outdated historical vector versions (e.g. v1 when SQLite document is v2) are rejected."""
    col_id = str(uuid.uuid4())
    col = Collection(id=col_id, name="Test Col 10")
    db_session.add(col)
    await db_session.commit()

    # Document in SQLite is at version 2
    doc = await create_db_document(db_session, col_id, status=DocumentStatus.INDEXED, version=2)

    # ChromaDB has an old chunk at version 1
    chunk_v1 = make_embedded_chunk(chunk_index=0, document_id=doc.id, collection_id=col_id, version=1)
    test_vector_store.store_chunks([chunk_v1], collection_id=col_id)

    mock_embed = MockEmbeddingService()
    mock_embed.embed_text = MagicMock(return_value=chunk_v1.embedding)

    service = RetrievalService(embedding_service=mock_embed, vector_store=test_vector_store)
    resp = await service.retrieve({"query": "find me", "collection_id": col_id}, db=db_session)

    assert resp.accepted_candidates == 0
    assert resp.rejected_candidates == 1
    assert resp.retrieval_metadata["rejection_reasons"]["version_mismatch"] == 1


@pytest.mark.asyncio
async def test_11_correct_active_version_accepted(db_session: AsyncSession, test_vector_store: VectorStoreService):
    """11. Chunks whose version matches the active SQLite document version are accepted."""
    col_id = str(uuid.uuid4())
    col = Collection(id=col_id, name="Test Col 11")
    db_session.add(col)
    await db_session.commit()

    # Document is version 2
    doc = await create_db_document(db_session, col_id, status=DocumentStatus.INDEXED, version=2)

    # Chunk is version 2
    chunk_v2 = make_embedded_chunk(chunk_index=0, document_id=doc.id, collection_id=col_id, version=2)
    test_vector_store.store_chunks([chunk_v2], collection_id=col_id)

    mock_embed = MockEmbeddingService()
    mock_embed.embed_text = MagicMock(return_value=chunk_v2.embedding)

    service = RetrievalService(embedding_service=mock_embed, vector_store=test_vector_store)
    resp = await service.retrieve({"query": "find me", "collection_id": col_id}, db=db_session)

    assert resp.accepted_candidates == 1
    assert resp.returned_results == 1
    assert resp.results[0].version == 2


@pytest.mark.asyncio
async def test_12_missing_version_metadata_rejected(db_session: AsyncSession, test_vector_store: VectorStoreService):
    """12. Chunks lacking version metadata are rejected by the version validation gate."""
    col_id = str(uuid.uuid4())
    col = Collection(id=col_id, name="Test Col 12")
    db_session.add(col)
    await db_session.commit()

    doc = await create_db_document(db_session, col_id, status=DocumentStatus.INDEXED, version=1)

    chunk = make_embedded_chunk(chunk_index=0, document_id=doc.id, collection_id=col_id, version=1)
    test_vector_store.store_chunks([chunk], collection_id=col_id)

    mock_embed = MockEmbeddingService()
    mock_embed.embed_text = MagicMock(return_value=chunk.embedding)

    # Simulate vector_store returning a chunk with version=None
    orig_query = test_vector_store.query_vectors

    def query_with_none_version(*args, **kwargs):
        res = orig_query(*args, **kwargs)
        res[0].version = None
        return res

    with patch.object(test_vector_store, "query_vectors", side_effect=query_with_none_version):
        service = RetrievalService(embedding_service=mock_embed, vector_store=test_vector_store)
        resp = await service.retrieve({"query": "find me", "collection_id": col_id}, db=db_session)

    assert resp.accepted_candidates == 0
    assert resp.rejected_candidates == 1
    assert resp.retrieval_metadata["rejection_reasons"]["missing_version_metadata"] == 1


@pytest.mark.asyncio
async def test_13_invalid_document_metadata_rejected(db_session: AsyncSession, test_vector_store: VectorStoreService):
    """13. Chunks with empty text or invalid chunk ID are rejected."""
    col_id = str(uuid.uuid4())
    col = Collection(id=col_id, name="Test Col 13")
    db_session.add(col)
    await db_session.commit()

    doc = await create_db_document(db_session, col_id, status=DocumentStatus.INDEXED, version=1)
    chunk = make_embedded_chunk(chunk_index=0, document_id=doc.id, collection_id=col_id, version=1)
    test_vector_store.store_chunks([chunk], collection_id=col_id)

    mock_embed = MockEmbeddingService()
    mock_embed.embed_text = MagicMock(return_value=chunk.embedding)

    # Simulate vector_store returning an empty text chunk
    orig_query = test_vector_store.query_vectors

    def query_with_empty_text(*args, **kwargs):
        res = orig_query(*args, **kwargs)
        res[0].text = ""
        return res

    with patch.object(test_vector_store, "query_vectors", side_effect=query_with_empty_text):
        service = RetrievalService(embedding_service=mock_embed, vector_store=test_vector_store)
        resp = await service.retrieve({"query": "find me", "collection_id": col_id}, db=db_session)

    assert resp.accepted_candidates == 0
    assert resp.rejected_candidates == 1
    assert resp.retrieval_metadata["rejection_reasons"]["invalid_metadata"] == 1


@pytest.mark.asyncio
async def test_14_similarity_threshold_enforced(db_session: AsyncSession, test_vector_store: VectorStoreService):
    """14. Candidates below the requested similarity threshold are filtered out."""
    col_id = str(uuid.uuid4())
    col = Collection(id=col_id, name="Test Col 14")
    db_session.add(col)
    await db_session.commit()

    doc = await create_db_document(db_session, col_id, status=DocumentStatus.INDEXED, version=1)

    c0 = make_embedded_chunk(chunk_index=0, document_id=doc.id, collection_id=col_id, version=1, seed=1)
    c1 = make_embedded_chunk(chunk_index=1, document_id=doc.id, collection_id=col_id, version=1, seed=100)
    test_vector_store.store_chunks([c0, c1], collection_id=col_id)

    mock_embed = MockEmbeddingService()
    # Query with exact embedding of c0
    mock_embed.embed_text = MagicMock(return_value=c0.embedding)

    service = RetrievalService(embedding_service=mock_embed, vector_store=test_vector_store)

    # Threshold = 0.95: only c0 matches
    resp = await service.retrieve(
        {"query": "find me", "collection_id": col_id, "similarity_threshold": 0.95},
        db=db_session,
    )

    assert resp.returned_results == 1
    assert resp.results[0].chunk_id == c0.chunk_id


@pytest.mark.asyncio
async def test_15_empty_candidate_results(db_session: AsyncSession, test_vector_store: VectorStoreService):
    """15. Empty candidate results from vector store return empty RetrievalResponse cleanly."""
    col_id = str(uuid.uuid4())
    col = Collection(id=col_id, name="Test Col 15")
    db_session.add(col)
    await db_session.commit()

    test_vector_store.get_or_create_collection(col_id)

    mock_embed = MockEmbeddingService()
    service = RetrievalService(embedding_service=mock_embed, vector_store=test_vector_store)

    resp = await service.retrieve({"query": "find me", "collection_id": col_id}, db=db_session)
    assert resp.total_candidates == 0
    assert resp.accepted_candidates == 0
    assert resp.returned_results == 0
    assert resp.results == []


@pytest.mark.asyncio
async def test_16_all_candidates_rejected(db_session: AsyncSession, test_vector_store: VectorStoreService):
    """16. When all candidates fail validation gates, returns 0 results with accurate rejection metrics."""
    col_id = str(uuid.uuid4())
    col = Collection(id=col_id, name="Test Col 16")
    db_session.add(col)
    await db_session.commit()

    doc_failed = await create_db_document(db_session, col_id, status=DocumentStatus.FAILED, version=1)
    doc_pending = await create_db_document(db_session, col_id, status=DocumentStatus.DELETION_PENDING, version=1)

    c1 = make_embedded_chunk(chunk_index=0, document_id=doc_failed.id, collection_id=col_id, version=1)
    c2 = make_embedded_chunk(chunk_index=1, document_id=doc_pending.id, collection_id=col_id, version=1)
    c3 = make_embedded_chunk(chunk_index=2, document_id="non_existent_doc", collection_id=col_id, version=1)
    test_vector_store.store_chunks([c1, c2, c3], collection_id=col_id)

    mock_embed = MockEmbeddingService()
    service = RetrievalService(embedding_service=mock_embed, vector_store=test_vector_store)

    resp = await service.retrieve({"query": "find me", "collection_id": col_id}, db=db_session)
    assert resp.total_candidates == 3
    assert resp.accepted_candidates == 0
    assert resp.rejected_candidates == 3
    assert resp.returned_results == 0
    assert len(resp.results) == 0


@pytest.mark.asyncio
async def test_17_multiple_valid_chunks_from_one_document(db_session: AsyncSession, test_vector_store: VectorStoreService):
    """17. Multiple relevant chunks from the same document are preserved without arbitrary document deduplication."""
    col_id = str(uuid.uuid4())
    col = Collection(id=col_id, name="Test Col 17")
    db_session.add(col)
    await db_session.commit()

    doc = await create_db_document(db_session, col_id, status=DocumentStatus.INDEXED, version=1)

    chunks = [
        make_embedded_chunk(chunk_index=i, document_id=doc.id, collection_id=col_id, version=1, seed=i)
        for i in range(3)
    ]
    test_vector_store.store_chunks(chunks, collection_id=col_id)

    mock_embed = MockEmbeddingService()
    mock_embed.embed_text = MagicMock(return_value=chunks[0].embedding)

    service = RetrievalService(embedding_service=mock_embed, vector_store=test_vector_store)
    resp = await service.retrieve({"query": "find me", "collection_id": col_id, "top_k": 3}, db=db_session)

    assert resp.returned_results == 3
    assert all(c.document_id == doc.id for c in resp.results)
    # Different chunk indices
    indices = [c.chunk_index for c in resp.results]
    assert indices == [0, 1, 2] or len(set(indices)) == 3


@pytest.mark.asyncio
async def test_18_duplicate_candidate_chunk_ids(db_session: AsyncSession, test_vector_store: VectorStoreService):
    """18. Duplicate chunk IDs returned by the vector store are deduplicated."""
    col_id = str(uuid.uuid4())
    col = Collection(id=col_id, name="Test Col 18")
    db_session.add(col)
    await db_session.commit()

    doc = await create_db_document(db_session, col_id, status=DocumentStatus.INDEXED, version=1)
    chunk = make_embedded_chunk(chunk_index=0, document_id=doc.id, collection_id=col_id, version=1)
    test_vector_store.store_chunks([chunk], collection_id=col_id)

    mock_embed = MockEmbeddingService()

    # Simulate vector store returning the same chunk twice
    orig_query = test_vector_store.query_vectors

    def query_with_duplicates(*args, **kwargs):
        res = orig_query(*args, **kwargs)
        duplicate = RetrievedChunk.model_validate(res[0])
        return VectorSearchResults([res[0], duplicate], total_candidates=2)

    with patch.object(test_vector_store, "query_vectors", side_effect=query_with_duplicates):
        service = RetrievalService(embedding_service=mock_embed, vector_store=test_vector_store)
        resp = await service.retrieve({"query": "find me", "collection_id": col_id}, db=db_session)

    assert resp.total_candidates == 2
    assert resp.accepted_candidates == 1
    assert resp.rejected_candidates == 1
    assert resp.returned_results == 1
    assert resp.retrieval_metadata["rejection_reasons"]["duplicate_chunk_id"] == 1


@pytest.mark.asyncio
async def test_19_batched_sqlite_validation(db_session: AsyncSession, test_vector_store: VectorStoreService):
    """19. Candidate validation uses a single batched SQLite query, not one query per chunk."""
    col_id = str(uuid.uuid4())
    col = Collection(id=col_id, name="Test Col 19")
    db_session.add(col)
    await db_session.commit()

    doc1 = await create_db_document(db_session, col_id, status=DocumentStatus.INDEXED, version=1)
    doc2 = await create_db_document(db_session, col_id, status=DocumentStatus.INDEXED, version=1)

    c1 = make_embedded_chunk(chunk_index=0, document_id=doc1.id, collection_id=col_id, version=1, seed=1)
    c2 = make_embedded_chunk(chunk_index=1, document_id=doc2.id, collection_id=col_id, version=1, seed=2)
    test_vector_store.store_chunks([c1, c2], collection_id=col_id)

    mock_embed = MockEmbeddingService()
    service = RetrievalService(embedding_service=mock_embed, vector_store=test_vector_store)

    orig_execute = db_session.execute
    execute_calls = []

    async def tracking_execute(*args, **kwargs):
        execute_calls.append(args)
        return await orig_execute(*args, **kwargs)

    with patch.object(db_session, "execute", side_effect=tracking_execute):
        resp = await service.retrieve({"query": "find me", "collection_id": col_id, "top_k": 2}, db=db_session)

    # Exactly 1 query executed for candidate document validation
    assert len(execute_calls) == 1
    assert resp.accepted_candidates == 2


@pytest.mark.asyncio
async def test_20_database_error_propagated_appropriately(db_session: AsyncSession, test_vector_store: VectorStoreService):
    """20. Database infrastructure failures raise RetrievalExecutionError and are not converted to empty results."""
    col_id = str(uuid.uuid4())
    col = Collection(id=col_id, name="Test Col 20")
    db_session.add(col)
    await db_session.commit()

    doc = await create_db_document(db_session, col_id, status=DocumentStatus.INDEXED, version=1)
    chunk = make_embedded_chunk(chunk_index=0, document_id=doc.id, collection_id=col_id, version=1)
    test_vector_store.store_chunks([chunk], collection_id=col_id)

    mock_embed = MockEmbeddingService()
    service = RetrievalService(embedding_service=mock_embed, vector_store=test_vector_store)

    # Simulate database disk I/O or lock error
    with patch.object(db_session, "execute", side_effect=OperationalError("database is locked", {}, None)):
        with pytest.raises(RetrievalExecutionError) as exc_info:
            await service.retrieve({"query": "find me", "collection_id": col_id}, db=db_session)
        assert "database is locked" in str(exc_info.value)


@pytest.mark.asyncio
async def test_21_missing_chromadb_collection_handled_correctly(db_session: AsyncSession, test_vector_store: VectorStoreService):
    """21. Missing ChromaDB collection raises CollectionNotFoundError as expected."""
    mock_embed = MockEmbeddingService()
    service = RetrievalService(embedding_service=mock_embed, vector_store=test_vector_store)

    non_existent_id = str(uuid.uuid4())
    with pytest.raises(CollectionNotFoundError):
        await service.retrieve({"query": "find me", "collection_id": non_existent_id}, db=db_session)


def test_22_no_second_embedding_model_instance_created():
    """22. Verifies that RetrievalService defaults reuse the class-level singleton model cache."""
    # Instantiating RetrievalService without passing an embedding model
    service = RetrievalService()
    assert isinstance(service.embedding_service, EmbeddingService)

    # Pre-populate class-level process cache
    mock_model = MagicMock()
    EmbeddingService._cached_model = mock_model

    try:
        # Access model via service
        model = service.embedding_service._get_model()
        assert model is mock_model
    finally:
        EmbeddingService.clear_cache()
