"""
tests/test_embedding_service.py
─────────────────────────────────────────────────────────────
Comprehensive unit and integration tests for Phase 5B:
Isolated Embedding Service and DocumentProcessor integration.

Design:
  - Uses deterministic mock SentenceTransformer by default (0 network, milliseconds).
  - Validates EmbeddedChunk Pydantic schema constraints.
  - Validates EmbeddingService input handling, singleton caching, and RAM guardrail.
  - Validates cosine similarity mathematical correctness.
  - Validates DocumentProcessor optional embedding integration and failure rollback.
  - Includes an optional real-model smoke test marked with @pytest.mark.real_model.
─────────────────────────────────────────────────────────────
"""

import math
import os
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import Settings
from backend.models.collection import Collection
from backend.models.document import Document
from backend.schemas.chunk import ChunkMetadata, DocumentChunk, ProcessedDocument
from backend.schemas.document import DocumentStatus
from backend.schemas.embedding import EmbeddedChunk
from backend.services.document_processor import (
    DocumentProcessor,
    DocumentProcessorError,
)
from backend.services.embedding_service import (
    EmbeddingInferenceError,
    EmbeddingService,
    EmbeddingServiceError,
    InvalidInputError,
    ModelLoadError,
)
from backend.services.file_storage import FileStorageService


# ── Mock Model for Fast Offline Testing ──────────────────────────────────────


class MockSentenceTransformer:
    """Deterministic, fast mock SentenceTransformer for unit tests."""

    def __init__(self, dimension: int = 384):
        self._dim = dimension

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim

    def encode(
        self,
        sentences,
        batch_size: int = 1,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
    ):
        if isinstance(sentences, str):
            sentences = [sentences]
        vectors = []
        for s in sentences:
            # Deterministic vector based on text hash
            seed = abs(hash(s)) % (2**31)
            rng = np.random.RandomState(seed)
            vec = rng.randn(self._dim).astype(np.float32)
            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            vectors.append(vec)
        return np.array(vectors)


@pytest.fixture
def mock_model() -> MockSentenceTransformer:
    return MockSentenceTransformer(dimension=384)


@pytest.fixture
def embedding_service(mock_model) -> EmbeddingService:
    return EmbeddingService(model=mock_model)


@pytest.fixture
def sample_chunk_metadata() -> ChunkMetadata:
    return ChunkMetadata(
        document_id="doc-1234",
        collection_id="col-5678",
        filename="test_doc.pdf",
        file_type="pdf",
        chunk_index=0,
        char_count=32,
        page_number=1,
        page_range=None,
        start_char=0,
        end_char=32,
    )


@pytest.fixture
def sample_document_chunk(sample_chunk_metadata) -> DocumentChunk:
    return DocumentChunk(
        chunk_id="doc-1234_c0000",
        document_id="doc-1234",
        collection_id="col-5678",
        filename="test_doc.pdf",
        file_type="pdf",
        chunk_index=0,
        chunk_text="This is a test chunk text for embedding.",
        char_count=40,
        page_number=1,
        page_range=None,
        metadata=sample_chunk_metadata,
    )


# ── 1. EmbeddedChunk Schema Tests ───────────────────────────────────────────


def test_embedded_chunk_valid_creation(sample_document_chunk):
    """Verify constructing EmbeddedChunk directly and via factory method."""
    vec = [0.1] * 384
    ec = EmbeddedChunk.from_document_chunk(
        chunk=sample_document_chunk,
        embedding=vec,
        embedding_dimension=384,
        model_name="BAAI/bge-small-en-v1.5",
    )
    assert ec.chunk_id == sample_document_chunk.chunk_id
    assert ec.document_id == sample_document_chunk.document_id
    assert ec.collection_id == sample_document_chunk.collection_id
    assert ec.chunk_text == sample_document_chunk.chunk_text
    assert ec.metadata == sample_document_chunk.metadata
    assert ec.embedding_dimension == 384
    assert ec.model_name == "BAAI/bge-small-en-v1.5"
    assert len(ec.embedding) == 384


def test_embedded_chunk_empty_or_whitespace_text_rejected(sample_chunk_metadata):
    """Verify empty or whitespace-only text is rejected."""
    vec = [0.1] * 384
    # Empty string is rejected by min_length or custom validator
    with pytest.raises(ValueError):
        EmbeddedChunk(
            chunk_id="c1",
            document_id="d1",
            collection_id="col1",
            chunk_text="",
            metadata=sample_chunk_metadata,
            embedding=vec,
            embedding_dimension=384,
            model_name="test-model",
        )

    # Whitespace-only string is rejected by custom validator
    with pytest.raises(ValueError, match="chunk_text cannot be empty or whitespace only"):
        EmbeddedChunk(
            chunk_id="c1",
            document_id="d1",
            collection_id="col1",
            chunk_text="   \n\t  ",
            metadata=sample_chunk_metadata,
            embedding=vec,
            embedding_dimension=384,
            model_name="test-model",
        )


def test_embedded_chunk_empty_vector_rejected(sample_document_chunk):
    """Verify empty embedding vector is rejected."""
    with pytest.raises(ValueError, match="embedding vector cannot be empty"):
        EmbeddedChunk(
            chunk_id="c1",
            document_id="d1",
            collection_id="col1",
            chunk_text="Valid chunk text",
            metadata=sample_document_chunk.metadata,
            embedding=[],
            embedding_dimension=0,
            model_name="test-model",
        )


def test_embedded_chunk_dimension_mismatch_rejected(sample_document_chunk):
    """Verify mismatch between embedding_dimension and actual vector length is rejected."""
    vec = [0.1] * 384
    with pytest.raises(ValueError, match="embedding_dimension .* must match len\\(embedding\\)"):
        EmbeddedChunk(
            chunk_id="c1",
            document_id="d1",
            collection_id="col1",
            chunk_text="Valid text",
            metadata=sample_document_chunk.metadata,
            embedding=vec,
            embedding_dimension=256,
            model_name="test-model",
        )


def test_embedded_chunk_non_finite_rejected(sample_document_chunk):
    """Verify NaN and Inf values in embedding vector are rejected."""
    vec_nan = [0.1] * 383 + [float("nan")]
    with pytest.raises(ValueError, match="must be finite"):
        EmbeddedChunk(
            chunk_id="c1",
            document_id="d1",
            collection_id="col1",
            chunk_text="Valid text",
            metadata=sample_document_chunk.metadata,
            embedding=vec_nan,
            embedding_dimension=384,
            model_name="test-model",
        )

    vec_inf = [0.1] * 383 + [float("inf")]
    with pytest.raises(ValueError, match="must be finite"):
        EmbeddedChunk(
            chunk_id="c1",
            document_id="d1",
            collection_id="col1",
            chunk_text="Valid text",
            metadata=sample_document_chunk.metadata,
            embedding=vec_inf,
            embedding_dimension=384,
            model_name="test-model",
        )


def test_embedded_chunk_serialization(sample_document_chunk):
    """Verify EmbeddedChunk serializes to dict and JSON cleanly."""
    vec = [0.5] * 384
    ec = EmbeddedChunk.from_document_chunk(
        chunk=sample_document_chunk,
        embedding=vec,
        embedding_dimension=384,
        model_name="BAAI/bge-small-en-v1.5",
    )
    d = ec.model_dump()
    assert d["chunk_id"] == sample_document_chunk.chunk_id
    assert d["embedding_dimension"] == 384
    assert len(d["embedding"]) == 384

    # JSON round-trip
    json_str = ec.model_dump_json()
    reconstructed = EmbeddedChunk.model_validate_json(json_str)
    assert reconstructed.chunk_id == ec.chunk_id
    assert reconstructed.embedding == ec.embedding


# ── 2. EmbeddingService Configuration & Singleton Caching ───────────────────


def test_embedding_service_defaults():
    """Verify default configuration attributes align with Settings."""
    service = EmbeddingService(model=MockSentenceTransformer())
    assert service.model_name == "BAAI/bge-small-en-v1.5"
    assert service.device == "cpu"
    assert service.batch_size == 1
    assert service.normalize is True
    assert service.dimension == 384


def test_embedding_service_custom_configuration():
    """Verify overrides for model_name, device, batch_size, and normalize."""
    custom_model = MockSentenceTransformer(dimension=512)
    service = EmbeddingService(
        model=custom_model,
        model_name="custom/model",
        device="cpu",
        batch_size=4,
        normalize=False,
    )
    assert service.model_name == "custom/model"
    assert service.batch_size == 4
    assert service.normalize is False
    assert service.dimension == 512


def test_embedding_service_singleton_caching():
    """Verify that _get_model() caches the model at class level and returns the same instance."""
    EmbeddingService.clear_cache()
    try:
        mock_loaded_model = MockSentenceTransformer(dimension=384)
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_loaded_model) as mock_st_init:
            service1 = EmbeddingService()
            m1 = service1._get_model()

            service2 = EmbeddingService()
            m2 = service2._get_model()

            assert m1 is m2
            assert m1 is mock_loaded_model
            # SentenceTransformer initialized exactly once across multiple service instances
            assert mock_st_init.call_count == 1
    finally:
        EmbeddingService.clear_cache()


def test_embedding_service_clear_cache():
    """Verify clear_cache() resets the cached model."""
    EmbeddingService._cached_model = MagicMock()
    assert EmbeddingService._cached_model is not None
    EmbeddingService.clear_cache()
    assert EmbeddingService._cached_model is None


def test_embedding_service_ram_guardrail_failure():
    """Verify that insufficient RAM (<500 MB) raises ModelLoadError."""
    EmbeddingService.clear_cache()
    try:
        # Mock psutil virtual_memory to report 200 MB available (below 500 MB threshold)
        mock_mem = MagicMock()
        mock_mem.available = 200 * 1024 * 1024  # 200 MB
        with patch("psutil.virtual_memory", return_value=mock_mem):
            service = EmbeddingService(min_ram_mb=500)
            with pytest.raises(ModelLoadError, match="Insufficient RAM to load embedding model"):
                service._get_model()
    finally:
        EmbeddingService.clear_cache()


# ── 3. embed_text() Tests ───────────────────────────────────────────────────


def test_embed_text_success(embedding_service):
    """Verify embed_text produces a unit-normalized vector of 384 floats."""
    text = "Artificial intelligence and machine learning in agentic RAG."
    vec = embedding_service.embed_text(text)

    assert isinstance(vec, list)
    assert len(vec) == 384
    assert all(isinstance(x, float) for x in vec)

    # Unit norm check: ||v|| == 1.0 within floating point precision
    norm = math.sqrt(sum(x * x for x in vec))
    assert pytest.approx(norm, rel=1e-5) == 1.0


def test_embed_text_invalid_inputs(embedding_service):
    """Verify embed_text rejects empty string, whitespace, and non-strings."""
    with pytest.raises(InvalidInputError, match="cannot be empty or whitespace only"):
        embedding_service.embed_text("")

    with pytest.raises(InvalidInputError, match="cannot be empty or whitespace only"):
        embedding_service.embed_text("    \n\t  ")

    with pytest.raises(InvalidInputError, match="must be a string"):
        embedding_service.embed_text(None)  # type: ignore

    with pytest.raises(InvalidInputError, match="must be a string"):
        embedding_service.embed_text(12345)  # type: ignore

    with pytest.raises(InvalidInputError, match="must be a string"):
        embedding_service.embed_text(["some text"])  # type: ignore


# ── 4. embed_batch() Tests ──────────────────────────────────────────────────


def test_embed_batch_success(embedding_service):
    """Verify embed_batch returns a list of normalized vectors matching input size."""
    texts = [
        "First sentence for embedding.",
        "Second document chunk text.",
        "Third knowledge assistant query.",
    ]
    vectors = embedding_service.embed_batch(texts, batch_size=2)

    assert len(vectors) == 3
    for vec in vectors:
        assert len(vec) == 384
        norm = math.sqrt(sum(x * x for x in vec))
        assert pytest.approx(norm, rel=1e-5) == 1.0


def test_embed_batch_invalid_inputs(embedding_service):
    """Verify embed_batch validates sequence structure and content."""
    # Empty list
    with pytest.raises(InvalidInputError, match="sequence cannot be empty"):
        embedding_service.embed_batch([])

    # Non-sequence
    with pytest.raises(InvalidInputError, match="must be a non-string sequence"):
        embedding_service.embed_batch("not a sequence")  # type: ignore

    # None
    with pytest.raises(InvalidInputError, match="must be a non-string sequence"):
        embedding_service.embed_batch(None)  # type: ignore

    # List with invalid elements
    with pytest.raises(InvalidInputError, match="cannot be empty or whitespace only"):
        embedding_service.embed_batch(["valid text", "   ", "another valid text"])

    with pytest.raises(InvalidInputError, match="must be a string"):
        embedding_service.embed_batch(["valid text", 100])  # type: ignore

    # Invalid batch size
    with pytest.raises(InvalidInputError, match="Batch size must be >= 1"):
        embedding_service.embed_batch(["valid text"], batch_size=0)


def test_embed_batch_inference_failure():
    """Verify model inference errors are wrapped in EmbeddingInferenceError."""
    broken_model = MagicMock()
    broken_model.encode.side_effect = RuntimeError("CUDA/CPU tensor allocation failure")
    service = EmbeddingService(model=broken_model)

    with pytest.raises(EmbeddingInferenceError, match="Embedding inference failed"):
        service.embed_batch(["valid test text"])


# ── 5. embed_chunks() Tests ─────────────────────────────────────────────────


def test_embed_chunks_success(embedding_service, sample_document_chunk):
    """Verify embed_chunks generates EmbeddedChunk instances preserving chunk IDs and metadata."""
    chunk2 = DocumentChunk(
        chunk_id="doc-1234_c0001",
        document_id="doc-1234",
        collection_id="col-5678",
        filename="test_doc.pdf",
        file_type="pdf",
        chunk_index=1,
        chunk_text="Second chunk content about evaluation metrics.",
        char_count=46,
        page_number=2,
        page_range=None,
        metadata=sample_document_chunk.metadata.model_copy(update={"chunk_index": 1, "page_number": 2}),
    )

    chunks = [sample_document_chunk, chunk2]
    embedded = embedding_service.embed_chunks(chunks)

    assert len(embedded) == 2
    for orig, ec in zip(chunks, embedded):
        assert isinstance(ec, EmbeddedChunk)
        assert ec.chunk_id == orig.chunk_id
        assert ec.document_id == orig.document_id
        assert ec.collection_id == orig.collection_id
        assert ec.chunk_text == orig.chunk_text
        assert ec.embedding_dimension == 384
        assert len(ec.embedding) == 384
        assert ec.model_name == embedding_service.model_name


def test_embed_chunks_empty_sequence(embedding_service):
    """Verify embed_chunks with empty sequence returns an empty list without error."""
    result = embedding_service.embed_chunks([])
    assert result == []


def test_embed_chunks_invalid_inputs(embedding_service):
    """Verify embed_chunks validates sequence items."""
    with pytest.raises(InvalidInputError, match="must be a sequence"):
        embedding_service.embed_chunks(None)  # type: ignore

    with pytest.raises(InvalidInputError, match="is not a DocumentChunk instance"):
        embedding_service.embed_chunks(["not a chunk"])  # type: ignore


# ── 6. compute_similarity() Tests ───────────────────────────────────────────


def test_compute_similarity_identical_vectors():
    """Verify cosine similarity of identical vectors is 1.0."""
    vec = [0.2, 0.5, 0.8, -0.3]
    sim = EmbeddingService.compute_similarity(vec, vec)
    assert pytest.approx(sim, rel=1e-5) == 1.0


def test_compute_similarity_opposite_vectors():
    """Verify cosine similarity of opposite vectors is -1.0."""
    v1 = [1.0, 0.0, 0.0]
    v2 = [-1.0, 0.0, 0.0]
    sim = EmbeddingService.compute_similarity(v1, v2)
    assert pytest.approx(sim, rel=1e-5) == -1.0


def test_compute_similarity_orthogonal_vectors():
    """Verify cosine similarity of orthogonal vectors is 0.0."""
    v1 = [1.0, 0.0, 0.0]
    v2 = [0.0, 1.0, 0.0]
    sim = EmbeddingService.compute_similarity(v1, v2)
    assert pytest.approx(sim, abs=1e-5) == 0.0


def test_compute_similarity_validation():
    """Verify error handling on mismatched dimensions, empty vectors, and NaNs."""
    with pytest.raises(InvalidInputError, match="dimensions do not match"):
        EmbeddingService.compute_similarity([1.0, 2.0], [1.0, 2.0, 3.0])

    with pytest.raises(InvalidInputError, match="Embeddings cannot be empty"):
        EmbeddingService.compute_similarity([], [])

    with pytest.raises(InvalidInputError, match="Embeddings cannot be None"):
        EmbeddingService.compute_similarity(None, [1.0])  # type: ignore

    with pytest.raises(InvalidInputError, match="must be finite"):
        EmbeddingService.compute_similarity([float("nan"), 1.0], [1.0, 1.0])

    with pytest.raises(InvalidInputError, match="must be finite"):
        EmbeddingService.compute_similarity([1.0, 1.0], [float("inf"), 1.0])


# ── 7. DocumentProcessor Integration Tests ──────────────────────────────────


@pytest.mark.asyncio
async def test_document_processor_embed_false_preserves_phase4(
    db_session: AsyncSession,
    test_upload_dir: Path,
):
    """Verify process_document(embed=False) keeps embedded_chunks=None and sets status='indexed'."""
    storage = FileStorageService(upload_dir=test_upload_dir)
    processor = DocumentProcessor(storage_service=storage)

    # 1. Seed collection and document
    col = Collection(name="Test Col")
    db_session.add(col)
    await db_session.flush()

    file_content = b"Dayananda Sagar University Regulations. Academic requirements."
    col_dir = test_upload_dir / col.id
    col_dir.mkdir(parents=True, exist_ok=True)
    stored_filename = "handbook_stored.txt"
    (col_dir / stored_filename).write_bytes(file_content)

    doc = Document(
        collection_id=col.id,
        filename="handbook.txt",
        stored_filename=stored_filename,
        file_type="txt",
        size_bytes=len(file_content),
        sha256="dummy_sha256_1",
        status=DocumentStatus.PENDING.value,
    )
    db_session.add(doc)
    await db_session.commit()
    await db_session.refresh(doc)

    # 2. Process with embed=False (Phase 4 baseline)
    result = await processor.process_document(doc.id, db_session, embed=False)

    assert result.status == DocumentStatus.INDEXED
    assert result.chunk_count >= 1
    assert result.embedded_chunks is None

    # Verify DB record
    await db_session.refresh(doc)
    assert doc.status == DocumentStatus.INDEXED.value
    assert doc.chunk_count == result.chunk_count
    assert doc.error is None


@pytest.mark.asyncio
async def test_document_processor_embed_true_success(
    db_session: AsyncSession,
    test_upload_dir: Path,
    embedding_service: EmbeddingService,
):
    """Verify process_document(embed=True) generates embedded_chunks and marks document 'indexed'."""
    storage = FileStorageService(upload_dir=test_upload_dir)
    processor = DocumentProcessor(
        storage_service=storage,
        embedding_service=embedding_service,
    )

    col = Collection(name="Embed Test Col")
    db_session.add(col)
    await db_session.flush()

    file_content = b"Intelligent Knowledge Assistant Using Agentic RAG and Large Language Models."
    col_dir = test_upload_dir / col.id
    col_dir.mkdir(parents=True, exist_ok=True)
    stored_filename = "research_stored.txt"
    (col_dir / stored_filename).write_bytes(file_content)

    doc = Document(
        collection_id=col.id,
        filename="research.txt",
        stored_filename=stored_filename,
        file_type="txt",
        size_bytes=len(file_content),
        sha256="dummy_sha256_2",
        status=DocumentStatus.PENDING.value,
    )
    db_session.add(doc)
    await db_session.commit()
    await db_session.refresh(doc)

    # Process with embed=True
    result = await processor.process_document(doc.id, db_session, embed=True)

    assert result.status == DocumentStatus.INDEXED
    assert result.chunk_count >= 1
    assert result.embedded_chunks is not None
    assert len(result.embedded_chunks) == result.chunk_count

    first_embedded = result.embedded_chunks[0]
    assert isinstance(first_embedded, EmbeddedChunk)
    assert first_embedded.chunk_id == result.chunks[0].chunk_id
    assert len(first_embedded.embedding) == 384
    assert first_embedded.embedding_dimension == 384
    assert first_embedded.model_name == embedding_service.model_name

    # Verify DB record status
    await db_session.refresh(doc)
    assert doc.status == DocumentStatus.INDEXED.value
    assert doc.error is None


@pytest.mark.asyncio
async def test_document_processor_embed_true_without_service_fails(
    db_session: AsyncSession,
    test_upload_dir: Path,
):
    """Verify requesting embed=True when embedding_service is None fails safely and marks document 'failed'."""
    storage = FileStorageService(upload_dir=test_upload_dir)
    processor = DocumentProcessor(storage_service=storage, embedding_service=None)

    col = Collection(name="Fail Col")
    db_session.add(col)
    await db_session.flush()

    file_content = b"Sample content"
    col_dir = test_upload_dir / col.id
    col_dir.mkdir(parents=True, exist_ok=True)
    stored_filename = "doc_stored.txt"
    (col_dir / stored_filename).write_bytes(file_content)

    doc = Document(
        collection_id=col.id,
        filename="doc.txt",
        stored_filename=stored_filename,
        file_type="txt",
        size_bytes=len(file_content),
        sha256="dummy_sha256_3",
        status=DocumentStatus.PENDING.value,
    )
    db_session.add(doc)
    await db_session.commit()
    await db_session.refresh(doc)

    with pytest.raises(DocumentProcessorError, match="no EmbeddingService is configured"):
        await processor.process_document(doc.id, db_session, embed=True)

    # Document must be transitioned to 'failed', NEVER left as 'processing' or 'indexed'
    await db_session.refresh(doc)
    assert doc.status == DocumentStatus.FAILED.value
    assert "no EmbeddingService is configured" in doc.error


@pytest.mark.asyncio
async def test_document_processor_embed_true_inference_failure_rollback(
    db_session: AsyncSession,
    test_upload_dir: Path,
):
    """Verify that if embedding fails during processing, document status becomes 'failed'."""
    broken_model = MagicMock()
    broken_model.encode.side_effect = RuntimeError("Out of memory during model inference")
    failing_service = EmbeddingService(model=broken_model)

    storage = FileStorageService(upload_dir=test_upload_dir)
    processor = DocumentProcessor(
        storage_service=storage,
        embedding_service=failing_service,
    )

    col = Collection(name="Rollback Col")
    db_session.add(col)
    await db_session.flush()

    file_content = b"Some content to embed"
    col_dir = test_upload_dir / col.id
    col_dir.mkdir(parents=True, exist_ok=True)
    stored_filename = "fail_doc_stored.txt"
    (col_dir / stored_filename).write_bytes(file_content)

    doc = Document(
        collection_id=col.id,
        filename="fail_doc.txt",
        stored_filename=stored_filename,
        file_type="txt",
        size_bytes=len(file_content),
        sha256="dummy_sha256_4",
        status=DocumentStatus.PENDING.value,
    )
    db_session.add(doc)
    await db_session.commit()
    await db_session.refresh(doc)

    with pytest.raises(DocumentProcessorError, match="Out of memory"):
        await processor.process_document(doc.id, db_session, embed=True)

    # Document must be transitioned to 'failed', NEVER left as 'indexed'
    await db_session.refresh(doc)
    assert doc.status == DocumentStatus.FAILED.value
    assert "Out of memory" in doc.error


# ── 8. Real Model Smoke Test (Marked) ───────────────────────────────────────


@pytest.mark.real_model
def test_real_model_inference_smoke():
    """
    Smoke test using the real BGE model cached at D:\\huggingface.
    Only runs if cache directory exists; skipped otherwise.
    """
    cache_dir = "D:/huggingface"
    if not os.path.exists(cache_dir):
        pytest.skip(f"Real model cache directory '{cache_dir}' does not exist.")

    # Instantiate real service pointing to D:/huggingface
    service = EmbeddingService(cache_dir=cache_dir, min_ram_mb=300)

    text1 = "Artificial intelligence and machine learning"
    text2 = "Deep learning and neural network architectures"
    text3 = "Culinary recipes and baking bread"

    v1 = service.embed_text(text1)
    v2 = service.embed_text(text2)
    v3 = service.embed_text(text3)

    assert len(v1) == 384
    assert len(v2) == 384
    assert len(v3) == 384

    # Real semantic similarity: Sim(AI, ML) > Sim(AI, Cooking)
    sim_tech = EmbeddingService.compute_similarity(v1, v2)
    sim_unrelated = EmbeddingService.compute_similarity(v1, v3)

    assert sim_tech > sim_unrelated
    assert sim_tech > 0.70
