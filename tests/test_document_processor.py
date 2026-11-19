"""
tests/test_document_processor.py
─────────────────────────────────────────────────────────────
Integration tests for DocumentProcessor service, including:
  - Phase 4 document processing endpoints (PDF, DOCX, TXT, OCR-free scanned PDF, 404s).
  - Phase 6B Step 3 integration: DocumentProcessor -> EmbeddingService -> VectorStoreService.
  - Safe versioned indexing, staging, verification, promotion, retirement, and rollback.
─────────────────────────────────────────────────────────────
"""

import io
import math
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, patch

import docx
import pymupdf as fitz
import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.collection import Collection
from backend.models.document import Document
from backend.schemas.chunk import DocumentChunk, ProcessedDocument
from backend.schemas.document import DocumentStatus
from backend.schemas.embedding import EmbeddedChunk
from backend.services.document_processor import (
    DocumentNotFoundError,
    DocumentProcessor,
    DocumentProcessorError,
)
from backend.services.embedding_service import EmbeddingService
from backend.services.file_storage import FileStorageService
from backend.services.text_chunker import TextChunker
from backend.services.vector_store import (
    VectorStorageError,
    VectorStoreService,
)


# ── Test Helpers & Mock Model ────────────────────────────────────────────────


def make_dummy_vector(dim: int = 384, seed: int = 0) -> List[float]:
    """Generate a deterministic unit-normalized vector for testing."""
    raw = [float(((i + seed) % 19) + 1) for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


class MockEmbeddingModel:
    """Mock SentenceTransformer model providing fast, deterministic CPU embeddings."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            return make_dummy_vector(self.dim, seed=len(texts))
        return [make_dummy_vector(self.dim, seed=i + len(t)) for i, t in enumerate(texts)]


@pytest.fixture
def mock_embedding_service() -> EmbeddingService:
    """Provides an isolated EmbeddingService with a deterministic mock model."""
    return EmbeddingService(model=MockEmbeddingModel(dim=384))


@pytest.fixture
def temp_chroma_dir(tmp_path: Path) -> Path:
    """Provides an isolated temporary directory for ChromaDB vector storage tests."""
    chroma_dir = tmp_path / "chromadb_proc_test"
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


# ── Fixtures for Legacy Phase 4 Endpoint Tests ───────────────────────────────


@pytest.fixture
def make_pdf_bytes() -> bytes:
    """Create multi-page PDF in memory."""
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((50, 50), "Dayananda Sagar University Regulations. Section 1: Academic Code. Page 1 text.")
    p2 = doc.new_page()
    p2.insert_text((50, 50), "Section 2: Minimum Credits, Evaluation System, and Degree Requirements. Page 2 text.")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
def make_scanned_pdf_bytes() -> bytes:
    """Create PDF with empty page (simulates scanned image PDF without text layer)."""
    doc = fitz.open()
    doc.new_page()  # Blank page
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
def make_docx_bytes() -> bytes:
    """Create DOCX with paragraphs and a table in memory."""
    doc = docx.Document()
    doc.add_heading("Agentic RAG Syllabus 2026", level=1)
    doc.add_paragraph("Comprehensive study of Retrieval-Augmented Generation using Python.")
    table = doc.add_table(rows=2, cols=2)
    table.rows[0].cells[0].text = "Phase"
    table.rows[0].cells[1].text = "Objective"
    table.rows[1].cells[0].text = "Phase 4"
    table.rows[1].cells[1].text = "Parsing, extraction, cleaning and chunking"

    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()


# ── Phase 4 Existing Endpoint Tests (Preserved) ──────────────────────────────


@pytest.mark.asyncio
async def test_process_pdf_end_to_end(client: AsyncClient, make_pdf_bytes: bytes):
    """Verify uploading a PDF, processing it, and inspecting chunk output and DB status."""
    col_resp = await client.post("/api/collections", json={"name": "PDF Process Col"})
    assert col_resp.status_code == 201
    col_id = col_resp.json()["id"]

    files = {"file": ("handbook.pdf", io.BytesIO(make_pdf_bytes), "application/pdf")}
    up_resp = await client.post(f"/api/collections/{col_id}/documents", files=files)
    assert up_resp.status_code == 201
    doc_id = up_resp.json()["id"]
    assert up_resp.json()["status"] == "pending"
    assert up_resp.json()["page_count"] == 0
    assert up_resp.json()["chunk_count"] == 0

    proc_resp = await client.post(f"/api/documents/{doc_id}/process")
    assert proc_resp.status_code == 200
    data = proc_resp.json()
    assert data["status"] == "indexed"
    assert data["page_count"] == 2
    assert data["chunk_count"] >= 1
    assert data["total_characters"] > 0
    assert len(data["chunks"]) == data["chunk_count"]

    first_chunk = data["chunks"][0]
    assert first_chunk["document_id"] == doc_id
    assert first_chunk["collection_id"] == col_id
    assert first_chunk["file_type"] == "pdf"
    assert first_chunk["page_number"] in (1, 2)
    assert first_chunk["char_count"] == len(first_chunk["chunk_text"])

    get_resp = await client.get(f"/api/documents/{doc_id}")
    assert get_resp.status_code == 200
    db_doc = get_resp.json()
    assert db_doc["status"] == "indexed"
    assert db_doc["page_count"] == 2
    assert db_doc["chunk_count"] == data["chunk_count"]
    assert db_doc["error"] is None


@pytest.mark.asyncio
async def test_process_docx_end_to_end(client: AsyncClient, make_docx_bytes: bytes):
    """Verify DOCX processing extracts paragraphs, tables, and sets page_number=None."""
    col_resp = await client.post("/api/collections", json={"name": "DOCX Process Col"})
    col_id = col_resp.json()["id"]

    files = {"file": ("syllabus.docx", io.BytesIO(make_docx_bytes), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
    up_resp = await client.post(f"/api/collections/{col_id}/documents", files=files)
    assert up_resp.status_code == 201
    doc_id = up_resp.json()["id"]

    proc_resp = await client.post(f"/api/documents/{doc_id}/process")
    assert proc_resp.status_code == 200
    data = proc_resp.json()
    assert data["status"] == "indexed"
    assert data["page_count"] == 1
    assert data["chunk_count"] >= 1

    for chunk in data["chunks"]:
        assert chunk["page_number"] is None
        assert chunk["page_range"] is None
        assert chunk["metadata"]["page_number"] is None


@pytest.mark.asyncio
async def test_process_txt_end_to_end(client: AsyncClient):
    """Verify TXT processing cleans and chunks text with page_number=None."""
    col_resp = await client.post("/api/collections", json={"name": "TXT Process Col"})
    col_id = col_resp.json()["id"]

    txt_content = "Regulations Chapter 1.\nParagraph 1 detailing student honor code.\n\nParagraph 2 rules."
    files = {"file": ("rules.txt", io.BytesIO(txt_content.encode("utf-8")), "text/plain")}
    up_resp = await client.post(f"/api/collections/{col_id}/documents", files=files)
    doc_id = up_resp.json()["id"]

    proc_resp = await client.post(f"/api/documents/{doc_id}/process")
    assert proc_resp.status_code == 200
    data = proc_resp.json()
    assert data["status"] == "indexed"
    assert data["chunk_count"] >= 1
    for chunk in data["chunks"]:
        assert chunk["page_number"] is None


@pytest.mark.asyncio
async def test_process_scanned_pdf_fails_gracefully(client: AsyncClient, make_scanned_pdf_bytes: bytes):
    """Verify scanned PDF with no text fails, records error, and status is not stuck in processing."""
    col_resp = await client.post("/api/collections", json={"name": "Scanned Col"})
    col_id = col_resp.json()["id"]

    files = {"file": ("scanned.pdf", io.BytesIO(make_scanned_pdf_bytes), "application/pdf")}
    up_resp = await client.post(f"/api/collections/{col_id}/documents", files=files)
    doc_id = up_resp.json()["id"]

    proc_resp = await client.post(f"/api/documents/{doc_id}/process")
    assert proc_resp.status_code == 422
    assert "ocr not supported" in proc_resp.json()["detail"].lower()

    get_resp = await client.get(f"/api/documents/{doc_id}")
    assert get_resp.status_code == 200
    db_doc = get_resp.json()
    assert db_doc["status"] == "failed"
    assert db_doc["error"] is not None
    assert "ocr not supported" in db_doc["error"].lower()


@pytest.mark.asyncio
async def test_process_nonexistent_document(client: AsyncClient):
    """Verify 404 when processing non-existent document ID."""
    resp = await client.post("/api/documents/missing-uuid-12345/process")
    assert resp.status_code == 404


# ── Phase 6B Step 3 Integration Tests (15 Mandatory Scenarios) ───────────────


async def create_test_doc_on_disk(
    db_session: AsyncSession,
    test_upload_dir: Path,
    content: bytes = b"Artificial Intelligence and Machine Learning regulations.\nParagraph 1.\n\nParagraph 2.",
    filename: str = "regulations.txt",
    file_type: str = "txt",
    col_name: str = "Test Integration Col",
) -> tuple[str, str, str]:
    """Helper to persist a collection, document record in SQLite, and source file on disk. Returns (col_id, doc_id, stored_filename)."""
    col = Collection(name=col_name)
    db_session.add(col)
    await db_session.flush()

    col_id = str(col.id)
    col_dir = test_upload_dir / col_id
    col_dir.mkdir(parents=True, exist_ok=True)
    stored_filename = f"stored_{filename}"
    (col_dir / stored_filename).write_bytes(content)

    doc = Document(
        collection_id=col_id,
        filename=filename,
        stored_filename=stored_filename,
        file_type=file_type,
        size_bytes=len(content),
        sha256=f"sha_{len(content)}_{col_id[:8]}",
        status=DocumentStatus.PENDING.value,
        version=1,
    )
    db_session.add(doc)
    await db_session.commit()
    doc_id = str(doc.id)
    return col_id, doc_id, stored_filename


@pytest.mark.asyncio
async def test_1_processing_without_embedding(
    db_session: AsyncSession,
    test_upload_dir: Path,
    mock_embedding_service: EmbeddingService,
    vector_service: VectorStoreService,
):
    """1. Test processing without embedding (embed=False) preserves Phase 4 behavior."""
    storage = FileStorageService(upload_dir=test_upload_dir)
    processor = DocumentProcessor(
        storage_service=storage,
        embedding_service=mock_embedding_service,
        vector_store=vector_service,
    )
    col_id, doc_id, _ = await create_test_doc_on_disk(db_session, test_upload_dir, col_name="Col 1")

    result = await processor.process_document(doc_id, db_session, embed=False)

    assert result.status == DocumentStatus.INDEXED
    assert result.chunk_count >= 1
    assert result.embedded_chunks is None
    # ChromaDB must have 0 vectors written
    assert vector_service.count(col_id) == 0


@pytest.mark.asyncio
async def test_2_successful_embedding_and_vector_storage(
    db_session: AsyncSession,
    test_upload_dir: Path,
    mock_embedding_service: EmbeddingService,
    vector_service: VectorStoreService,
):
    """2. Test successful embedding and vector storage (embed=True)."""
    storage = FileStorageService(upload_dir=test_upload_dir)
    processor = DocumentProcessor(
        storage_service=storage,
        embedding_service=mock_embedding_service,
        vector_store=vector_service,
    )
    col_id, doc_id, _ = await create_test_doc_on_disk(db_session, test_upload_dir, col_name="Col 2")

    result = await processor.process_document(doc_id, db_session, embed=True)

    assert result.status == DocumentStatus.INDEXED
    assert result.chunk_count >= 1
    assert result.embedded_chunks is not None
    assert len(result.embedded_chunks) == result.chunk_count
    # All vectors stored in ChromaDB
    assert vector_service.count(col_id) == result.chunk_count

    # Verify SQLite status
    res = await db_session.execute(select(Document).where(Document.id == doc_id))
    doc = res.scalar_one()
    assert doc.status == DocumentStatus.INDEXED.value
    assert doc.version == 1
    assert doc.chunk_count == result.chunk_count


@pytest.mark.asyncio
async def test_3_correct_collection_mapping(
    db_session: AsyncSession,
    test_upload_dir: Path,
    mock_embedding_service: EmbeddingService,
    vector_service: VectorStoreService,
):
    """3. Test correct collection mapping rule ('col_' + UUID with _)."""
    storage = FileStorageService(upload_dir=test_upload_dir)
    processor = DocumentProcessor(
        storage_service=storage,
        embedding_service=mock_embedding_service,
        vector_store=vector_service,
    )
    col_id, doc_id, _ = await create_test_doc_on_disk(db_session, test_upload_dir, col_name="Col 3")

    await processor.process_document(doc_id, db_session, embed=True)

    expected_chroma_col_name = f"col_{col_id.replace('-', '_')}"
    existing_cols = vector_service.list_collections()
    assert expected_chroma_col_name in existing_cols


@pytest.mark.asyncio
async def test_4_correct_document_and_chunk_metadata(
    db_session: AsyncSession,
    test_upload_dir: Path,
    mock_embedding_service: EmbeddingService,
    vector_service: VectorStoreService,
):
    """4. Test stored chunks contain correct metadata fields and values."""
    storage = FileStorageService(upload_dir=test_upload_dir)
    processor = DocumentProcessor(
        storage_service=storage,
        embedding_service=mock_embedding_service,
        vector_store=vector_service,
    )
    col_id, doc_id, _ = await create_test_doc_on_disk(
        db_session,
        test_upload_dir,
        content=b"Sample content for metadata test verification.",
        filename="meta_test.txt",
        col_name="Col 4",
    )

    result = await processor.process_document(doc_id, db_session, embed=True)
    chunk_ids = [c.chunk_id for c in result.chunks]

    records = vector_service.get_chunks_by_ids(col_id, chunk_ids)
    assert len(records["metadatas"]) == len(chunk_ids)

    first_meta = records["metadatas"][0]
    assert first_meta["document_id"] == doc_id
    assert first_meta["collection_id"] == col_id
    assert first_meta["filename"] == "meta_test.txt"
    assert first_meta["file_type"] == "txt"
    assert first_meta["chunk_index"] == 0
    assert first_meta["version"] == 1
    assert isinstance(first_meta["char_count"], int)
    # Scalar types only
    for k, v in first_meta.items():
        assert isinstance(v, (str, int, float, bool))


@pytest.mark.asyncio
async def test_5_correct_vector_count_after_indexing(
    db_session: AsyncSession,
    test_upload_dir: Path,
    mock_embedding_service: EmbeddingService,
    vector_service: VectorStoreService,
):
    """5. Test vector count in ChromaDB exactly matches chunk count."""
    storage = FileStorageService(upload_dir=test_upload_dir)
    small_chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    processor = DocumentProcessor(
        storage_service=storage,
        chunker=small_chunker,
        embedding_service=mock_embedding_service,
        vector_store=vector_service,
    )
    long_content = b"Paragraph one of test content. " * 15
    col_id, doc_id, _ = await create_test_doc_on_disk(
        db_session, test_upload_dir, content=long_content, col_name="Col 5"
    )

    result = await processor.process_document(doc_id, db_session, embed=True)
    assert result.chunk_count > 1
    assert vector_service.count(col_id) == result.chunk_count


@pytest.mark.asyncio
async def test_6_repeated_indexing(
    db_session: AsyncSession,
    test_upload_dir: Path,
    mock_embedding_service: EmbeddingService,
    vector_service: VectorStoreService,
):
    """6. Test repeated indexing of unchanged document updates in place without duplicate accumulation."""
    storage = FileStorageService(upload_dir=test_upload_dir)
    processor = DocumentProcessor(
        storage_service=storage,
        embedding_service=mock_embedding_service,
        vector_store=vector_service,
    )
    col_id, doc_id, _ = await create_test_doc_on_disk(db_session, test_upload_dir, col_name="Col 6")

    # Initial indexing (version 1)
    res1 = await processor.process_document(doc_id, db_session, embed=True)
    initial_count = res1.chunk_count
    assert vector_service.count(col_id) == initial_count

    # Second indexing (staged as version 2, then replaces version 1)
    res2 = await processor.process_document(doc_id, db_session, embed=True)
    assert res2.chunk_count == initial_count
    # Count must NOT double; old version retired
    assert vector_service.count(col_id) == initial_count

    res = await db_session.execute(select(Document).where(Document.id == doc_id))
    doc = res.scalar_one()
    assert doc.version == 2
    assert doc.status == DocumentStatus.INDEXED.value


@pytest.mark.asyncio
async def test_7_successful_reindexing(
    db_session: AsyncSession,
    test_upload_dir: Path,
    mock_embedding_service: EmbeddingService,
    vector_service: VectorStoreService,
):
    """7. Test successful re-indexing after content modification."""
    storage = FileStorageService(upload_dir=test_upload_dir)
    chunker = TextChunker(chunk_size=100, chunk_overlap=10)
    processor = DocumentProcessor(
        storage_service=storage,
        chunker=chunker,
        embedding_service=mock_embedding_service,
        vector_store=vector_service,
    )
    col_id, doc_id, stored_filename = await create_test_doc_on_disk(
        db_session,
        test_upload_dir,
        content=b"Short original text content.",
        col_name="Col 7",
    )

    # Initial indexing: 1 chunk
    res1 = await processor.process_document(doc_id, db_session, embed=True)
    assert res1.chunk_count == 1
    assert vector_service.count(col_id) == 1

    # Update file on disk with much longer content
    stored_path = storage.get_stored_file_path(col_id, stored_filename)
    stored_path.write_bytes(b"Much longer revised content paragraph. " * 10)

    # Re-index
    res2 = await processor.process_document(doc_id, db_session, embed=True)
    assert res2.chunk_count > 1
    # Vector count in Chroma matches new chunk count
    assert vector_service.count(col_id) == res2.chunk_count

    res = await db_session.execute(select(Document).where(Document.id == doc_id))
    doc = res.scalar_one()
    assert doc.version == 2
    assert doc.chunk_count == res2.chunk_count


@pytest.mark.asyncio
async def test_8_failed_embedding(
    db_session: AsyncSession,
    test_upload_dir: Path,
    vector_service: VectorStoreService,
):
    """8. Test embedding failure marks document 'failed' and leaves 0 vectors in ChromaDB."""
    broken_model = MagicMock()
    broken_model.encode.side_effect = RuntimeError("GPU/CPU memory exhaustion during inference")
    failing_embedding_service = EmbeddingService(model=broken_model)

    storage = FileStorageService(upload_dir=test_upload_dir)
    processor = DocumentProcessor(
        storage_service=storage,
        embedding_service=failing_embedding_service,
        vector_store=vector_service,
    )
    col_id, doc_id, _ = await create_test_doc_on_disk(db_session, test_upload_dir, col_name="Col 8")

    with pytest.raises(DocumentProcessorError, match="memory exhaustion"):
        await processor.process_document(doc_id, db_session, embed=True)

    res = await db_session.execute(select(Document).where(Document.id == doc_id))
    doc = res.scalar_one()
    assert doc.status == DocumentStatus.FAILED.value
    assert "memory exhaustion" in doc.error
    assert vector_service.count(col_id) == 0


@pytest.mark.asyncio
async def test_9_chromadb_storage_failure(
    db_session: AsyncSession,
    test_upload_dir: Path,
    mock_embedding_service: EmbeddingService,
):
    """9. Test ChromaDB storage failure marks document 'failed' and rolls back."""
    failing_vector_store = MagicMock(spec=VectorStoreService)
    failing_vector_store.store_chunks.side_effect = VectorStorageError("ChromaDB disk write I/O error")

    storage = FileStorageService(upload_dir=test_upload_dir)
    processor = DocumentProcessor(
        storage_service=storage,
        embedding_service=mock_embedding_service,
        vector_store=failing_vector_store,
    )
    col_id, doc_id, _ = await create_test_doc_on_disk(db_session, test_upload_dir, col_name="Col 9")

    with pytest.raises(DocumentProcessorError, match="ChromaDB disk write I/O error"):
        await processor.process_document(doc_id, db_session, embed=True)

    res = await db_session.execute(select(Document).where(Document.id == doc_id))
    doc = res.scalar_one()
    assert doc.status == DocumentStatus.FAILED.value
    assert "disk write I/O error" in doc.error


@pytest.mark.asyncio
async def test_10_partial_batch_insertion_failure(
    db_session: AsyncSession,
    test_upload_dir: Path,
    mock_embedding_service: EmbeddingService,
    vector_service: VectorStoreService,
):
    """10. Test that mid-batch insertion failure cleans up staged chunks and leaves no orphans."""
    storage = FileStorageService(upload_dir=test_upload_dir)
    processor = DocumentProcessor(
        storage_service=storage,
        embedding_service=mock_embedding_service,
        vector_store=vector_service,
    )
    col_id, doc_id, _ = await create_test_doc_on_disk(db_session, test_upload_dir, col_name="Col 10")

    original_store = vector_service.store_chunks

    def faulty_store(chunks, collection_id=None):
        if len(chunks) > 0:
            original_store(chunks[:1], collection_id=collection_id)
        raise VectorStorageError("Mid-batch failure simulated")

    with patch.object(vector_service, "store_chunks", side_effect=faulty_store):
        with pytest.raises(DocumentProcessorError, match="Mid-batch failure simulated"):
            await processor.process_document(doc_id, db_session, embed=True)

    res = await db_session.execute(select(Document).where(Document.id == doc_id))
    doc = res.scalar_one()
    assert doc.status == DocumentStatus.FAILED.value

    # Staged partial chunks must have been deleted by the processor cleanup handler
    assert vector_service.count(col_id) == 0


@pytest.mark.asyncio
async def test_11_sqlite_transaction_failure(
    db_session: AsyncSession,
    test_upload_dir: Path,
    mock_embedding_service: EmbeddingService,
    vector_service: VectorStoreService,
):
    """11. Test SQLite transaction failure cleans up unpromoted staged vectors."""
    storage = FileStorageService(upload_dir=test_upload_dir)
    processor = DocumentProcessor(
        storage_service=storage,
        embedding_service=mock_embedding_service,
        vector_store=vector_service,
    )
    col_id, doc_id, _ = await create_test_doc_on_disk(db_session, test_upload_dir, col_name="Col 11")

    commit_count = 0
    real_commit = db_session.commit

    async def flaky_commit():
        nonlocal commit_count
        commit_count += 1
        if commit_count == 2:  # Promotion commit
            raise RuntimeError("Database connection lost during commit")
        await real_commit()

    with patch.object(db_session, "commit", side_effect=flaky_commit):
        with pytest.raises(DocumentProcessorError, match="Database connection lost"):
            await processor.process_document(doc_id, db_session, embed=True)

    # Unpromoted staged vectors in ChromaDB must be cleaned up
    assert vector_service.count(col_id) == 0


@pytest.mark.asyncio
async def test_12_preservation_of_old_valid_index_during_failed_reindexing(
    db_session: AsyncSession,
    test_upload_dir: Path,
    mock_embedding_service: EmbeddingService,
    vector_service: VectorStoreService,
):
    """
    12. CRITICAL SAFETY: Verify that a failed re-indexing operation NEVER destroys
    the previously valid index in ChromaDB.
    """
    storage = FileStorageService(upload_dir=test_upload_dir)
    processor = DocumentProcessor(
        storage_service=storage,
        embedding_service=mock_embedding_service,
        vector_store=vector_service,
    )
    col_id, doc_id, stored_filename = await create_test_doc_on_disk(
        db_session,
        test_upload_dir,
        content=b"Original valid document index version 1 content.",
        col_name="Col 12",
    )

    # Step 1: Initial indexing succeeds (Version 1)
    res1 = await processor.process_document(doc_id, db_session, embed=True)
    assert res1.status == DocumentStatus.INDEXED
    v1_count = res1.chunk_count
    assert v1_count >= 1
    assert vector_service.count(col_id) == v1_count

    v1_ids = [c.chunk_id for c in res1.chunks]
    v1_records_before = vector_service.get_chunks_by_ids(col_id, v1_ids)
    assert len(v1_records_before["ids"]) == v1_count

    # Step 2: Attempt re-indexing with simulated failure during vector storage
    stored_path = storage.get_stored_file_path(col_id, stored_filename)
    stored_path.write_bytes(b"Revised content that will fail during storage.")

    with patch.object(vector_service, "store_chunks", side_effect=VectorStorageError("Disk full during v2")):
        with pytest.raises(DocumentProcessorError, match="Disk full during v2"):
            await processor.process_document(doc_id, db_session, embed=True)

    # Step 3: Verify that the OLD valid index was PRESERVED in ChromaDB!
    res = await db_session.execute(select(Document).where(Document.id == doc_id))
    doc = res.scalar_one()
    assert doc.status == DocumentStatus.FAILED.value
    # Version remains 1 in SQLite
    assert doc.version == 1

    # ChromaDB count MUST STILL BE v1_count (old index was NOT wiped!)
    assert vector_service.count(col_id) == v1_count

    # The actual version 1 chunks must still exist intact with their original text
    v1_records_after = vector_service.get_chunks_by_ids(col_id, v1_ids)
    assert len(v1_records_after["ids"]) == v1_count
    assert v1_records_after["documents"] == v1_records_before["documents"]


@pytest.mark.asyncio
async def test_13_retry_and_recovery_after_failure(
    db_session: AsyncSession,
    test_upload_dir: Path,
    mock_embedding_service: EmbeddingService,
    vector_service: VectorStoreService,
):
    """13. Test retry after failure successfully indexes and recovers."""
    storage = FileStorageService(upload_dir=test_upload_dir)
    processor = DocumentProcessor(
        storage_service=storage,
        embedding_service=mock_embedding_service,
        vector_store=vector_service,
    )
    col_id, doc_id, _ = await create_test_doc_on_disk(db_session, test_upload_dir, col_name="Col 13")

    # Step 1: Cause initial failure
    with patch.object(mock_embedding_service, "embed_chunks", side_effect=RuntimeError("Transient error")):
        with pytest.raises(DocumentProcessorError):
            await processor.process_document(doc_id, db_session, embed=True)

    res = await db_session.execute(select(Document).where(Document.id == doc_id))
    doc = res.scalar_one()
    assert doc.status == DocumentStatus.FAILED.value
    assert vector_service.count(col_id) == 0

    # Step 2: Retry with working service -> must recover cleanly
    retry_result = await processor.process_document(doc_id, db_session, embed=True)
    assert retry_result.status == DocumentStatus.INDEXED
    assert vector_service.count(col_id) == retry_result.chunk_count

    res = await db_session.execute(select(Document).where(Document.id == doc_id))
    doc = res.scalar_one()
    assert doc.status == DocumentStatus.INDEXED.value
    assert doc.error is None


@pytest.mark.asyncio
async def test_14_document_deletion_and_vector_cleanup(
    db_session: AsyncSession,
    test_upload_dir: Path,
    mock_embedding_service: EmbeddingService,
    vector_service: VectorStoreService,
):
    """14. Test document deletion cleans up its vectors from ChromaDB."""
    storage = FileStorageService(upload_dir=test_upload_dir)
    processor = DocumentProcessor(
        storage_service=storage,
        embedding_service=mock_embedding_service,
        vector_store=vector_service,
    )
    col_id, doc_id, _ = await create_test_doc_on_disk(db_session, test_upload_dir, col_name="Col 14")

    # Index document
    result = await processor.process_document(doc_id, db_session, embed=True)
    assert vector_service.count(col_id) == result.chunk_count

    # Delete vectors via processor
    deleted_count = processor.delete_document_vectors(col_id, doc_id)
    assert deleted_count == result.chunk_count
    assert vector_service.count(col_id) == 0


@pytest.mark.asyncio
async def test_15_collection_isolation(
    db_session: AsyncSession,
    test_upload_dir: Path,
    mock_embedding_service: EmbeddingService,
    vector_service: VectorStoreService,
):
    """15. Test vector store operations in one collection do not affect other collections."""
    storage = FileStorageService(upload_dir=test_upload_dir)
    processor = DocumentProcessor(
        storage_service=storage,
        embedding_service=mock_embedding_service,
        vector_store=vector_service,
    )
    col_a_id, doc_a_id, _ = await create_test_doc_on_disk(
        db_session, test_upload_dir, content=b"Document in collection Alpha.", col_name="Col Alpha"
    )
    col_b_id, doc_b_id, _ = await create_test_doc_on_disk(
        db_session, test_upload_dir, content=b"Document in collection Beta.", col_name="Col Beta"
    )

    res_a = await processor.process_document(doc_a_id, db_session, embed=True)
    res_b = await processor.process_document(doc_b_id, db_session, embed=True)

    assert vector_service.count(col_a_id) == res_a.chunk_count
    assert vector_service.count(col_b_id) == res_b.chunk_count

    # Delete Doc A vectors
    processor.delete_document_vectors(col_a_id, doc_a_id)
    assert vector_service.count(col_a_id) == 0

    # Collection B must remain 100% intact
    assert vector_service.count(col_b_id) == res_b.chunk_count
