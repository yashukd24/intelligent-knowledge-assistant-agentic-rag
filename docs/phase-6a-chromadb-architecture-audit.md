# Phase 6A — ChromaDB Architecture and Compatibility Audit Report

**Project:** Intelligent Knowledge Assistant Using Agentic RAG and Large Language Models  
**Institution:** Dayananda Sagar University  
**Audit Scope:** Phase 6A — ChromaDB Architecture Planning & Compatibility Audit  
**Date:** 2026-11-07  
**Status:** ARCHITECTURE DESIGN & AUDIT COMPLETE (Ready for Review)

---

## 1. Executive Summary

Phase 6A conducts the technical architecture planning and compatibility investigation for integrating **ChromaDB** into the Intelligent Knowledge Assistant. This phase bridges Phase 5 (Embedding Generation with `BAAI/bge-small-en-v1.5` yielding 384-dimensional normalized vectors) and Phase 7 (Semantic Retrieval).

### Core Audit Outcomes
1. **Critical Dependency Finding:**
   The legacy placeholder version (`chromadb==0.5.23`) pinned in `requirements.txt` contains a fatal dependency conflict: it restricts `tokenizers<=0.20.3` and relies on `chroma-hnswlib==0.7.6` which lacks Python 3.13 Windows x64 binary wheels. Installing `0.5.23` would force-downgrade `tokenizers`, breaking `transformers==4.57.6` (which requires `tokenizers>=0.22.0`).
2. **Recommended Version:**
   **`chromadb==1.5.9`** is officially recommended. It provides pre-compiled `abi3` binary wheels (`chromadb-1.5.9-cp39-abi3-win_amd64.whl`), embeds the compiled Rust engine directly, supports Python 3.13 natively without C++ build tools, satisfies `tokenizers>=0.13.2` without upper bounds, supports Pydantic v2 and NumPy 2.x, and executes CPU-first.
3. **Storage Boundary:**
   Strict separation of concerns is defined: SQLite manages relational metadata, ingestion lifecycle states (`pending`, `processing`, `indexed`, `failed`), and deduplication; ChromaDB persists 384-dimensional dense vectors, chunk texts, and provenance metadata.
4. **Failure & Consistency Model:**
   Because distributed 2-phase transactions across SQLite and ChromaDB do not exist, an explicit sequential two-phase ingestion workflow is architected with automatic ChromaDB partial-vector cleanup and SQLite `failed` state guarantees.

---

## 2. Current Architecture Assessment

### 2.1 Existing State (Phases 1–5B)
- **Runtime Environment:** Python 3.13.3 (Windows x64) inside isolated `.venv`.
- **Relational Storage:** SQLite 3 with SQLAlchemy 2.0 (async via `aiosqlite`).
  - `collections` table: manages knowledge collections with cascade deletion.
  - `documents` table: tracks uploaded files, content hashes (SHA-256), page/chunk counts, and processing status (`pending`, `processing`, `indexed`, `failed`).
- **Document Processing Pipeline (`DocumentProcessor`):**
  - Parses PDF (PyMuPDF), DOCX (`python-docx`), and TXT into structured pages.
  - Cleans and segments text using semantic sliding-window chunking (default: 512 chars, 64 overlap).
  - Emits structured `DocumentChunk` and `ProcessedDocument` Pydantic models.
- **Embedding Generation (`EmbeddingService`):**
  - CPU-first Sentence Transformers model: `BAAI/bge-small-en-v1.5`.
  - Process-level singleton caching with thread-safe double-checked locking.
  - Pre-flight RAM safety check (500 MB threshold).
  - Produces 384-dimensional unit-normalized `float32` vectors.
  - Outputs `EmbeddedChunk` instances preserving full chunk provenance metadata.
- **Regression Baseline:** 102 passing unit and integration tests (0 regressions).

---

## 3. ChromaDB Compatibility & Version Resolution

### 3.1 Audit of Legacy Candidate (`chromadb==0.5.23`)
A `pip install --dry-run` against `chromadb==0.5.23` in the active virtual environment revealed two critical blockers:
1. **Fatal Tokenizers Conflict:**
   - `chromadb==0.5.23` metadata declares: `tokenizers<=0.20.3,>=0.13.2`.
   - `transformers==4.57.6` (Phase 5 ML stack) requires: `tokenizers<=0.23.0,>=0.22.0`.
   - *Conflict:* Downgrading `tokenizers` to `0.20.3` immediately breaks `transformers`, resulting in `ImportError` or ABI crashes during BGE model loading.
2. **Missing Python 3.13 Windows Wheels for `chroma-hnswlib`:**
   - `chromadb==0.5.23` depends on `chroma-hnswlib==0.7.6`.
   - PyPI provides no pre-built wheels for `chroma-hnswlib` on Python 3.13 Windows x64.
   - Pip attempts to compile from source (`chroma_hnswlib-0.7.6.tar.gz`), requiring Microsoft Visual C++ Build Tools (MSVC 14.x) and failing if compilers or build headers are absent.

### 3.2 Audit of Intermediate Releases (`chromadb==0.6.3`)
- `chromadb==0.6.3` continues to rely on `chroma-hnswlib==0.7.6.tar.gz`, triggering the identical C++ source compilation requirement and potential compiler mismatches on Python 3.13.

### 3.3 Audit of Modern Release (`chromadb==1.5.9`)
A dry-run resolution for `chromadb==1.5.9` completed with status `Success` and verified the following compatibility matrix:

| Component | Project Requirement | ChromaDB 1.5.9 Status | Evaluation |
|---|---|---|---|
| **Python Version** | Python 3.13.3 (Windows x64) | Compatible via `cp39-abi3` wheel | Pre-compiled binary, zero compiler needed |
| **NumPy** | 2.5.4 (installed in `.venv`) | `numpy>=1.22.5` | Fully satisfied; zero downgrade |
| **Pydantic** | 2.12.5 (installed in `.venv`) | `pydantic>=2.0` | Fully satisfied |
| **Tokenizers** | 0.22.2 (installed in `.venv`) | `tokenizers>=0.13.2` | Fully satisfied; no upper-bound conflict |
| **ONNX Runtime** | 1.31.1 (installed in `.venv`) | `onnxruntime>=1.14.1` | Fully satisfied |
| **Uvicorn / FastAPI** | 0.30.0 / 0.111.0 | `uvicorn>=0.18.3`, `fastapi>=0.95.2` | Fully satisfied |
| **Persistence** | Local on-disk persistence | Supported via `PersistentClient` | Built-in SQLite + Rust vector storage |
| **Hardware Mode** | CPU-first (AMD Ryzen 7 7730U) | Supported | Pure CPU SIMD vector distance engine |

### Recommendation
**Adopt `chromadb==1.5.9`** as the pinned vector database dependency for Phase 6B. It delivers a modern, pre-compiled Rust architecture that eliminates C++ build dependencies on Windows and maintains 100% compatibility with Phase 5 ML libraries.

---

## 4. Storage Architecture & Separation of Concerns

```
┌────────────────────────────────────────────────────────────────────────┐
│                          FastAPI Application                           │
└──────────────────┬─────────────────────────────────┬───────────────────┘
                   │                                 │
                   ▼                                 ▼
┌──────────────────────────────────────┐  ┌──────────────────────────────┐
│         SQLite 3 (aiosqlite)         │  │    ChromaDB (Persistent)     │
│       Relational System of Record    │  │   Vector & Semantic Index    │
├──────────────────────────────────────┤  ├──────────────────────────────┤
│ Collections:                         │  │ ChromaDB Collections:        │
│  - id (UUID pk)                      │  │  - name: col_{collection_id} │
│  - name (unique str)                 │  │                              │
│  - description                       │  │ Records (per chunk):         │
│                                      │  │  - id: chunk_id              │
│ Documents:                           │  │  - embedding: List[float]    │
│  - id (UUID pk)                      │  │  - document: chunk_text      │
│  - collection_id (FK)                │  │  - metadata: {               │
│  - filename, stored_filename         │  │      document_id: str,       │
│  - sha256 (deduplication)            │  │      collection_id: str,     │
│  - page_count, chunk_count           │  │      chunk_index: int,       │
│  - status (pending/processing/etc.)  │  │      page_number: int,       │
│  - error (traceback / details)       │  │      file_type: str,         │
│                                      │  │      model_name: str         │
│                                      │  │    }                         │
└──────────────────────────────────────┘  └──────────────────────────────┘
```

### 4.1 Responsibility Matrix

| Entity / Concern | Stored In | Schema Details | Access Patterns |
|---|---|---|---|
| **Collection Entities** | SQLite | `collections` table | CRUD, name uniqueness check, UI display |
| **Document State & Lifecycle** | SQLite | `documents.status`, `error` | Processing state machine, error recording |
| **File Provenance & Hashes** | SQLite | `documents.sha256`, `size_bytes` | Duplicate upload prevention |
| **Page & Chunk Quantities** | SQLite | `page_count`, `chunk_count` | System statistics, fast pagination |
| **Dense Vector Embeddings** | ChromaDB | 384-dimensional float vectors | Cosine / HNSW index search |
| **Chunk Text for Retrieval** | ChromaDB | `document` field | Feeding context chunks to Claude in Phase 9 |
| **Chunk-Level Provenance** | ChromaDB | `metadata` dictionary | Page citations, source file attribution |

### 4.2 Identifier Mapping
- **Collection Mapping:**
  ChromaDB requires collection names matching `^[a-zA-Z0-9][a-zA-Z0-9_-]{1,61}[a-zA-Z0-9]$`.
  SQLite collection IDs are standard 36-character UUID strings (e.g., `123e4567-e89b-12d3-a456-426614174000`).
  *Mapping rule:*
  ```python
  def get_chroma_collection_name(collection_id: str) -> str:
      return f"col_{collection_id.replace('-', '_')}"
  ```
  Result: 40 characters, valid alphanumeric characters, guaranteed collision-free.
- **Chunk Mapping:**
  SQLite chunk references are deterministic deterministic string keys generated by Phase 4: `{document_id}_c{chunk_index:04d}` (e.g., `doc-1234_c0000`).
  ChromaDB item IDs use this exact `chunk_id` string directly.
  *Guarantee:* Complete 1:1 mapping between `DocumentChunk.chunk_id`, `EmbeddedChunk.chunk_id`, and ChromaDB record `id`.

---

## 5. Persistence and Lifecycle Management

### 5.1 Storage Directory Structure
ChromaDB local persistence operates under `data/chromadb/`:
```
data/
├── chromadb/                  <-- chroma_persist_dir (Git ignored)
│   ├── chroma.sqlite3         <-- Internal ChromaDB metadata & index store
│   └── [uuid]/                <-- HNSW vector segment directories
├── knowledge_assistant.db     <-- Application SQLite database
└── uploads/                   <-- Uploaded files directory
```

### 5.2 Environment Configuration
In `backend/config.py`:
```python
chroma_persist_dir: str = "./data/chromadb"
chroma_collection_prefix: str = "col_"
chroma_telemetry: bool = False
```
Configurable via `.env` without machine-specific hardcoded paths:
```env
CHROMA_PERSIST_DIR=./data/chromadb
```

### 5.3 Service Architecture: `VectorStoreService`
An isolated service `VectorStoreService` in `backend/services/vector_store.py` will manage ChromaDB operations:
- **Thread-safe process singleton:** Holds a single `chromadb.PersistentClient` initialized lazily or during FastAPI startup.
- **Methods:**
  - `get_or_create_collection(collection_id: str) -> Collection`
  - `upsert_chunks(collection_id: str, chunks: List[EmbeddedChunk]) -> int`
  - `delete_document_chunks(collection_id: str, document_id: str) -> int`
  - `delete_collection(collection_id: str) -> bool`
  - `query_similar(collection_id: str, query_embedding: List[float], top_k: int) -> List[dict]` (Prepared for Phase 7)
  - `count_chunks(collection_id: str) -> int`

### 5.4 Application Lifecycle Integration (`backend/main.py`)
In `lifespan(app: FastAPI)`:
- **Startup:**
  1. Ensure `Path(settings.chroma_persist_dir)` exists.
  2. Initialize `VectorStoreService` and verify client readiness.
- **Shutdown:**
  1. Flush and release any active vector store client handles.

---

## 6. Memory and Performance Constraints

### 6.1 Hardware Profile
- CPU: AMD Ryzen 7 7730U (8 cores, 16 threads, AVX2 support).
- Usable RAM: ~13.84 GB.
- Embedding Dimension: 384 (`float32` = 1,536 bytes per vector).

### 6.2 Insertion Batch Sizing
- While embedding inference runs at `batch_size=1` to minimize PyTorch activation spikes, ChromaDB vector indexing benefits from batch insertion to reduce SQLite overhead in Chroma's backing store.
- **Recommendation:**
  Insert chunks into ChromaDB in batches of **50 to 100 chunks**.
  *Memory footprint per batch:*
  - 100 vectors $\times 1.536\text{ KB} \approx 153.6\text{ KB}$ vector data.
  - 100 chunk texts $\times \approx 512\text{ bytes} \approx 51.2\text{ KB}$ text data.
  - Total RAM per insertion batch: **< 1 MB**.
  This guarantees negligible memory pressure while ensuring high indexing throughput.

---

## 7. Failure Handling & Consistency Model

Because SQLite and ChromaDB do not share an atomic transaction manager, partial failures must be handled explicitly.

### 7.1 Ingestion Flow & Rollback
1. **Transition to Processing:**
   Document status set to `processing` in SQLite (committed).
2. **Parsing & Chunking:**
   Extracted into `DocumentChunk` instances.
3. **Embedding:**
   `EmbeddingService.embed_chunks()` computes vectors.
4. **Vector Store Insertion:**
   `VectorStoreService.upsert_chunks(collection_id, embedded_chunks)` executes.
5. **On ChromaDB Failure (e.g., Disk Full, Corruption, Timeout):**
   - Catch `VectorStoreError`.
   - Execute best-effort cleanup: `vector_store.delete_document_chunks(collection_id, document_id)`.
   - In SQLite, transition document record to `status = 'failed'` with `error = str(exc)` and commit.
   - Document is **never left in `processing` or marked `indexed`**.
6. **On ChromaDB Success:**
   - In SQLite, update document: `status = 'indexed'`, `chunk_count = len(chunks)`, `page_count = pages`, `error = None`.
   - Commit SQLite transaction.

### 7.2 Deletion Cascades
1. **Document Deletion (`DELETE /api/documents/{id}`):**
   - Retrieve document record from SQLite to obtain `collection_id`.
   - Delete all corresponding vectors from ChromaDB: `vector_store.delete_document_chunks(collection_id, doc_id)`.
   - Delete document record from SQLite (committed).
   - Remove physical upload file from disk.
2. **Collection Deletion (`DELETE /api/collections/{id}`):**
   - Delete the entire ChromaDB collection: `vector_store.delete_collection(collection_id)`.
   - Delete SQLite collection record (cascading to documents).
   - Delete collection storage directory from disk.

---

## 8. Proposed Testing Strategy for Phase 6B

All tests must be fast, offline, and completely isolated from production data:
1. **Isolated Test Fixture (`tests/conftest.py`):**
   - Use a function-scoped `tmp_path / "test_chromadb"` directory for ChromaDB persistence during tests.
   - Alternatively, inject `chromadb.EphemeralClient()` for pure in-memory unit tests.
2. **Test Cases (`tests/test_vector_store.py`):**
   - `test_chromadb_initialization`: verifies persistent directory creation and client initialization.
   - `test_collection_lifecycle`: creates, retrieves, and deletes collections.
   - `test_upsert_embedded_chunks`: verifies embedding vectors (384-dim) and metadata are stored accurately.
   - `test_upsert_idempotency`: re-upserting identical chunk IDs overwrites without duplicate record errors.
   - `test_metadata_filtering`: verifies querying by `document_id` isolates specific subsets.
   - `test_delete_document_chunks`: verifies deleting a document removes all its chunk vectors.
   - `test_delete_collection`: verifies deleting a collection completely purges the ChromaDB index.
   - `test_document_processor_vector_store_integration`: end-to-end test verifying document processing with `embed=True` populates ChromaDB and sets status `indexed`.
   - `test_document_processor_vector_store_failure_rollback`: verifies ChromaDB failure sets document status `failed`.

---

## 9. Implementation Sequence for Phase 6B

When authorized, Phase 6B implementation will proceed in the following controlled steps:

```
Step 1: Dependency Update
  └─ Add chromadb==1.5.9 to requirements.txt
  └─ Install into .venv (verify pip check & regression test)

Step 2: Configuration
  └─ Add ChromaDB settings & validation in backend/config.py

Step 3: Service Layer Implementation
  └─ Create backend/services/vector_store.py (VectorStoreService)
  └─ Export in backend/services/__init__.py

Step 4: Pipeline Integration
  └─ Update DocumentProcessor to inject VectorStoreService
  └─ Add vector store deletion calls to document/collection deletion routers

Step 5: Test Suite Creation
  └─ Add tests/test_vector_store.py
  └─ Verify 100% test pass rate across all suites

Step 6: Completion Report
  └─ Generate docs/phase-6b-vector-store-report.md
```

---

## 10. Audit Conclusion

The compatibility investigation resolves the legacy dependency conflict identified with `0.5.23` and provides an unencumbered path forward using `chromadb==1.5.9`. The proposed architecture preserves SQLite as the single relational source of truth while leveraging ChromaDB for high-performance dense vector persistence and retrieval.

**Recommendation:** Proceed to Phase 6B installation and implementation upon user authorization.
