# Phase 7.1 — Semantic Retrieval Architecture Audit Report

**Project:** Intelligent Knowledge Assistant Using Agentic RAG and Large Language Models  
**Phase:** Phase 7.1 (Semantic Retrieval and Query Service — Architecture Audit)  
**Date:** 2026-11-09  
**Status:** Completed (Read-Only Audit)  
**Author:** Senior Python Backend & Vector Retrieval Systems Architect  

---

## Executive Summary

Phase 6 (comprising VectorStoreService, EmbeddingService, DocumentProcessor integration, crash-safe deletion, and recovery) has achieved full verification and operational acceptance. Phase 7 initiates the **Semantic Retrieval and Query Service**, transitioning the system from data ingestion and persistence to grounded query retrieval.

This audit evaluates the existing codebase, defines strict safety guarantees against stale or ghost vectors, establishes mathematical alignment between ChromaDB distance metrics and cosine similarity, and specifies the complete service and API architecture for Phase 7.

---

## 1. Existing Architecture & Codebase Inspection

The current workspace architecture was inspected across all schema, model, service, router, and configuration files.

| Component / Layer | Source File | Key Responsibilities & Line References |
|---|---|---|
| **Chunk Schemas** | [`backend/schemas/chunk.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/schemas/chunk.py) | Defines `TextPage` (lines 18–40), `ChunkMetadata` (lines 42–69), `DocumentChunk` (lines 71–105), and `ProcessedDocument` (lines 107–129). Provenance fields include `document_id`, `collection_id`, `filename`, `file_type`, `chunk_index`, `page_number`, `page_range`, `start_char`, `end_char`, and `version`. |
| **Embedding Schema** | [`backend/schemas/embedding.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/schemas/embedding.py) | Defines `EmbeddedChunk` (lines 15–81) with dense vector validation (`validate_embedding` at line 38 rejects empty, non-numeric, NaN, and Inf values) and dimension validation (`validate_dimension` at line 50). `from_document_chunk` factory at line 61. |
| **Document Schemas** | [`backend/schemas/document.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/schemas/document.py) | Defines `DocumentStatus` enum (lines 15–21: `PENDING`, `PROCESSING`, `INDEXED`, `FAILED`, `DELETION_PENDING`), `DocumentResponse` (lines 23–39), and `DocumentListResponse` (lines 41–43). |
| **Collection Schemas** | [`backend/schemas/collection.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/schemas/collection.py) | Defines `CollectionBase`, `CollectionCreate`, `CollectionUpdate`, `CollectionResponse` (with active `document_count`), and `CollectionListResponse` (lines 14–76). |
| **Embedding Service** | [`backend/services/embedding_service.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/services/embedding_service.py) | Manages `SentenceTransformer` lifecycle (lines 57–380). Uses thread-safe double-checked singleton model caching (`_cached_model` at lines 62, 136–141), pre-flight RAM safety check (`_check_ram_safety` at line 107), unit L2 normalization (`normalize=True` at line 94), `embed_text` (lines 195–216), `embed_batch` (lines 217–274), `embed_chunks` (lines 276–323), and `compute_similarity` (lines 326–380). |
| **Vector Store Service** | [`backend/services/vector_store.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/services/vector_store.py) | Embedded ChromaDB `PersistentClient` wrapper (lines 68–560). Handles collection name hashing/sanitization (lines 111–142), cosine distance initialization (`metadata={"hnsw:space": "cosine"}` at line 161), scalar metadata extraction (`_sanitize_metadata` at lines 258–292), batch upsert (`store_chunks` at lines 340–404), document vector deletion (lines 406–484), chunk ID deletion (lines 486–519), and chunk ID lookup (`get_chunks_by_ids` at lines 521–560). |
| **Document Processor** | [`backend/services/document_processor.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/services/document_processor.py) | Coordinates text extraction, chunking, embedding, staged versioning, and ChromaDB persistence (lines 52–347). Enforces target versioning (lines 141–157) and storage verification before SQLite status promotion to `indexed` (lines 223–242). |
| **Document Recovery** | [`backend/services/document_recovery.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/services/document_recovery.py) | Discovers `DELETION_PENDING` documents and executes isolated cleanup across ChromaDB, disk files, and SQLite with `StaleDataError` handling (lines 48–178). |
| **Database Models** | [`backend/models/document.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/models/document.py)<br>[`backend/models/collection.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/models/collection.py) | SQLAlchemy 2.0 ORM entities. `Document` includes `id`, `collection_id`, `filename`, `file_type`, `sha256`, `status`, `version`, `page_count`, and `chunk_count` with composite index `ix_doc_collection_sha256` (lines 21–105). `Collection` defines name uniqueness and cascade deletion (lines 21–60). |
| **System Configuration** | [`backend/config.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/config.py) | Configures `top_k_results: int = 5` (line 84), `similarity_threshold: float = 0.3` (line 85), `embedding_model: str = "BAAI/bge-small-en-v1.5"` (line 67), `embedding_device: str = "cpu"` (line 68), `embedding_dimension: int = 384` (line 69), `embedding_batch_size: int = 1` (line 70), and `embedding_normalize: bool = True` (line 71). |
| **FastAPI Application** | [`backend/main.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/main.py) | Single-process runtime (`workers=1`), lifespan management with bounded startup recovery, CORS, and centralized exception handling (lines 58–180). |
| **Routers** | [`backend/routers/collections.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/routers/collections.py)<br>[`backend/routers/documents.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/routers/documents.py) | REST API endpoints for collection management, active document counts, document upload, staged replacement (`replace=true`), and crash-safe two-phase deletion. |

---

## 2. Current Implementation Capabilities (10 Verification Points)

### 1. How Vectors Are Generated
Vectors are generated using the `EmbeddingService` class ([`backend/services/embedding_service.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/services/embedding_service.py#L217-L274)). Texts are encoded via `SentenceTransformer.encode()` with `normalize_embeddings=True` and `convert_to_numpy=True` (line 258). The output is a list of Python native `float` values of length 384. Every vector is strictly unit-normalized ($\|\mathbf{v}\|_2 = 1.0$), ensuring that cosine similarity is numerically identical to the dot product.

### 2. How Query Embeddings Can Reuse the Existing Embedding Service
The existing `EmbeddingService.embed_text(text: str) -> List[float]` ([`backend/services/embedding_service.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/services/embedding_service.py#L195-L216)) directly serves query embedding needs:
- It calls `embed_batch([text], batch_size=1)`.
- It accesses `EmbeddingService._cached_model` ([`backend/services/embedding_service.py:136`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/services/embedding_service.py#L136)), which holds the shared process-level `SentenceTransformer` singleton.
- **Zero additional memory** is consumed because no duplicate model is loaded.
- Input validation rejects empty strings, whitespace-only queries, and non-string inputs with `InvalidInputError` (lines 209–212).

### 3. How Vectors Are Stored in ChromaDB
Vectors are persisted in embedded ChromaDB ([`backend/services/vector_store.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/services/vector_store.py#L340-L404)):
- Chunks are pre-validated for dimension (384), non-finiteness (NaN/Inf), and empty text (lines 306–339).
- Chunks are upserted into ChromaDB in batches of 50 via `col.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)` (lines 383–396).
- Storage is disk-backed under `./data/chromadb/` using `chromadb.PersistentClient` (lines 93, 105).

### 4. How Collection Names Are Generated
Collection names are computed by [`VectorStoreService.get_collection_name(collection_id)`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/services/vector_store.py#L111-L142):
- **Rule:** `"col_" + collection_id.strip().replace("-", "_")`.
- Example: SQLite UUID `123e4567-e89b-12d3-a456-426614174000` maps to `col_123e4567_e89b_12d3_a456_426614174000`.
- Validates length between 3 and 63 characters and conforms to ChromaDB naming constraints.

### 5. How Chunk IDs Are Generated
Chunk IDs are constructed deterministically in [`backend/services/text_chunker.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/services/text_chunker.py#L165-L185):
- **Rule:** `{document_id}_v{version}_c{chunk_index:04d}`.
- Example: `d7e9b012-3456-7890-abcd-ef0123456789_v1_c0000`.
- When version is omitted, it defaults to `{document_id}_c{chunk_index:04d}`.
- This deterministic ID guarantees that upserting chunks is idempotent and version-isolated.

### 6. Which Metadata Fields Are Stored
Metadata is sanitized into scalar values by [`VectorStoreService._sanitize_metadata`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/services/vector_store.py#L258-L292):
- Mandatory fields: `document_id` (str), `collection_id` (str), `chunk_index` (int), `char_count` (int), `start_char` (int), `end_char` (int), `filename` (str), `file_type` (str), `model_name` (str).
- Optional fields (included only when non-None): `page_number` (int), `page_range` (str), `version` (int).
- ChromaDB strictly rejects `None`, lists, and nested dictionaries. The sanitizer guarantees only `str`, `int`, `float`, and `bool` reach ChromaDB.

### 7. How Document Versions Are Represented
- **In SQLite:** The `documents` table stores `version: Mapped[int]` ([`backend/models/document.py:76`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/models/document.py#L76)), initialized to `1` and incremented on replacement.
- **In Chunk ID:** Encoded directly into the string: `..._v{version}_c...`.
- **In ChromaDB Metadata:** Stored as integer metadata `version: int` ([`backend/services/vector_store.py:290`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/services/vector_store.py#L290)).

### 8. How Document Deletion Affects Stored Vectors
When a document is deleted ([`backend/routers/documents.py:385`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/routers/documents.py#L385) or [`backend/services/document_recovery.py:134`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/services/document_recovery.py#L134)):
- `VectorStoreService.delete_document_vectors(collection_id, document_id)` queries ChromaDB for all chunks where `document_id == document_id` across all versions and calls `col.delete(ids=target_ids)`.
- If the collection or chunks are missing, it returns `0` gracefully without raising an exception ([`backend/services/vector_store.py:435, 446`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/services/vector_store.py#L435)).

### 9. How Collection Deletion Affects Stored Vectors
When a collection is deleted ([`backend/routers/collections.py:229`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/routers/collections.py#L229)):
- `VectorStoreService.delete_collection(collection_id)` invokes `self.client.delete_collection(name=col_name)`.
- The entire ChromaDB collection namespace and its underlying HNSW index are purged from disk.
- If already deleted, it catches `not found` and returns `False` safely without error ([`backend/services/vector_store.py:215`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/services/vector_store.py#L215)).

### 10. Whether Existing Methods Already Support Similarity Search
**NO.** `VectorStoreService` currently possesses **zero similarity search methods**.
- Existing methods are strictly administrative: `store_chunks`, `delete_document_vectors`, `delete_chunks_by_ids`, `get_chunks_by_ids`, `get_collection`, `get_or_create_collection`, `delete_collection`, `list_collections`, and `count`.
- ChromaDB's underlying `Collection` instance supports `.query(query_embeddings=..., n_results=..., where=..., include=...)`, but `VectorStoreService` lacks any wrapper or query interface.

---

## 3. Missing Functionality for Phase 7

To deliver semantic retrieval, the following components must be built in Phase 7:

1. **`VectorStoreService.query_vectors(...)`**:
   A method on `VectorStoreService` to execute vector similarity queries against ChromaDB with collection isolation, metadata filtering (`where`), and structured exception handling.
2. **Distance-to-Similarity Conversion**:
   ChromaDB configured with `{"hnsw:space": "cosine"}` returns cosine distance $d \in [0.0, 2.0]$. The system must convert this to cosine similarity $s = 1.0 - d \in [-1.0, 1.0]$.
3. **Candidate Oversampling and Filtering**:
   Because ChromaDB may contain stale vectors (from documents pending deletion, unpromoted re-indexing runs, or outdated versions), retrieval must oversample candidates ($N_{\text{candidates}} = \min(\text{top\_k} \times 3, 50)$) before SQLite filtering.
4. **SQLite Document and Version Ground Truth Gate**:
   An authoritative validation step that queries SQLite to reject candidates whose parent document is missing, not in `status == 'indexed'`, or has a version mismatch.
5. **Retrieval Service Orchestrator (`backend/services/retrieval_service.py`)**:
   Coordinates pre-flight validation, query embedding, vector search, candidate validation, threshold filtering, and provenance packaging.
6. **Retrieval Schemas (`backend/schemas/retrieval.py`)**:
   Pydantic models for `RetrievalQuery`, `RetrievedChunk`, and `RetrievalResult`.
7. **FastAPI Query Endpoint (`POST /api/collections/{collection_id}/query`)**:
   REST API route exposing semantic retrieval to the frontend and downstream agent tools.

---

## 4. Retrieval Safety Requirements

The retrieval system must operate under the **Zero-Trust Vector Store Principle**:
> *ChromaDB is treated as an unconstrained candidate index, never as an authoritative source of truth. Every candidate chunk must be validated against SQLite before being admitted as evidence.*

### Mandatory Safety Gates:

```
ChromaDB Candidate Result
          │
          ▼
┌──────────────────────────────────────────────┐
│ Gate 1: Collection Namespace Isolation      │  Chunk must belong to target collection
└─────────────────────┬────────────────────────┘
                      ▼
┌──────────────────────────────────────────────┐
│ Gate 2: SQLite Document Existence Check      │  Document must exist in SQLite database
└─────────────────────┬────────────────────────┘
                      ▼
┌──────────────────────────────────────────────┐
│ Gate 3: Document Status Gate                 │  Status must strictly equal 'indexed'
│                                              │  (Rejects 'deletion_pending', 'failed',
│                                              │   'processing', 'pending')
└─────────────────────┬────────────────────────┘
                      ▼
┌──────────────────────────────────────────────┐
│ Gate 4: Document Version Matching            │  chunk.metadata['version'] == doc.version
│                                              │  (Rejects outdated historical vectors)
└─────────────────────┬────────────────────────┘
                      ▼
┌──────────────────────────────────────────────┐
│ Gate 5: Similarity Threshold Check           │  similarity_score >= threshold
└─────────────────────┬────────────────────────┘
                      ▼
       Admitted Grounded Chunk
```

### Safety Rules:
1. **Target Collection Confinement:** Queries are routed strictly to the collection specified by `collection_id`. Cross-collection retrieval is physically impossible due to collection namespace partitioning.
2. **Status Discrimination:**
   - `indexed` $\rightarrow$ **ACCEPT**
   - `deletion_pending` $\rightarrow$ **REJECT** (in-flight deletion)
   - `processing` $\rightarrow$ **REJECT** (unpromoted ingestion)
   - `failed` $\rightarrow$ **REJECT** (ingestion error)
   - `pending` $\rightarrow$ **REJECT** (not yet processed)
   - Missing from SQLite $\rightarrow$ **REJECT** (orphaned/ghost vector)
3. **Version Discrimination:** If a document is updated to Version 2, any lingering Version 1 vectors in ChromaDB must be discarded if returned by similarity search.
4. **Provenance Preservation:** Every returned chunk must retain:
   - `chunk_id`
   - `document_id`
   - `collection_id`
   - `filename`
   - `file_type`
   - `page_number` (int or null)
   - `page_range` (str or null)
   - `start_char` and `end_char`
   - `version`
   - `similarity_score`

---

## 5. End-to-End Retrieval Design Proposal

```
User Query (string)
         │
         ▼
[Step 1: Input Validation]
  - Query length (1 .. 2048 chars)
  - Non-empty, non-whitespace
  - top_k bounds (1 .. 50)
  - threshold bounds (-1.0 .. 1.0)
         │
         ▼
[Step 2: Collection Pre-flight in SQLite]
  - Verify collection_id exists (HTTP 404 if not found)
  - If optional document_id filter provided:
      Verify document exists in collection and is 'indexed'
         │
         ▼
[Step 3: Query Embedding Generation]
  - Call EmbeddingService.embed_text(query_text)
  - Uses cached SentenceTransformer ("BAAI/bge-small-en-v1.5")
  - Returns 384-dimensional unit-normalized vector
         │
         ▼
[Step 4: ChromaDB Similarity Search]
  - Calculate candidate oversampling:
      n_candidates = min(max(top_k * 3, 15), 50)
  - Construct metadata filter:
      where = {"document_id": document_id} if filtered else None
  - Call VectorStoreService.query_vectors(
        collection_id=col_id,
        query_vector=query_vec,
        n_results=n_candidates,
        where=where,
    )
         │
         ▼
[Step 5: Metric Conversion & Pre-filtering]
  - Convert ChromaDB cosine distance d to similarity:
      similarity = 1.0 - d
  - Discard candidates where similarity < similarity_threshold
         │
         ▼
[Step 6: Batch SQLite Document & Version Validation]
  - Extract unique document_ids from surviving candidates
  - Single batch query:
      SELECT id, status, version, filename FROM documents
      WHERE id IN (:doc_ids) AND collection_id = :col_id
  - Reject candidate if:
      - Document not found in SQLite
      - doc.status != 'indexed'
      - candidate.metadata.get('version') != doc.version
         │
         ▼
[Step 7: Ranking, Truncation, & Provenance Packaging]
  - Sort validated candidates by similarity descending
  - Truncate to requested top_k
  - Package into RetrievedChunk schemas with full metadata
         │
         ▼
RetrievalResponse (HTTP 200)
```

### Proposed Configuration Defaults

| Parameter | Configuration Key | Proposed Default | Valid Range | Rationale |
|---|---|---|---|---|
| `top_k` | `top_k_results` | **5** | 1 to 50 | Balances context window efficiency with recall. Existing config default at [`backend/config.py:84`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/config.py#L84). |
| `similarity_metric` | `hnsw:space` | **cosine** | cosine | Standard for dense text retrieval with normalized embeddings. Initialized in [`backend/services/vector_store.py:161`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/services/vector_store.py#L161). |
| `similarity_threshold` | `similarity_threshold` | **0.30** | -1.0 to 1.0 | Filters out noise and tangential matches. Existing config default at [`backend/config.py:85`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/config.py#L85). *(Measured validation threshold in Phase 20)*. |
| `candidate_oversampling_multiplier` | N/A (Internal) | **3x** | 2x to 5x | Guarantees that if stale/historical chunks are discarded, sufficient valid candidates remain to satisfy `top_k`. |
| `max_candidate_limit` | N/A (Internal) | **50** | 10 to 100 | Prevents excessive memory allocation and slow SQLite IN-clauses during retrieval. |
| `max_query_length` | N/A (Schema) | **2048 chars** | 1 to 4096 | Protects against memory denial-of-service and aligns with sentence-transformer input token bounds. |

---

## 6. Proposed Service and Schema Design

### A. Schemas: `backend/schemas/retrieval.py`

```python
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class RetrievalQuery(BaseModel):
    """Input query request for semantic retrieval."""
    query_text: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="Search query string.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of relevant chunks to return.",
    )
    similarity_threshold: float = Field(
        default=0.3,
        ge=-1.0,
        le=1.0,
        description="Minimum cosine similarity score required for candidate chunks.",
    )
    document_id: Optional[str] = Field(
        default=None,
        description="Optional document UUID filter to restrict search to a single document.",
    )

    @field_validator("query_text")
    @classmethod
    def validate_query_text(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Query text cannot be empty or whitespace only.")
        return stripped


class RetrievedChunk(BaseModel):
    """A verified, grounded chunk returned by the semantic retrieval engine."""
    chunk_id: str = Field(..., description="Unique chunk identifier.")
    document_id: str = Field(..., description="Source document UUID.")
    collection_id: str = Field(..., description="Parent collection UUID.")
    filename: str = Field(..., description="Original filename of the source document.")
    file_type: str = Field(..., description="File format extension (pdf, docx, txt).")
    chunk_index: int = Field(..., ge=0, description="0-based sequence index.")
    chunk_text: str = Field(..., description="Extracted text content.")
    page_number: Optional[int] = Field(default=None, description="1-based PDF page number.")
    page_range: Optional[str] = Field(default=None, description="PDF page span.")
    start_char: int = Field(default=0, description="Character start offset.")
    end_char: int = Field(default=0, description="Character end offset.")
    version: int = Field(default=1, description="Document version number.")
    similarity_score: float = Field(..., description="Cosine similarity score in range [-1.0, 1.0].")
    distance: float = Field(..., description="Raw ChromaDB HNSW cosine distance.")

    model_config = ConfigDict(from_attributes=True)


class RetrievalResponse(BaseModel):
    """Structured response containing ranked, verified retrieval results."""
    collection_id: str = Field(..., description="Collection searched.")
    query: str = Field(..., description="Original query text.")
    total_candidates_found: int = Field(..., description="Raw candidates returned by ChromaDB.")
    verified_chunks_count: int = Field(..., description="Count of chunks surviving SQLite validation and thresholding.")
    chunks: List[RetrievedChunk] = Field(default_factory=list, description="Ranked list of verified chunks.")
```

### B. Vector Store Extension: `VectorStoreService.query_vectors`

To be added to [`backend/services/vector_store.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/services/vector_store.py):

```python
def query_vectors(
    self,
    collection_id: str,
    query_vector: Sequence[float],
    n_results: int = 10,
    where: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Perform nearest-neighbor similarity search against a ChromaDB collection.

    Args:
        collection_id: SQLite collection UUID.
        query_vector: Unit-normalized dense embedding vector (dimension 384).
        n_results: Maximum candidate count to retrieve.
        where: Optional ChromaDB metadata filter dictionary.

    Returns:
        Dict containing 'ids', 'documents', 'metadatas', and 'distances'.

    Raises:
        CollectionNotFoundError: If the collection does not exist.
        InvalidVectorError: If query_vector is invalid, empty, or has wrong dimension.
        VectorStorageError: If ChromaDB query fails.
    """
```

### C. Retrieval Service: `backend/services/retrieval_service.py`

Coordinates the end-to-end flow:
- Injected with `EmbeddingService`, `VectorStoreService`, and `Settings`.
- Method: `async def retrieve(self, collection_id: str, query: RetrievalQuery, db: AsyncSession) -> RetrievalResponse`.
- Executes the SQLite batch validation query in $O(1)$ database trips.

---

## 7. API Design Recommendation

### REST Endpoint Specification

```http
POST /api/collections/{collection_id}/query
Content-Type: application/json
```

#### Request Payload:
```json
{
  "query_text": "What are the core principles of Agentic RAG?",
  "top_k": 5,
  "similarity_threshold": 0.3,
  "document_id": null
}
```

#### Successful Response (HTTP 200 OK):
```json
{
  "collection_id": "c7a8b901-2345-6789-abcd-ef0123456789",
  "query": "What are the core principles of Agentic RAG?",
  "total_candidates_found": 8,
  "verified_chunks_count": 3,
  "chunks": [
    {
      "chunk_id": "doc_001_v1_c0002",
      "document_id": "doc_001",
      "collection_id": "c7a8b901-2345-6789-abcd-ef0123456789",
      "filename": "agent_architecture.pdf",
      "file_type": "pdf",
      "chunk_index": 2,
      "chunk_text": "The agent treats retrieval as an iterative loop, not a one-shot operation...",
      "page_number": 3,
      "page_range": "3",
      "start_char": 1024,
      "end_char": 1536,
      "version": 1,
      "similarity_score": 0.8421,
      "distance": 0.1579
    }
  ]
}
```

#### Error Status Codes:
- `400 Bad Request`: Empty query, NaN/Inf vector values, or invalid JSON parameters.
- `404 Not Found`: Collection ID does not exist in SQLite, or specified `document_id` filter is not found in the collection.
- `422 Unprocessable Entity`: Pydantic validation failure (e.g., `top_k > 50` or `top_k < 1`).
- `500 Internal Server Error`: Sentence-transformers encoding failure or unrecoverable ChromaDB error.

---

## 8. Hardware and Performance Considerations

### System Specifications
- **CPU:** AMD Ryzen 7 7730U (8 cores, 16 threads, Zen 3 architecture, 2.0 GHz base, up to 4.5 GHz boost).
- **RAM:** 16 GB DDR4 (shared with AMD Radeon integrated GPU).
- **PyTorch:** CPU-only build (`torch==2.15.0+cpu`).
- **Embedding Model:** `BAAI/bge-small-en-v1.5` (384 dimensions, ~133 MB on-disk weights, ~250–350 MB RAM in memory).
- **Concurrency Mode:** Single Uvicorn process (`workers=1`).

### Performance Profile & Guardrails:
1. **Model Reuse (Zero Duplicate Allocation):**
   - The `EmbeddingService` process singleton pattern ([`backend/services/embedding_service.py:136`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/services/embedding_service.py#L136)) ensures that query embeddings use the identical model weights already loaded for document ingestion.
   - No second instance of `SentenceTransformer` is ever loaded.
2. **CPU-First Latency Budget:**
   - **Query Embedding:** Encoding a single query string of 50–200 characters on an AMD Ryzen 7 7730U takes approximately **10 to 25 milliseconds**.
   - **ChromaDB Vector Search:** Querying an HNSW index of 5,000 vectors for top-15 candidates takes **3 to 8 milliseconds**.
   - **SQLite Batch Validation:** A single `SELECT ... WHERE id IN (:doc_ids)` query over SQLite WAL takes **1 to 3 milliseconds**.
   - **Total Retrieval Latency:** Expected end-to-end response time is **under 40 milliseconds**, well within interactive SLA limits (< 200 ms).
3. **RAM Protection:**
   - Query candidate limits are bounded at `n_candidates <= 50`, preventing excessive memory allocation.
   - Batch database lookups prevent high-cardinality ORM memory usage.

---

## 9. SQLite Validation Strategy (Anti-Stale & Ghost Vector Prevention)

### Why SQLite Validation is Mandatory
ChromaDB is a pure approximate nearest-neighbor vector index. It does not enforce relational integrity, foreign key constraints, or distributed transactions with SQLite:
- A document deletion could fail midway, leaving ChromaDB vectors staged or SQLite records in `DELETION_PENDING`.
- A re-indexing run could fail before version promotion, leaving unpromoted candidate vectors in ChromaDB.
- An operator might delete an item directly in the database during maintenance.

### The Validation Algorithm ($O(1)$ Database Overhead):
1. **Candidate Ingestion:** ChromaDB returns $M$ raw candidates ($M \le 50$).
2. **Key Extraction:** Collect all unique `document_id` strings: `doc_ids = list({chunk.document_id for chunk in candidates})`.
3. **Single SQL Query:**
   ```python
   stmt = select(
       Document.id,
       Document.status,
       Document.version,
       Document.filename,
   ).where(
       Document.id.in_(doc_ids),
       Document.collection_id == collection_id,
   )
   result = await db.execute(stmt)
   valid_docs = {row.id: row for row in result.all()}
   ```
4. **Validation Filter Loop:**
   For each candidate chunk:
   ```python
   doc = valid_docs.get(candidate.document_id)
   if doc is None:
       continue  # Ghost vector: document deleted from SQLite
   if doc.status != DocumentStatus.INDEXED.value:
       continue  # In-flight deletion, failed, or processing
   if candidate.version != doc.version:
       continue  # Outdated version vector
   ```
5. **Outcome:** Stale, orphaned, processing, failed, and historical vectors are eliminated with zero data leakage.

---

## 10. Comprehensive Testing Strategy

A dedicated test suite `tests/test_retrieval.py` must be implemented in Phase 7 to verify the following 17 test scenarios:

| # | Test Scenario | Description & Assertions |
|---|---|---|
| **1** | Relevant Semantic Query | High semantic match returns candidate with `similarity_score > 0.6` and correct content. |
| **2** | Irrelevant Query | Unrelated query returns no results (if thresholded) or chunks with low similarity (`< 0.3`). |
| **3** | Empty Query String | `query_text=""` or whitespace rejected with HTTP 422 / 400 validation error. |
| **4** | Invalid Query Type / Bounds | Overly long query (> 2048 chars) or invalid JSON types rejected with 422. |
| **5** | Collection Isolation | Chunks from Collection B are never returned when querying Collection A. |
| **6** | Document Filtering | When `document_id="doc_A"` is passed, only chunks from `doc_A` are returned. |
| **7** | Indexed Document Success | Fully indexed document chunks pass validation and return full provenance metadata. |
| **8** | Deletion-Pending Rejection | Document in `status='deletion_pending'` in SQLite has its ChromaDB vectors rejected. |
| **9** | Failed Document Rejection | Document in `status='failed'` in SQLite has its ChromaDB vectors rejected. |
| **10** | Processing Document Rejection | Document in `status='processing'` in SQLite has its ChromaDB vectors rejected. |
| **11** | Outdated Version Rejection | Document is at version 2; candidate chunks from version 1 in ChromaDB are rejected. |
| **12** | Missing SQLite Document | ChromaDB contains vector whose document was deleted from SQLite; chunk is rejected. |
| **13** | Missing ChromaDB Collection | Querying non-existent collection returns empty results or 404 cleanly without 500 error. |
| **14** | Empty Collection | Collection with 0 indexed documents returns `{"chunks": [], "verified_chunks_count": 0}` with HTTP 200. |
| **15** | Non-Finite Vector Rejection | Injected NaN or Inf query embeddings raise `InvalidVectorError` and return HTTP 400. |
| **16** | No Matching Results | Valid query where no chunks exceed `similarity_threshold` returns empty chunk list with HTTP 200. |
| **17** | Multiple Chunks from Same Document | Verifies multiple relevant chunks from one document preserve distinct `chunk_index`, `page_number`, and text. |

*Safety Rule:* All unit and integration tests must run against temporary SQLite databases (`:memory:` or isolated temp paths) and temporary ChromaDB test directories. Production data must never be touched.

---

## 11. Risks and Unresolved Architectural Questions

1. **BGE Model Query Instruction Prefix:**
   - The `BAAI/bge-small-en-v1.5` model documentation suggests prepending an instruction prefix to queries for asymmetric retrieval tasks:
     `"Represent this sentence for searching relevant passages: "`
   - *Analysis:* Testing should evaluate whether this instruction improves retrieval precision on our specific corpus or if raw query strings provide sufficient accuracy.
   - *Recommendation:* Support an optional `use_query_instruction: bool = False` configuration flag.
2. **ChromaDB Cosine Space Inversion:**
   - ChromaDB calculates cosine distance as $d = 1.0 - \text{cosine\_similarity}$.
   - Ensure tests explicitly assert that higher similarity scores correspond to closer semantic matches and that scores lie in $[-1.0, 1.0]$.
3. **Chunk Deduplication vs. Redundancy:**
   - If sliding window overlap (64 chars) causes two adjacent chunks from the same page to both match, should they both be returned or deduplicated?
   - *Recommendation for Phase 7:* Return top-K ranked chunks without aggressive deduplication; downstream Agent Synthesizer (Phase 15) handles contextual merging.

---

## 12. Recommended Implementation Sequence for Phase 7

```
Phase 7.1: Architecture Audit (THIS DELIVERABLE)
     │
     ▼
Phase 7.2: Vector Store Query Capabilities & Schemas
  - Implement VectorStoreService.query_vectors()
  - Define backend/schemas/retrieval.py (RetrievalQuery, RetrievedChunk, RetrievalResponse)
  - Unit tests for vector_store query method
     │
     ▼
Phase 7.3: Retrieval Service & SQLite Anti-Stale Validation
  - Implement backend/services/retrieval_service.py
  - Wire EmbeddingService singleton with pre-flight checks
  - Implement batch SQLite document and version ground-truth gate
  - Unit tests for retrieval service with all 17 test scenarios
     │
     ▼
Phase 7.4: FastAPI REST API Integration
  - Create endpoint POST /api/collections/{collection_id}/query
  - Wire dependency injection (get_retrieval_service, get_db)
  - Integration tests for HTTP status codes, error handling, and CORS
     │
     ▼
Phase 7.5: Read-Only Audit & Verification
  - Independent verification of retrieval performance, safety gates, and test coverage
```

---

## 13. Acceptance Checklist

- [x] Inspected all 14 required source files and models with exact line citations.
- [x] Analyzed and documented the 10 current vector architecture points.
- [x] Formulated strict Zero-Trust retrieval safety requirements and anti-stale gates.
- [x] Designed end-to-end retrieval flow from user input to verified response.
- [x] Specified Pydantic schemas for `RetrievalQuery`, `RetrievedChunk`, and `RetrievalResponse`.
- [x] Defined `VectorStoreService.query_vectors` method signature and contract.
- [x] Detailed the batch SQLite validation query pattern.
- [x] Evaluated AMD Ryzen 7 7730U CPU-first hardware and memory constraints.
- [x] Formulated 17-scenario comprehensive testing strategy.
- [x] Identified open technical questions (BGE prefix, cosine distance formula).
- [x] Proposed phased implementation roadmap (7.1 to 7.5).
- [x] Zero application source files, databases, or dependencies modified.
