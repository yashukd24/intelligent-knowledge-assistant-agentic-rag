"""
backend/services/vector_store.py
─────────────────────────────────────────────────────────────
Vector Store Service using embedded ChromaDB PersistentClient.

Key Features:
  - ChromaDB version: 1.5.9
  - Storage: In-process PersistentClient at data/chromadb/
  - Metric: Cosine distance ({"hnsw:space": "cosine"})
  - Dimension: 384 dimensions (BAAI/bge-small-en-v1.5)
  - External embeddings only (raw vectors passed)
  - Idempotent upserts via deterministic chunk_ids
  - Document-level vector deletion
  - Pure CPU execution (no CUDA, no Chroma HTTP server)
─────────────────────────────────────────────────────────────
"""

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import chromadb
from chromadb.config import Settings as ChromaSettings

from backend.config import Settings, get_settings
from backend.schemas.embedding import EmbeddedChunk
from backend.schemas.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


# ── Exception Hierarchy ──────────────────────────────────────────────────────


class VectorStoreError(Exception):
    """Base exception for all vector store errors."""
    pass


class CollectionNotFoundError(VectorStoreError):
    """Raised when a specified collection does not exist in ChromaDB."""
    pass


class VectorDimensionMismatchError(VectorStoreError):
    """Raised when vector dimension does not match the expected dimension (384)."""
    pass


class InvalidVectorError(VectorStoreError):
    """Raised when vector contains non-numeric, NaN, or infinite values."""
    pass


class InvalidChunkDataError(VectorStoreError):
    """Raised when chunk text, ID, or metadata fails validation."""
    pass


class VectorStorageError(VectorStoreError):
    """Raised when a ChromaDB storage or persistence operation fails."""
    pass


class VectorSearchResults(list):
    """
    Structured list of RetrievedChunk instances returned by query_vectors,
    with attached metadata distinguishing raw ChromaDB candidate count
    from results remaining after threshold filtering.
    """

    def __init__(
        self,
        chunks: Sequence[RetrievedChunk],
        total_candidates: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(chunks)
        self.total_candidates = total_candidates
        self.returned_results = len(chunks)
        self.metadata = metadata or {}

    @property
    def results(self) -> List[RetrievedChunk]:
        """Alias returning candidate chunks as a standard Python list."""
        return list(self)


# ── Vector Store Service ─────────────────────────────────────────────────────


class VectorStoreService:
    """
    Service wrapper around embedded ChromaDB PersistentClient.
    Provides idempotent chunk storage, document deletion, and vector retrieval.
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        client: Optional[Any] = None,
        settings: Optional[Settings] = None,
        batch_size: int = 50,
        expected_dimension: int = 384,
    ):
        """
        Initialize the VectorStoreService.

        Args:
            persist_dir: Directory path for ChromaDB on-disk persistence.
            client: Optional injected ChromaDB client (e.g. for testing).
            settings: Optional application settings. Falls back to get_settings().
            batch_size: Batch size for chunk upserts (default 50).
            expected_dimension: Required vector dimensionality (default 384).
        """
        self.settings = settings or get_settings()
        self.persist_dir = Path(persist_dir or self.settings.chroma_persist_dir)
        self.batch_size = max(1, batch_size)
        self.expected_dimension = expected_dimension

        if client is not None:
            self.client = client
        else:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            chroma_settings = ChromaSettings(
                anonymized_telemetry=False,
                is_persistent=True,
            )
            self.client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=chroma_settings,
            )

    @staticmethod
    def get_collection_name(collection_id: str) -> str:
        """
        Map a SQLite collection UUID to a valid ChromaDB collection name.

        Approved mapping rule:
          'col_' + collection_id with hyphens replaced by underscores.
          Example: '123e4567-e89b-12d3-a456-426614174000'
                -> 'col_123e4567_e89b_12d3_a456_426614174000'

        Args:
            collection_id: Non-empty collection identifier string.

        Returns:
            Sanitized ChromaDB collection name.

        Raises:
            InvalidChunkDataError: If collection_id is empty or invalid.
        """
        if not collection_id or not isinstance(collection_id, str) or not collection_id.strip():
            raise InvalidChunkDataError("collection_id must be a non-empty string.")

        cleaned = collection_id.strip()
        sanitized = cleaned.replace("-", "_")
        name = f"col_{sanitized}"

        if len(name) < 3 or len(name) > 63:
            raise InvalidChunkDataError(
                f"Resulting collection name '{name}' has invalid length ({len(name)}). "
                "Must be between 3 and 63 characters."
            )

        return name

    def get_or_create_collection(self, collection_id: str) -> Any:
        """
        Retrieve an existing collection or create a new one configured with cosine distance.

        Args:
            collection_id: SQLite collection UUID.

        Returns:
            ChromaDB Collection object.

        Raises:
            VectorStorageError: If ChromaDB fails to create or retrieve collection.
        """
        col_name = self.get_collection_name(collection_id)
        try:
            return self.client.get_or_create_collection(
                name=col_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise VectorStorageError(
                f"Failed to get or create collection '{col_name}': {exc}"
            ) from exc

    def get_collection(self, collection_id: str) -> Any:
        """
        Retrieve an existing ChromaDB collection.

        Args:
            collection_id: SQLite collection UUID.

        Returns:
            ChromaDB Collection object.

        Raises:
            CollectionNotFoundError: If collection does not exist in ChromaDB.
            VectorStorageError: If ChromaDB query fails.
        """
        col_name = self.get_collection_name(collection_id)
        try:
            return self.client.get_collection(name=col_name)
        except Exception as exc:
            # Check if this is a "does not exist" error from Chroma
            exc_str = str(exc).lower()
            if "does not exist" in exc_str or "not found" in exc_str:
                raise CollectionNotFoundError(
                    f"Collection '{col_name}' (collection_id: '{collection_id}') does not exist."
                ) from exc
            raise VectorStorageError(
                f"Error retrieving collection '{col_name}': {exc}"
            ) from exc

    def delete_collection(self, collection_id: str) -> bool:
        """
        Delete a collection from ChromaDB.

        Args:
            collection_id: SQLite collection UUID.

        Returns:
            True if collection was deleted, False if it did not exist.

        Raises:
            VectorStorageError: If deletion fails unexpectedly.
        """
        col_name = self.get_collection_name(collection_id)
        try:
            self.client.delete_collection(name=col_name)
            return True
        except Exception as exc:
            exc_str = str(exc).lower()
            if "does not exist" in exc_str or "not found" in exc_str:
                return False
            raise VectorStorageError(
                f"Failed to delete collection '{col_name}': {exc}"
            ) from exc

    def list_collections(self) -> List[str]:
        """
        List all collection names currently in ChromaDB.

        Returns:
            List of collection name strings.
        """
        try:
            raw_collections = self.client.list_collections()
            names = []
            for item in raw_collections:
                names.append(item.name if hasattr(item, "name") else str(item))
            return names
        except Exception as exc:
            raise VectorStorageError(f"Failed to list collections: {exc}") from exc

    def count(self, collection_id: str) -> int:
        """
        Return the total number of vectors in a collection.
        Returns 0 if the collection does not yet exist.

        Args:
            collection_id: SQLite collection UUID.

        Returns:
            Total item count in the collection.
        """
        try:
            col = self.get_collection(collection_id)
            return col.count()
        except CollectionNotFoundError:
            return 0
        except Exception as exc:
            raise VectorStorageError(
                f"Failed to get count for collection '{collection_id}': {exc}"
            ) from exc

    def _sanitize_metadata(self, chunk: EmbeddedChunk) -> Dict[str, Union[str, int, float, bool]]:
        """
        Extract and validate scalar-only metadata for ChromaDB.
        ChromaDB supports only str, int, float, bool.
        None, lists, and nested dictionaries are strictly excluded.

        Args:
            chunk: EmbeddedChunk instance.

        Returns:
            Dictionary of scalar metadata values.
        """
        meta: Dict[str, Union[str, int, float, bool]] = {
            "document_id": str(chunk.document_id),
            "collection_id": str(chunk.collection_id),
            "chunk_index": int(chunk.metadata.chunk_index),
            "char_count": int(chunk.metadata.char_count),
            "start_char": int(chunk.metadata.start_char),
            "end_char": int(chunk.metadata.end_char),
            "filename": str(chunk.metadata.filename),
            "file_type": str(chunk.metadata.file_type),
            "model_name": str(chunk.model_name),
        }

        # Optional page information — include only when present
        if chunk.metadata.page_number is not None:
            meta["page_number"] = int(chunk.metadata.page_number)
        if chunk.metadata.page_range is not None:
            meta["page_range"] = str(chunk.metadata.page_range)

        # Optional document version
        if hasattr(chunk.metadata, "version") and chunk.metadata.version is not None:
            meta["version"] = int(chunk.metadata.version)

        return meta

    def _validate_chunk(self, chunk: EmbeddedChunk) -> None:
        """
        Validate an EmbeddedChunk before inserting into ChromaDB.

        Args:
            chunk: EmbeddedChunk instance to validate.

        Raises:
            InvalidChunkDataError: If text, ID, or structure is invalid.
            InvalidVectorError: If vector is empty, contains non-numeric or non-finite values.
            VectorDimensionMismatchError: If dimension does not match expected_dimension (384).
        """
        if not isinstance(chunk, EmbeddedChunk):
            raise InvalidChunkDataError(
                f"Expected EmbeddedChunk instance, got {type(chunk).__name__}"
            )

        if not chunk.chunk_id or not str(chunk.chunk_id).strip():
            raise InvalidChunkDataError("chunk_id must be a non-empty string.")

        if not chunk.chunk_text or not str(chunk.chunk_text).strip():
            raise InvalidChunkDataError(
                f"chunk_text for chunk '{chunk.chunk_id}' cannot be empty or whitespace only."
            )

        if not chunk.embedding:
            raise InvalidVectorError(
                f"Embedding vector for chunk '{chunk.chunk_id}' cannot be empty."
            )

        if len(chunk.embedding) != self.expected_dimension:
            raise VectorDimensionMismatchError(
                f"Chunk '{chunk.chunk_id}' embedding dimension {len(chunk.embedding)} "
                f"does not match expected dimension {self.expected_dimension}."
            )

        for idx, val in enumerate(chunk.embedding):
            if not isinstance(val, (int, float)):
                raise InvalidVectorError(
                    f"Vector value at index {idx} in chunk '{chunk.chunk_id}' is non-numeric: {type(val)}"
                )
            if math.isnan(val) or math.isinf(val):
                raise InvalidVectorError(
                    f"Non-finite value ({val}) detected at index {idx} in chunk '{chunk.chunk_id}'."
                )

    def store_chunks(
        self,
        chunks: Sequence[EmbeddedChunk],
        collection_id: Optional[str] = None,
    ) -> int:
        """
        Store embedded chunks in ChromaDB using idempotent upsert operations.

        Args:
            chunks: Sequence of EmbeddedChunk instances.
            collection_id: Optional collection UUID override. If omitted, derived from chunks.

        Returns:
            Number of chunks successfully stored.

        Raises:
            InvalidChunkDataError: If input sequence or chunks fail validation.
            InvalidVectorError: If vectors contain NaN, Inf, or non-numeric values.
            VectorDimensionMismatchError: If vector dimensions mismatch expected dimension.
            VectorStorageError: If ChromaDB upsert fails.
        """
        if not chunks:
            return 0

        target_col_id = collection_id or chunks[0].collection_id
        if not target_col_id or not str(target_col_id).strip():
            raise InvalidChunkDataError("No collection_id provided or found in chunks.")

        # Pre-validate all chunks before writing any to avoid partial batch corruption
        for chunk in chunks:
            if collection_id and chunk.collection_id != collection_id:
                raise InvalidChunkDataError(
                    f"Chunk '{chunk.chunk_id}' collection_id '{chunk.collection_id}' "
                    f"does not match target collection '{collection_id}'."
                )
            self._validate_chunk(chunk)

        col = self.get_or_create_collection(target_col_id)

        total_stored = 0
        num_chunks = len(chunks)

        # Process in bounded batches to conserve memory and maintain responsiveness
        for i in range(0, num_chunks, self.batch_size):
            batch = chunks[i : i + self.batch_size]
            ids = [str(c.chunk_id) for c in batch]
            embeddings = [list(c.embedding) for c in batch]
            documents = [str(c.chunk_text) for c in batch]
            metadatas = [self._sanitize_metadata(c) for c in batch]

            try:
                col.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )
                total_stored += len(batch)
            except Exception as exc:
                raise VectorStorageError(
                    f"Failed to upsert batch {i // self.batch_size + 1} "
                    f"(items {i} to {i + len(batch)}) into collection '{target_col_id}': {exc}"
                ) from exc

        return total_stored

    def delete_document_vectors(
        self,
        collection_id: str,
        document_id: str,
        version: Optional[int] = None,
        exclude_version: Optional[int] = None,
    ) -> int:
        """
        Delete vector records belonging to a specific document in a collection.
        Optionally filters by document version or excludes a specific version.

        Args:
            collection_id: SQLite collection UUID.
            document_id: UUID of document whose vectors should be deleted.
            version: Optional version number to selectively delete.
            exclude_version: Optional version number to preserve (delete all others).

        Returns:
            Number of vector records deleted.

        Raises:
            InvalidChunkDataError: If document_id is empty or invalid.
            VectorStorageError: If deletion fails.
        """
        if not document_id or not isinstance(document_id, str) or not document_id.strip():
            raise InvalidChunkDataError("document_id must be a non-empty string.")

        try:
            col = self.get_collection(collection_id)
        except CollectionNotFoundError:
            return 0

        try:
            # Query matching chunk IDs and metadatas
            doc_id_clean = document_id.strip()
            existing = col.get(where={"document_id": doc_id_clean}, include=["metadatas"])
            raw_ids = existing.get("ids", [])
            metadatas = existing.get("metadatas", []) or []

            if not raw_ids:
                return 0

            target_ids = []
            for cid, meta in zip(raw_ids, metadatas):
                chunk_ver = meta.get("version") if meta else None
                if chunk_ver is None:
                    # Parse version from chunk_id pattern: {doc_id}_v{version}_c{idx}
                    parts = cid.split("_")
                    for p in parts:
                        if p.startswith("v") and p[1:].isdigit():
                            try:
                                chunk_ver = int(p[1:])
                                break
                            except ValueError:
                                pass

                if version is not None and chunk_ver != version:
                    continue
                if exclude_version is not None and chunk_ver == exclude_version:
                    continue
                target_ids.append(cid)

            if not target_ids:
                return 0

            col.delete(ids=target_ids)
            logger.info(
                "Deleted %d vector chunks for document '%s' (version=%s, exclude_version=%s) from collection '%s'.",
                len(target_ids),
                doc_id_clean,
                version,
                exclude_version,
                collection_id,
            )
            return len(target_ids)
        except Exception as exc:
            raise VectorStorageError(
                f"Failed to delete vectors for document '{document_id}' in collection '{collection_id}': {exc}"
            ) from exc

    def delete_chunks_by_ids(
        self,
        collection_id: str,
        chunk_ids: List[str],
    ) -> int:
        """
        Delete specific vector records by their chunk IDs.

        Args:
            collection_id: SQLite collection UUID.
            chunk_ids: List of chunk ID strings to delete.

        Returns:
            Number of chunk IDs requested for deletion.

        Raises:
            CollectionNotFoundError: If collection does not exist.
            VectorStorageError: If deletion fails.
        """
        if not chunk_ids:
            return 0

        try:
            col = self.get_collection(collection_id)
        except CollectionNotFoundError:
            return 0

        try:
            col.delete(ids=chunk_ids)
            return len(chunk_ids)
        except Exception as exc:
            raise VectorStorageError(
                f"Failed to delete chunks by IDs from collection '{collection_id}': {exc}"
            ) from exc

    def get_chunks_by_ids(
        self,
        collection_id: str,
        chunk_ids: List[str],
    ) -> Dict[str, Any]:
        """
        Retrieve stored vector records by chunk IDs for verification and citations.

        Args:
            collection_id: SQLite collection UUID.
            chunk_ids: List of chunk ID strings.

        Returns:
            Dictionary with keys: 'ids', 'embeddings', 'documents', 'metadatas'.

        Raises:
            CollectionNotFoundError: If collection does not exist.
            VectorStorageError: If retrieval fails.
        """
        if not chunk_ids:
            return {"ids": [], "embeddings": [], "documents": [], "metadatas": []}

        col = self.get_collection(collection_id)
        try:
            result = col.get(
                ids=chunk_ids,
                include=["embeddings", "documents", "metadatas"],
            )
            return {
                "ids": result.get("ids", []),
                "embeddings": [list(emb) for emb in result.get("embeddings", [])]
                if result.get("embeddings") is not None
                else [],
                "documents": result.get("documents", []),
                "metadatas": result.get("metadatas", []),
            }
        except Exception as exc:
            raise VectorStorageError(
                f"Failed to retrieve chunks by IDs from collection '{collection_id}': {exc}"
            ) from exc

    def query_vectors(
        self,
        collection_id: str,
        query_vector: Sequence[float],
        top_k: int = 5,
        document_ids: Optional[Union[str, Sequence[str]]] = None,
        where: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> VectorSearchResults:
        """
        Query vector records in a ChromaDB collection using nearest-neighbor similarity search.

        Converts raw ChromaDB cosine distance (d) to cosine similarity:
            similarity = 1.0 - distance

        Args:
            collection_id: SQLite collection UUID.
            query_vector: Dense embedding vector of dimension expected_dimension (384).
            top_k: Maximum candidate count to return (must be >= 1).
            document_ids: Optional single document ID or sequence of document IDs to filter by.
            where: Optional additional ChromaDB metadata filter dictionary.
            similarity_threshold: Optional minimum cosine similarity in [-1.0, 1.0].
                Candidates below this threshold are omitted.

        Returns:
            VectorSearchResults (a list of RetrievedChunk instances with attached
            total_candidates and returned_results properties).

        Raises:
            CollectionNotFoundError: If the collection does not exist in ChromaDB.
            InvalidChunkDataError: If collection_id, top_k, document_ids, or threshold is invalid.
            InvalidVectorError: If query_vector contains non-numeric, NaN, or Inf values.
            VectorDimensionMismatchError: If query_vector dimension does not match expected_dimension.
            VectorStorageError: If ChromaDB query fails unexpectedly.
        """
        # 1. Validate collection_id
        if not collection_id or not isinstance(collection_id, str) or not collection_id.strip():
            raise InvalidChunkDataError("collection_id must be a non-empty string.")

        # 2. Validate query_vector
        if query_vector is None or not isinstance(query_vector, Sequence) or isinstance(query_vector, (str, bytes)):
            raise InvalidVectorError("query_vector must be a sequence of floats.")
        if len(query_vector) == 0:
            raise InvalidVectorError("query_vector cannot be empty.")
        if len(query_vector) != self.expected_dimension:
            raise VectorDimensionMismatchError(
                f"Query vector dimension ({len(query_vector)}) does not match expected dimension ({self.expected_dimension})."
            )
        for idx, val in enumerate(query_vector):
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise InvalidVectorError(f"Vector value at index {idx} is non-numeric: {type(val)}")
            if math.isnan(val) or math.isinf(val):
                raise InvalidVectorError(f"Non-finite value ({val}) detected at index {idx} in query vector.")

        # 3. Validate top_k
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise InvalidChunkDataError(f"top_k must be an integer >= 1, got {top_k}.")

        # 4. Validate similarity_threshold
        if similarity_threshold is not None:
            if not isinstance(similarity_threshold, (int, float)) or isinstance(similarity_threshold, bool):
                raise InvalidChunkDataError("similarity_threshold must be a numeric float.")
            if math.isnan(similarity_threshold) or math.isinf(similarity_threshold):
                raise InvalidChunkDataError("similarity_threshold must be a finite number (not NaN or Inf).")
            if similarity_threshold < -1.0 or similarity_threshold > 1.0:
                raise InvalidChunkDataError(
                    f"similarity_threshold must be within [-1.0, 1.0], got {similarity_threshold}."
                )

        # 5. Build and validate filters
        combined_where: Optional[Dict[str, Any]] = None
        doc_filter: Optional[Dict[str, Any]] = None

        if document_ids is not None:
            if isinstance(document_ids, str):
                cleaned_ids = [document_ids.strip()]
            elif isinstance(document_ids, Sequence):
                cleaned_ids = [str(did).strip() for did in document_ids]
            else:
                raise InvalidChunkDataError("document_ids must be a string or sequence of strings.")

            if not cleaned_ids or any(not did for did in cleaned_ids):
                raise InvalidChunkDataError("document_ids cannot be empty or contain blank entries.")

            if len(cleaned_ids) != len(set(cleaned_ids)):
                raise InvalidChunkDataError("Duplicate document IDs are not permitted in document_ids filter.")

            if len(cleaned_ids) == 1:
                doc_filter = {"document_id": cleaned_ids[0]}
            else:
                doc_filter = {"document_id": {"$in": cleaned_ids}}

        if where is not None:
            if not isinstance(where, dict):
                raise InvalidChunkDataError("where must be a dictionary.")

        if doc_filter is not None and where is not None:
            if "$and" in where and isinstance(where["$and"], list):
                combined_where = {"$and": [doc_filter] + where["$and"]}
            else:
                combined_where = {"$and": [doc_filter, where]}
        elif doc_filter is not None:
            combined_where = doc_filter
        elif where is not None:
            combined_where = where

        # 6. Retrieve collection from ChromaDB
        col = self.get_collection(collection_id)

        # 7. Check for empty collection
        if col.count() == 0:
            return VectorSearchResults(
                [],
                total_candidates=0,
                metadata={
                    "collection_id": collection_id,
                    "top_k": top_k,
                    "similarity_threshold": similarity_threshold,
                    "filters": combined_where,
                },
            )

        # 8. Query ChromaDB
        try:
            raw = col.query(
                query_embeddings=[[float(x) for x in query_vector]],
                n_results=top_k,
                where=combined_where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStorageError(
                f"Failed to query vectors in collection '{collection_id}': {exc}"
            ) from exc

        # 9. Extract and transform candidate records
        raw_ids = (raw.get("ids") or [[]])[0]
        raw_docs = (raw.get("documents") or [[]])[0]
        raw_metas = (raw.get("metadatas") or [[]])[0]
        raw_dists = (raw.get("distances") or [[]])[0]

        total_candidates = len(raw_ids)
        candidates: List[RetrievedChunk] = []

        for cid, doc_text, meta, dist in zip(raw_ids, raw_docs, raw_metas, raw_dists):
            distance_val = float(dist)
            # Cosine distance to similarity: similarity = 1.0 - distance
            similarity_val = max(-1.0, min(1.0, 1.0 - distance_val))

            if similarity_threshold is not None and similarity_val < similarity_threshold:
                continue

            meta_dict = meta or {}
            chunk = RetrievedChunk(
                chunk_id=cid,
                document_id=str(meta_dict.get("document_id", "")),
                collection_id=str(meta_dict.get("collection_id", collection_id)),
                text=str(doc_text or ""),
                similarity_score=similarity_val,
                distance=distance_val,
                chunk_index=int(meta_dict.get("chunk_index", 0)),
                version=int(meta_dict["version"]) if "version" in meta_dict and meta_dict["version"] is not None else None,
                filename=str(meta_dict["filename"]) if "filename" in meta_dict and meta_dict["filename"] is not None else None,
                file_type=str(meta_dict["file_type"]) if "file_type" in meta_dict and meta_dict["file_type"] is not None else None,
                start_char=int(meta_dict["start_char"]) if "start_char" in meta_dict and meta_dict["start_char"] is not None else None,
                end_char=int(meta_dict["end_char"]) if "end_char" in meta_dict and meta_dict["end_char"] is not None else None,
                page_number=int(meta_dict["page_number"]) if "page_number" in meta_dict and meta_dict["page_number"] is not None else None,
                page_range=str(meta_dict["page_range"]) if "page_range" in meta_dict and meta_dict["page_range"] is not None else None,
            )
            candidates.append(chunk)

        return VectorSearchResults(
            candidates,
            total_candidates=total_candidates,
            metadata={
                "collection_id": collection_id,
                "top_k": top_k,
                "similarity_threshold": similarity_threshold,
                "filters": combined_where,
            },
        )
