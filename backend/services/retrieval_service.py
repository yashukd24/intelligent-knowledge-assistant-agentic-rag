"""
backend/services/retrieval_service.py
─────────────────────────────────────────────────────────────
Semantic Retrieval Service with authoritative SQLite validation.

Orchestrates:
  1. Input query validation (RetrievalQuery).
  2. Dense query vector generation via existing EmbeddingService.
  3. Nearest-neighbor candidate search via VectorStoreService.query_vectors().
  4. Single-trip batched SQLite validation:
     - Document existence in SQLite
     - Collection ownership match
     - Document status == 'indexed' (rejects pending, processing, failed, deletion_pending)
     - Document version match (rejects historical vectors & missing version metadata)
     - Provenance & metadata integrity
     - Duplicate chunk ID elimination
  5. Construction of grounded RetrievalResponse with rejection diagnostics.
─────────────────────────────────────────────────────────────
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Set, Union
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.config import Settings, get_settings
from backend.models.document import Document
from backend.schemas.document import DocumentStatus
from backend.schemas.retrieval import RetrievalQuery, RetrievalResponse, RetrievedChunk
from backend.services.embedding_service import (
    EmbeddingInferenceError,
    EmbeddingService,
    InvalidInputError,
)
from backend.services.vector_store import (
    CollectionNotFoundError,
    InvalidChunkDataError,
    InvalidVectorError,
    VectorDimensionMismatchError,
    VectorStoreError,
    VectorStoreService,
)

logger = logging.getLogger(__name__)


# ── Exception Hierarchy ──────────────────────────────────────────────────────


class RetrievalServiceError(Exception):
    """Base exception for all semantic retrieval service errors."""
    pass


class InvalidRetrievalQueryError(RetrievalServiceError):
    """Raised when a retrieval query or its parameters fail validation."""
    pass


class RetrievalExecutionError(RetrievalServiceError):
    """Raised when query embedding, vector search, or database validation fails."""
    pass


# ── Retrieval Service ────────────────────────────────────────────────────────


class RetrievalService:
    """
    Semantic Retrieval Service coordinating query embedding, vector search,
    and authoritative SQLite document & version validation gates.
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[VectorStoreService] = None,
        session_factory: Optional[async_sessionmaker[AsyncSession]] = None,
        settings: Optional[Settings] = None,
    ):
        """
        Initialize the RetrievalService using dependency injection.

        Args:
            embedding_service: Optional shared EmbeddingService instance.
                Defaults to a new service instance reusing the process model cache.
            vector_store: Optional VectorStoreService instance.
            session_factory: Optional async sessionmaker for database queries.
            settings: Optional application settings. Falls back to get_settings().
        """
        self.settings = settings or get_settings()
        self.embedding_service = embedding_service or EmbeddingService(settings=self.settings)
        self.vector_store = vector_store or VectorStoreService(settings=self.settings)
        self.session_factory = session_factory

    async def retrieve(
        self,
        query: Union[RetrievalQuery, Dict[str, Any]],
        db: Optional[AsyncSession] = None,
    ) -> RetrievalResponse:
        """
        Execute grounded semantic retrieval for a user query.

        Workflow:
          1. Validate RetrievalQuery schema and parameters.
          2. Generate unit-normalized query embedding via EmbeddingService.
          3. Query candidate vectors from ChromaDB via VectorStoreService.
          4. Execute single-trip batched SQLite lookup for candidate documents.
          5. Validate all candidates against authoritative SQLite records:
             - Document exists
             - Belongs to target collection
             - Status is INDEXED (rejects DELETION_PENDING, FAILED, PROCESSING, PENDING)
             - Version matches active SQLite version
             - Discards duplicate chunk IDs
          6. Truncate validated candidates to top_k and return RetrievalResponse.

        Args:
            query: Validated RetrievalQuery instance or dictionary of query arguments.
            db: Optional active async database session. If omitted, session_factory
                or AsyncSessionLocal is used.

        Returns:
            RetrievalResponse with verified, grounded chunks and diagnostic metadata.

        Raises:
            InvalidRetrievalQueryError: If query parameters fail validation.
            CollectionNotFoundError: If target collection does not exist in ChromaDB.
            RetrievalExecutionError: If database or model inference fails.
        """
        # 1. Validate query input
        if isinstance(query, RetrievalQuery):
            validated_query = query
        elif isinstance(query, dict):
            try:
                validated_query = RetrievalQuery.model_validate(query)
            except Exception as val_err:
                raise InvalidRetrievalQueryError(f"Invalid retrieval query parameters: {val_err}") from val_err
        else:
            raise InvalidRetrievalQueryError(
                f"Expected RetrievalQuery instance or dict, got {type(query).__name__}"
            )

        query_text = validated_query.query
        collection_id_str = str(validated_query.collection_id)
        top_k = validated_query.top_k
        similarity_threshold = validated_query.similarity_threshold
        document_ids = (
            [str(did) for did in validated_query.document_ids]
            if validated_query.document_ids
            else None
        )

        # 2. Generate query embedding via existing EmbeddingService
        try:
            query_vector = self.embedding_service.embed_text(query_text)
        except InvalidInputError as inp_err:
            raise InvalidRetrievalQueryError(f"Query text rejected by embedding service: {inp_err}") from inp_err
        except EmbeddingInferenceError as inf_err:
            raise RetrievalExecutionError(f"Query embedding inference failed: {inf_err}") from inf_err
        except Exception as emb_err:
            raise RetrievalExecutionError(f"Unexpected error generating query embedding: {emb_err}") from emb_err

        # Verify numerical integrity and dimension
        if len(query_vector) != self.embedding_service.dimension:
            raise RetrievalExecutionError(
                f"Generated query vector dimension ({len(query_vector)}) does not match "
                f"expected embedding dimension ({self.embedding_service.dimension})."
            )

        # 3. Candidate vector search in ChromaDB with oversampling
        # Oversample candidates to ensure top_k valid chunks remain after SQLite filtering
        candidate_k = min(max(top_k * 3, top_k), 100)

        try:
            candidates = self.vector_store.query_vectors(
                collection_id=collection_id_str,
                query_vector=query_vector,
                top_k=candidate_k,
                document_ids=document_ids,
                similarity_threshold=similarity_threshold,
            )
        except CollectionNotFoundError:
            # Re-raise directly to preserve standard collection error semantics
            raise
        except (InvalidVectorError, VectorDimensionMismatchError, InvalidChunkDataError) as vec_err:
            raise InvalidRetrievalQueryError(f"Vector search parameter error: {vec_err}") from vec_err
        except VectorStoreError as vs_err:
            raise RetrievalExecutionError(f"ChromaDB vector search failed: {vs_err}") from vs_err
        except Exception as exc:
            raise RetrievalExecutionError(f"Unexpected error during vector query: {exc}") from exc

        raw_candidate_count = candidates.total_candidates

        # If zero candidates returned from ChromaDB, return empty response immediately
        if len(candidates) == 0:
            return RetrievalResponse(
                query=query_text,
                collection_id=collection_id_str,
                results=[],
                total_candidates=raw_candidate_count,
                returned_results=0,
                rejected_candidates=0,
                accepted_candidates=0,
                retrieval_metadata={
                    "top_k": top_k,
                    "similarity_threshold": similarity_threshold,
                    "rejection_reasons": {},
                },
            )

        # 4. Batched SQLite Document & Version Validation
        # Collect distinct document IDs across all candidates
        candidate_doc_ids = list({c.document_id for c in candidates if c.document_id})

        # Determine database session to use
        session_to_close: Optional[AsyncSession] = None
        if db is not None:
            session = db
        elif self.session_factory is not None:
            session = self.session_factory()
            session_to_close = session
        else:
            from backend.database import AsyncSessionLocal
            session = AsyncSessionLocal()
            session_to_close = session

        try:
            # Single batched query across all candidate document IDs
            stmt = select(
                Document.id,
                Document.collection_id,
                Document.status,
                Document.version,
                Document.filename,
                Document.file_type,
            ).where(
                Document.id.in_(candidate_doc_ids)
            )

            result = await session.execute(stmt)
            db_docs = {str(row.id): row for row in result.all()}
        except SQLAlchemyError as db_err:
            logger.error("Database query failed during candidate validation: %s", db_err, exc_info=True)
            raise RetrievalExecutionError(f"Database error during candidate validation: {db_err}") from db_err
        except Exception as exc:
            logger.error("Unexpected failure during candidate SQLite validation: %s", exc, exc_info=True)
            raise RetrievalExecutionError(f"Unexpected error validating candidates against database: {exc}") from exc
        finally:
            if session_to_close is not None:
                await session_to_close.close()

        # 5. Validate every candidate against authoritative SQLite records
        accepted_candidates: List[RetrievedChunk] = []
        rejection_reasons: Dict[str, int] = {
            "duplicate_chunk_id": 0,
            "document_not_found": 0,
            "wrong_collection": 0,
            "not_indexed": 0,
            "version_mismatch": 0,
            "missing_version_metadata": 0,
            "invalid_metadata": 0,
            "below_threshold": 0,
        }
        rejected_count = 0
        seen_chunk_ids: Set[str] = set()

        for candidate in candidates:
            # Gate 1: Duplicate chunk ID elimination
            if candidate.chunk_id in seen_chunk_ids:
                rejection_reasons["duplicate_chunk_id"] += 1
                rejected_count += 1
                continue

            # Gate 2: Document existence in SQLite
            doc_row = db_docs.get(candidate.document_id)
            if doc_row is None:
                rejection_reasons["document_not_found"] += 1
                rejected_count += 1
                continue

            # Gate 3: Collection ownership verification
            if str(doc_row.collection_id) != collection_id_str:
                rejection_reasons["wrong_collection"] += 1
                rejected_count += 1
                continue

            # Gate 4: Document status gate (accept ONLY 'indexed')
            if doc_row.status != DocumentStatus.INDEXED.value:
                rejection_reasons["not_indexed"] += 1
                rejected_count += 1
                continue

            # Gate 5: Document version consistency
            if candidate.version is None:
                rejection_reasons["missing_version_metadata"] += 1
                rejected_count += 1
                continue

            if candidate.version != doc_row.version:
                rejection_reasons["version_mismatch"] += 1
                rejected_count += 1
                continue

            # Gate 6: Provenance & metadata integrity
            if not candidate.text or not candidate.text.strip():
                rejection_reasons["invalid_metadata"] += 1
                rejected_count += 1
                continue

            if not candidate.chunk_id or not candidate.chunk_id.strip():
                rejection_reasons["invalid_metadata"] += 1
                rejected_count += 1
                continue

            # Gate 7: Enforce similarity threshold strictly
            if similarity_threshold is not None and candidate.similarity_score < similarity_threshold:
                rejection_reasons["below_threshold"] += 1
                rejected_count += 1
                continue

            # Candidate passes all applicable gates
            seen_chunk_ids.add(candidate.chunk_id)
            # F-01: Backfill authoritative provenance from SQLite if missing on candidate
            if candidate.filename is None and doc_row.filename:
                candidate.filename = doc_row.filename
            if candidate.file_type is None and doc_row.file_type:
                candidate.file_type = doc_row.file_type
            accepted_candidates.append(candidate)

        # 6. Truncate validated candidates to requested top_k
        # Ranking order (descending similarity) is preserved from vector search
        final_results = accepted_candidates[:top_k]

        return RetrievalResponse(
            query=query_text,
            collection_id=collection_id_str,
            results=final_results,
            total_candidates=raw_candidate_count,
            returned_results=len(final_results),
            rejected_candidates=rejected_count,
            accepted_candidates=len(accepted_candidates),
            retrieval_metadata={
                "top_k": top_k,
                "similarity_threshold": similarity_threshold,
                "rejection_reasons": {k: v for k, v in rejection_reasons.items() if v > 0},
                "candidate_oversampling": len(candidates),
            },
        )
