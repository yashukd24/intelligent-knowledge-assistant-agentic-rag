"""
backend/services/rag_service.py
─────────────────────────────────────────────────────────────
RAGService: Orchestrates the conventional RAG pipeline:
  1. Validates RAG query parameters.
  2. Delegates semantic vector search and SQLite candidate validation
     to the existing RetrievalService.
  3. Formats source-grounded context with deterministic citations ([S1], [S2])
     and budget enforcement via ContextBuilder.
  4. Constructs an injection-resistant, grounded prompt via PromptBuilder.
  5. Returns a structured RAGResponse ready for Phase 9 LLM synthesis.

Constraints:
  - Reuses existing RetrievalService (no duplicate embeddings or ChromaDB clients).
  - Strictly deterministic; no external LLM API calls are made in Phase 8.
─────────────────────────────────────────────────────────────
"""

import logging
from typing import Any, Dict, Optional, Union
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import Settings, get_settings
from backend.schemas.rag import RAGQuery, RAGResponse
from backend.schemas.retrieval import RetrievalQuery
from backend.services.context_builder import ContextBuilder
from backend.services.prompt_builder import PromptBuilder
from backend.services.retrieval_service import (
    InvalidRetrievalQueryError,
    RetrievalExecutionError,
    RetrievalService,
)
from backend.services.vector_store import CollectionNotFoundError

logger = logging.getLogger(__name__)


class RAGServiceError(Exception):
    """Base exception for RAGService errors."""
    pass


class InvalidRAGQueryError(RAGServiceError):
    """Raised when RAG query parameters fail validation."""
    pass


class RAGExecutionError(RAGServiceError):
    """Raised when retrieval or context/prompt assembly encounters an internal error."""
    pass


class RAGService:
    """
    Conventional RAG pipeline service orchestrating retrieval, context assembly,
    and prompt generation.
    """

    def __init__(
        self,
        retrieval_service: Optional[RetrievalService] = None,
        context_builder: Optional[ContextBuilder] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        settings: Optional[Settings] = None,
    ):
        """
        Initialize RAGService via dependency injection.

        Args:
            retrieval_service: Existing RetrievalService instance.
            context_builder: ContextBuilder instance for structured context assembly.
            prompt_builder: PromptBuilder instance for prompt construction.
            settings: Application configuration settings.
        """
        self.settings = settings or get_settings()
        self.retrieval_service = retrieval_service or RetrievalService(settings=self.settings)
        self.context_builder = context_builder or ContextBuilder(
            default_max_context_chars=getattr(self.settings, "rag_default_context_budget", 12000)
        )
        self.prompt_builder = prompt_builder or PromptBuilder()

    async def build_rag_context(
        self,
        query: Union[RAGQuery, Dict[str, Any]],
        db: Optional[AsyncSession] = None,
    ) -> RAGResponse:
        """
        Execute the conventional RAG pipeline to produce structured context
        and an injection-resistant prompt.

        Workflow:
          1. Validate RAGQuery input.
          2. Formulate internal RetrievalQuery and query RetrievalService.
          3. Format retrieved chunks into context with [S1], [S2] source tags.
          4. Build deterministic, grounded LLM prompt.
          5. Return validated RAGResponse.

        Args:
            query: Validated RAGQuery instance or dictionary of query arguments.
            db: Optional active async database session for SQLite validation.

        Returns:
            RAGResponse containing context, sources, citations, and prompt text.

        Raises:
            InvalidRAGQueryError: If query parameters fail validation.
            CollectionNotFoundError: If target collection does not exist in ChromaDB.
            RAGExecutionError: If retrieval or context assembly fails.
        """
        # 1. Validate query input
        if isinstance(query, RAGQuery):
            validated_query = query
        elif isinstance(query, dict):
            try:
                validated_query = RAGQuery.model_validate(query)
            except Exception as val_err:
                raise InvalidRAGQueryError(f"Invalid RAG query parameters: {val_err}") from val_err
        else:
            raise InvalidRAGQueryError(
                f"Expected RAGQuery instance or dict, got {type(query).__name__}"
            )

        collection_id_str = str(validated_query.collection_id)

        # 2. Formulate internal RetrievalQuery for RetrievalService
        retrieval_query = RetrievalQuery(
            query=validated_query.query,
            collection_id=validated_query.collection_id,
            top_k=validated_query.top_k,
            document_ids=validated_query.document_ids,
            similarity_threshold=validated_query.similarity_threshold,
        )

        # 3. Execute grounded semantic retrieval
        try:
            retrieval_resp = await self.retrieval_service.retrieve(query=retrieval_query, db=db)
        except InvalidRetrievalQueryError as inv_err:
            raise InvalidRAGQueryError(f"Retrieval query validation error: {inv_err}") from inv_err
        except CollectionNotFoundError:
            # Re-raise directly to preserve standard collection error semantics
            raise
        except RetrievalExecutionError as ret_err:
            raise RAGExecutionError(f"Retrieval execution error: {ret_err}") from ret_err
        except Exception as exc:
            raise RAGExecutionError(f"Unexpected error during retrieval: {exc}") from exc

        # 4. Assemble source-grounded context with budget enforcement
        try:
            context_result = self.context_builder.build_context(
                chunks=retrieval_resp.results,
                max_context_chars=validated_query.max_context_chars,
                expected_collection_id=collection_id_str,
            )
        except Exception as ctx_err:
            raise RAGExecutionError(f"Context assembly failed: {ctx_err}") from ctx_err

        # 5. Construct grounded prompt
        try:
            prompt_text = self.prompt_builder.build_prompt(
                query=validated_query.query,
                assembled_context=context_result.assembled_context,
                sources=context_result.sources,
            )
        except Exception as prm_err:
            raise RAGExecutionError(f"Prompt construction failed: {prm_err}") from prm_err

        # 6. Assemble and return validated RAGResponse
        return RAGResponse(
            query=validated_query.query,
            collection_id=collection_id_str,
            retrieved_chunk_count=len(retrieval_resp.results),
            assembled_context=context_result.assembled_context,
            context_character_count=context_result.context_character_count,
            sources=context_result.sources,
            source_metadata=context_result.source_metadata,
            prompt_text=prompt_text,
            excluded_chunks=context_result.excluded_chunks,
            retrieval_metadata=retrieval_resp.retrieval_metadata,
        )
