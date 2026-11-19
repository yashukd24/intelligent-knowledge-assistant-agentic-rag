"""
backend/routers/rag.py
─────────────────────────────────────────────────────────────
Conventional RAG REST endpoints.

Routes:
  POST /api/collections/{collection_id}/rag/context — Assemble source-grounded
       context and injection-resistant prompt for downstream LLM generation.
─────────────────────────────────────────────────────────────
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import Settings, get_settings
from backend.database import get_db
from backend.models.collection import Collection
from backend.routers.retrieval import get_retrieval_service, get_vector_store
from backend.schemas.rag import RAGContextRequest, RAGResponse
from backend.services.context_builder import ContextBuilder
from backend.services.prompt_builder import PromptBuilder
from backend.services.rag_service import (
    InvalidRAGQueryError,
    RAGExecutionError,
    RAGService,
)
from backend.services.retrieval_service import RetrievalService
from backend.services.vector_store import CollectionNotFoundError, VectorStoreService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/collections", tags=["RAG"])


def get_context_builder(settings: Settings = Depends(get_settings)) -> ContextBuilder:
    """FastAPI dependency yielding a ContextBuilder instance."""
    return ContextBuilder(
        default_max_context_chars=getattr(settings, "rag_default_context_budget", 12000)
    )


def get_prompt_builder() -> PromptBuilder:
    """FastAPI dependency yielding a PromptBuilder instance."""
    return PromptBuilder()


def get_rag_service(
    settings: Settings = Depends(get_settings),
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    context_builder: ContextBuilder = Depends(get_context_builder),
    prompt_builder: PromptBuilder = Depends(get_prompt_builder),
) -> RAGService:
    """FastAPI dependency yielding a RAGService instance."""
    return RAGService(
        retrieval_service=retrieval_service,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        settings=settings,
    )


@router.post(
    "/{collection_id}/rag/context",
    response_model=RAGResponse,
    status_code=status.HTTP_200_OK,
    summary="Assemble source-grounded conventional RAG context and prompt",
    description=(
        "Executes grounded semantic vector search across indexed documents in "
        "the specified collection, validates candidate chunks against SQLite, "
        "assembles a structured context block respecting character budget limits, "
        "and constructs an injection-resistant prompt ready for Phase 9 LLM synthesis."
    ),
)
async def build_rag_context_endpoint(
    collection_id: UUID,
    payload: RAGContextRequest,
    db: AsyncSession = Depends(get_db),
    rag_service: RAGService = Depends(get_rag_service),
) -> RAGResponse:
    """
    Assemble source-grounded context and prompt for a knowledge collection query.

    - **collection_id**: UUID of the knowledge collection (from URL path, authoritative).
    - **payload**: Query parameters including natural-language question, top_k,
      optional document ID filters, similarity threshold, and max context characters.
    """
    # 1. Reject conflicting collection ID in request body if provided
    if payload.collection_id is not None and payload.collection_id != collection_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Collection ID in request body ({payload.collection_id}) does not match "
                f"authoritative collection ID in URL path ({collection_id})."
            ),
        )

    # 2. Authoritative verification of collection existence in SQLite
    col_str = str(collection_id)
    collection = await db.get(Collection, col_str)
    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection '{collection_id}' not found.",
        )

    # 3. Formulate internal RAGQuery with authoritative URL collection_id
    internal_query = payload.to_rag_query(collection_id=collection_id)

    # 4. Delegate execution to RAGService
    try:
        response = await rag_service.build_rag_context(query=internal_query, db=db)
        return response
    except CollectionNotFoundError:
        # Collection exists in SQLite but has zero vectors in ChromaDB yet (empty collection)
        logger.info(
            "ChromaDB collection missing for SQLite collection %s; returning empty RAG context.",
            collection_id,
        )
        empty_prompt = rag_service.prompt_builder.build_prompt(
            query=payload.query,
            assembled_context="",
            sources=[],
        )
        return RAGResponse(
            query=payload.query,
            collection_id=col_str,
            retrieved_chunk_count=0,
            assembled_context="",
            context_character_count=0,
            sources=[],
            source_metadata={},
            prompt_text=empty_prompt,
            excluded_chunks=[],
            retrieval_metadata={
                "top_k": payload.top_k,
                "similarity_threshold": payload.similarity_threshold,
                "collection_empty": True,
            },
        )
    except InvalidRAGQueryError as inv_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(inv_err),
        )
    except RAGExecutionError as exec_err:
        logger.error("RAG pipeline execution error on collection %s: %s", collection_id, exec_err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to assemble RAG context and prompt due to an internal execution error.",
        )
    except Exception as exc:
        logger.exception("Unexpected error in RAG endpoint on collection %s: %s", collection_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the RAG request.",
        )
