"""
backend/routers/retrieval.py
─────────────────────────────────────────────────────────────
Semantic Retrieval REST endpoints.

Routes:
  POST /api/collections/{collection_id}/query — Query collection via semantic search
─────────────────────────────────────────────────────────────
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import Settings, get_settings
from backend.database import get_db
from backend.models.collection import Collection
from backend.schemas.retrieval import RetrievalQueryRequest, RetrievalResponse
from backend.services.retrieval_service import (
    InvalidRetrievalQueryError,
    RetrievalExecutionError,
    RetrievalService,
)
from backend.services.vector_store import CollectionNotFoundError, VectorStoreService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/collections", tags=["Retrieval"])


def get_vector_store(settings: Settings = Depends(get_settings)) -> VectorStoreService:
    """FastAPI dependency yielding a VectorStoreService instance."""
    return VectorStoreService(settings=settings)


def get_retrieval_service(
    settings: Settings = Depends(get_settings),
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> RetrievalService:
    """FastAPI dependency yielding a RetrievalService instance."""
    return RetrievalService(settings=settings, vector_store=vector_store)


@router.post(
    "/{collection_id}/query",
    response_model=RetrievalResponse,
    status_code=status.HTTP_200_OK,
    summary="Semantic retrieval from a knowledge collection",
    description=(
        "Executes grounded semantic vector search across indexed documents in "
        "the specified collection. All candidates are validated against SQLite "
        "for existence, collection ownership, active version freshness, and "
        "INDEXED lifecycle status before being returned."
    ),
)
async def query_collection(
    collection_id: UUID,
    payload: RetrievalQueryRequest,
    db: AsyncSession = Depends(get_db),
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
) -> RetrievalResponse:
    """
    Query a knowledge collection using natural language.

    - **collection_id**: UUID of the knowledge collection (from URL path, authoritative).
    - **payload**: Query parameters including natural-language query string, top_k,
      optional document ID filters, and optional similarity threshold.
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

    # 3. Formulate internal RetrievalQuery with authoritative URL collection_id
    internal_query = payload.to_retrieval_query(collection_id=collection_id)

    # 4. Delegate execution to RetrievalService
    try:
        response = await retrieval_service.retrieve(query=internal_query, db=db)
        return response
    except CollectionNotFoundError:
        # Collection exists in SQLite but has zero vectors in ChromaDB yet (empty collection)
        logger.info("ChromaDB collection missing for SQLite collection %s; returning empty results.", collection_id)
        return RetrievalResponse(
            query=payload.query,
            collection_id=col_str,
            results=[],
            total_candidates=0,
            returned_results=0,
            rejected_candidates=0,
            accepted_candidates=0,
            retrieval_metadata={
                "top_k": payload.top_k,
                "similarity_threshold": payload.similarity_threshold,
                "rejection_reasons": {},
                "note": "Empty collection in vector store",
            },
        )
    except InvalidRetrievalQueryError as q_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(q_err),
        ) from q_err
    except RetrievalExecutionError as exec_err:
        logger.error(
            "Retrieval execution failure for collection %s: %s",
            collection_id,
            exec_err,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Retrieval service encountered an internal execution error.",
        ) from exec_err
    except Exception as exc:
        logger.error(
            "Unexpected error in query_collection for collection %s: %s",
            collection_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during retrieval.",
        ) from exc
