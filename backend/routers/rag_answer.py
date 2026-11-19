"""
backend/routers/rag_answer.py
─────────────────────────────────────────────────────────────
FastAPI REST router for conventional RAG answer generation.

Routes:
  POST /api/collections/{collection_id}/rag/answer — Retrieve relevant context,
       query Claude API, validate citations, and return structured answer.
─────────────────────────────────────────────────────────────
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import Settings, get_settings
from backend.database import get_db
from backend.models.collection import Collection
from backend.routers.rag import get_rag_service
from backend.schemas.rag_answer import RAGAnswerRequest, RAGAnswerResponse
from backend.services.answer_generator import AnswerGenerator, AnswerStatus
from backend.services.anthropic_provider import AnthropicProvider
from backend.services.citation_validator import CitationValidator
from backend.services.llm_provider import LLMProvider
from backend.services.rag_answer_service import RAGAnswerService
from backend.services.rag_service import (
    InvalidRAGQueryError,
    RAGExecutionError,
    RAGService,
)
from backend.services.vector_store import CollectionNotFoundError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/collections", tags=["RAG Answer"])


def get_llm_provider(settings: Settings = Depends(get_settings)) -> LLMProvider:
    """FastAPI dependency yielding an LLM completion provider based on configuration."""
    provider_name = getattr(settings, "llm_provider", "ollama").lower().strip()
    if provider_name == "ollama":
        from backend.services.ollama_provider import OllamaProvider
        return OllamaProvider(settings=settings)
    elif provider_name == "anthropic":
        from backend.services.anthropic_provider import AnthropicProvider
        return AnthropicProvider(settings=settings)
    else:
        logger.warning("Unrecognized LLM provider '%s'; defaulting to Ollama.", provider_name)
        from backend.services.ollama_provider import OllamaProvider
        return OllamaProvider(settings=settings)


def get_citation_validator() -> CitationValidator:
    """FastAPI dependency yielding a CitationValidator instance."""
    return CitationValidator()


def get_answer_generator(
    llm_provider: LLMProvider = Depends(get_llm_provider),
    citation_validator: CitationValidator = Depends(get_citation_validator),
) -> AnswerGenerator:
    """FastAPI dependency yielding an AnswerGenerator instance."""
    return AnswerGenerator(
        llm_provider=llm_provider,
        citation_validator=citation_validator,
    )


def get_rag_answer_service(
    rag_service: RAGService = Depends(get_rag_service),
    answer_generator: AnswerGenerator = Depends(get_answer_generator),
    settings: Settings = Depends(get_settings),
) -> RAGAnswerService:
    """FastAPI dependency yielding a RAGAnswerService instance."""
    return RAGAnswerService(
        rag_service=rag_service,
        answer_generator=answer_generator,
        settings=settings,
    )


@router.post(
    "/{collection_id}/rag/answer",
    response_model=RAGAnswerResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate source-grounded answer using Claude and semantic retrieval",
    description=(
        "Executes grounded semantic vector search across indexed documents in "
        "the specified collection, validates candidate chunks against SQLite, "
        "assembles a structured context block respecting character budget limits, "
        "sends the grounded prompt to Anthropic Claude, validates citations in the "
        "generated text, and returns a structured answer response."
    ),
)
async def generate_rag_answer_endpoint(
    collection_id: UUID,
    payload: RAGAnswerRequest,
    db: AsyncSession = Depends(get_db),
    rag_answer_service: RAGAnswerService = Depends(get_rag_answer_service),
) -> RAGAnswerResponse:
    """
    Generate a grounded answer for a natural-language question.

    - **collection_id**: UUID of the knowledge collection (from URL path, authoritative).
    - **payload**: Query parameters including natural-language question, top_k,
      optional document ID filters, similarity threshold, and max output tokens.
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

    # 3. Delegate execution to RAGAnswerService
    try:
        response = await rag_answer_service.answer_question(
            collection_id=collection_id,
            request=payload,
            db=db,
        )

        # Map provider-level controlled error statuses to proper HTTP error codes
        if response.status == AnswerStatus.TIMEOUT:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Language model request timed out.",
            )
        elif response.status == AnswerStatus.RATE_LIMITED:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Language model rate limit or quota exceeded.",
            )
        elif response.status == AnswerStatus.PROVIDER_ERROR:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Language model provider service error.",
            )

        return response
    except HTTPException:
        # Re-raise explicit HTTPExceptions directly
        raise
    except CollectionNotFoundError:
        # Collection exists in SQLite but has zero vectors in ChromaDB yet
        logger.info(
            "ChromaDB collection missing for SQLite collection %s; returning empty answer.",
            collection_id,
        )
        return RAGAnswerResponse(
            query=payload.query,
            answer=STANDARD_INSUFFICIENT_EVIDENCE_ANSWER,
            status=AnswerStatus.INSUFFICIENT_EVIDENCE,
            collection_id=col_str,
            sources=[],
            valid_citations=[],
            invalid_citations=[],
            uncited_sources=[],
            provider="rule_based",
            model="none",
            usage=None,
            generation_latency_ms=0.0,
            total_latency_ms=0.0,
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
            detail="Failed to execute RAG answer pipeline due to an internal execution error.",
        )
    except Exception as exc:
        logger.exception("Unexpected error in RAG answer endpoint on collection %s: %s", collection_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the RAG answer request.",
        )
