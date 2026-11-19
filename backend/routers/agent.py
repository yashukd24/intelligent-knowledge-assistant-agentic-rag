"""
backend/routers/agent.py
─────────────────────────────────────────────────────────────
FastAPI REST router for Agentic RAG Controller (Phase 10).

Routes:
  POST /api/collections/{collection_id}/agent/run — Execute the Agentic RAG
       state machine across planning, retrieval, evaluation, and synthesis.
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
from backend.routers.rag_answer import get_answer_generator, get_citation_validator
from backend.schemas.agent_state import (
    AgentEvent,
    AgentEventType,
    AgentRunRequest,
    AgentRunResponse,
    AgentStatus,
)
from backend.schemas.rag_answer import ExecutionTrace, TraceStage, TraceStageStatus
from backend.services.agent_controller import (
    AgentController,
    AgentControllerError,
    AgentExecutionError,
    InvalidStateTransitionError,
)
from backend.services.answer_generator import (
    AnswerGenerator,
    STANDARD_INSUFFICIENT_EVIDENCE_ANSWER,
)
from backend.services.citation_validator import CitationValidator
from backend.services.rag_service import (
    InvalidRAGQueryError,
    RAGExecutionError,
    RAGService,
)
from backend.services.vector_store import CollectionNotFoundError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/collections", tags=["Agentic RAG"])


def get_agent_controller(
    rag_service: RAGService = Depends(get_rag_service),
    answer_generator: AnswerGenerator = Depends(get_answer_generator),
    citation_validator: CitationValidator = Depends(get_citation_validator),
    settings: Settings = Depends(get_settings),
) -> AgentController:
    """FastAPI dependency yielding an AgentController instance."""
    return AgentController(
        rag_service=rag_service,
        answer_generator=answer_generator,
        citation_validator=citation_validator,
        settings=settings,
    )


@router.post(
    "/{collection_id}/agent/run",
    response_model=AgentRunResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute Agentic RAG state machine query",
    description=(
        "Executes a deterministic plain-Python state machine: initializes AgentState, "
        "generates an execution plan, retrieves candidates via RetrievalService, "
        "evaluates evidence sufficiency, synthesizes grounded answer, validates citations, "
        "and returns observable agent events and execution trace."
    ),
)
async def run_agent_endpoint(
    collection_id: UUID,
    payload: AgentRunRequest,
    db: AsyncSession = Depends(get_db),
    agent_controller: AgentController = Depends(get_agent_controller),
) -> AgentRunResponse:
    """
    Execute Agentic RAG state machine for a question against a collection.
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

    # 3. Delegate execution to AgentController
    try:
        response = await agent_controller.run(
            request=payload,
            collection_id=collection_id,
            db=db,
        )

        # Check if the execution failed due to provider-level timeout or server errors
        if response.status == AgentStatus.FAILED:
            error_msg = response.error_details or "Agent execution failed."
            err_lower = error_msg.lower()
            if "timeout" in err_lower or "timed out" in err_lower:
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail=f"Language model request timed out: {error_msg}",
                )
            elif "rate limit" in err_lower or "quota" in err_lower:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Language model rate limit exceeded: {error_msg}",
                )
            elif "provider" in err_lower or "502" in err_lower:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Language model provider error: {error_msg}",
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Agent execution failed: {error_msg}",
                )

        return response

    except HTTPException:
        # Re-raise explicit HTTP exceptions
        raise
    except CollectionNotFoundError:
        # Collection exists in SQLite but has zero vectors in ChromaDB yet
        logger.info(
            "ChromaDB collection missing for SQLite collection %s; returning rule-based refusal.",
            collection_id,
        )
        empty_events = [
            AgentEvent(
                event_type=AgentEventType.QUERY_UNDERSTOOD,
                step=1,
                payload={"query": payload.query, "collection_id": col_str},
                message="Query received for empty collection.",
            ),
            AgentEvent(
                event_type=AgentEventType.EVIDENCE_INSUFFICIENT,
                step=2,
                payload={"reason": "Collection empty in vector store."},
                message="Collection contains 0 indexed document vectors.",
            ),
            AgentEvent(
                event_type=AgentEventType.AGENT_DONE,
                step=3,
                payload={"outcome": "empty_collection_refusal"},
                message="Completed with zero-evidence refusal.",
            ),
        ]
        return AgentRunResponse(
            session_id=str(collection_id),
            trace_id=f"empty-{col_str[:8]}",
            collection_id=col_str,
            query=payload.query,
            status=AgentStatus.COMPLETED,
            answer=STANDARD_INSUFFICIENT_EVIDENCE_ANSWER,
            is_sufficient=False,
            evidence_score=0.0,
            iteration_count=0,
            sources=[],
            valid_citations=[],
            invalid_citations=[],
            events=empty_events,
            execution_trace=ExecutionTrace(
                trace_id=f"empty-{col_str[:8]}",
                stages=[
                    TraceStage(
                        id="step-1-query-understood",
                        name="Query Understood",
                        status=TraceStageStatus.COMPLETED,
                        details={"collection_id": col_str},
                    ),
                    TraceStage(
                        id="step-2-evidence-insufficient",
                        name="Evidence Insufficient",
                        status=TraceStageStatus.COMPLETED,
                        details={"collection_empty": True},
                    ),
                    TraceStage(
                        id="step-3-agent-done",
                        name="Agent Done",
                        status=TraceStageStatus.COMPLETED,
                        details={"refusal": True},
                    ),
                ],
                total_duration_ms=0.0,
            ),
            total_latency_ms=0.0,
        )
    except InvalidRAGQueryError as inv_err:
        try:
            await db.rollback()
        except Exception:
            pass
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(inv_err),
        )
    except (AgentExecutionError, RAGExecutionError) as exec_err:
        try:
            await db.rollback()
        except Exception:
            pass
        logger.error("Agent execution error on collection %s: %s", collection_id, exec_err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution encountered an internal error: {exec_err}",
        )
    except Exception as exc:
        try:
            await db.rollback()
        except Exception:
            pass
        raise
