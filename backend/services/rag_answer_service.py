"""
backend/services/rag_answer_service.py
─────────────────────────────────────────────────────────────
RAGAnswerService: Orchestrates end-to-end grounded question answering:
  1. Reuses existing RAGService to execute retrieval, SQLite validation,
     context assembly, and injection-resistant prompt construction.
  2. Evaluates evidence availability; bypasses LLM calls when zero evidence exists.
  3. Dispatches prompt to AnswerGenerator (backed by AnthropicProvider or mock).
  4. Validates citations and constructs a comprehensive RAGAnswerResponse.
─────────────────────────────────────────────────────────────
"""

from datetime import datetime, timezone
import logging
import time
from typing import Any, Dict, Optional, Union
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import Settings, get_settings
from backend.schemas.rag import RAGQuery
from backend.schemas.rag_answer import (
    ExecutionTrace,
    RAGAnswerRequest,
    RAGAnswerResponse,
    TraceStage,
    TraceStageStatus,
)
from backend.services.answer_generator import AnswerGenerator, AnswerStatus
from backend.services.rag_service import (
    InvalidRAGQueryError,
    RAGExecutionError,
    RAGService,
)
from backend.services.vector_store import CollectionNotFoundError

logger = logging.getLogger(__name__)


class RAGAnswerServiceError(Exception):
    """Base exception for RAG answer service errors."""
    pass


class RAGAnswerService:
    """
    Orchestrates grounded question answering across RAGService and AnswerGenerator.
    """

    def __init__(
        self,
        rag_service: RAGService,
        answer_generator: AnswerGenerator,
        settings: Optional[Settings] = None,
    ):
        """
        Initialize RAGAnswerService via dependency injection.

        Args:
            rag_service: Existing RAGService instance.
            answer_generator: AnswerGenerator instance backed by LLMProvider.
            settings: Optional Settings singleton.
        """
        self.rag_service = rag_service
        self.answer_generator = answer_generator
        self.settings = settings or get_settings()

    async def answer_question(
        self,
        collection_id: UUID,
        request: Union[RAGAnswerRequest, Dict[str, Any]],
        db: Optional[AsyncSession] = None,
    ) -> RAGAnswerResponse:
        """
        Execute end-to-end grounded answer generation for a question.

        Workflow:
          1. Validate request payload into internal RAGQuery.
          2. Execute RAGService.build_rag_context to obtain context and prompt.
          3. Inspect retrieved evidence:
             - If 0 chunks retrieved or no sources fit budget, bypass Claude.
             - If evidence exists, invoke AnswerGenerator with prompt & sources.
          4. Validate citations and assemble final RAGAnswerResponse.

        Args:
            collection_id: Authoritative collection UUID.
            request: Validated RAGAnswerRequest instance or dictionary.
            db: Optional active async database session for SQLite validation.

        Returns:
            Structured RAGAnswerResponse model.

        Raises:
            CollectionNotFoundError: If target collection does not exist in ChromaDB.
            InvalidRAGQueryError: If query parameters fail validation.
            RAGExecutionError: If pipeline execution encounters an internal error.
        """
        t0 = time.perf_counter()
        t0_iso = datetime.now(timezone.utc).isoformat()
        trace_id = str(uuid4())
        stages: list[TraceStage] = []

        # Stage 1: User Query Received
        t_s1 = time.perf_counter()
        raw_query_text = request.query if isinstance(request, RAGAnswerRequest) else (request.get("query", "") if isinstance(request, dict) else "")
        stages.append(
            TraceStage(
                id="stage-1-query",
                name="User Query Received",
                status=TraceStageStatus.COMPLETED,
                started_at=t0_iso,
                completed_at=datetime.now(timezone.utc).isoformat(),
                duration_ms=round((time.perf_counter() - t_s1) * 1000.0, 2),
                details={
                    "query_length": len(raw_query_text),
                    "query_preview": raw_query_text[:120] if len(raw_query_text) > 120 else raw_query_text,
                },
            )
        )

        # Stage 2: Query Validation
        t_s2 = time.perf_counter()
        t_s2_iso = datetime.now(timezone.utc).isoformat()
        if isinstance(request, RAGAnswerRequest):
            payload = request
        elif isinstance(request, dict):
            try:
                payload = RAGAnswerRequest.model_validate(request)
            except Exception as val_err:
                stages.append(
                    TraceStage(
                        id="stage-2-validation",
                        name="Query Validation",
                        status=TraceStageStatus.FAILED,
                        started_at=t_s2_iso,
                        completed_at=datetime.now(timezone.utc).isoformat(),
                        duration_ms=round((time.perf_counter() - t_s2) * 1000.0, 2),
                        error_message=str(val_err),
                    )
                )
                raise InvalidRAGQueryError(f"Invalid RAG answer query parameters: {val_err}") from val_err
        else:
            stages.append(
                TraceStage(
                    id="stage-2-validation",
                    name="Query Validation",
                    status=TraceStageStatus.FAILED,
                    started_at=t_s2_iso,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    duration_ms=round((time.perf_counter() - t_s2) * 1000.0, 2),
                    error_message=f"Expected RAGAnswerRequest or dict, got {type(request).__name__}",
                )
            )
            raise InvalidRAGQueryError(f"Expected RAGAnswerRequest or dict, got {type(request).__name__}")

        stages.append(
            TraceStage(
                id="stage-2-validation",
                name="Query Validation",
                status=TraceStageStatus.COMPLETED,
                started_at=t_s2_iso,
                completed_at=datetime.now(timezone.utc).isoformat(),
                duration_ms=round((time.perf_counter() - t_s2) * 1000.0, 2),
                details={
                    "top_k": payload.top_k,
                    "max_context_chars": payload.max_context_chars,
                    "similarity_threshold": payload.similarity_threshold,
                    "document_filters_count": len(payload.document_ids) if payload.document_ids else 0,
                },
            )
        )

        # Stage 3: Collection Selection
        t_s3 = time.perf_counter()
        t_s3_iso = datetime.now(timezone.utc).isoformat()
        rag_query = RAGQuery(
            query=payload.query,
            collection_id=collection_id,
            top_k=payload.top_k,
            document_ids=payload.document_ids,
            similarity_threshold=payload.similarity_threshold,
            max_context_chars=payload.max_context_chars,
        )
        stages.append(
            TraceStage(
                id="stage-3-collection",
                name="Collection Selection",
                status=TraceStageStatus.COMPLETED,
                started_at=t_s3_iso,
                completed_at=datetime.now(timezone.utc).isoformat(),
                duration_ms=round((time.perf_counter() - t_s3) * 1000.0, 2),
                details={
                    "collection_id": str(collection_id),
                    "scope": "filtered_documents" if payload.document_ids else "entire_collection",
                },
            )
        )

        # 2. Build RAG context and prompt using existing Phase 8 pipeline
        t_ctx = time.perf_counter()
        t_ctx_iso = datetime.now(timezone.utc).isoformat()
        try:
            rag_context_resp = await self.rag_service.build_rag_context(query=rag_query, db=db)
        except CollectionNotFoundError as c_err:
            stages.append(
                TraceStage(
                    id="stage-4-retrieval",
                    name="Document Retrieval",
                    status=TraceStageStatus.FAILED,
                    started_at=t_ctx_iso,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    duration_ms=round((time.perf_counter() - t_ctx) * 1000.0, 2),
                    error_message=f"Collection '{collection_id}' not found.",
                )
            )
            raise
        except InvalidRAGQueryError as inv_err:
            stages.append(
                TraceStage(
                    id="stage-4-retrieval",
                    name="Document Retrieval",
                    status=TraceStageStatus.FAILED,
                    started_at=t_ctx_iso,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    duration_ms=round((time.perf_counter() - t_ctx) * 1000.0, 2),
                    error_message=str(inv_err),
                )
            )
            raise
        except RAGExecutionError as exec_err:
            stages.append(
                TraceStage(
                    id="stage-4-retrieval",
                    name="Document Retrieval",
                    status=TraceStageStatus.FAILED,
                    started_at=t_ctx_iso,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    duration_ms=round((time.perf_counter() - t_ctx) * 1000.0, 2),
                    error_message=str(exec_err),
                )
            )
            raise
        except Exception as exc:
            logger.exception("RAG context assembly error during answer generation: %s", exc)
            stages.append(
                TraceStage(
                    id="stage-4-retrieval",
                    name="Document Retrieval",
                    status=TraceStageStatus.FAILED,
                    started_at=t_ctx_iso,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    duration_ms=round((time.perf_counter() - t_ctx) * 1000.0, 2),
                    error_message=str(exc),
                )
            )
            raise RAGExecutionError(f"Failed to assemble RAG context: {exc}") from exc

        # Stage 4: Document Retrieval
        retrieval_meta = rag_context_resp.retrieval_metadata or {}
        raw_candidates_count = retrieval_meta.get("candidate_count", rag_context_resp.retrieved_chunk_count)
        stages.append(
            TraceStage(
                id="stage-4-retrieval",
                name="Document Retrieval",
                status=TraceStageStatus.COMPLETED,
                started_at=t_ctx_iso,
                completed_at=datetime.now(timezone.utc).isoformat(),
                duration_ms=None,
                details={
                    "retrieved_chunk_count": rag_context_resp.retrieved_chunk_count,
                    "top_k_requested": payload.top_k,
                    "candidate_count": raw_candidates_count,
                    "vector_db": "ChromaDB (cosine similarity)",
                    "timing_note": "Sub-stage duration unmeasured (included in total pipeline latency)",
                },
            )
        )

        # Stage 5: Similarity Filtering
        excluded_chunks = getattr(rag_context_resp, "excluded_chunks", [])
        stages.append(
            TraceStage(
                id="stage-5-filtering",
                name="Similarity Filtering",
                status=TraceStageStatus.COMPLETED,
                started_at=datetime.now(timezone.utc).isoformat(),
                completed_at=datetime.now(timezone.utc).isoformat(),
                duration_ms=None,
                details={
                    "similarity_threshold": payload.similarity_threshold,
                    "passing_sources_count": len(rag_context_resp.sources),
                    "excluded_chunks_count": len(excluded_chunks),
                    "max_similarity_score": round(max([s.similarity_score for s in rag_context_resp.sources], default=0.0), 4),
                    "timing_note": "Sub-stage duration unmeasured (evaluated inside SQLite validation gate)",
                },
            )
        )

        # Stage 6: Context Construction
        has_evidence = len(rag_context_resp.sources) > 0 and bool(rag_context_resp.assembled_context.strip())
        stages.append(
            TraceStage(
                id="stage-6-context",
                name="Context Construction",
                status=TraceStageStatus.COMPLETED,
                started_at=datetime.now(timezone.utc).isoformat(),
                completed_at=datetime.now(timezone.utc).isoformat(),
                duration_ms=None,
                details={
                    "character_budget": payload.max_context_chars,
                    "assembled_context_chars": rag_context_resp.context_character_count,
                    "sources_included": [s.source_id for s in rag_context_resp.sources],
                    "has_evidence": has_evidence,
                    "timing_note": "Sub-stage duration unmeasured (deterministic context formatting)",
                },
            )
        )

        # Stage 7: LLM Answer Generation
        t_gen = time.perf_counter()
        t_gen_iso = datetime.now(timezone.utc).isoformat()
        answer_result = await self.answer_generator.generate_answer(
            prompt=rag_context_resp.prompt_text,
            sources=rag_context_resp.sources,
            has_evidence=has_evidence,
            max_tokens=payload.max_output_tokens,
            temperature=payload.temperature,
        )
        gen_duration_ms = round((time.perf_counter() - t_gen) * 1000.0, 2)

        if not has_evidence:
            stages.append(
                TraceStage(
                    id="stage-7-generation",
                    name="LLM Answer Generation",
                    status=TraceStageStatus.SKIPPED,
                    started_at=t_gen_iso,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    duration_ms=gen_duration_ms,
                    details={
                        "reason": "Bypassed LLM inference because zero relevant evidence was found.",
                        "rule_based_refusal": True,
                    },
                )
            )
        elif answer_result.status in (AnswerStatus.PROVIDER_ERROR, AnswerStatus.TIMEOUT, AnswerStatus.RATE_LIMITED):
            stages.append(
                TraceStage(
                    id="stage-7-generation",
                    name="LLM Answer Generation",
                    status=TraceStageStatus.FAILED,
                    started_at=t_gen_iso,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    duration_ms=gen_duration_ms,
                    details={
                        "provider": answer_result.provider,
                        "model": answer_result.model,
                        "status": answer_result.status.value,
                    },
                    error_message=answer_result.error_message or f"Generation failed with status {answer_result.status.value}",
                )
            )
        else:
            stages.append(
                TraceStage(
                    id="stage-7-generation",
                    name="LLM Answer Generation",
                    status=TraceStageStatus.COMPLETED,
                    started_at=t_gen_iso,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    duration_ms=gen_duration_ms,
                    details={
                        "provider": answer_result.provider,
                        "model": answer_result.model,
                        "prompt_tokens": answer_result.usage.prompt_tokens if answer_result.usage else None,
                        "completion_tokens": answer_result.usage.completion_tokens if answer_result.usage else None,
                        "total_tokens": answer_result.usage.total_tokens if answer_result.usage else None,
                        "temperature": payload.temperature,
                    },
                )
            )

        # Stage 8: Citation Validation
        t_cit = time.perf_counter()
        t_cit_iso = datetime.now(timezone.utc).isoformat()
        citations = answer_result.citations
        if not has_evidence or answer_result.status != AnswerStatus.ANSWERED:
            stages.append(
                TraceStage(
                    id="stage-8-citation",
                    name="Citation Validation",
                    status=TraceStageStatus.SKIPPED,
                    started_at=t_cit_iso,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    duration_ms=0.0,
                    details={
                        "reason": "Citation validation skipped because answer generation was bypassed or failed.",
                    },
                )
            )
        else:
            stages.append(
                TraceStage(
                    id="stage-8-citation",
                    name="Citation Validation",
                    status=TraceStageStatus.COMPLETED,
                    started_at=t_cit_iso,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    duration_ms=round((time.perf_counter() - t_cit) * 1000.0, 2),
                    details={
                        "valid_citations": citations.valid_citations,
                        "invalid_citations": citations.invalid_citations,
                        "uncited_sources": citations.uncited_sources,
                        "citation_count": len(citations.cited_sources),
                        "integrity_status": "warning" if citations.invalid_citations else "verified",
                    },
                )
            )

        t1 = time.perf_counter()
        total_latency_ms = round((t1 - t0) * 1000.0, 2)

        # Stage 9: Final Answer Returned
        stages.append(
            TraceStage(
                id="stage-9-final",
                name="Final Answer Returned",
                status=TraceStageStatus.COMPLETED,
                started_at=datetime.now(timezone.utc).isoformat(),
                completed_at=datetime.now(timezone.utc).isoformat(),
                duration_ms=round((t1 - t_cit) * 1000.0, 2),
                details={
                    "status": answer_result.status.value,
                    "answer_length": len(answer_result.answer),
                    "total_latency_ms": total_latency_ms,
                    "sources_count": len(rag_context_resp.sources),
                },
            )
        )

        execution_trace = ExecutionTrace(
            trace_id=trace_id,
            stages=stages,
            total_duration_ms=total_latency_ms,
        )

        return RAGAnswerResponse(
            query=payload.query,
            answer=answer_result.answer,
            status=answer_result.status,
            collection_id=str(collection_id),
            sources=rag_context_resp.sources,
            valid_citations=citations.valid_citations,
            invalid_citations=citations.invalid_citations,
            uncited_sources=citations.uncited_sources,
            provider=answer_result.provider,
            model=answer_result.model,
            usage=answer_result.usage,
            generation_latency_ms=answer_result.latency_ms,
            total_latency_ms=total_latency_ms,
            retrieval_metadata=rag_context_resp.retrieval_metadata,
            error_message=answer_result.error_message,
            execution_trace=execution_trace,
        )
