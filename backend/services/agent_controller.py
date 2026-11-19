"""
backend/services/agent_controller.py
─────────────────────────────────────────────────────────────
AgentController: Deterministic plain-Python asynchronous state machine
governing the Agentic RAG lifecycle (Phase 10).

Constraints & Architectural Rules:
  - NO LangChain, NO LangGraph, NO third-party agent framework.
  - Plain Python state machine with explicit Pydantic v2 AgentState.
  - Deterministic state transition guards: terminal states (COMPLETED, FAILED)
    strictly reject further transitions.
  - Reuses existing project services (RAGService, AnswerGenerator,
    CitationValidator) via controlled dependency injection.
  - Emits structured, immutable AgentEvent objects for total observability.
  - Correctly captures asyncio.CancelledError and preserves cancellation semantics.
  - Safe zero-evidence refusal generation without LLM API calls.
─────────────────────────────────────────────────────────────
"""

import asyncio
from datetime import datetime, timezone
import logging
import time
from typing import Any, Dict, Optional, Set
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import Settings, get_settings
from backend.schemas.agent_state import (
    AgentEvent,
    AgentEventType,
    AgentRunRequest,
    AgentRunResponse,
    AgentState,
    AgentStatus,
)
from backend.schemas.query_plan import (
    ExecutionPlan,
    PlanStep,
    PlanStepType,
    QueryAnalysis,
    SubQuery,
)
from backend.schemas.rag import RAGQuery, RAGResponse, SourceReference
from backend.schemas.rag_answer import ExecutionTrace, TraceStage, TraceStageStatus
from backend.schemas.retrieval import RetrievalQuery, RetrievedChunk
from backend.schemas.tool import ToolCall, ToolResult, ToolStatus
from backend.services.answer_generator import (
    AnswerGenerationResult,
    AnswerGenerator,
    AnswerStatus,
    STANDARD_INSUFFICIENT_EVIDENCE_ANSWER,
)
from backend.services.answer_verifier import AnswerVerifier
from backend.services.citation_validator import CitationValidator
from backend.services.context_builder import ContextBuilder
from backend.services.evidence_evaluator import EvidenceEvaluator
from backend.services.prompt_builder import PromptBuilder
from backend.services.query_refiner import QueryRefiner
from backend.services.llm_provider import (
    LLMAuthenticationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMServiceUnavailableError,
    LLMTimeoutError,
)
from backend.services.query_planner import QueryPlannerService
from backend.services.query_understanding import QueryUnderstandingService
from backend.services.rag_service import (
    InvalidRAGQueryError,
    RAGExecutionError,
    RAGService,
)
from backend.services.tool_registry import (
    CalculatorTool,
    MetadataFilterTool,
    SemanticSearchTool,
    ToolRegistry,
)
from backend.services.vector_store import CollectionNotFoundError

logger = logging.getLogger(__name__)


# ── Exception Hierarchy ──────────────────────────────────────────────────────


class AgentControllerError(Exception):
    """Base exception for all AgentController errors."""
    pass


class InvalidStateTransitionError(AgentControllerError):
    """Raised when an illegal or unsupported state transition is attempted."""
    pass


class AgentExecutionError(AgentControllerError):
    """Raised when an internal agent execution step fails unrecoverably."""
    pass


class AgentMaxIterationsError(AgentControllerError):
    """Raised when bounded loop execution exceeds the configured max iterations."""
    pass


# ── Transition Matrix ────────────────────────────────────────────────────────

VALID_TRANSITIONS: Dict[AgentStatus, Set[AgentStatus]] = {
    AgentStatus.INITIALIZING: {AgentStatus.PLANNING, AgentStatus.FAILED},
    AgentStatus.PLANNING: {AgentStatus.RETRIEVING, AgentStatus.FAILED},
    AgentStatus.RETRIEVING: {AgentStatus.EVALUATING, AgentStatus.FAILED},
    AgentStatus.EVALUATING: {
        AgentStatus.SYNTHESIZING,
        AgentStatus.REFINING,
        AgentStatus.COMPLETED,
        AgentStatus.FAILED,
    },
    AgentStatus.REFINING: {AgentStatus.RETRIEVING, AgentStatus.FAILED},
    AgentStatus.SYNTHESIZING: {AgentStatus.COMPLETED, AgentStatus.FAILED},
    AgentStatus.COMPLETED: set(),  # Terminal state: strictly no outgoing transitions
    AgentStatus.FAILED: set(),     # Terminal state: strictly no outgoing transitions
}


# ── Agent Controller ─────────────────────────────────────────────────────────


class AgentController:
    """
    Plain-Python state machine orchestrating the Agentic RAG lifecycle.
    Manages deterministic state progression, tool invocation, evidence evaluation,
    and answer synthesis.
    """

    def __init__(
        self,
        rag_service: Optional[RAGService] = None,
        answer_generator: Optional[AnswerGenerator] = None,
        citation_validator: Optional[CitationValidator] = None,
        query_understanding: Optional[QueryUnderstandingService] = None,
        query_planner: Optional[QueryPlannerService] = None,
        tool_registry: Optional[ToolRegistry] = None,
        evidence_evaluator: Optional[EvidenceEvaluator] = None,
        query_refiner: Optional[QueryRefiner] = None,
        answer_verifier: Optional[AnswerVerifier] = None,
        settings: Optional[Settings] = None,
    ):
        """
        Initialize AgentController with dependency-injected services.
        """
        self.settings = settings or get_settings()
        self.rag_service = rag_service or RAGService(settings=self.settings)
        self.citation_validator = citation_validator or CitationValidator()
        self.context_builder = getattr(self.rag_service, "context_builder", None) or ContextBuilder()
        self.prompt_builder = getattr(self.rag_service, "prompt_builder", None) or PromptBuilder()
        self.query_understanding = query_understanding or QueryUnderstandingService()
        self.query_planner = query_planner or QueryPlannerService()
        self.evidence_evaluator = evidence_evaluator or EvidenceEvaluator()
        self.query_refiner = query_refiner or QueryRefiner()
        self.answer_verifier = answer_verifier or AnswerVerifier(citation_validator=self.citation_validator)

        if tool_registry is not None:
            self.tool_registry = tool_registry
        else:
            retrieval_svc = getattr(self.rag_service, "retrieval_service", None)
            self.tool_registry = ToolRegistry(
                tools=[
                    SemanticSearchTool(retrieval_service=retrieval_svc, rag_service=self.rag_service),
                    MetadataFilterTool(retrieval_service=retrieval_svc),
                    CalculatorTool(),
                ]
            )

        if answer_generator is not None:
            self.answer_generator = answer_generator
        else:
            try:
                from backend.routers.rag_answer import get_llm_provider
                provider = get_llm_provider(settings=self.settings)
                self.answer_generator = AnswerGenerator(
                    llm_provider=provider,
                    citation_validator=self.citation_validator,
                )
            except Exception:
                # Fallback for lightweight testing environments without provider setup
                self.answer_generator = None  # type: ignore

    def transition_to(
        self,
        state: AgentState,
        new_status: AgentStatus,
        event_type: AgentEventType,
        message: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> AgentEvent:
        """
        Execute an explicit, validated state transition and emit a frozen AgentEvent.

        Args:
            state: Active AgentState instance.
            new_status: Target AgentStatus.
            event_type: Event classification for audit log.
            message: Human-readable description of event.
            payload: Optional dictionary of event data.

        Returns:
            The created immutable AgentEvent.

        Raises:
            InvalidStateTransitionError: If the transition violates VALID_TRANSITIONS.
        """
        current_status = state.status

        # Check validity against transition matrix
        allowed_targets = VALID_TRANSITIONS.get(current_status, set())
        if new_status not in allowed_targets:
            err_msg = (
                f"Illegal state transition from {current_status.value} to {new_status.value}. "
                f"Allowed transitions: {[s.value for s in allowed_targets]}."
            )
            logger.error("State machine violation: %s", err_msg)
            raise InvalidStateTransitionError(err_msg)

        # Increment step counter and update status
        state.current_step += 1
        state.status = new_status
        state.updated_at = datetime.now(timezone.utc).isoformat()

        # Create immutable event record
        event = AgentEvent(
            event_id=str(uuid4()),
            event_type=event_type,
            timestamp=state.updated_at,
            step=state.current_step,
            payload=payload or {},
            message=message,
        )
        state.agent_events.append(event)
        logger.debug(
            "[AgentState %s] Step %d: %s -> %s (Event: %s)",
            state.trace_id[:8],
            state.current_step,
            current_status.value,
            new_status.value,
            event_type.value,
        )
        return event

    def _emit_event(
        self,
        state: AgentState,
        event_type: AgentEventType,
        message: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> AgentEvent:
        """
        Emit an event within the current state without triggering a status change.
        """
        state.current_step += 1
        state.updated_at = datetime.now(timezone.utc).isoformat()
        event = AgentEvent(
            event_id=str(uuid4()),
            event_type=event_type,
            timestamp=state.updated_at,
            step=state.current_step,
            payload=payload or {},
            message=message,
        )
        state.agent_events.append(event)
        return event

    async def run(
        self,
        request: AgentRunRequest,
        collection_id: UUID,
        db: Optional[AsyncSession] = None,
    ) -> AgentRunResponse:
        """
        Execute the Phase 10 Agentic RAG baseline pipeline.

        Workflow:
          1. Initialize AgentState and emit QUERY_UNDERSTOOD.
          2. Transition to PLANNING and emit PLAN_CREATED.
          3. Transition to RETRIEVING and invoke RetrievalService.
          4. Transition to EVALUATING and assess evidence sufficiency.
          5. If insufficient: emit EVIDENCE_INSUFFICIENT, synthesize refusal, COMPLETED.
          6. If sufficient: transition to SYNTHESIZING, generate answer, validate citations, COMPLETED.
          7. On failure: transition to FAILED, record error, emit AGENT_FAILED.

        Args:
            request: Validated AgentRunRequest.
            collection_id: Authoritative UUID of the target collection.
            db: Optional active async database session.

        Returns:
            AgentRunResponse with answer, status, events, and execution trace.

        Raises:
            asyncio.CancelledError: If the async task is cancelled externally.
            AgentControllerError: On internal unrecoverable failure.
        """
        start_time = time.perf_counter()
        raw_query = request.query.strip()
        # Bound maximum refinements to 2 (ensuring at most 3 total retrieval rounds: 1 initial + up to 2 refinements)
        configured_max_iter = getattr(self.settings, "agent_max_iterations", 5)
        max_iter = min(configured_max_iter, 2)

        # 1. Query Understanding & Analysis
        analysis = self.query_understanding.analyze(raw_query)

        # Initialize AgentState
        state = AgentState(
            session_id=str(uuid4()),
            trace_id=str(uuid4()),
            collection_id=str(collection_id),
            original_query=raw_query,
            normalized_query=analysis.normalized_query,
            user_role=getattr(request, "user_role", None) or "student",
            status=AgentStatus.INITIALIZING,
            max_iterations=max_iter,
            iteration_count=0,
            decomposed_queries=[sq.text for sq in analysis.subqueries],
            subqueries=analysis.subqueries,
            query_analysis=analysis,
        )

        self._emit_event(
            state=state,
            event_type=AgentEventType.QUERY_UNDERSTOOD,
            message=(
                f"Query understood: intent={analysis.primary_intent.value}"
                + (f" (college_intent={analysis.college_intent.value})" if analysis.college_intent else "")
                + f", role={state.user_role}, is_complex={analysis.is_complex}."
            ),
            payload={
                "original_query": raw_query,
                "normalized_query": analysis.normalized_query,
                "intent": analysis.primary_intent.value,
                "college_intent": analysis.college_intent.value if analysis.college_intent else None,
                "user_role": state.user_role,
                "target_categories": analysis.target_categories,
                "target_departments": analysis.target_departments,
                "is_complex": analysis.is_complex,
                "subquery_count": len(analysis.subqueries),
                "complexity_factors": analysis.complexity_factors,
                "collection_id": str(collection_id),
            },
        )

        # If complex query with multiple subqueries, emit QUERY_DECOMPOSED
        if analysis.is_complex and len(analysis.subqueries) > 1:
            self._emit_event(
                state=state,
                event_type=AgentEventType.QUERY_DECOMPOSED,
                message=f"Decomposed query into {len(analysis.subqueries)} independent information requirements.",
                payload={
                    "subqueries": [
                        {"id": sq.id, "text": sq.text, "intent": sq.intent.value, "order": sq.order}
                        for sq in analysis.subqueries
                    ],
                    "factors": analysis.complexity_factors,
                },
            )

        try:
            # 2. Enter PLANNING state
            plan = self.query_planner.create_plan(analysis)
            state.structured_plan = plan
            state.execution_plan = [f"{s.step_number}. {s.name}" for s in plan.steps]

            self.transition_to(
                state=state,
                new_status=AgentStatus.PLANNING,
                event_type=AgentEventType.PLAN_CREATED,
                message=plan.plan_summary,
                payload={
                    "plan_id": plan.plan_id,
                    "is_decomposed": plan.is_decomposed,
                    "subquery_count": plan.subquery_count,
                    "comparison_objective": plan.comparison_objective,
                    "steps": [
                        {
                            "step_number": s.step_number,
                            "type": s.step_type.value,
                            "name": s.name,
                            "subquery_id": s.subquery_id,
                        }
                        for s in plan.steps
                    ],
                },
            )

            # 3. Enter RETRIEVING state
            subqueries_to_retrieve = state.subqueries if state.subqueries else [
                SubQuery(
                    parent_query=state.original_query,
                    text=state.normalized_query,
                    intent=analysis.primary_intent,
                    order=0,
                )
            ]

            unique_chunks: Dict[str, RetrievedChunk] = {}
            total_raw_chunks = 0

            # Execute retrieval steps using ToolRegistry
            for idx, sq in enumerate(subqueries_to_retrieve, start=1):
                matching_step = next(
                    (s for s in (state.structured_plan.steps if state.structured_plan else []) if s.subquery_id == sq.id),
                    PlanStep(
                        step_number=idx,
                        step_type=PlanStepType.RETRIEVAL,
                        name=f"Subquery Retrieval {idx}",
                        subquery_id=sq.id,
                        description=f"Execute targeted retrieval for: '{sq.text}'",
                    ),
                )

                # Deterministic tool selection based on plan step and query requirements
                selected_tool_name = self.tool_registry.select_tool(
                    step=matching_step,
                    query_intent=sq.intent,
                    subquery_text=sq.text,
                    has_document_filter=bool(request.document_ids),
                )

                if selected_tool_name not in state.selected_tools:
                    state.selected_tools.append(selected_tool_name)

                # Transition or emit TOOL_SELECTED
                if state.status == AgentStatus.PLANNING:
                    self.transition_to(
                        state=state,
                        new_status=AgentStatus.RETRIEVING,
                        event_type=AgentEventType.TOOL_SELECTED,
                        message=f"Selected tool: {selected_tool_name}",
                        payload={
                            "tool": selected_tool_name,
                            "subquery_count": len(subqueries_to_retrieve),
                            "top_k": request.top_k,
                            "similarity_threshold": request.similarity_threshold,
                        },
                    )
                else:
                    self._emit_event(
                        state=state,
                        event_type=AgentEventType.TOOL_SELECTED,
                        message=f"Selected tool: {selected_tool_name} for subquery {idx}",
                        payload={"tool": selected_tool_name, "subquery": sq.text},
                    )

                # Prepare validated arguments based on tool type
                if selected_tool_name == "metadata_filter":
                    tool_args = {
                        "collection_id": collection_id,
                        "document_ids": request.document_ids or [],
                        "query": sq.text,
                        "top_k": request.top_k,
                        "similarity_threshold": request.similarity_threshold,
                    }
                elif selected_tool_name == "calculator":
                    tool_args = {"expression": sq.text}
                else:
                    tool_args = {
                        "query": sq.text,
                        "collection_id": collection_id,
                        "top_k": request.top_k,
                        "similarity_threshold": request.similarity_threshold,
                        "document_ids": request.document_ids,
                    }

                # Record tool call in audit state
                tool_call = ToolCall(
                    tool_name=selected_tool_name,
                    arguments=tool_args,
                    step_number=matching_step.step_number,
                )
                state.tool_calls.append(tool_call)

                # Execute through central ToolRegistry with validation, timeouts, and call limits
                tool_result = await self.tool_registry.execute_tool(
                    tool_name=selected_tool_name,
                    arguments=tool_args,
                    context={"db": db},
                    call_count=state.tool_call_count,
                    step_number=matching_step.step_number,
                )
                state.tool_results.append(tool_result)
                state.tool_call_count += 1

                if tool_result.status == ToolStatus.SUCCESS:
                    self._emit_event(
                        state=state,
                        event_type=AgentEventType.TOOL_EXECUTED,
                        message=f"Tool '{selected_tool_name}' executed successfully ({tool_result.latency_ms:.1f}ms).",
                        payload={
                            "tool": selected_tool_name,
                            "call_id": tool_result.call_id,
                            "latency_ms": tool_result.latency_ms,
                        },
                    )
                    # Accumulate retrieved chunks from output
                    if tool_result.output and hasattr(tool_result.output, "chunks"):
                        out_chunks = tool_result.output.chunks
                        total_raw_chunks += len(out_chunks)
                        for chunk in out_chunks:
                            if chunk.chunk_id not in unique_chunks:
                                unique_chunks[chunk.chunk_id] = chunk
                            else:
                                if chunk.similarity_score > unique_chunks[chunk.chunk_id].similarity_score:
                                    unique_chunks[chunk.chunk_id] = chunk
                else:
                    self._emit_event(
                        state=state,
                        event_type=AgentEventType.TOOL_FAILED,
                        message=f"Tool '{selected_tool_name}' failed with status {tool_result.status.value}: {tool_result.error_message}",
                        payload={
                            "tool": selected_tool_name,
                            "call_id": tool_result.call_id,
                            "status": tool_result.status.value,
                            "error": tool_result.error_message,
                        },
                    )

            # Sort deduplicated chunks by similarity descending
            deduplicated_chunks = list(unique_chunks.values())
            deduplicated_chunks.sort(key=lambda c: c.similarity_score, reverse=True)
            state.retrieved_chunks = deduplicated_chunks

            # Format combined context with deterministic [S1], [S2] source identifiers
            context_result = self.context_builder.build_context(
                chunks=deduplicated_chunks,
                max_context_chars=request.max_context_chars,
            )

            comp_obj = state.structured_plan.comparison_objective if state.structured_plan else None
            prompt_text = self.prompt_builder.build_prompt(
                query=state.normalized_query,
                assembled_context=context_result.assembled_context,
                sources=context_result.sources,
                comparison_objective=comp_obj,
            )

            state.sources = context_result.sources
            self._emit_event(
                state=state,
                event_type=AgentEventType.RETRIEVAL_COMPLETE,
                message=f"Retrieved {len(state.sources)} source candidate(s) from collection.",
                payload={
                    "retrieved_chunk_count": len(deduplicated_chunks),
                    "sources_count": len(state.sources),
                    "total_tool_calls": state.tool_call_count,
                },
            )

            # 4. Enter EVALUATING state
            assembled_context = context_result.assembled_context
            eval_report = self.evidence_evaluator.evaluate(
                query=state.normalized_query,
                chunks=deduplicated_chunks,
                subqueries=state.subqueries if state.subqueries else None,
                assembled_context=assembled_context,
                similarity_threshold=request.similarity_threshold,
            )
            state.evidence_evaluation = eval_report
            state.is_sufficient = eval_report.is_sufficient
            state.evidence_score = eval_report.overall_score

            self.transition_to(
                state=state,
                new_status=AgentStatus.EVALUATING,
                event_type=AgentEventType.EVIDENCE_EVALUATED if eval_report.is_sufficient else AgentEventType.EVIDENCE_INSUFFICIENT,
                message=(
                    f"Evidence evaluated: {'sufficient' if eval_report.is_sufficient else 'insufficient'} "
                    f"(score={eval_report.overall_score:.4f}, reason={eval_report.reason.value}, "
                    f"qualifying_chunks={eval_report.qualifying_chunk_count})."
                ),
                payload={
                    "is_sufficient": eval_report.is_sufficient,
                    "evidence_score": eval_report.overall_score,
                    "reason": eval_report.reason.value,
                    "explanation": eval_report.explanation,
                    "source_count": len(state.sources),
                    "coverage_ratio": eval_report.coverage_ratio,
                    "qualifying_chunk_count": eval_report.qualifying_chunk_count,
                },
            )

            # 5. Adaptive Refinement Loop (Phase 14)
            while not state.is_sufficient and self.query_refiner.can_refine(
                state.iteration_count, eval_report, max_iterations=state.max_iterations
            ):
                refined_query = self.query_refiner.refine(
                    current_iteration=state.iteration_count,
                    original_query=state.normalized_query,
                    evaluation=eval_report,
                    subqueries=state.subqueries if state.subqueries else None,
                    current_top_k=request.top_k,
                    current_similarity_threshold=request.similarity_threshold,
                )
                state.iteration_count += 1
                state.refinement_history.append(refined_query)

                self.transition_to(
                    state=state,
                    new_status=AgentStatus.REFINING,
                    event_type=AgentEventType.QUERY_REFINED,
                    message=f"Query refined (attempt {refined_query.iteration}): {refined_query.strategy.value}",
                    payload={
                        "iteration": refined_query.iteration,
                        "strategy": refined_query.strategy.value,
                        "refined_query_text": refined_query.refined_query_text,
                        "rationale": refined_query.rationale,
                        "adjusted_top_k": refined_query.adjusted_top_k,
                        "adjusted_similarity_threshold": refined_query.adjusted_similarity_threshold,
                    },
                )

                # Transition REFINING -> RETRIEVING for adaptive re-retrieval
                self.transition_to(
                    state=state,
                    new_status=AgentStatus.RETRIEVING,
                    event_type=AgentEventType.TOOL_SELECTED,
                    message=f"Executing adaptive re-retrieval with strategy {refined_query.strategy.value}.",
                    payload={
                        "tool": "semantic_search",
                        "query": refined_query.refined_query_text,
                        "top_k": refined_query.adjusted_top_k or request.top_k,
                    },
                )

                re_retrieve_args = {
                    "query": refined_query.refined_query_text,
                    "collection_id": collection_id,
                    "top_k": refined_query.adjusted_top_k or request.top_k,
                    "similarity_threshold": (
                        refined_query.adjusted_similarity_threshold
                        if refined_query.adjusted_similarity_threshold is not None
                        else request.similarity_threshold
                    ),
                    "document_ids": request.document_ids,
                }

                re_tool_call = ToolCall(
                    tool_name="semantic_search",
                    arguments=re_retrieve_args,
                    step_number=state.current_step,
                )
                state.tool_calls.append(re_tool_call)

                re_tool_result = await self.tool_registry.execute_tool(
                    tool_name="semantic_search",
                    arguments=re_retrieve_args,
                    context={"db": db},
                    call_count=state.tool_call_count,
                    step_number=state.current_step,
                )
                state.tool_results.append(re_tool_result)
                state.tool_call_count += 1

                if re_tool_result.status == ToolStatus.SUCCESS and re_tool_result.output and hasattr(re_tool_result.output, "chunks"):
                    for chunk in re_tool_result.output.chunks:
                        if chunk.chunk_id not in unique_chunks:
                            unique_chunks[chunk.chunk_id] = chunk
                        else:
                            if chunk.similarity_score > unique_chunks[chunk.chunk_id].similarity_score:
                                unique_chunks[chunk.chunk_id] = chunk

                # Rebuild deduplicated chunks & context
                deduplicated_chunks = list(unique_chunks.values())
                deduplicated_chunks.sort(key=lambda c: c.similarity_score, reverse=True)
                state.retrieved_chunks = deduplicated_chunks

                context_result = self.context_builder.build_context(
                    chunks=deduplicated_chunks,
                    max_context_chars=request.max_context_chars,
                )
                state.sources = context_result.sources
                assembled_context = context_result.assembled_context

                # Re-evaluate evidence
                eval_report = self.evidence_evaluator.evaluate(
                    query=state.normalized_query,
                    chunks=deduplicated_chunks,
                    subqueries=state.subqueries if state.subqueries else None,
                    assembled_context=assembled_context,
                    similarity_threshold=refined_query.adjusted_similarity_threshold or request.similarity_threshold,
                )
                state.evidence_evaluation = eval_report
                state.is_sufficient = eval_report.is_sufficient
                state.evidence_score = eval_report.overall_score

                self.transition_to(
                    state=state,
                    new_status=AgentStatus.EVALUATING,
                    event_type=AgentEventType.EVIDENCE_EVALUATED if eval_report.is_sufficient else AgentEventType.EVIDENCE_INSUFFICIENT,
                    message=(
                        f"Evidence re-evaluated: {'sufficient' if eval_report.is_sufficient else 'insufficient'} "
                        f"(score={eval_report.overall_score:.4f}, reason={eval_report.reason.value}, "
                        f"qualifying_chunks={eval_report.qualifying_chunk_count})."
                    ),
                    payload={
                        "is_sufficient": eval_report.is_sufficient,
                        "evidence_score": eval_report.overall_score,
                        "reason": eval_report.reason.value,
                        "iteration": state.iteration_count,
                        "qualifying_chunk_count": eval_report.qualifying_chunk_count,
                    },
                )

            # 6. Safe Refusal if Evidence Remains Insufficient After Refinements
            if not state.is_sufficient:
                state.answer = STANDARD_INSUFFICIENT_EVIDENCE_ANSWER
                self.transition_to(
                    state=state,
                    new_status=AgentStatus.COMPLETED,
                    event_type=AgentEventType.AGENT_DONE,
                    message="Execution completed with safe rule-based refusal (zero LLM inference).",
                    payload={
                        "outcome": "insufficient_evidence_refusal",
                        "reason": eval_report.reason.value,
                        "refinements_attempted": state.iteration_count,
                    },
                )
                total_duration = round((time.perf_counter() - start_time) * 1000.0, 2)
                return self._build_response(state=state, total_duration_ms=total_duration)

            # 7. Enter SYNTHESIZING state
            comp_obj = state.structured_plan.comparison_objective if state.structured_plan else None
            synth_message = (
                f"Initiating grounded comparative synthesis for: '{comp_obj}'."
                if comp_obj
                else "Initiating grounded answer generation from context."
            )
            self.transition_to(
                state=state,
                new_status=AgentStatus.SYNTHESIZING,
                event_type=AgentEventType.ANSWER_SYNTHESIZED,
                message=synth_message,
                payload={
                    "character_budget": request.max_context_chars,
                    "comparison_objective": comp_obj,
                },
            )

            # Rebuild prompt with the latest refined context, sources, and comparison objective
            comp_obj = state.structured_plan.comparison_objective if state.structured_plan else None
            prompt_text = self.prompt_builder.build_prompt(
                query=state.normalized_query,
                assembled_context=assembled_context,
                sources=state.sources,
                comparison_objective=comp_obj,
            )

            answer_result = await self.answer_generator.generate_answer(
                prompt=prompt_text,
                sources=state.sources,
                has_evidence=True,
                max_tokens=request.max_output_tokens,
                temperature=request.temperature,
            )

            # Check for generation errors
            if answer_result.status in (
                AnswerStatus.PROVIDER_ERROR,
                AnswerStatus.TIMEOUT,
                AnswerStatus.RATE_LIMITED,
            ):
                err_msg = answer_result.error_message or f"Generation failed with status {answer_result.status.value}"
                state.error_details = err_msg
                self.transition_to(
                    state=state,
                    new_status=AgentStatus.FAILED,
                    event_type=AgentEventType.AGENT_FAILED,
                    message=f"Answer generation error: {err_msg}",
                    payload={"status": answer_result.status.value, "error": err_msg},
                )
                total_duration = round((time.perf_counter() - start_time) * 1000.0, 2)
                return self._build_response(state=state, total_duration_ms=total_duration)

            # Record answer and verify semantic grounding (Phase 15)
            state.answer = answer_result.answer

            verification = self.answer_verifier.verify(
                answer_text=answer_result.answer,
                sources=state.sources,
                chunks=state.retrieved_chunks,
                assembled_context=assembled_context,
            )
            state.verification = verification
            state.citations = verification.valid_citations
            state.invalid_citations = verification.invalid_citations

            self._emit_event(
                state=state,
                event_type=AgentEventType.ANSWER_VERIFIED,
                message=(
                    f"Answer verified: {'grounded' if verification.is_verified else 'warning'} "
                    f"({verification.grounded_sentences}/{verification.total_sentences} sentences grounded, "
                    f"score={verification.grounding_score:.2f})."
                ),
                payload={
                    "is_verified": verification.is_verified,
                    "grounding_score": verification.grounding_score,
                    "valid_citations": verification.valid_citations,
                    "invalid_citations": verification.invalid_citations,
                    "hallucination_warning": verification.hallucination_warning,
                    "grounding_explanation": verification.grounding_explanation,
                },
            )
            self._emit_event(
                state=state,
                event_type=AgentEventType.CITATIONS_GENERATED,
                message=f"Deterministic citations linked to {len(state.sources)} source reference(s).",
                payload={
                    "citation_count": len(verification.valid_citations),
                    "valid_citations": verification.valid_citations,
                },
            )

            # 7. Complete Execution
            self.transition_to(
                state=state,
                new_status=AgentStatus.COMPLETED,
                event_type=AgentEventType.AGENT_DONE,
                message="Agent execution successfully completed.",
                payload={"outcome": "answered", "valid_citations": state.citations},
            )

            total_duration = round((time.perf_counter() - start_time) * 1000.0, 2)
            return self._build_response(state=state, total_duration_ms=total_duration)

        except asyncio.CancelledError:
            # Preserve cancellation semantics: record state but re-raise CancelledError
            logger.warning("AgentController task cancelled for trace %s", state.trace_id)
            state.status = AgentStatus.FAILED
            state.error_details = "Execution cancelled."
            state.agent_events.append(
                AgentEvent(
                    event_id=str(uuid4()),
                    event_type=AgentEventType.AGENT_FAILED,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    step=state.current_step + 1,
                    payload={"cancelled": True},
                    message="Agent execution was cancelled.",
                )
            )
            raise

        except CollectionNotFoundError as c_err:
            logger.warning("Collection '%s' not found during agent execution: %s", collection_id, c_err)
            state.status = AgentStatus.FAILED
            state.error_details = str(c_err)
            state.agent_events.append(
                AgentEvent(
                    event_id=str(uuid4()),
                    event_type=AgentEventType.AGENT_FAILED,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    step=state.current_step + 1,
                    payload={"error_type": "CollectionNotFoundError"},
                    message=str(c_err),
                )
            )
            raise

        except Exception as exc:
            logger.exception("Unexpected error during agent execution: %s", exc)
            state.status = AgentStatus.FAILED
            state.error_details = str(exc)
            state.agent_events.append(
                AgentEvent(
                    event_id=str(uuid4()),
                    event_type=AgentEventType.AGENT_FAILED,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    step=state.current_step + 1,
                    payload={"error_type": type(exc).__name__, "error": str(exc)},
                    message=f"Agent failed with error: {exc}",
                )
            )
            total_duration = round((time.perf_counter() - start_time) * 1000.0, 2)
            return self._build_response(state=state, total_duration_ms=total_duration)

    def _build_response(self, state: AgentState, total_duration_ms: float) -> AgentRunResponse:
        """
        Construct a structured AgentRunResponse from the terminal AgentState,
        including a backward-compatible ExecutionTrace for frontend visualization.
        """
        # Map agent events into TraceStages compatible with TraceDrawer
        stages = []
        for ev in state.agent_events:
            stage_status = TraceStageStatus.COMPLETED
            if ev.event_type == AgentEventType.AGENT_FAILED:
                stage_status = TraceStageStatus.FAILED

            stages.append(
                TraceStage(
                    id=f"step-{ev.step}-{ev.event_type.value.lower()}",
                    name=ev.event_type.value.replace("_", " ").title(),
                    status=stage_status,
                    started_at=ev.timestamp,
                    completed_at=ev.timestamp,
                    duration_ms=None,
                    details={**ev.payload, "event_message": ev.message},
                    error_message=ev.message if stage_status == TraceStageStatus.FAILED else None,
                )
            )

        execution_trace = ExecutionTrace(
            trace_id=state.trace_id,
            stages=stages,
            total_duration_ms=total_duration_ms,
        )

        return AgentRunResponse(
            session_id=state.session_id,
            trace_id=state.trace_id,
            collection_id=state.collection_id,
            query=state.original_query,
            user_role=state.user_role,
            college_intent=(
                state.query_analysis.college_intent.value
                if state.query_analysis and state.query_analysis.college_intent
                else None
            ),
            status=state.status,
            answer=state.answer or (f"Execution failed: {state.error_details}" if state.error_details else ""),
            is_sufficient=state.is_sufficient,
            evidence_score=state.evidence_score,
            evidence_evaluation=state.evidence_evaluation,
            iteration_count=state.iteration_count,
            refinement_history=state.refinement_history,
            sources=state.sources,
            valid_citations=state.citations,
            invalid_citations=state.invalid_citations,
            verification=state.verification,
            decomposed_queries=state.decomposed_queries,
            structured_plan=state.structured_plan,
            comparison_objective=state.structured_plan.comparison_objective if state.structured_plan else None,
            tool_calls=state.tool_calls,
            tool_results=state.tool_results,
            events=state.agent_events,
            execution_trace=execution_trace,
            total_latency_ms=total_duration_ms,
            error_details=state.error_details,
        )
