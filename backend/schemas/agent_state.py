"""
backend/schemas/agent_state.py
─────────────────────────────────────────────────────────────
Pydantic v2 schemas for Agentic RAG state machine (Phase 10):
  - AgentStatus and AgentEventType lifecycle enums.
  - AgentEvent: Immutable structured transition and action record.
  - AgentState: Complete state container for plain-Python state machine.
  - AgentRunRequest: Client request payload for agent execution.
  - AgentRunResponse: Observable client response containing answer,
    citations, events, and backward-compatible ExecutionTrace.
─────────────────────────────────────────────────────────────
"""

from datetime import datetime, timezone
from enum import Enum
import math
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.schemas.evidence import EvidenceEvaluation
from backend.schemas.query_plan import (
    ExecutionPlan,
    QueryAnalysis,
    QueryIntent,
    SubQuery,
)
from backend.schemas.query_refinement import RefinedQuery
from backend.schemas.synthesis import SynthesisVerification
from backend.schemas.rag import (
    DEFAULT_CONTEXT_BUDGET,
    MAX_CONTEXT_BUDGET,
    MIN_CONTEXT_BUDGET,
    SourceReference,
)
from backend.schemas.rag_answer import ExecutionTrace, TraceStage, TraceStageStatus
from backend.schemas.retrieval import MAX_QUERY_LENGTH, MAX_TOP_K, RetrievedChunk
from backend.schemas.tool import ToolCall, ToolResult


class AgentStatus(str, Enum):
    """Lifecycle status of the agent state machine."""
    INITIALIZING = "INITIALIZING"
    PLANNING = "PLANNING"
    RETRIEVING = "RETRIEVING"
    EVALUATING = "EVALUATING"
    REFINING = "REFINING"
    SYNTHESIZING = "SYNTHESIZING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class AgentEventType(str, Enum):
    """Granular observable event types emitted during state transitions."""
    QUERY_UNDERSTOOD = "QUERY_UNDERSTOOD"
    PLAN_CREATED = "PLAN_CREATED"
    QUERY_DECOMPOSED = "QUERY_DECOMPOSED"
    TOOL_SELECTED = "TOOL_SELECTED"
    TOOL_EXECUTED = "TOOL_EXECUTED"
    TOOL_FAILED = "TOOL_FAILED"
    RETRIEVAL_COMPLETE = "RETRIEVAL_COMPLETE"
    EVIDENCE_EVALUATED = "EVIDENCE_EVALUATED"
    EVIDENCE_INSUFFICIENT = "EVIDENCE_INSUFFICIENT"
    QUERY_REFINED = "QUERY_REFINED"
    ANSWER_SYNTHESIZED = "ANSWER_SYNTHESIZED"
    ANSWER_VERIFIED = "ANSWER_VERIFIED"
    CITATIONS_GENERATED = "CITATIONS_GENERATED"
    AGENT_DONE = "AGENT_DONE"
    AGENT_FAILED = "AGENT_FAILED"


class AgentEvent(BaseModel):
    """
    Immutable structured record of an agent decision, tool execution, or state change.
    Payloads are frozen to prevent accidental mutation.
    """
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: AgentEventType
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    step: int = Field(default=0, ge=0)
    payload: Dict[str, Any] = Field(default_factory=dict)
    message: str

    model_config = ConfigDict(frozen=True)


class AgentState(BaseModel):
    """
    Authoritative state model for the plain-Python Agentic RAG state machine.
    Maintains full lifecycle context, evidence, and event history across Phases 10–15.
    """
    session_id: str = Field(default_factory=lambda: str(uuid4()), description="Client session UUID.")
    trace_id: str = Field(default_factory=lambda: str(uuid4()), description="Trace identifier.")
    collection_id: str = Field(..., description="Target collection UUID string.")
    original_query: str = Field(..., description="Raw natural language question.")
    normalized_query: str = Field(..., description="Cleaned, whitespace-normalized query.")
    user_role: str = Field(default="student", description="Active user role: student, faculty, new_admission, administrator.")

    status: AgentStatus = Field(default=AgentStatus.INITIALIZING, description="Current machine status.")
    current_step: int = Field(default=0, ge=0, description="Step sequence counter.")
    iteration_count: int = Field(default=0, ge=0, description="Refinement iteration loop counter.")
    max_iterations: int = Field(default=5, ge=1, le=10, description="Hard upper bound on execution loops.")

    execution_plan: List[str] = Field(default_factory=list, description="Ordered plan steps.")
    decomposed_queries: List[str] = Field(default_factory=list, description="Text list of derived subqueries.")
    subqueries: List[SubQuery] = Field(default_factory=list, description="Derived SubQuery objects.")
    query_analysis: Optional[QueryAnalysis] = Field(default=None, description="Detailed query analysis.")
    structured_plan: Optional[ExecutionPlan] = Field(default=None, description="Structured execution plan.")
    selected_tools: List[str] = Field(default_factory=list, description="Names of tools invoked.")
    tool_calls: List[ToolCall] = Field(default_factory=list, description="Ordered history of tool invocations.")
    tool_results: List[ToolResult] = Field(default_factory=list, description="Ordered history of tool execution outcomes.")
    tool_call_count: int = Field(default=0, ge=0, description="Total count of tool calls executed.")
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list, description="Chunks retrieved from vector store.")
    sources: List[SourceReference] = Field(default_factory=list, description="Structured citation sources.")

    evidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Assessed sufficiency score.")
    is_sufficient: bool = Field(default=False, description="Sufficiency gate decision.")
    evidence_evaluation: Optional[EvidenceEvaluation] = Field(default=None, description="Detailed Phase 13 evidence evaluation report.")
    refinement_history: List[RefinedQuery] = Field(default_factory=list, description="Ordered history of adaptive query refinements.")

    answer: Optional[str] = Field(default=None, description="Synthesized answer text or refusal.")
    citations: List[str] = Field(default_factory=list, description="Valid citation identifiers (e.g. [S1]).")
    invalid_citations: List[str] = Field(default_factory=list, description="Ungrounded citation identifiers.")
    verification: Optional[SynthesisVerification] = Field(default=None, description="Detailed Phase 15 answer verification and grounding report.")

    agent_events: List[AgentEvent] = Field(default_factory=list, description="Ordered audit trail of events.")
    error_details: Optional[str] = Field(default=None, description="Diagnostic error details if failed.")

    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    model_config = ConfigDict(from_attributes=True)


class AgentRunRequest(BaseModel):
    """
    Request parameters for invoking the Agentic RAG state machine.
    """
    query: str = Field(..., description="Natural language question.")
    top_k: int = Field(default=5, ge=1, le=MAX_TOP_K, description="Candidate retrieval count.")
    document_ids: Optional[List[UUID]] = Field(default=None, description="Optional document filter.")
    similarity_threshold: Optional[float] = Field(default=None, description="Cosine similarity threshold in [-1.0, 1.0].")
    max_context_chars: int = Field(default=DEFAULT_CONTEXT_BUDGET, description="Character budget for context.")
    max_output_tokens: Optional[int] = Field(default=None, description="Maximum completion tokens.")
    temperature: Optional[float] = Field(default=None, description="Sampling temperature.")
    collection_id: Optional[UUID] = Field(default=None, description="Optional body collection UUID.")
    user_role: Optional[str] = Field(default="student", description="Role of the user (student, faculty, new_admission, administrator)")
    category: Optional[str] = Field(default=None, description="Optional target college category")
    department: Optional[str] = Field(default=None, description="Optional target department")

    model_config = ConfigDict(from_attributes=True)

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("query must be a string.")
        if not v.strip():
            raise ValueError("query cannot be empty or whitespace only.")
        if len(v) > MAX_QUERY_LENGTH:
            raise ValueError(
                f"query length ({len(v)}) exceeds maximum allowed limit of {MAX_QUERY_LENGTH} characters."
            )
        return v.strip()

    @field_validator("document_ids")
    @classmethod
    def validate_document_ids(cls, v: Optional[List[UUID]]) -> Optional[List[UUID]]:
        if v is not None and len(v) != len(set(v)):
            raise ValueError("Duplicate document IDs are not permitted in document_ids filter.")
        return v

    @field_validator("similarity_threshold")
    @classmethod
    def validate_similarity_threshold(cls, v: Optional[float]) -> Optional[float]:
        if v is not None:
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise ValueError("similarity_threshold must be a numeric float.")
            if math.isnan(v) or math.isinf(v):
                raise ValueError("similarity_threshold must be a finite number (not NaN or Inf).")
            if v < -1.0 or v > 1.0:
                raise ValueError(
                    f"similarity_threshold must be within cosine similarity range [-1.0, 1.0], got {v}."
                )
            return float(v)
        return v

    @field_validator("max_context_chars")
    @classmethod
    def validate_max_context_chars(cls, v: int) -> int:
        if v < MIN_CONTEXT_BUDGET or v > MAX_CONTEXT_BUDGET:
            raise ValueError(
                f"max_context_chars must be between {MIN_CONTEXT_BUDGET} and {MAX_CONTEXT_BUDGET}, got {v}."
            )
        return v


class AgentRunResponse(BaseModel):
    """
    Structured outcome of an Agentic RAG run.
    Contains the final answer, citation validation, observable events,
    and a backward-compatible ExecutionTrace for visual inspection.
    """
    session_id: str = Field(..., description="Unique session identifier.")
    trace_id: str = Field(..., description="Unique trace identifier.")
    collection_id: str = Field(..., description="Target collection UUID string.")
    query: str = Field(..., description="User query.")
    user_role: str = Field(default="student", description="Active user role.")
    college_intent: Optional[str] = Field(default=None, description="Classified college institutional intent.")
    status: AgentStatus = Field(..., description="Terminal agent status: COMPLETED or FAILED.")
    answer: str = Field(..., description="Final synthesized answer or refusal.")
    is_sufficient: bool = Field(..., description="True if evidence was sufficient for answering.")
    evidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Sufficiency score.")
    evidence_evaluation: Optional[EvidenceEvaluation] = Field(default=None, description="Detailed Phase 13 evidence evaluation report.")
    iteration_count: int = Field(default=0, ge=0, description="Total refinement iterations executed.")
    refinement_history: List[RefinedQuery] = Field(default_factory=list, description="Ordered history of adaptive query refinements.")
    sources: List[SourceReference] = Field(default_factory=list, description="Sources in context.")
    valid_citations: List[str] = Field(default_factory=list, description="Verified citation IDs.")
    invalid_citations: List[str] = Field(default_factory=list, description="Unverified citation IDs.")
    verification: Optional[SynthesisVerification] = Field(default=None, description="Detailed Phase 15 answer verification and grounding report.")
    decomposed_queries: List[str] = Field(default_factory=list, description="Derived subqueries if decomposed.")
    structured_plan: Optional[ExecutionPlan] = Field(default=None, description="Detailed execution plan.")
    comparison_objective: Optional[str] = Field(default=None, description="Explicit comparative objective if query is comparative.")
    tool_calls: List[ToolCall] = Field(default_factory=list, description="Tool calls executed during run.")
    tool_results: List[ToolResult] = Field(default_factory=list, description="Tool execution results produced during run.")
    events: List[AgentEvent] = Field(default_factory=list, description="Chronological event log.")
    execution_trace: Optional[ExecutionTrace] = Field(
        default=None,
        description="Structured activity trace compatible with existing Activity Trace visualizer.",
    )
    total_latency_ms: float = Field(default=0.0, ge=0.0, description="Total execution time in milliseconds.")
    error_details: Optional[str] = Field(default=None, description="Diagnostic error string if failed.")
    limitations_note: str = Field(
        default="Citation validation verifies reference existence, not semantic entailment.",
        description="Scope disclaimer on citation validity.",
    )

    model_config = ConfigDict(from_attributes=True)
