"""
backend/schemas/rag_answer.py
─────────────────────────────────────────────────────────────
Pydantic v2 schemas for conventional RAG answer generation requests
and structured responses containing generated answers, citation
validation, token consumption, and retrieval metadata.
─────────────────────────────────────────────────────────────
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.schemas.rag import (
    DEFAULT_CONTEXT_BUDGET,
    MAX_CONTEXT_BUDGET,
    MIN_CONTEXT_BUDGET,
    RAGQuery,
    SourceReference,
)
from backend.services.answer_generator import AnswerStatus
from backend.services.llm_provider import TokenUsage


class TraceStageStatus(str, Enum):
    """Execution status of an individual RAG pipeline activity stage."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TraceStage(BaseModel):
    """Structured record of an individual stage in the RAG execution lifecycle."""
    id: str = Field(..., description="Unique machine identifier for the stage.")
    name: str = Field(..., description="Human-readable title of the stage.")
    status: TraceStageStatus = Field(..., description="Status: pending, running, completed, failed, skipped.")
    started_at: Optional[str] = Field(default=None, description="ISO timestamp when stage execution started.")
    completed_at: Optional[str] = Field(default=None, description="ISO timestamp when stage execution completed.")
    duration_ms: Optional[float] = Field(default=None, description="Execution duration of the stage in milliseconds.")
    details: Dict[str, Any] = Field(default_factory=dict, description="Observable metadata, parameters, and decision summary.")
    error_message: Optional[str] = Field(default=None, description="Error diagnostics if stage execution failed.")

    model_config = ConfigDict(from_attributes=True)


class ExecutionTrace(BaseModel):
    """Complete structured execution trace across all 9 RAG pipeline stages."""
    trace_id: str = Field(..., description="Unique UUID tracking this execution trace.")
    stages: List[TraceStage] = Field(default_factory=list, description="Ordered list of execution stages.")
    total_duration_ms: float = Field(default=0.0, ge=0.0, description="End-to-end trace latency in milliseconds.")

    model_config = ConfigDict(from_attributes=True)


class RAGAnswerRequest(BaseModel):
    """
    Request parameters for generating a grounded RAG answer from a knowledge collection.
    collection_id from the URL path remains authoritative.
    """
    query: str = Field(
        ...,
        description="Natural language question to answer using retrieved collection context.",
    )
    top_k: int = Field(
        default=5,
        description="Maximum number of candidate chunks to retrieve.",
    )
    document_ids: Optional[List[UUID]] = Field(
        default=None,
        description="Optional list of document UUIDs to restrict retrieval scope.",
    )
    similarity_threshold: Optional[float] = Field(
        default=None,
        description="Minimum cosine similarity score in [-1.0, 1.0]. Candidates below this threshold are omitted.",
    )
    max_context_chars: int = Field(
        default=DEFAULT_CONTEXT_BUDGET,
        description="Maximum character budget for the assembled context text.",
    )
    max_output_tokens: Optional[int] = Field(
        default=None,
        description="Maximum output tokens for LLM answer generation.",
    )
    temperature: Optional[float] = Field(
        default=None,
        description="Sampling temperature for answer generation in [0.0, 1.0].",
    )
    collection_id: Optional[UUID] = Field(
        default=None,
        description="Optional collection UUID in request body. If provided, must match the URL path collection_id.",
    )

    model_config = ConfigDict(from_attributes=True)

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        return RAGQuery.validate_query(v)

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, v: int) -> int:
        return RAGQuery.validate_top_k(v)

    @field_validator("document_ids")
    @classmethod
    def validate_document_ids(cls, v: Optional[List[UUID]]) -> Optional[List[UUID]]:
        return RAGQuery.validate_document_ids(v)

    @field_validator("similarity_threshold")
    @classmethod
    def validate_similarity_threshold(cls, v: Optional[float]) -> Optional[float]:
        return RAGQuery.validate_similarity_threshold(v)

    @field_validator("max_context_chars")
    @classmethod
    def validate_max_context_chars(cls, v: int) -> int:
        return RAGQuery.validate_max_context_chars(v)

    @field_validator("max_output_tokens")
    @classmethod
    def validate_max_output_tokens(cls, v: Optional[int]) -> Optional[int]:
        if v is not None:
            if not isinstance(v, int) or isinstance(v, bool):
                raise ValueError("max_output_tokens must be an integer.")
            if v < 1:
                raise ValueError("max_output_tokens must be >= 1.")
            if v > 8192:
                raise ValueError("max_output_tokens cannot exceed 8192.")
        return v

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: Optional[float]) -> Optional[float]:
        if v is not None:
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise ValueError("temperature must be a numeric float.")
            if v < 0.0 or v > 1.0:
                raise ValueError("temperature must be within range [0.0, 1.0].")
            return float(v)
        return v


class RAGAnswerResponse(BaseModel):
    """
    Structured response containing grounded answer, citation validation,
    source metadata, and generation diagnostics.
    """
    query: str = Field(..., description="Original user query.")
    answer: str = Field(..., description="Grounded answer text generated by Claude or refusal message.")
    status: AnswerStatus = Field(..., description="Generation status: answered, insufficient_evidence, provider_error, timeout, rate_limited.")
    collection_id: str = Field(..., description="Target knowledge collection UUID string.")
    sources: List[SourceReference] = Field(default_factory=list, description="Retrieved sources available in context.")
    valid_citations: List[str] = Field(default_factory=list, description="Citation IDs cited in text and verified in sources.")
    invalid_citations: List[str] = Field(default_factory=list, description="Citation IDs cited in text but not present in sources.")
    uncited_sources: List[str] = Field(default_factory=list, description="Sources present in context that were not cited in answer.")
    provider: str = Field(..., description="LLM provider name.")
    model: str = Field(..., description="Model identifier used for generation.")
    usage: Optional[TokenUsage] = Field(default=None, description="Prompt, completion, and total token usage.")
    generation_latency_ms: float = Field(default=0.0, ge=0.0, description="Latency of LLM completion in milliseconds.")
    total_latency_ms: float = Field(default=0.0, ge=0.0, description="Total pipeline latency (retrieval + prompt + LLM) in milliseconds.")
    retrieval_metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata from underlying semantic retrieval.")
    error_message: Optional[str] = Field(default=None, description="Diagnostic error message if generation encountered an error.")
    limitations_note: str = Field(
        default="Citation validation verifies reference existence, not semantic entailment.",
        description="Disclaimer on citation validation scope.",
    )
    execution_trace: Optional[ExecutionTrace] = Field(
        default=None,
        description="Detailed activity trace capturing timing, observables, and decisions across all 9 pipeline stages.",
    )

    model_config = ConfigDict(from_attributes=True)
