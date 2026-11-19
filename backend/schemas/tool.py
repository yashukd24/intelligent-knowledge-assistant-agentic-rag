"""
backend/schemas/tool.py
─────────────────────────────────────────────────────────────
Pydantic v2 schemas for Phase 12 Tool Registry & Selection:
  - ToolStatus: Granular status enum for tool invocations.
  - ToolParameter: Structural metadata for tool input parameters.
  - ToolDefinition: Complete manifest of an approved tool.
  - ToolCall: Immutable audit record of a tool invocation.
  - ToolResult: Immutable outcome record of a tool execution.
  - SemanticSearchInput & SemanticSearchOutput: Schemas for semantic_search.
  - MetadataFilterInput & MetadataFilterOutput: Schemas for metadata_filter.
  - CalculatorInput & CalculatorOutput: Schemas for calculator.
─────────────────────────────────────────────────────────────
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from backend.schemas.retrieval import RetrievedChunk


class ToolStatus(str, Enum):
    """Execution status outcome for tool invocations."""
    SUCCESS = "SUCCESS"
    INVALID_ARGUMENTS = "INVALID_ARGUMENTS"
    EXECUTION_ERROR = "EXECUTION_ERROR"
    TIMEOUT = "TIMEOUT"
    TOOL_NOT_FOUND = "TOOL_NOT_FOUND"
    CALL_LIMIT_EXCEEDED = "CALL_LIMIT_EXCEEDED"


class ToolParameter(BaseModel):
    """Structural metadata for a single tool parameter."""
    name: str = Field(..., description="Parameter identifier.")
    type: str = Field(..., description="Data type representation.")
    description: str = Field(..., description="Semantic purpose of the parameter.")
    required: bool = Field(default=True, description="Whether the parameter is mandatory.")
    default: Optional[Any] = Field(default=None, description="Default value if optional.")

    model_config = ConfigDict(frozen=True)


class ToolDefinition(BaseModel):
    """Manifest describing an approved agent tool."""
    name: str = Field(..., description="Unique tool name.")
    description: str = Field(..., description="Clear human-readable description.")
    parameters: List[ToolParameter] = Field(default_factory=list, description="Input parameter specifications.")
    timeout_seconds: float = Field(default=10.0, ge=0.1, le=60.0, description="Per-execution timeout.")
    is_safe: bool = Field(default=True, description="Indicates tool is sandboxed and approved.")

    model_config = ConfigDict(frozen=True)


class ToolCall(BaseModel):
    """Immutable audit record of an initiated tool call."""
    call_id: str = Field(default_factory=lambda: f"call-{uuid4().hex[:8]}")
    tool_name: str = Field(..., description="Target tool name.")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Raw input arguments.")
    step_number: Optional[int] = Field(default=None, description="Linked execution plan step.")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    model_config = ConfigDict(frozen=True)


class ToolResult(BaseModel):
    """Immutable outcome record of a completed or failed tool invocation."""
    call_id: str = Field(..., description="Matching ToolCall identifier.")
    tool_name: str = Field(..., description="Name of the invoked tool.")
    status: ToolStatus = Field(..., description="Execution outcome classification.")
    output: Optional[Any] = Field(default=None, description="Validated output payload if successful.")
    error_message: Optional[str] = Field(default=None, description="Diagnostic error details if failed.")
    latency_ms: float = Field(default=0.0, ge=0.0, description="Tool execution duration in milliseconds.")
    completed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    model_config = ConfigDict(frozen=True)


# ── Tool-Specific Schemas ────────────────────────────────────────────────────


class SemanticSearchInput(BaseModel):
    """Input payload for semantic_search tool."""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query string.")
    collection_id: UUID = Field(..., description="Target collection UUID.")
    top_k: int = Field(default=5, ge=1, le=50, description="Maximum candidate chunks to retrieve.")
    similarity_threshold: Optional[float] = Field(
        default=None, ge=-1.0, le=1.0, description="Minimum cosine similarity score."
    )
    document_ids: Optional[List[UUID]] = Field(
        default=None, description="Optional document ID filter."
    )

    model_config = ConfigDict(extra="forbid")


class SemanticSearchOutput(BaseModel):
    """Output payload from semantic_search tool."""
    chunks: List[RetrievedChunk] = Field(default_factory=list, description="Validated retrieved chunks.")
    candidate_count: int = Field(default=0, ge=0, description="Number of returned candidate chunks.")
    max_similarity: float = Field(default=0.0, ge=-1.0, le=1.0, description="Highest similarity score.")
    query: str = Field(..., description="Query executed.")
    collection_id: str = Field(..., description="Collection UUID string.")

    model_config = ConfigDict(from_attributes=True)


class MetadataFilterInput(BaseModel):
    """Input payload for metadata_filter tool."""
    collection_id: UUID = Field(..., description="Target collection UUID.")
    document_ids: List[UUID] = Field(..., min_length=1, description="Required document IDs to filter by.")
    query: Optional[str] = Field(default=None, max_length=1000, description="Optional query to rank within documents.")
    top_k: int = Field(default=5, ge=1, le=50, description="Maximum candidate chunks to return.")
    similarity_threshold: Optional[float] = Field(
        default=None, ge=-1.0, le=1.0, description="Optional similarity threshold."
    )

    model_config = ConfigDict(extra="forbid")


class MetadataFilterOutput(BaseModel):
    """Output payload from metadata_filter tool."""
    chunks: List[RetrievedChunk] = Field(default_factory=list, description="Filtered chunks.")
    filtered_chunk_count: int = Field(default=0, ge=0, description="Count of chunks matching filter.")
    matched_document_count: int = Field(default=0, ge=0, description="Count of distinct matched documents.")
    collection_id: str = Field(..., description="Collection UUID string.")

    model_config = ConfigDict(from_attributes=True)


class CalculatorInput(BaseModel):
    """Input payload for safe sandboxed calculator tool."""
    expression: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Mathematical arithmetic expression using +, -, *, /, %, **.",
    )

    model_config = ConfigDict(extra="forbid")


class CalculatorOutput(BaseModel):
    """Output payload from safe sandboxed calculator tool."""
    result: float = Field(..., description="Computed numerical result.")
    expression: str = Field(..., description="Normalized evaluated expression.")
    rounded_result: float = Field(..., description="Result rounded to 4 decimal places.")

    model_config = ConfigDict(from_attributes=True)
