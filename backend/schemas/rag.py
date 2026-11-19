"""
backend/schemas/rag.py
─────────────────────────────────────────────────────────────
Pydantic v2 schemas for conventional Retrieval-Augmented Generation (RAG):
  - RAG query requests and internal query parameters.
  - Source references, excluded chunks, and structured context.
  - RAG responses containing assembled context, citations, and prompt.
─────────────────────────────────────────────────────────────
"""

import math
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.schemas.retrieval import MAX_QUERY_LENGTH, MAX_TOP_K

# Context character budget bounds
DEFAULT_CONTEXT_BUDGET: int = 12000
MIN_CONTEXT_BUDGET: int = 500
MAX_CONTEXT_BUDGET: int = 60000


class RAGQuery(BaseModel):
    """
    Internal query parameters for conventional RAG pipeline execution.
    """
    query: str = Field(
        ...,
        description="Natural language question to answer using retrieved context.",
    )
    collection_id: UUID = Field(
        ...,
        description="UUID of the target knowledge collection (authoritative).",
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
    user_role: Optional[str] = Field(
        default="student",
        description="User role: student, faculty, new_admission, administrator",
    )
    category: Optional[str] = Field(
        default=None,
        description="Optional category scope filter",
    )
    department: Optional[str] = Field(
        default=None,
        description="Optional department scope filter",
    )

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
        return v

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, v: int) -> int:
        if not isinstance(v, int) or isinstance(v, bool):
            raise ValueError("top_k must be an integer.")
        if v < 1:
            raise ValueError(f"top_k must be >= 1, got {v}.")
        if v > MAX_TOP_K:
            raise ValueError(f"top_k cannot exceed maximum limit of {MAX_TOP_K}, got {v}.")
        return v

    @field_validator("document_ids")
    @classmethod
    def validate_document_ids(cls, v: Optional[List[UUID]]) -> Optional[List[UUID]]:
        if v is not None:
            if len(v) != len(set(v)):
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
        if not isinstance(v, int) or isinstance(v, bool):
            raise ValueError("max_context_chars must be an integer.")
        if v < MIN_CONTEXT_BUDGET:
            raise ValueError(
                f"max_context_chars must be at least {MIN_CONTEXT_BUDGET} characters, got {v}."
            )
        if v > MAX_CONTEXT_BUDGET:
            raise ValueError(
                f"max_context_chars cannot exceed maximum limit of {MAX_CONTEXT_BUDGET} characters, got {v}."
            )
        return v


class RAGContextRequest(BaseModel):
    """
    Search and context assembly request parameters for the RAG API.
    collection_id is supplied via URL path (authoritative), but may optionally appear
    in the body. If present in the body, it must match the URL path parameter.
    """
    query: str = Field(
        ...,
        description="Natural language question to answer using retrieved context.",
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
    user_role: Optional[str] = Field(
        default="student",
        description="User role: student, faculty, new_admission, administrator",
    )
    category: Optional[str] = Field(
        default=None,
        description="Optional category scope filter",
    )
    department: Optional[str] = Field(
        default=None,
        description="Optional department scope filter",
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

    def to_rag_query(self, collection_id: UUID) -> RAGQuery:
        """Convert request payload to internal RAGQuery with authoritative collection_id."""
        return RAGQuery(
            query=self.query,
            collection_id=collection_id,
            top_k=self.top_k,
            document_ids=self.document_ids,
            similarity_threshold=self.similarity_threshold,
            max_context_chars=self.max_context_chars,
            user_role=self.user_role,
            category=self.category,
            department=self.department,
        )


class SourceReference(BaseModel):
    """
    Deterministic source citation metadata mapping a context identifier (e.g. [S1])
    to the underlying retrieved chunk and document.
    """
    source_id: str = Field(..., description="Deterministic citation identifier (e.g., [S1], [S2]).")
    document_id: str = Field(..., description="UUID string of source document.")
    chunk_id: str = Field(..., description="Unique chunk identifier.")
    filename: Optional[str] = Field(default=None, description="Original filename of source document.")
    page_number: Optional[int] = Field(default=None, description="1-based page number if available.")
    page_range: Optional[str] = Field(default=None, description="Page range span if available.")
    chunk_index: int = Field(..., ge=0, description="0-based sequence index within document.")
    similarity_score: float = Field(..., description="Computed cosine similarity in [-1.0, 1.0].")
    char_count: int = Field(..., ge=0, description="Character length of chunk text.")

    model_config = ConfigDict(from_attributes=True)


class ExcludedChunkInfo(BaseModel):
    """
    Information on a candidate chunk that was excluded during context assembly.
    """
    chunk_id: str = Field(..., description="Identifier of the excluded chunk.")
    document_id: str = Field(..., description="UUID string of source document.")
    reason: str = Field(
        ...,
        description="Reason for exclusion: budget_exceeded, duplicate_chunk_id, cross_collection_mismatch.",
    )
    char_count: int = Field(..., ge=0, description="Character length of the excluded chunk text.")

    model_config = ConfigDict(from_attributes=True)


class RAGResponse(BaseModel):
    """
    Structured response containing assembled source-grounded context,
    deterministic citations, and complete LLM prompt ready for Phase 9 generation.
    """
    query: str = Field(..., description="Original user query.")
    collection_id: str = Field(..., description="Target collection identifier.")
    retrieved_chunk_count: int = Field(..., ge=0, description="Count of chunks returned by semantic retrieval.")
    assembled_context: str = Field(..., description="Structured, formatted context text with source identifiers.")
    context_character_count: int = Field(..., ge=0, description="Total character length of assembled context.")
    sources: List[SourceReference] = Field(default_factory=list, description="Ordered source references mapped to [S1], [S2], etc.")
    source_metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary keyed by source_id.")
    prompt_text: str = Field(..., description="Fully assembled, grounded LLM prompt ready for generation.")
    excluded_chunks: List[ExcludedChunkInfo] = Field(default_factory=list, description="Chunks excluded due to budget or duplication.")
    retrieval_metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata from underlying semantic retrieval.")

    model_config = ConfigDict(from_attributes=True)
