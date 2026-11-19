"""
backend/schemas/retrieval.py
─────────────────────────────────────────────────────────────
Pydantic v2 schemas for semantic retrieval queries, candidate
chunks, and retrieval responses.
─────────────────────────────────────────────────────────────
"""

import math
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


MAX_QUERY_LENGTH = 2048
MAX_TOP_K = 100


class RetrievalQuery(BaseModel):
    """
    Search query parameters for semantic retrieval from a knowledge collection.
    """
    query: str = Field(
        ...,
        description="Search query string.",
    )
    collection_id: UUID = Field(
        ...,
        description="UUID of the target knowledge collection.",
    )
    top_k: int = Field(
        default=5,
        description="Maximum number of candidate chunks to return.",
    )
    document_ids: Optional[List[UUID]] = Field(
        default=None,
        description="Optional list of document UUIDs to restrict retrieval scope.",
    )
    similarity_threshold: Optional[float] = Field(
        default=None,
        description="Minimum cosine similarity score in [-1.0, 1.0]. Candidates below this threshold are omitted.",
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


class RetrievalQueryRequest(BaseModel):
    """
    Search query parameters for semantic retrieval via REST API.
    collection_id is supplied via URL path, but may optionally appear in the body.
    If present in the body, it must match the URL path parameter.
    """
    query: str = Field(
        ...,
        description="Search query string.",
    )
    top_k: int = Field(
        default=5,
        description="Maximum number of candidate chunks to return.",
    )
    document_ids: Optional[List[UUID]] = Field(
        default=None,
        description="Optional list of document UUIDs to restrict retrieval scope.",
    )
    similarity_threshold: Optional[float] = Field(
        default=None,
        description="Minimum cosine similarity score in [-1.0, 1.0]. Candidates below this threshold are omitted.",
    )
    collection_id: Optional[UUID] = Field(
        default=None,
        description="Optional collection UUID in request body. If provided, must match the URL path collection_id.",
    )

    model_config = ConfigDict(from_attributes=True)

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        return RetrievalQuery.validate_query(v)

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, v: int) -> int:
        return RetrievalQuery.validate_top_k(v)

    @field_validator("document_ids")
    @classmethod
    def validate_document_ids(cls, v: Optional[List[UUID]]) -> Optional[List[UUID]]:
        return RetrievalQuery.validate_document_ids(v)

    @field_validator("similarity_threshold")
    @classmethod
    def validate_similarity_threshold(cls, v: Optional[float]) -> Optional[float]:
        return RetrievalQuery.validate_similarity_threshold(v)

    def to_retrieval_query(self, collection_id: UUID) -> "RetrievalQuery":
        """Convert request payload to internal RetrievalQuery with authoritative collection_id."""
        return RetrievalQuery(
            query=self.query,
            collection_id=collection_id,
            top_k=self.top_k,
            document_ids=self.document_ids,
            similarity_threshold=self.similarity_threshold,
        )


class RetrievedChunk(BaseModel):
    """
    Candidate text chunk retrieved from the vector store with similarity metric.
    """
    chunk_id: str = Field(..., description="Unique chunk identifier.")
    document_id: str = Field(..., description="UUID of source document.")
    collection_id: str = Field(..., description="UUID of parent collection.")
    text: str = Field(..., description="Cleaned text content of the chunk.")
    similarity_score: float = Field(..., description="Computed cosine similarity in [-1.0, 1.0].")
    distance: float = Field(..., description="Raw ChromaDB HNSW cosine distance.")
    chunk_index: int = Field(..., ge=0, description="0-based sequence index.")
    version: Optional[int] = Field(default=None, description="Source document version.")
    filename: Optional[str] = Field(default=None, description="Original filename.")
    file_type: Optional[str] = Field(default=None, description="Document format extension (pdf, docx, txt).")
    start_char: Optional[int] = Field(default=None, description="Character start offset.")
    end_char: Optional[int] = Field(default=None, description="Character end offset.")
    page_number: Optional[int] = Field(default=None, description="1-based PDF page number.")
    page_range: Optional[str] = Field(default=None, description="PDF page span string.")

    model_config = ConfigDict(from_attributes=True)

    @property
    def chunk_text(self) -> str:
        """Alias for compatibility with DocumentChunk schema."""
        return self.text


class RetrievalResponse(BaseModel):
    """
    Structured response containing vector search candidates.
    Note: These are raw vector search candidates. Full SQLite document and
    version validation is performed by RetrievalService in Phase 7.3.
    """
    query: str = Field(..., description="Original search query string.")
    collection_id: str = Field(..., description="Target collection identifier.")
    results: List[RetrievedChunk] = Field(default_factory=list, description="Candidate chunks passing threshold.")
    total_candidates: int = Field(..., ge=0, description="Total raw candidates returned by ChromaDB.")
    returned_results: int = Field(..., ge=0, description="Count of candidate chunks returned after threshold filtering.")
    rejected_candidates: int = Field(default=0, ge=0, description="Count of candidate chunks rejected during SQLite validation.")
    accepted_candidates: int = Field(default=0, ge=0, description="Count of candidate chunks verified and accepted by SQLite validation.")
    retrieval_metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata about the query and filters.")

    model_config = ConfigDict(from_attributes=True)
