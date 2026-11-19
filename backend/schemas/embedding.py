"""
backend/schemas/embedding.py
─────────────────────────────────────────────────────────────
Pydantic v2 schema for chunks with dense vector embeddings.
─────────────────────────────────────────────────────────────
"""

import math
from typing import List, Sequence
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.schemas.chunk import ChunkMetadata, DocumentChunk


class EmbeddedChunk(BaseModel):
    """
    Represents a DocumentChunk paired with its dense vector embedding
    and model provenance metadata.
    """
    chunk_id: str = Field(..., description="Unique deterministic identifier (matches source DocumentChunk)")
    document_id: str = Field(..., description="UUID of source document")
    collection_id: str = Field(..., description="UUID of parent collection")
    chunk_text: str = Field(..., min_length=1, description="Cleaned text content of the chunk")
    metadata: ChunkMetadata = Field(..., description="Granular chunk metadata")
    embedding: List[float] = Field(..., description="Dense vector embedding")
    embedding_dimension: int = Field(..., gt=0, description="Dimension of the embedding vector")
    model_name: str = Field(..., min_length=1, description="Name of the embedding model used")

    model_config = ConfigDict(from_attributes=True)

    @field_validator("chunk_text")
    @classmethod
    def validate_chunk_text(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("chunk_text cannot be empty or whitespace only")
        return v

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, v: List[float]) -> List[float]:
        if not v:
            raise ValueError("embedding vector cannot be empty")
        for idx, val in enumerate(v):
            if not isinstance(val, (int, float)):
                raise ValueError(f"embedding value at index {idx} must be a numeric float, got {type(val)}")
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"embedding value at index {idx} must be finite (not NaN or Inf)")
        return [float(x) for x in v]

    @field_validator("embedding_dimension")
    @classmethod
    def validate_dimension(cls, v: int, info) -> int:
        embedding = info.data.get("embedding")
        if embedding is not None and len(embedding) != v:
            raise ValueError(
                f"embedding_dimension ({v}) must match len(embedding) ({len(embedding)})"
            )
        return v

    @classmethod
    def from_document_chunk(
        cls,
        chunk: DocumentChunk,
        embedding: Sequence[float],
        embedding_dimension: int,
        model_name: str,
    ) -> "EmbeddedChunk":
        """
        Factory classmethod to construct an EmbeddedChunk from a DocumentChunk
        and its computed embedding vector.
        """
        return cls(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            collection_id=chunk.collection_id,
            chunk_text=chunk.chunk_text,
            metadata=chunk.metadata,
            embedding=list(embedding),
            embedding_dimension=embedding_dimension,
            model_name=model_name,
        )
