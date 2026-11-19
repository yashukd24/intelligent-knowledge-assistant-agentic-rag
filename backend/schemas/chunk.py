"""
backend/schemas/chunk.py
─────────────────────────────────────────────────────────────
Pydantic v2 data models for extracted pages, text chunks, and
processed document output.
─────────────────────────────────────────────────────────────
"""

from typing import TYPE_CHECKING, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.schemas.document import DocumentStatus

if TYPE_CHECKING:
    from backend.schemas.embedding import EmbeddedChunk


class TextPage(BaseModel):
    """
    Represents raw extracted text from a single document page or logical block.
    For PDF, page_number is 1-based.
    For DOCX and TXT, page_number is None (unless physical page breaks exist).
    """
    page_number: Optional[int] = Field(
        default=None,
        description="1-based page number for paginated formats (PDF). None for unpaginated flow formats (DOCX, TXT).",
    )
    text: str = Field(..., description="Raw or pre-cleaned text content of the page")
    char_count: int = Field(default=0, description="Character length of text")

    model_config = ConfigDict(from_attributes=True)

    @field_validator("char_count", mode="before")
    @classmethod
    def set_char_count(cls, v, info):
        if v is None or v == 0:
            text = info.data.get("text", "")
            return len(text)
        return v


class ChunkMetadata(BaseModel):
    """
    Structured metadata accompanying each text chunk for provenance,
    grounding, citations, and downstream vector store indexing.
    """
    document_id: str = Field(..., description="UUID of source document")
    collection_id: str = Field(..., description="UUID of parent collection")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="Extension: pdf, docx, or txt")
    chunk_index: int = Field(..., ge=0, description="0-based sequential chunk index")
    char_count: int = Field(..., gt=0, description="Exact character count of chunk text")
    page_number: Optional[int] = Field(
        default=None,
        description="Source page number for PDF. Null for DOCX and TXT.",
    )
    page_range: Optional[str] = Field(
        default=None,
        description="Page span if chunk straddles multiple PDF pages (e.g. '1-2'). Null for DOCX/TXT.",
    )
    start_char: int = Field(default=0, ge=0, description="Start character offset in normalized document")
    end_char: int = Field(default=0, ge=0, description="End character offset in normalized document")
    version: Optional[int] = Field(
        default=None,
        description="Source document version corresponding to this chunk",
    )

    model_config = ConfigDict(from_attributes=True)


class DocumentChunk(BaseModel):
    """
    A single granular text chunk ready for Phase 5 embedding and Phase 6 vector indexing.
    Note: Chunk size is character-based. It does NOT guarantee a token count.
    """
    chunk_id: str = Field(..., description="Unique deterministic identifier (e.g. {doc_id}_c0001)")
    document_id: str = Field(..., description="UUID of source document")
    collection_id: str = Field(..., description="UUID of parent collection")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="Document format: pdf, docx, txt")
    chunk_index: int = Field(..., ge=0, description="0-based chunk sequence index")
    chunk_text: str = Field(..., min_length=1, description="Cleaned text content of the chunk")
    char_count: int = Field(..., gt=0, description="Character count of chunk_text")
    page_number: Optional[int] = Field(
        default=None,
        description="PDF 1-based page number. None for DOCX and TXT.",
    )
    page_range: Optional[str] = Field(
        default=None,
        description="PDF page range string (e.g. '1-2') if spanning boundaries. None for DOCX/TXT.",
    )
    metadata: ChunkMetadata = Field(..., description="Granular chunk metadata")
    version: Optional[int] = Field(
        default=None,
        description="Source document version corresponding to this chunk",
    )

    model_config = ConfigDict(from_attributes=True)

    @field_validator("char_count")
    @classmethod
    def validate_char_count(cls, v: int, info):
        chunk_text = info.data.get("chunk_text")
        if chunk_text is not None and v != len(chunk_text):
            raise ValueError(f"char_count ({v}) must match len(chunk_text) ({len(chunk_text)})")
        return v


class ProcessedDocument(BaseModel):
    """
    Structured in-memory output produced by DocumentProcessor.
    Contains full extraction, cleaning, and chunking results.
    """
    document_id: str = Field(..., description="UUID of document record")
    collection_id: str = Field(..., description="UUID of parent collection")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="Format: pdf, docx, txt")
    page_count: int = Field(default=0, ge=0, description="Total pages extracted (PDF) or 1 (DOCX/TXT)")
    chunk_count: int = Field(default=0, ge=0, description="Total chunks generated")
    total_characters: int = Field(default=0, ge=0, description="Total characters in normalized document text")
    chunks: List[DocumentChunk] = Field(default_factory=list, description="Ordered list of generated chunks")
    embedded_chunks: Optional[List["EmbeddedChunk"]] = Field(
        default=None,
        description="Generated embeddings for chunks if embedding was performed",
    )
    status: DocumentStatus = Field(default=DocumentStatus.INDEXED, description="Ingestion status after processing")
    error: Optional[str] = Field(default=None, description="Error message if processing failed")

    model_config = ConfigDict(from_attributes=True)
