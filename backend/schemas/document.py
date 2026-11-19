"""
backend/schemas/document.py
─────────────────────────────────────────────────────────────
Pydantic v2 schemas for Document upload, status, and listings.
─────────────────────────────────────────────────────────────
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"
    DELETION_PENDING = "deletion_pending"


class DocumentResponse(BaseModel):
    id: str = Field(description="Unique UUID of the document")
    collection_id: str = Field(description="Foreign key of the parent collection")
    filename: str = Field(description="Original sanitized filename")
    file_type: str = Field(description="File extension without dot, e.g. pdf, docx, txt")
    sha256: str = Field(description="SHA-256 hash of the document content")
    size_bytes: int = Field(description="File size in bytes")
    page_count: int = Field(default=0, description="Page count (populated in Phase 4)")
    chunk_count: int = Field(default=0, description="Chunk count (populated in Phase 4)")
    status: DocumentStatus = Field(description="Current ingestion lifecycle status")
    error: Optional[str] = Field(default=None, description="Error message if ingestion failed")
    version: int = Field(default=1, description="Document revision version")
    category: Optional[str] = Field(default=None, description="College institutional category")
    department: Optional[str] = Field(default=None, description="Department name if applicable")
    academic_year: Optional[str] = Field(default=None, description="Academic year")
    effective_date: Optional[str] = Field(default=None, description="Effective policy date")
    expiry_date: Optional[str] = Field(default=None, description="Expiry date if applicable")
    section_heading: Optional[str] = Field(default=None, description="Section heading")
    created_at: datetime = Field(description="Upload timestamp")
    updated_at: datetime = Field(description="Last update timestamp")

    model_config = ConfigDict(from_attributes=True)


class DocumentMetadataUpdate(BaseModel):
    category: Optional[str] = Field(default=None, description="Institutional category")
    department: Optional[str] = Field(default=None, description="Department name")
    academic_year: Optional[str] = Field(default=None, description="Academic year")
    effective_date: Optional[str] = Field(default=None, description="Effective date")
    expiry_date: Optional[str] = Field(default=None, description="Expiry date")
    section_heading: Optional[str] = Field(default=None, description="Section heading")


class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse] = Field(description="List of documents in the collection")
    total: int = Field(description="Total count of documents")
