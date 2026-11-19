"""
backend/schemas/collection.py
─────────────────────────────────────────────────────────────
Pydantic v2 schemas for Knowledge Collections.
─────────────────────────────────────────────────────────────
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CollectionBase(BaseModel):
    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Unique name of the collection",
    )
    description: Optional[str] = Field(
        default=None,
        description="Optional human-readable description",
    )

    @field_validator("name")
    @classmethod
    def validate_name_not_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Collection name cannot be empty or only whitespace")
        return stripped


class CollectionCreate(CollectionBase):
    pass


class CollectionUpdate(BaseModel):
    name: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="New name of the collection",
    )
    description: Optional[str] = Field(
        default=None,
        description="New description",
    )

    @field_validator("name")
    @classmethod
    def validate_name_not_empty(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            stripped = v.strip()
            if not stripped:
                raise ValueError("Collection name cannot be empty or only whitespace")
            return stripped
        return v


class CollectionResponse(CollectionBase):
    id: str = Field(description="Unique UUID of the collection")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")
    document_count: int = Field(
        default=0,
        description="Total active documents in collection (excluding deletion-pending)",
    )

    model_config = ConfigDict(from_attributes=True)


class CollectionListResponse(BaseModel):
    collections: List[CollectionResponse] = Field(description="List of collections")
    total: int = Field(description="Total number of collections")
