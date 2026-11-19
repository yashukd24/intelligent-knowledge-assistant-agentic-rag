"""
backend/models/document.py
─────────────────────────────────────────────────────────────
SQLAlchemy 2.0 ORM model for uploaded documents.
─────────────────────────────────────────────────────────────
"""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.models.base import Base

if TYPE_CHECKING:
    from backend.models.collection import Collection


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    collection_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    stored_filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    file_type: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
    )
    sha256: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
    )
    size_bytes: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    page_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    chunk_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(20),
        default="pending",
        nullable=False,
        index=True,
    )
    error: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    version: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False,
    )
    category: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )
    department: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )
    academic_year: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
    )
    effective_date: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
    )
    expiry_date: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
    )
    section_heading: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationship to parent collection
    collection: Mapped["Collection"] = relationship(
        "Collection",
        back_populates="documents",
    )

    # Composite indexes and unique constraints for fast scoped lookups & integrity
    __table_args__ = (
        UniqueConstraint("collection_id", "sha256", name="uq_doc_collection_sha256"),
        Index("ix_doc_collection_sha256", "collection_id", "sha256"),
        Index("ix_doc_collection_filename", "collection_id", "filename"),
    )

    def __repr__(self) -> str:
        return f"<Document(id='{self.id}', filename='{self.filename}', status='{self.status}')>"
