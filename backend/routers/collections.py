"""
backend/routers/collections.py
─────────────────────────────────────────────────────────────
Knowledge Collections REST endpoints.

Routes:
  POST   /api/collections        — Create a new collection
  GET    /api/collections        — List all collections with document counts
  GET    /api/collections/{id}   — Get collection details
  DELETE /api/collections/{id}   — Delete collection, child documents & files
─────────────────────────────────────────────────────────────
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.exc import StaleDataError

from backend.database import get_db
from backend.models.collection import Collection
from backend.models.document import Document
from backend.schemas.collection import (
    CollectionCreate,
    CollectionListResponse,
    CollectionResponse,
)
from backend.schemas.document import DocumentStatus
from backend.services.collection_lock import collection_lock_registry
from backend.services.file_storage import FileStorageService
from backend.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/collections", tags=["Collections"])


def get_storage_service() -> FileStorageService:
    return FileStorageService()


def get_vector_store() -> VectorStoreService:
    return VectorStoreService()


@router.post(
    "",
    response_model=CollectionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new knowledge collection",
)
async def create_collection(
    payload: CollectionCreate,
    db: AsyncSession = Depends(get_db),
) -> CollectionResponse:
    """Create a new knowledge collection. Name must be unique."""
    # Check for existing collection with identical name
    query = select(Collection).where(Collection.name == payload.name)
    result = await db.execute(query)
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Collection with name '{payload.name}' already exists.",
        )

    collection = Collection(
        name=payload.name,
        description=payload.description,
    )
    db.add(collection)
    try:
        await db.flush()
        await db.refresh(collection)
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Collection with this name already exists.",
        ) from exc

    return CollectionResponse(
        id=collection.id,
        name=collection.name,
        description=collection.description,
        created_at=collection.created_at,
        updated_at=collection.updated_at,
        document_count=0,
    )


@router.get(
    "",
    response_model=CollectionListResponse,
    summary="List all knowledge collections",
)
async def list_collections(
    db: AsyncSession = Depends(get_db),
) -> CollectionListResponse:
    """Retrieve all collections along with their indexed document counts."""
    query = (
        select(Collection, func.count(Document.id).label("doc_count"))
        .outerjoin(
            Document,
            (Collection.id == Document.collection_id)
            & (Document.status != DocumentStatus.DELETION_PENDING.value),
        )
        .group_by(Collection.id)
        .order_by(Collection.created_at.desc())
    )
    result = await db.execute(query)
    rows = result.all()

    items = [
        CollectionResponse(
            id=col.id,
            name=col.name,
            description=col.description,
            created_at=col.created_at,
            updated_at=col.updated_at,
            document_count=doc_count,
        )
        for col, doc_count in rows
    ]

    return CollectionListResponse(
        collections=items,
        total=len(items),
    )


@router.get(
    "/{collection_id}",
    response_model=CollectionResponse,
    summary="Get collection details",
)
async def get_collection(
    collection_id: str,
    db: AsyncSession = Depends(get_db),
) -> CollectionResponse:
    """Get single collection by UUID, reporting active documents only."""
    query = (
        select(Collection, func.count(Document.id).label("doc_count"))
        .outerjoin(
            Document,
            (Collection.id == Document.collection_id)
            & (Document.status != DocumentStatus.DELETION_PENDING.value),
        )
        .where(Collection.id == collection_id)
        .group_by(Collection.id)
    )
    result = await db.execute(query)
    row = result.first()
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection '{collection_id}' not found.",
        )

    col, doc_count = row
    return CollectionResponse(
        id=col.id,
        name=col.name,
        description=col.description,
        created_at=col.created_at,
        updated_at=col.updated_at,
        document_count=doc_count,
    )


@router.delete(
    "/{collection_id}",
    summary="Delete collection, child documents, vectors, and files",
)
async def delete_collection(
    collection_id: str,
    db: AsyncSession = Depends(get_db),
    storage: FileStorageService = Depends(get_storage_service),
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> dict:
    """
    Recoverable collection deletion workflow:
    1. Look up collection in SQLite (HTTP 404 if not found).
    2. Stage DELETION_PENDING on all child documents and commit.
    3. Delete ChromaDB vector collection across all versions and chunks.
    4. Delete collection physical upload storage directory.
    5. Delete the SQLite collection record (cascades to child documents) and commit.
    6. Return HTTP 200 only after external cleanup and DB commit succeed.
    """
    async with collection_lock_registry.acquire(collection_id):
        query = select(Collection).where(Collection.id == collection_id)
        result = await db.execute(query)
        collection = result.scalar_one_or_none()
        if not collection:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection '{collection_id}' not found.",
            )

        collection_name = collection.name

        # Step 1: Stage DELETION_PENDING on all child documents and commit
        try:
            stmt = (
                update(Document)
                .where(Document.collection_id == collection_id)
                .values(
                    status=DocumentStatus.DELETION_PENDING.value,
                    error=f"Collection '{collection_name}' deletion in progress",
                )
            )
            await db.execute(stmt)
            await db.commit()
        except Exception as exc:
            await db.rollback()
            logger.error(
                "Failed to stage DELETION_PENDING for child documents of collection '%s' (%s): %s",
                collection_name,
                collection_id,
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initiate deletion for collection '{collection_name}': {exc}",
            ) from exc

        # Step 2: Delete ChromaDB vector collection
        try:
            vector_store.delete_collection(collection_id)
        except Exception as exc:
            logger.error(
                "Vector store collection deletion failed for '%s' (%s): %s",
                collection_name,
                collection_id,
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Vector store deletion failed for collection '{collection_name}': {exc}",
            ) from exc

        # Step 3: Delete physical upload storage directory
        dir_cleaned = storage.delete_collection_storage(collection_id)
        if not dir_cleaned:
            logger.error(
                "Filesystem deletion failed for collection storage directory: %s",
                collection_id,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Filesystem deletion failed for collection '{collection_name}': storage cleanup could not be completed.",
            )

        # Step 4: Delete collection record from SQLite (cascades to child documents)
        try:
            await db.delete(collection)
            await db.commit()
        except StaleDataError:
            # A concurrent request already committed deletion of this collection
            await db.rollback()
            logger.info(
                "Collection '%s' (%s) was already deleted by a concurrent transaction.",
                collection_name,
                collection_id,
            )
        except Exception as exc:
            await db.rollback()
            logger.error(
                "Failed to commit collection deletion from DB for '%s' (%s): %s",
                collection_name,
                collection_id,
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database transaction failed during final collection deletion for '{collection_name}'.",
            ) from exc

        return {
            "status": "deleted",
            "id": collection_id,
            "message": f"Collection '{collection_name}' and all associated documents deleted.",
        }
