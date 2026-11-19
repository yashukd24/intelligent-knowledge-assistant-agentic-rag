"""
backend/routers/documents.py
─────────────────────────────────────────────────────────────
Document Upload and Management REST endpoints.

Routes:
  POST   /api/collections/{collection_id}/documents — Upload and ingest document
  GET    /api/collections/{collection_id}/documents — List documents in collection
  GET    /api/documents/{document_id}               — Get document status & metadata
  DELETE /api/documents/{document_id}               — Delete document & disk file
─────────────────────────────────────────────────────────────
"""

import logging
from typing import List, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.exc import StaleDataError

from backend.database import get_db
from backend.models.collection import Collection
from backend.models.document import Document
from backend.schemas.chunk import ProcessedDocument
from backend.schemas.document import (
    DocumentListResponse,
    DocumentMetadataUpdate,
    DocumentResponse,
    DocumentStatus,
)
from backend.services.collection_lock import collection_lock_registry
from backend.services.document_processor import (
    DocumentNotFoundError,
    DocumentProcessor,
    DocumentProcessorError,
)
from backend.services.file_storage import (
    EmptyFileError,
    FileStorageService,
    FileTooLargeError,
    InvalidFileError,
    StorageError,
)
from backend.services.vector_store import (
    VectorStorageError,
    VectorStoreService,
)

logger = logging.getLogger(__name__)

# Router for collection-scoped document operations
collection_docs_router = APIRouter(prefix="/api/collections/{collection_id}/documents", tags=["Documents"])

# Router for direct document operations
docs_router = APIRouter(prefix="/api/documents", tags=["Documents"])


def get_storage_service() -> FileStorageService:
    return FileStorageService()


def get_vector_store() -> VectorStoreService:
    return VectorStoreService()


def get_document_processor(
    storage: FileStorageService = Depends(get_storage_service),
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> DocumentProcessor:
    from backend.services.embedding_service import EmbeddingService
    return DocumentProcessor(
        storage_service=storage,
        embedding_service=EmbeddingService(),
        vector_store=vector_store,
    )


@collection_docs_router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload document into a knowledge collection",
)
async def upload_document(
    collection_id: str,
    response: Response,
    file: UploadFile = File(..., description="PDF, DOCX, or TXT document file"),
    replace: bool = Query(
        default=False,
        description="If True, replace an existing document having the same filename in this collection",
    ),
    category: Optional[str] = Query(default=None, description="College institutional category"),
    department: Optional[str] = Query(default=None, description="Department name if applicable"),
    academic_year: Optional[str] = Query(default=None, description="Academic year (e.g. 2025-2026)"),
    effective_date: Optional[str] = Query(default=None, description="Policy effective date"),
    expiry_date: Optional[str] = Query(default=None, description="Policy expiry date"),
    section_heading: Optional[str] = Query(default=None, description="Section heading"),
    db: AsyncSession = Depends(get_db),
    storage: FileStorageService = Depends(get_storage_service),
) -> DocumentResponse:
    """
    Upload and validate a document into a collection.
    Enforces format signatures, 50MB size limit, non-empty content,
    and duplicate SHA-256 detection within the collection.
    """
    # 1. Verify parent collection exists
    col_query = select(Collection).where(Collection.id == collection_id)
    col_res = await db.execute(col_query)
    collection = col_res.scalar_one_or_none()
    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Target collection '{collection_id}' not found.",
        )

    # 2. Save and stream-validate file
    try:
        stored_info = await storage.save_uploaded_file(collection_id, file)
    except EmptyFileError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        ) from exc
    except InvalidFileError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except FileTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=str(exc),
        ) from exc
    except StorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Storage validation failed: {exc}",
        ) from exc
    except Exception as exc:
        logger.error("Unexpected error saving file: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file.",
        ) from exc

    # 3. Check for identical content duplicate (same SHA-256 in this collection)
    sha_query = select(Document).where(
        Document.collection_id == collection_id,
        Document.sha256 == stored_info.sha256,
    )
    sha_res = await db.execute(sha_query)
    dup_sha_doc = sha_res.scalar_one_or_none()
    if dup_sha_doc:
        # Cleanup uploaded file immediately
        storage.delete_stored_file(collection_id, stored_info.stored_filename)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Document with identical content already exists in collection: "
                f"'{dup_sha_doc.filename}' (ID: {dup_sha_doc.id})"
            ),
        )

    # 4. Check for same filename in this collection
    fn_query = select(Document).where(
        Document.collection_id == collection_id,
        Document.filename == stored_info.original_filename,
    )
    fn_res = await db.execute(fn_query)
    dup_fn_doc = fn_res.scalar_one_or_none()

    if dup_fn_doc:
        if not replace:
            # Cleanup uploaded file and reject
            storage.delete_stored_file(collection_id, stored_info.stored_filename)
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"A document named '{stored_info.original_filename}' already exists "
                    f"in this collection. Pass '?replace=true' to replace it."
                ),
            )
        else:
            # Replace existing document: staged replacement workflow
            # Preserve old stored filename until DB transaction commits successfully
            old_stored = dup_fn_doc.stored_filename
            dup_fn_doc.stored_filename = stored_info.stored_filename
            dup_fn_doc.file_type = stored_info.file_type
            dup_fn_doc.sha256 = stored_info.sha256
            dup_fn_doc.size_bytes = stored_info.size_bytes
            dup_fn_doc.status = DocumentStatus.PENDING.value
            dup_fn_doc.error = None
            dup_fn_doc.version += 1

            if category is not None:
                dup_fn_doc.category = category
            if department is not None:
                dup_fn_doc.department = department
            if academic_year is not None:
                dup_fn_doc.academic_year = academic_year
            if effective_date is not None:
                dup_fn_doc.effective_date = effective_date
            if expiry_date is not None:
                dup_fn_doc.expiry_date = expiry_date
            if section_heading is not None:
                dup_fn_doc.section_heading = section_heading

            try:
                await db.commit()
                await db.refresh(dup_fn_doc)
            except IntegrityError as exc:
                await db.rollback()
                # Clean up new upload, preserving old file and database state
                storage.delete_stored_file(collection_id, stored_info.stored_filename)
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Replacement failed: identical content already exists under another document in this collection.",
                ) from exc
            except Exception as exc:
                await db.rollback()
                # Clean up new upload, preserving old file and database state
                storage.delete_stored_file(collection_id, stored_info.stored_filename)
                logger.error("Failed to replace document in DB: %s", exc, exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Database transaction failed during document replacement.",
                ) from exc

            # Commit succeeded: safe to delete old stored file from disk
            storage.delete_stored_file(collection_id, old_stored)

            response.status_code = status.HTTP_200_OK
            return DocumentResponse.model_validate(dup_fn_doc)

    # 5. New document creation
    doc = Document(
        collection_id=collection_id,
        filename=stored_info.original_filename,
        stored_filename=stored_info.stored_filename,
        file_type=stored_info.file_type,
        sha256=stored_info.sha256,
        size_bytes=stored_info.size_bytes,
        status=DocumentStatus.PENDING.value,
        version=1,
        category=category,
        department=department,
        academic_year=academic_year,
        effective_date=effective_date,
        expiry_date=expiry_date,
        section_heading=section_heading,
    )
    db.add(doc)
    try:
        await db.commit()
        await db.refresh(doc)
    except IntegrityError as exc:
        await db.rollback()
        storage.delete_stored_file(collection_id, stored_info.stored_filename)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Document with identical content already exists in collection.",
        ) from exc
    except Exception as exc:
        await db.rollback()
        storage.delete_stored_file(collection_id, stored_info.stored_filename)
        logger.error("Failed to commit document to DB: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database transaction failed during document indexing.",
        ) from exc

    return DocumentResponse.model_validate(doc)


@collection_docs_router.get(
    "",
    response_model=DocumentListResponse,
    summary="List all documents in a collection",
)
async def list_documents(
    collection_id: str,
    category: Optional[str] = Query(default=None, description="Filter by category"),
    department: Optional[str] = Query(default=None, description="Filter by department"),
    search: Optional[str] = Query(default=None, description="Search by filename or section heading"),
    db: AsyncSession = Depends(get_db),
) -> DocumentListResponse:
    """Retrieve all documents belonging to a knowledge collection with optional filtering."""
    # Verify collection exists
    col_query = select(Collection).where(Collection.id == collection_id)
    col_res = await db.execute(col_query)
    if not col_res.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection '{collection_id}' not found.",
        )

    doc_query = select(Document).where(Document.collection_id == collection_id)

    if category:
        doc_query = doc_query.where(Document.category == category)
    if department:
        doc_query = doc_query.where(Document.department == department)
    if search:
        search_term = f"%{search.strip()}%"
        doc_query = doc_query.where(
            Document.filename.ilike(search_term) | Document.section_heading.ilike(search_term)
        )

    doc_query = doc_query.order_by(Document.created_at.desc())
    doc_res = await db.execute(doc_query)
    docs = doc_res.scalars().all()

    items = [DocumentResponse.model_validate(d) for d in docs]
    return DocumentListResponse(documents=items, total=len(items))


@docs_router.patch(
    "/{document_id}/metadata",
    response_model=DocumentResponse,
    summary="Update document metadata (category, department, dates)",
)
async def update_document_metadata(
    document_id: str,
    payload: DocumentMetadataUpdate,
    db: AsyncSession = Depends(get_db),
) -> DocumentResponse:
    """Update institutional metadata for a document."""
    query = select(Document).where(Document.id == document_id)
    result = await db.execute(query)
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )

    if payload.category is not None:
        doc.category = payload.category
    if payload.department is not None:
        doc.department = payload.department
    if payload.academic_year is not None:
        doc.academic_year = payload.academic_year
    if payload.effective_date is not None:
        doc.effective_date = payload.effective_date
    if payload.expiry_date is not None:
        doc.expiry_date = payload.expiry_date
    if payload.section_heading is not None:
        doc.section_heading = payload.section_heading

    await db.commit()
    await db.refresh(doc)
    return DocumentResponse.model_validate(doc)


@docs_router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get single document status and metadata",
)
async def get_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
) -> DocumentResponse:
    """Retrieve metadata and ingestion status for a document."""
    query = select(Document).where(Document.id == document_id)
    result = await db.execute(query)
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )
    return DocumentResponse.model_validate(doc)


@docs_router.post(
    "/{document_id}/process",
    response_model=ProcessedDocument,
    summary="Process document (extract text, clean, chunk, and embed)",
)
async def process_document_route(
    document_id: str,
    embed: bool = Query(default=True, description="Whether to generate embeddings and store in ChromaDB"),
    db: AsyncSession = Depends(get_db),
    processor: DocumentProcessor = Depends(get_document_processor),
) -> ProcessedDocument:
    """
    Process an uploaded document through the complete pipeline:
    Extract text -> Clean -> Chunk -> (Optionally Embed & Vector Store) -> Update SQLite document status to 'indexed'.
    """
    try:
        should_embed = embed and (processor.embedding_service is not None)
        return await processor.process_document(document_id, db, embed=should_embed)
    except DocumentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except DocumentProcessorError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc


@docs_router.delete(
    "/{document_id}",
    summary="Delete a document and its stored file",
)
async def delete_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    storage: FileStorageService = Depends(get_storage_service),
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> dict:
    """
    Recoverable document deletion workflow:
    1. Look up document in SQLite (404 if not found).
    2. Transition status to DELETION_PENDING and commit to SQLite before external cleanup.
    3. Delete all ChromaDB vectors across all versions for this document.
    4. Delete the stored file from disk and verify the Boolean result.
    5. Delete the SQLite document record and commit.
    6. Return success only after all cleanup phases succeed.
    """
    query = select(Document).where(Document.id == document_id)
    result = await db.execute(query)
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )

    collection_id = doc.collection_id
    stored_filename = doc.stored_filename
    doc_filename = doc.filename

    async with collection_lock_registry.acquire(collection_id):
        # Re-verify document exists under lock in case collection was deleted concurrently
        query_locked = select(Document).where(Document.id == document_id)
        res_locked = await db.execute(query_locked)
        doc = res_locked.scalar_one_or_none()
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{document_id}' not found.",
            )

        # Step 1: Mark document DELETION_PENDING and commit to SQLite before external cleanup
        if doc.status != DocumentStatus.DELETION_PENDING.value:
            doc.status = DocumentStatus.DELETION_PENDING.value
            doc.error = None
            try:
                await db.commit()
                await db.refresh(doc)
            except Exception as exc:
                await db.rollback()
                logger.error(
                    "Failed to commit DELETION_PENDING status for document '%s' (%s): %s",
                    doc_filename,
                    document_id,
                    exc,
                    exc_info=True,
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to initiate deletion for document '{doc_filename}': {exc}",
                ) from exc

        # Step 2: Delete all ChromaDB vectors belonging to this document across all versions
        try:
            vector_store.delete_document_vectors(collection_id, document_id)
        except Exception as exc:
            logger.error(
                "Vector store deletion failed for document '%s' (%s) in collection '%s': %s",
                doc_filename,
                document_id,
                collection_id,
                exc,
                exc_info=True,
            )
            try:
                doc.error = f"Vector deletion failed: {exc}"
                await db.commit()
            except Exception:
                await db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Vector store deletion failed for document '{doc_filename}': {exc}",
            ) from exc

        # Step 3: Delete stored physical file and verify Boolean return value
        file_deleted = storage.delete_stored_file(collection_id, stored_filename)
        if not file_deleted:
            logger.error(
                "Filesystem deletion failed for document '%s' (%s), file '%s' in collection '%s'.",
                doc_filename,
                document_id,
                stored_filename,
                collection_id,
            )
            try:
                doc.error = f"Filesystem deletion failed for file: {stored_filename}"
                await db.commit()
            except Exception:
                await db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Filesystem deletion failed for document '{doc_filename}': file cleanup could not be completed.",
            )

        # Step 4: Delete document record from SQLite
        try:
            await db.delete(doc)
            await db.commit()
        except StaleDataError:
            # A concurrent request already committed deletion of this row
            await db.rollback()
            logger.info(
                "Document '%s' (%s) was already deleted by a concurrent transaction.",
                doc_filename,
                document_id,
            )
        except Exception as exc:
            await db.rollback()
            logger.error(
                "Failed to commit final document deletion from DB for '%s' (%s): %s",
                doc_filename,
                document_id,
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database transaction failed during final document deletion for '{doc_filename}'.",
            ) from exc

        return {
            "status": "deleted",
            "id": document_id,
            "message": f"Document '{doc_filename}' deleted successfully.",
        }
