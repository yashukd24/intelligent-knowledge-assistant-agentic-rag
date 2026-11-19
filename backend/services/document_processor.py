"""
backend/services/document_processor.py
─────────────────────────────────────────────────────────────
Document Processing Pipeline Orchestrator.

Coordinates:
  1. Retrieve document metadata from database.
  2. Inspect state & version for safe versioned staging.
  3. Transition status: pending -> processing.
  4. Resolve safe stored file path.
  5. Parse document pages (PDF, DOCX, TXT) via DocumentParser.
  6. Clean text via TextCleaner and generate versioned chunks via TextChunker.
  7. If embed=True:
       a. Generate embeddings via EmbeddingService.
       b. Store vectors in ChromaDB under target version via VectorStoreService.
       c. Verify stored vectors on disk.
  8. Promote target version in SQLite: status='indexed', version=target_version.
  9. Retire obsolete previous-version vectors only AFTER promotion.
  10. Guaranteed failure rollback: cleans up staged chunks, never destroys old valid index.
─────────────────────────────────────────────────────────────
"""

import asyncio
import logging
from typing import TYPE_CHECKING, List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.document import Document
from backend.schemas.chunk import ProcessedDocument
from backend.schemas.document import DocumentStatus
from backend.services.collection_lock import collection_lock_registry
from backend.services.document_parser import DocumentParser, ParsingError
from backend.services.file_storage import FileStorageService
from backend.services.text_chunker import TextChunker

if TYPE_CHECKING:
    from backend.services.embedding_service import EmbeddingService
    from backend.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)


class DocumentProcessorError(Exception):
    """Base exception for document processing pipeline failures."""
    pass


class DocumentNotFoundError(DocumentProcessorError):
    """Raised when the requested document ID does not exist in SQLite."""
    pass


class DocumentProcessor:
    """
    Coordinates end-to-end extraction, cleaning, chunking, embedding,
    and safe versioned vector storage of uploaded documents.
    """

    def __init__(
        self,
        storage_service: Optional[FileStorageService] = None,
        chunker: Optional[TextChunker] = None,
        embedding_service: Optional["EmbeddingService"] = None,
        vector_store: Optional["VectorStoreService"] = None,
    ):
        self.storage = storage_service or FileStorageService()
        self.chunker = chunker or TextChunker()
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def delete_document_vectors(self, collection_id: str, document_id: str) -> int:
        """
        Delete all vector records for a document from ChromaDB if vector_store is configured.

        Args:
            collection_id: Parent collection UUID.
            document_id: UUID of document whose vectors should be removed.

        Returns:
            Number of deleted vector records, or 0 if vector_store is not configured.
        """
        if self.vector_store is not None:
            return self.vector_store.delete_document_vectors(collection_id, document_id)
        return 0

    async def process_document(
        self,
        document_id: str,
        db: AsyncSession,
        embed: bool = False,
    ) -> ProcessedDocument:
        """
        Process an uploaded document through the complete extraction, chunking,
        and optional embedding and vector storage pipeline.

        Lifecycle:
          1. Validates document exists in SQLite.
          2. Determines target indexing version (staged versioning).
          3. Sets status to 'processing' (committed).
          4. Locates file and parses pages.
          5. Chunks text into versioned DocumentChunk instances.
          6. If embed=True:
             - Generates embeddings via embedding_service.
             - Stores vectors in vector_store under target_version.
             - Verifies disk storage of staged chunks.
          7. Promotes target version in SQLite: status='indexed', version=target_version.
          8. Retires old version vectors from ChromaDB (if re-indexing).
          9. On any failure: rolls back, deletes staged chunks, preserves previous valid index.

        Args:
            document_id: UUID of document to process.
            db: Active async database session.
            embed: If True, generate embeddings and store in ChromaDB (requires services).

        Returns:
            ProcessedDocument containing all generated chunks, optional embeddings, and metadata.

        Raises:
            DocumentNotFoundError: if document_id does not exist.
            DocumentProcessorError: if extraction, chunking, embedding, or storage fails.
        """
        # 1. Fetch document from database
        query = select(Document).where(Document.id == document_id)
        result = await db.execute(query)
        doc = result.scalar_one_or_none()

        if not doc:
            raise DocumentNotFoundError(f"Document '{document_id}' not found in database.")

        collection_id = doc.collection_id
        stored_filename = doc.stored_filename
        original_filename = doc.filename
        file_type = doc.file_type

        # 2. Reject documents marked for deletion
        if doc.status == DocumentStatus.DELETION_PENDING.value:
            raise DocumentProcessorError(
                f"Cannot process document '{document_id}': document is pending deletion."
            )

        # 3. Determine target version using staged versioning strategy
        previous_status = doc.status
        previous_version = doc.version or 1
        is_reindex = (previous_status == DocumentStatus.INDEXED.value) or (doc.chunk_count > 0)
        target_version = (previous_version + 1) if is_reindex else max(1, previous_version)

        # 4. Transition status: pending/failed/indexed -> processing
        doc.status = DocumentStatus.PROCESSING.value
        doc.error = None
        await db.commit()
        await db.refresh(doc)
        logger.info(
            "Document %s set to 'processing' (is_reindex=%s, target_version=%d).",
            document_id,
            is_reindex,
            target_version,
        )

        staged_chunk_ids: List[str] = []

        try:
            # 4. Locate stored file on disk
            file_path = self.storage.get_stored_file_path(collection_id, stored_filename)

            # 5. Extract text pages
            pages = DocumentParser.parse_file(file_path, file_type)

            # 6. Chunk text with word-boundary awareness and target version tagging
            chunks = self.chunker.chunk_document(
                document_id=document_id,
                collection_id=collection_id,
                filename=original_filename,
                file_type=file_type,
                pages=pages,
                version=target_version,
            )

            # Page count: actual page count for PDF, or 1 for DOCX/TXT
            if file_type == "pdf":
                page_count = len(pages)
            else:
                page_count = 1

            chunk_count = len(chunks)
            total_chars = sum(c.char_count for c in chunks)

            # 7. Embedding Pipeline (when embed=True)
            embedded_chunks = None
            if embed:
                if self.embedding_service is None:
                    raise DocumentProcessorError(
                        "Embedding was requested (embed=True) but no EmbeddingService is configured on DocumentProcessor."
                    )

                # Step 7a: Generate dense embeddings offloaded to worker thread (unblocks event loop)
                embedded_chunks = await asyncio.to_thread(
                    self.embedding_service.embed_chunks,
                    chunks,
                )

            # Synchronized Persistence & Promotion Critical Section
            # Acquires per-collection lifecycle lock to prevent TOCTOU race with collection deletion
            async with collection_lock_registry.acquire(collection_id):
                # Revalidate document and collection state in SQLite under the lock
                chk_query = select(Document.status).where(
                    Document.id == document_id,
                    Document.collection_id == collection_id,
                )
                chk_res = await db.execute(chk_query)
                current_status = chk_res.scalar_one_or_none()

                if current_status is None:
                    raise DocumentProcessorError(
                        f"Cannot complete processing for document '{original_filename}': "
                        f"document or collection '{collection_id}' was deleted."
                    )
                if current_status == DocumentStatus.DELETION_PENDING.value:
                    raise DocumentProcessorError(
                        f"Cannot complete processing for document '{original_filename}': "
                        f"collection deletion is in progress."
                    )

                # Step 7b: Vector storage in ChromaDB if vector_store is configured
                if embed and self.vector_store is not None and embedded_chunks:
                    staged_chunk_ids = [c.chunk_id for c in embedded_chunks]

                    try:
                        self.vector_store.store_chunks(
                            embedded_chunks,
                            collection_id=collection_id,
                        )
                    except Exception as store_err:
                        # Clean up only the staged chunks for this failed attempt
                        if staged_chunk_ids:
                            try:
                                self.vector_store.delete_chunks_by_ids(
                                    collection_id=collection_id,
                                    chunk_ids=staged_chunk_ids,
                                )
                            except Exception as cleanup_err:
                                logger.warning(
                                    "Failed to clean up staged chunks after storage failure: %s",
                                    cleanup_err,
                                )
                        raise DocumentProcessorError(
                            f"Vector storage failed for document '{original_filename}': {store_err}"
                        ) from store_err

                    # Step 7c: Verify storage completeness on disk
                    records = self.vector_store.get_chunks_by_ids(
                        collection_id=collection_id,
                        chunk_ids=staged_chunk_ids,
                    )
                    found_count = len(records.get("ids", []))
                    if found_count != len(staged_chunk_ids):
                        if staged_chunk_ids:
                            try:
                                self.vector_store.delete_chunks_by_ids(
                                    collection_id=collection_id,
                                    chunk_ids=staged_chunk_ids,
                                )
                            except Exception:
                                pass
                        raise DocumentProcessorError(
                            f"Vector store verification failed for document '{original_filename}': "
                            f"expected {len(staged_chunk_ids)} chunks, found {found_count} on disk."
                        )

                # 8. Promote target version in SQLite
                doc.status = DocumentStatus.INDEXED.value
                doc.page_count = page_count
                doc.chunk_count = chunk_count
                doc.version = target_version
                doc.error = None

                try:
                    await db.commit()
                    await db.refresh(doc)
                except Exception as commit_err:
                    await db.rollback()
                    # Clean up unpromoted staged chunks from ChromaDB
                    if self.vector_store is not None and staged_chunk_ids:
                        try:
                            self.vector_store.delete_chunks_by_ids(
                                collection_id=collection_id,
                                chunk_ids=staged_chunk_ids,
                            )
                        except Exception as cleanup_err:
                            logger.warning(
                                "Failed to clean up staged chunks after DB commit failure: %s",
                                cleanup_err,
                            )
                    raise DocumentProcessorError(
                        f"Database commit failed during document indexing: {commit_err}"
                    ) from commit_err

                # 9. Post-promotion cleanup: Retire previous version vectors if this was a re-index
                if self.vector_store is not None and is_reindex:
                    try:
                        deleted_old = self.vector_store.delete_document_vectors(
                            collection_id=collection_id,
                            document_id=document_id,
                            exclude_version=target_version,
                        )
                        logger.info(
                            "Retired %d obsolete vector chunks from previous versions for doc %s.",
                            deleted_old,
                            document_id,
                        )
                    except Exception as retire_err:
                        logger.warning(
                            "Failed to retire obsolete vectors for document %s (new index remains valid): %s",
                            document_id,
                            retire_err,
                        )

            logger.info(
                "Document %s processed successfully: %d pages, %d chunks, %d chars, version=%d%s.",
                document_id,
                page_count,
                chunk_count,
                total_chars,
                target_version,
                f", {len(embedded_chunks)} vectors stored" if (embedded_chunks and self.vector_store) else "",
            )

            return ProcessedDocument(
                document_id=document_id,
                collection_id=collection_id,
                filename=original_filename,
                file_type=file_type,
                page_count=page_count,
                chunk_count=chunk_count,
                total_characters=total_chars,
                chunks=chunks,
                embedded_chunks=embedded_chunks,
                status=DocumentStatus.INDEXED,
                error=None,
            )

        except Exception as exc:
            # 10. Guaranteed failure handling: NEVER leave document stuck in 'processing'
            logger.error("Processing failed for document %s: %s", document_id, exc, exc_info=True)
            await db.rollback()

            # Clean up staged chunks from ChromaDB if any were partially written
            if self.vector_store is not None and staged_chunk_ids:
                try:
                    self.vector_store.delete_chunks_by_ids(
                        collection_id=collection_id,
                        chunk_ids=staged_chunk_ids,
                    )
                except Exception as cleanup_err:
                    logger.warning(
                        "Failed to clean up staged chunks during failure handling: %s",
                        cleanup_err,
                    )

            # In a fresh query/update, record the failure status in DB
            try:
                fetch_query = select(Document).where(Document.id == document_id)
                res = await db.execute(fetch_query)
                failed_doc = res.scalar_one_or_none()
                if failed_doc and failed_doc.status != DocumentStatus.DELETION_PENDING.value:
                    failed_doc.status = DocumentStatus.FAILED.value
                    failed_doc.error = str(exc)
                    # NOTE: failed_doc.version is NOT incremented; it retains previous_version,
                    # ensuring linkage to previous valid index vectors (if any) is preserved.
                    await db.commit()
            except Exception as db_err:
                logger.error("Failed to record error status for doc %s: %s", document_id, db_err)

            raise DocumentProcessorError(f"Failed to process document '{original_filename}': {exc}") from exc
