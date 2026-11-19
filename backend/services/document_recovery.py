"""
backend/services/document_recovery.py
─────────────────────────────────────────────────────────────
Durable deletion recovery service for pending document cleanup.

Discovers documents stuck in DELETION_PENDING status and completes
their asynchronous cleanup across ChromaDB, the local filesystem,
and SQLite.

Key guarantees:
  - Session Isolation: each document is processed in its own dedicated
    database session, preventing one failure from invalidating subsequent records.
  - Idempotent: safe to run repeatedly across any state.
  - Non-blocking: logs individual failures and continues processing.
  - Safe: strictly targets DELETION_PENDING documents only.
  - Bounded: supports optional batch limits to avoid stalling startup.
─────────────────────────────────────────────────────────────
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
)
from sqlalchemy.orm.exc import StaleDataError

from backend.models.document import Document
from backend.schemas.document import DocumentStatus
from backend.services.collection_lock import collection_lock_registry
from backend.services.file_storage import FileStorageService
from backend.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)


@dataclass
class DeletionRecoveryReport:
    """Summary report of a batch deletion recovery run."""
    total_found: int = 0
    succeeded: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    errors: Dict[str, str] = field(default_factory=dict)


class DocumentRecoveryService:
    """
    Service responsible for discovering and recovering incomplete document deletions.
    Can be invoked during application startup or via operator CLI script.
    """

    def __init__(
        self,
        storage_service: Optional[FileStorageService] = None,
        vector_store: Optional[VectorStoreService] = None,
        session_factory: Optional[async_sessionmaker[AsyncSession]] = None,
    ):
        self.storage = storage_service or FileStorageService()
        self.vector_store = vector_store or VectorStoreService()
        self.session_factory = session_factory

    async def recover_pending_deletions(
        self,
        db: Optional[AsyncSession] = None,
        limit: Optional[int] = None,
    ) -> DeletionRecoveryReport:
        """
        Scan for documents in DELETION_PENDING status and complete their cleanup.
        Each document is handled within an isolated database session.

        Args:
            db: Optional active async database session. If provided, its engine
                is used to construct per-document sessions.
            limit: Optional maximum number of pending documents to process.

        Returns:
            DeletionRecoveryReport with counts, succeeded IDs, and failed IDs.
        """
        # Determine session factory for isolated per-document transactions
        if self.session_factory is not None:
            factory = self.session_factory
        elif db is not None:
            factory = async_sessionmaker(
                bind=db.bind,
                expire_on_commit=False,
                autocommit=False,
                autoflush=False,
            )
        else:
            from backend.database import AsyncSessionLocal
            factory = AsyncSessionLocal

        # 1. Scan for pending documents (fetching scalar tuples, not bound ORM instances)
        async with factory() as scan_session:
            query = (
                select(
                    Document.id,
                    Document.collection_id,
                    Document.stored_filename,
                    Document.filename,
                )
                .where(Document.status == DocumentStatus.DELETION_PENDING.value)
                .order_by(Document.updated_at.asc())
            )
            if limit is not None and limit > 0:
                query = query.limit(limit)

            res = await scan_session.execute(query)
            pending_items = res.all()

        report = DeletionRecoveryReport(
            total_found=len(pending_items),
            succeeded=[],
            failed=[],
            errors={},
        )

        if not pending_items:
            logger.debug("No pending document deletions found during recovery sweep.")
            return report

        logger.info(
            "Found %d document(s) in DELETION_PENDING state; starting isolated recovery sweep.",
            len(pending_items),
        )

        # 2. Process each pending document in an isolated session
        for doc_id, col_id, stored_fn, fn in pending_items:
            try:
                async with collection_lock_registry.acquire(col_id):
                    # Revalidate document and status inside the collection lock
                    async with factory() as check_session:
                        chk_doc = await check_session.get(Document, doc_id)
                        if chk_doc is None:
                            # Document already deleted concurrently
                            report.succeeded.append(doc_id)
                            logger.info("Pending document %s was already deleted concurrently.", doc_id)
                            continue
                        if chk_doc.status != DocumentStatus.DELETION_PENDING.value:
                            # Document is no longer pending deletion
                            logger.info(
                                "Pending document %s status is '%s', skipping recovery.",
                                doc_id,
                                chk_doc.status,
                            )
                            continue

                    # A. Delete ChromaDB vectors across all versions (idempotent)
                    try:
                        self.vector_store.delete_document_vectors(col_id, doc_id)
                    except Exception as vec_err:
                        raise RuntimeError(f"ChromaDB vector cleanup failed: {vec_err}") from vec_err

                    # B. Delete stored file (delete_stored_file returns True if unlinked or already missing)
                    file_deleted = self.storage.delete_stored_file(col_id, stored_fn)
                    if not file_deleted:
                        raise RuntimeError(f"Filesystem file cleanup failed for stored file '{stored_fn}'")

                    # C. Delete document record in its own dedicated session
                    async with factory() as item_session:
                        doc = await item_session.get(Document, doc_id)
                        if doc is not None:
                            try:
                                await item_session.delete(doc)
                                await item_session.commit()
                            except StaleDataError:
                                # Row was already deleted by a concurrent process
                                await item_session.rollback()
                                logger.info("Pending document %s was already deleted concurrently.", doc_id)

                    report.succeeded.append(doc_id)
                    logger.info("Successfully recovered and deleted pending document %s ('%s').", doc_id, fn)

            except Exception as exc:
                report.failed.append(doc_id)
                report.errors[doc_id] = str(exc)
                logger.warning(
                    "Recovery failed for pending document %s ('%s'): %s. Document retained in DELETION_PENDING.",
                    doc_id,
                    fn,
                    exc,
                )
                # Record error in an isolated session so it can be audited
                try:
                    async with factory() as err_session:
                        doc_err = await err_session.get(Document, doc_id)
                        if doc_err is not None:
                            doc_err.error = f"Recovery attempt failed: {exc}"
                            await err_session.commit()
                except Exception as record_err:
                    logger.error("Failed to record recovery error for doc %s: %s", doc_id, record_err)
                continue

        return report
