"""
scripts/recover_deletions.py
─────────────────────────────────────────────────────────────
Manual operator utility to recover pending document deletions.

Scans the SQLite database for documents in 'deletion_pending' status,
removes their ChromaDB vectors, unlinks their physical files,
and purges the SQLite rows.

Usage:
    python scripts/recover_deletions.py
    python scripts/recover_deletions.py --limit 50
─────────────────────────────────────────────────────────────
"""

import argparse
import asyncio
import logging
import sys

from backend.database import AsyncSessionLocal, close_db
from backend.services.document_recovery import DocumentRecoveryService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("recover_deletions")


async def main(limit: int | None = None) -> int:
    logger.info("Starting manual document deletion recovery sweep (limit=%s)...", limit)
    recovery_svc = DocumentRecoveryService(session_factory=AsyncSessionLocal)
    report = await recovery_svc.recover_pending_deletions(limit=limit)

    logger.info(
        "Recovery sweep finished: %d found, %d succeeded, %d failed.",
        report.total_found,
        len(report.succeeded),
        len(report.failed),
    )

    if report.failed:
        logger.warning("Failed document IDs: %s", report.failed)
        for doc_id, err in report.errors.items():
            logger.error("  Document %s: %s", doc_id, err)
        return 1

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recover incomplete document deletions.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of pending documents to process.",
    )
    args = parser.parse_args()

    try:
        exit_code = asyncio.run(main(limit=args.limit))
    finally:
        asyncio.run(close_db())

    sys.exit(exit_code)
