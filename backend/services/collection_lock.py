"""
backend/services/collection_lock.py
─────────────────────────────────────────────────────────────
Per-collection asynchronous lifecycle lock registry with strict
reference counting and safe event-loop isolation.

Guarantees:
  1. Mutual exclusion between collection deletion, document ingestion,
     and recovery sweeps on the same collection.
  2. Zero memory leaks: lock entries are automatically evicted when
     active holders and waiters reach zero.
  3. Safe event-loop isolation: entries are isolated per event-loop ID
     without WeakKeyDictionary circular references or cross-loop lock reuse.
  4. Safe cancellation and exception handling: reference count and
     locks are safely released even if awaiting tasks are cancelled.
─────────────────────────────────────────────────────────────
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Optional

logger = logging.getLogger(__name__)


class _LockEntry:
    __slots__ = ("lock", "ref_count")

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.ref_count: int = 0


class CollectionLockRegistry:
    """
    Registry for per-collection asynchronous locks.
    Maintains loop-isolated stores with reference-counted entries to prevent
    both TOCTOU races and unbounded dictionary growth.
    """

    def __init__(self) -> None:
        # Maps event loop id (int) -> dict[collection_id, _LockEntry]
        # Using loop id avoids circular reference retention on the loop object.
        self._loop_stores: Dict[int, Dict[str, _LockEntry]] = {}

    def _get_entries(self, loop: asyncio.AbstractEventLoop) -> Dict[str, _LockEntry]:
        loop_id = id(loop)
        entries = self._loop_stores.get(loop_id)
        if entries is None:
            entries = {}
            self._loop_stores[loop_id] = entries
        return entries

    @asynccontextmanager
    async def acquire(self, collection_id: str) -> AsyncGenerator[None, None]:
        """
        Acquire the lifecycle lock for a collection.
        Automatically manages registration, reference counting, and eviction.
        """
        if not collection_id or not isinstance(collection_id, str):
            raise ValueError("collection_id must be a non-empty string.")

        loop = asyncio.get_running_loop()
        loop_id = id(loop)
        entries = self._get_entries(loop)

        # 1. Register intent & increment reference count synchronously
        if collection_id not in entries:
            entries[collection_id] = _LockEntry()
        entry = entries[collection_id]
        entry.ref_count += 1
        lock = entry.lock

        # 2. Acquire the collection lock
        acquired = False
        try:
            await lock.acquire()
            acquired = True
            yield
        finally:
            if acquired:
                lock.release()

            # 3. Decrement reference count and evict if no coroutines wait or hold
            entry.ref_count -= 1
            if entry.ref_count <= 0:
                if entries.get(collection_id) is entry:
                    del entries[collection_id]
                # If no locks remain on this loop, clean up the loop store
                if not entries and loop_id in self._loop_stores:
                    del self._loop_stores[loop_id]

    async def get_active_count(self) -> int:
        """Return the number of registered collection lock entries for the active loop."""
        loop = asyncio.get_running_loop()
        entries = self._get_entries(loop)
        return len(entries)

    async def get_ref_count(self, collection_id: str) -> int:
        """Return the current reference count for a collection lock on the active loop."""
        loop = asyncio.get_running_loop()
        entries = self._get_entries(loop)
        entry = entries.get(collection_id)
        return entry.ref_count if entry else 0

    def reset(self) -> None:
        """Explicitly clear stores (useful in test teardown)."""
        try:
            loop = asyncio.get_running_loop()
            self._loop_stores.pop(id(loop), None)
        except RuntimeError:
            self._loop_stores.clear()


# Module-level singleton
collection_lock_registry = CollectionLockRegistry()
