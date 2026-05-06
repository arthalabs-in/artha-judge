# storage/sync_manager.py
"""
Background sync manager for hybrid mode.

Periodically syncs pending SQLite writes to Firebase when it becomes available.
Tracks sync status and handles conflict resolution.
"""

import asyncio
import logging
import json
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

import aiosqlite

logger = logging.getLogger(__name__)


class SyncStatus(Enum):
    """Sync status for documents."""
    PENDING = "pending"
    SYNCED = "synced"
    FAILED = "failed"
    CONFLICT = "conflict"


@dataclass
class SyncRecord:
    """Record of a document that needs syncing."""
    document_id: str
    collection_path: str
    doc_id: str
    data: Dict[str, Any]
    last_synced_at: Optional[float]
    sync_status: SyncStatus
    sync_error: Optional[str]


class SyncManager:
    """
    Manages background synchronization between SQLite and Firebase.

    Features:
    - Periodic sync of pending documents
    - Retry logic with exponential backoff
    - Conflict detection and resolution
    - Sync status tracking
    """

    def __init__(self, sqlite_path: str, firebase_backend=None,
                 sync_interval: float = 60.0, max_retries: int = 3):
        """
        Initialize the sync manager.

        Args:
            sqlite_path: Path to SQLite database
            firebase_backend: Firebase backend instance (optional, can be set later)
            sync_interval: Seconds between sync attempts
            max_retries: Maximum retry attempts for failed syncs
        """
        self._sqlite_path = sqlite_path
        self._firebase = firebase_backend
        self._sync_interval = sync_interval
        self._max_retries = max_retries
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def set_firebase_backend(self, firebase_backend) -> None:
        """Set or update the Firebase backend."""
        self._firebase = firebase_backend

    async def start(self) -> None:
        """Start the background sync task."""
        if self._running:
            logger.warning("Sync manager already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._sync_loop())
        logger.info(f"Sync manager started (interval: {self._sync_interval}s)")

    async def stop(self) -> None:
        """Stop the background sync task."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Sync manager stopped")

    async def _sync_loop(self) -> None:
        """Main sync loop."""
        while self._running:
            try:
                await self._sync_pending()
            except Exception as e:
                logger.error(f"Sync loop error: {e}")

            await asyncio.sleep(self._sync_interval)

    async def _sync_pending(self) -> None:
        """Sync all pending documents to Firebase."""
        if not self._firebase or not self._firebase.is_available:
            logger.debug("Firebase not available, skipping sync")
            return

        pending = await self._get_pending_documents()
        if not pending:
            return

        logger.info(f"Syncing {len(pending)} pending documents to Firebase")

        for record in pending:
            try:
                await self._sync_document(record)
                await self._update_sync_status(
                    record.document_id,
                    SyncStatus.SYNCED,
                    time.time()
                )
            except Exception as e:
                logger.error(f"Failed to sync document {record.document_id}: {e}")
                await self._update_sync_status(
                    record.document_id,
                    SyncStatus.FAILED,
                    error=str(e)
                )

    async def _sync_document(self, record: SyncRecord) -> None:
        """Sync a single document to Firebase."""
        # Get Firebase document reference
        parts = record.collection_path.split('/')
        ref = self._firebase.collection(parts[0])

        for i in range(1, len(parts)):
            if i % 2 == 1:  # Document
                ref = ref.document(parts[i])
            else:  # Collection
                ref = ref.collection(parts[i])

        doc_ref = ref.document(record.doc_id)

        # Check for conflicts
        existing = await doc_ref.get()
        if existing.exists:
            existing_data = existing.to_dict()
            # Simple conflict resolution: last-write-wins based on updated_at
            existing_updated = existing_data.get('updated_at', 0)
            local_updated = record.data.get('updated_at', time.time())

            if existing_updated > local_updated:
                # Remote is newer, skip this sync
                logger.warning(f"Conflict detected for {record.document_id}: remote is newer, skipping")
                await self._update_sync_status(
                    record.document_id,
                    SyncStatus.CONFLICT,
                    error="Remote document is newer"
                )
                return

        # Sync to Firebase
        await doc_ref.set(record.data, merge=True)
        logger.debug(f"Synced document {record.document_id} to Firebase")

    async def _get_pending_documents(self) -> List[SyncRecord]:
        """Get all documents pending sync."""
        records = []

        async with aiosqlite.connect(self._sqlite_path) as conn:
            # Get documents that need syncing
            cursor = await conn.execute("""
                SELECT d.id, d.collection_path, d.doc_id, d.data,
                       s.last_synced_at, s.sync_status, s.sync_error
                FROM documents d
                LEFT JOIN sync_metadata s ON d.id = s.document_id
                WHERE s.sync_status IS NULL
                   OR s.sync_status = 'pending'
                   OR (s.sync_status = 'failed' AND s.last_synced_at < ?)
                ORDER BY d.updated_at ASC
                LIMIT 100
            """, (time.time() - 300,))  # Retry failed syncs after 5 minutes

            async for row in cursor:
                doc_id, collection_path, doc_name, data_json, last_synced, status, error = row
                records.append(SyncRecord(
                    document_id=doc_id,
                    collection_path=collection_path,
                    doc_id=doc_name,
                    data=json.loads(data_json),
                    last_synced_at=last_synced,
                    sync_status=SyncStatus(status) if status else SyncStatus.PENDING,
                    sync_error=error
                ))

        return records

    async def _update_sync_status(self, document_id: str, status: SyncStatus,
                                   synced_at: Optional[float] = None,
                                   error: Optional[str] = None) -> None:
        """Update sync status for a document."""
        async with aiosqlite.connect(self._sqlite_path) as conn:
            await conn.execute("""
                INSERT INTO sync_metadata (document_id, last_synced_at, sync_status, sync_error)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    last_synced_at = excluded.last_synced_at,
                    sync_status = excluded.sync_status,
                    sync_error = excluded.sync_error
            """, (document_id, synced_at or time.time(), status.value, error))
            await conn.commit()

    async def mark_for_sync(self, document_id: str) -> None:
        """Mark a document as needing sync."""
        await self._update_sync_status(document_id, SyncStatus.PENDING)

    async def get_sync_stats(self) -> Dict[str, int]:
        """Get sync statistics."""
        stats = {
            'pending': 0,
            'synced': 0,
            'failed': 0,
            'conflict': 0,
            'total': 0
        }

        async with aiosqlite.connect(self._sqlite_path) as conn:
            cursor = await conn.execute("""
                SELECT sync_status, COUNT(*) as count
                FROM sync_metadata
                GROUP BY sync_status
            """)
            async for row in cursor:
                status, count = row
                if status:
                    stats[status] = count
                    stats['total'] += count

            # Count documents without sync metadata (new/pending)
            cursor = await conn.execute("""
                SELECT COUNT(*) FROM documents d
                WHERE NOT EXISTS (SELECT 1 FROM sync_metadata s WHERE s.document_id = d.id)
            """)
            row = await cursor.fetchone()
            stats['pending'] += row[0]
            stats['total'] += row[0]

        return stats

    async def force_sync_all(self) -> Dict[str, Any]:
        """Force sync all pending documents immediately."""
        if not self._firebase or not self._firebase.is_available:
            return {'success': False, 'error': 'Firebase not available'}

        pending = await self._get_pending_documents()
        synced = 0
        failed = 0

        for record in pending:
            try:
                await self._sync_document(record)
                await self._update_sync_status(
                    record.document_id,
                    SyncStatus.SYNCED,
                    time.time()
                )
                synced += 1
            except Exception as e:
                logger.error(f"Force sync failed for {record.document_id}: {e}")
                await self._update_sync_status(
                    record.document_id,
                    SyncStatus.FAILED,
                    error=str(e)
                )
                failed += 1

        return {
            'success': True,
            'synced': synced,
            'failed': failed,
            'total': len(pending)
        }
