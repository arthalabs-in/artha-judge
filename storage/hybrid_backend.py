# storage/hybrid_backend.py
"""
Hybrid storage backend with automatic fallback.

Writes to both SQLite (always) and Firebase (when available).
Reads from Firebase first, falls back to SQLite on failure.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, AsyncIterator, Tuple

from .base import (
    StorageBackendInterface,
    CollectionReference,
    DocumentReference,
    DocumentSnapshot,
    Query,
    ServerTimestampSentinel,
    SERVER_TIMESTAMP,
)
from .sqlite_backend import SQLiteBackend, SQLiteCollectionReference, SQLiteDocumentReference, SQLiteQuery
from .firebase_backend import FirebaseBackend, FirebaseCollectionReference, FirebaseDocumentReference, FirebaseQuery

logger = logging.getLogger(__name__)


class HybridQuery(Query):
    """Hybrid query that tries Firebase first, falls back to SQLite."""

    def __init__(self, firebase_query: Optional[FirebaseQuery],
                 sqlite_query: SQLiteQuery,
                 firebase_available: bool):
        self._firebase_query = firebase_query
        self._sqlite_query = sqlite_query
        self._firebase_available = firebase_available

    def where(self, field: str, op: str, value: Any) -> 'HybridQuery':
        new_sqlite = self._sqlite_query.where(field, op, value)
        new_firebase = self._firebase_query.where(field, op, value) if self._firebase_query else None
        return HybridQuery(new_firebase, new_sqlite, self._firebase_available)

    def order_by(self, field: str, direction: str = 'ASCENDING') -> 'HybridQuery':
        new_sqlite = self._sqlite_query.order_by(field, direction)
        new_firebase = self._firebase_query.order_by(field, direction) if self._firebase_query else None
        return HybridQuery(new_firebase, new_sqlite, self._firebase_available)

    def limit(self, count: int) -> 'HybridQuery':
        new_sqlite = self._sqlite_query.limit(count)
        new_firebase = self._firebase_query.limit(count) if self._firebase_query else None
        return HybridQuery(new_firebase, new_sqlite, self._firebase_available)

    async def get(self) -> List[DocumentSnapshot]:
        # Try Firebase first
        if self._firebase_available and self._firebase_query:
            try:
                return await self._firebase_query.get()
            except Exception as e:
                logger.warning(f"Firebase query failed, falling back to SQLite: {e}")

        # Fallback to SQLite
        return await self._sqlite_query.get()

    async def stream(self) -> AsyncIterator[DocumentSnapshot]:
        # Try Firebase first
        if self._firebase_available and self._firebase_query:
            try:
                async for doc in self._firebase_query.stream():
                    yield doc
                return
            except Exception as e:
                logger.warning(f"Firebase stream failed, falling back to SQLite: {e}")

        # Fallback to SQLite
        async for doc in self._sqlite_query.stream():
            yield doc


class HybridDocumentReference(DocumentReference):
    """Hybrid document reference with dual-write and fallback read."""

    def __init__(self, firebase_ref: Optional[FirebaseDocumentReference],
                 sqlite_ref: SQLiteDocumentReference,
                 firebase_available: bool):
        self._firebase_ref = firebase_ref
        self._sqlite_ref = sqlite_ref
        self._firebase_available = firebase_available

    @property
    def id(self) -> str:
        return self._sqlite_ref.id

    @property
    def path(self) -> str:
        return self._sqlite_ref.path

    async def get(self) -> DocumentSnapshot:
        # Try Firebase first
        if self._firebase_available and self._firebase_ref:
            try:
                return await self._firebase_ref.get()
            except Exception as e:
                logger.warning(f"Firebase get failed, falling back to SQLite: {e}")

        # Fallback to SQLite
        return await self._sqlite_ref.get()

    async def set(self, data: Dict[str, Any], merge: bool = False) -> None:
        # Always write to SQLite first (guaranteed local)
        await self._sqlite_ref.set(data, merge)

        # Then try Firebase (best-effort)
        if self._firebase_available and self._firebase_ref:
            try:
                await self._firebase_ref.set(data, merge)
            except Exception as e:
                logger.warning(f"Firebase set failed (SQLite succeeded): {e}")
                # TODO: Mark for sync in sync_metadata table

    async def update(self, data: Dict[str, Any]) -> None:
        # Always update SQLite first
        await self._sqlite_ref.update(data)

        # Then try Firebase
        if self._firebase_available and self._firebase_ref:
            try:
                await self._firebase_ref.update(data)
            except Exception as e:
                logger.warning(f"Firebase update failed (SQLite succeeded): {e}")

    async def delete(self) -> None:
        # Delete from SQLite first
        await self._sqlite_ref.delete()

        # Then try Firebase
        if self._firebase_available and self._firebase_ref:
            try:
                await self._firebase_ref.delete()
            except Exception as e:
                logger.warning(f"Firebase delete failed (SQLite succeeded): {e}")

    def collection(self, name: str) -> 'HybridCollectionReference':
        firebase_subcol = self._firebase_ref.collection(name) if self._firebase_ref else None
        sqlite_subcol = self._sqlite_ref.collection(name)
        return HybridCollectionReference(firebase_subcol, sqlite_subcol, self._firebase_available)


class HybridCollectionReference(CollectionReference):
    """Hybrid collection reference with dual-write and fallback read."""

    def __init__(self, firebase_collection: Optional[FirebaseCollectionReference],
                 sqlite_collection: SQLiteCollectionReference,
                 firebase_available: bool):
        self._firebase_collection = firebase_collection
        self._sqlite_collection = sqlite_collection
        self._firebase_available = firebase_available

    @property
    def path(self) -> str:
        return self._sqlite_collection.path

    def document(self, doc_id: Optional[str] = None) -> HybridDocumentReference:
        sqlite_ref = self._sqlite_collection.document(doc_id)
        firebase_ref = self._firebase_collection.document(sqlite_ref.id) if self._firebase_collection else None
        return HybridDocumentReference(firebase_ref, sqlite_ref, self._firebase_available)

    async def add(self, data: Dict[str, Any]) -> Tuple[Any, HybridDocumentReference]:
        # Add to SQLite first
        timestamp, sqlite_ref = await self._sqlite_collection.add(data)

        # Create hybrid reference
        firebase_ref = None
        if self._firebase_available and self._firebase_collection:
            try:
                # Use same doc ID for consistency
                firebase_ref = self._firebase_collection.document(sqlite_ref.id)
                await firebase_ref.set(data)
                firebase_ref = self._firebase_collection.document(sqlite_ref.id)
            except Exception as e:
                logger.warning(f"Firebase add failed (SQLite succeeded): {e}")

        hybrid_ref = HybridDocumentReference(firebase_ref, sqlite_ref, self._firebase_available)
        return (timestamp, hybrid_ref)

    def where(self, field: str, op: str, value: Any) -> HybridQuery:
        sqlite_query = self._sqlite_collection.where(field, op, value)
        firebase_query = self._firebase_collection.where(field, op, value) if self._firebase_collection else None
        return HybridQuery(firebase_query, sqlite_query, self._firebase_available)

    def order_by(self, field: str, direction: str = 'ASCENDING') -> HybridQuery:
        sqlite_query = self._sqlite_collection.order_by(field, direction)
        firebase_query = self._firebase_collection.order_by(field, direction) if self._firebase_collection else None
        return HybridQuery(firebase_query, sqlite_query, self._firebase_available)

    def limit(self, count: int) -> HybridQuery:
        sqlite_query = self._sqlite_collection.limit(count)
        firebase_query = self._firebase_collection.limit(count) if self._firebase_collection else None
        return HybridQuery(firebase_query, sqlite_query, self._firebase_available)

    async def stream(self) -> AsyncIterator[DocumentSnapshot]:
        # Try Firebase first
        if self._firebase_available and self._firebase_collection:
            try:
                async for doc in self._firebase_collection.stream():
                    yield doc
                return
            except Exception as e:
                logger.warning(f"Firebase stream failed, falling back to SQLite: {e}")

        # Fallback to SQLite
        async for doc in self._sqlite_collection.stream():
            yield doc

    async def list_documents(self) -> List[HybridDocumentReference]:
        # Try Firebase first
        if self._firebase_available and self._firebase_collection:
            try:
                firebase_refs = await self._firebase_collection.list_documents()
                return [
                    HybridDocumentReference(
                        ref,
                        self._sqlite_collection.document(ref.id),
                        self._firebase_available
                    )
                    for ref in firebase_refs
                ]
            except Exception as e:
                logger.warning(f"Firebase list_documents failed, falling back to SQLite: {e}")

        # Fallback to SQLite
        sqlite_refs = await self._sqlite_collection.list_documents()
        return [
            HybridDocumentReference(None, ref, False)
            for ref in sqlite_refs
        ]


class HybridBackend(StorageBackendInterface):
    """
    Hybrid storage backend with dual-write and automatic fallback.

    - Always writes to SQLite first (guaranteed local storage)
    - Then writes to Firebase (best-effort, when available)
    - Reads from Firebase first, falls back to SQLite on failure
    """

    def __init__(self, sqlite_backend: SQLiteBackend, firebase_backend: FirebaseBackend):
        self._sqlite = sqlite_backend
        self._firebase = firebase_backend
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize both backends."""
        # SQLite is always initialized
        await self._sqlite.initialize()

        # Firebase initialization is optional (may fail)
        try:
            await self._firebase.initialize()
            logger.info("Hybrid backend: Firebase initialized successfully")
        except Exception as e:
            logger.warning(f"Hybrid backend: Firebase initialization failed, using SQLite only: {e}")

        self._initialized = True
        logger.info(f"Hybrid backend initialized (Firebase available: {self._firebase.is_available})")

    async def close(self) -> None:
        """Close both backends."""
        await self._sqlite.close()
        if self._firebase.is_available:
            await self._firebase.close()
        self._initialized = False
        logger.info("Hybrid backend closed")

    def collection(self, name: str) -> HybridCollectionReference:
        sqlite_col = self._sqlite.collection(name)
        firebase_col = self._firebase.collection(name) if self._firebase.is_available else None
        return HybridCollectionReference(firebase_col, sqlite_col, self._firebase.is_available)

    def server_timestamp(self) -> ServerTimestampSentinel:
        return SERVER_TIMESTAMP

    @property
    def is_available(self) -> bool:
        # Hybrid is available if at least SQLite is available
        return self._initialized and self._sqlite.is_available

    @property
    def backend_type(self) -> str:
        return 'hybrid'

    @property
    def firebase_available(self) -> bool:
        """Check if Firebase backend is currently available."""
        return self._firebase.is_available

    @property
    def sqlite_backend(self) -> SQLiteBackend:
        """Access the underlying SQLite backend."""
        return self._sqlite

    @property
    def firebase_backend(self) -> FirebaseBackend:
        """Access the underlying Firebase backend."""
        return self._firebase
