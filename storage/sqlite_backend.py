# storage/sqlite_backend.py
"""
SQLite implementation of the storage abstraction.

Uses aiosqlite for async operations and stores documents as JSON blobs.
Supports path-based collection hierarchy for subcollections.
"""

import os
import json
import time
import uuid
import logging
from typing import Any, Dict, List, Optional, AsyncIterator, Tuple
from pathlib import Path

import aiosqlite

from .base import (
    StorageBackendInterface,
    CollectionReference,
    DocumentReference,
    DocumentSnapshot,
    Query,
    ServerTimestampSentinel,
    SERVER_TIMESTAMP,
)

logger = logging.getLogger(__name__)


def _resolve_timestamps(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively resolve SERVER_TIMESTAMP sentinels to actual timestamps.
    """
    resolved = {}
    current_time = time.time()

    for key, value in data.items():
        if isinstance(value, ServerTimestampSentinel):
            resolved[key] = current_time
        elif isinstance(value, dict):
            if value.get("__sentinel__") == "SERVER_TIMESTAMP":
                resolved[key] = current_time
            else:
                resolved[key] = _resolve_timestamps(value)
        elif isinstance(value, list):
            resolved[key] = [
                _resolve_timestamps(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            resolved[key] = value

    return resolved


def _deep_merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class SQLiteQuery(Query):
    """SQLite implementation of Query interface."""

    def __init__(self, backend: 'SQLiteBackend', collection_path: str):
        self._backend = backend
        self._collection_path = collection_path
        self._filters: List[Tuple[str, str, Any]] = []
        self._order_by_field: Optional[str] = None
        self._order_direction: str = 'ASCENDING'
        self._limit_count: Optional[int] = None

    def where(self, field: str, op: str, value: Any) -> 'SQLiteQuery':
        new_query = SQLiteQuery(self._backend, self._collection_path)
        new_query._filters = self._filters.copy()
        new_query._filters.append((field, op, value))
        new_query._order_by_field = self._order_by_field
        new_query._order_direction = self._order_direction
        new_query._limit_count = self._limit_count
        return new_query

    def order_by(self, field: str, direction: str = 'ASCENDING') -> 'SQLiteQuery':
        new_query = SQLiteQuery(self._backend, self._collection_path)
        new_query._filters = self._filters.copy()
        new_query._order_by_field = field
        new_query._order_direction = direction
        new_query._limit_count = self._limit_count
        return new_query

    def limit(self, count: int) -> 'SQLiteQuery':
        new_query = SQLiteQuery(self._backend, self._collection_path)
        new_query._filters = self._filters.copy()
        new_query._order_by_field = self._order_by_field
        new_query._order_direction = self._order_direction
        new_query._limit_count = count
        return new_query

    async def get(self) -> List[DocumentSnapshot]:
        results = []
        async for doc in self.stream():
            results.append(doc)
        return results

    async def stream(self) -> AsyncIterator[DocumentSnapshot]:
        async with self._backend._get_connection() as conn:
            # Build base query
            query = "SELECT doc_id, data FROM documents WHERE collection_path = ?"
            params: List[Any] = [self._collection_path]

            # Fetch all documents and filter in Python (json_extract is limited)
            cursor = await conn.execute(query, params)
            rows = await cursor.fetchall()

            # Filter results
            filtered_rows = []
            for row in rows:
                doc_id, data_json = row
                data = json.loads(data_json)

                # Apply filters
                matches = True
                for field, op, value in self._filters:
                    field_value = self._get_nested_field(data, field)

                    if op == '==':
                        matches = field_value == value
                    elif op == '!=':
                        matches = field_value != value
                    elif op == '<':
                        matches = field_value is not None and field_value < value
                    elif op == '<=':
                        matches = field_value is not None and field_value <= value
                    elif op == '>':
                        matches = field_value is not None and field_value > value
                    elif op == '>=':
                        matches = field_value is not None and field_value >= value
                    elif op == 'in':
                        matches = field_value in value
                    elif op == 'array-contains':
                        matches = isinstance(field_value, list) and value in field_value

                    if not matches:
                        break

                if matches:
                    filtered_rows.append((doc_id, data))

            # Apply ordering
            if self._order_by_field:
                reverse = self._order_direction == 'DESCENDING'
                filtered_rows.sort(
                    key=lambda x: self._get_nested_field(x[1], self._order_by_field) or '',
                    reverse=reverse
                )

            # Apply limit
            if self._limit_count:
                filtered_rows = filtered_rows[:self._limit_count]

            # Yield results
            for doc_id, data in filtered_rows:
                snapshot = DocumentSnapshot(
                    _id=doc_id,
                    _exists=True,
                    _data=data,
                    _path=f"{self._collection_path}/{doc_id}"
                )
                # Set reference
                snapshot._reference = SQLiteDocumentReference(
                    self._backend, self._collection_path, doc_id
                )
                yield snapshot

    def _get_nested_field(self, data: Dict[str, Any], field: str) -> Any:
        """Get a potentially nested field value using dot notation."""
        parts = field.split('.')
        value = data
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None
        return value


class SQLiteDocumentReference(DocumentReference):
    """SQLite implementation of DocumentReference."""

    def __init__(self, backend: 'SQLiteBackend', collection_path: str, doc_id: str):
        self._backend = backend
        self._collection_path = collection_path
        self._doc_id = doc_id

    @property
    def id(self) -> str:
        return self._doc_id

    @property
    def path(self) -> str:
        return f"{self._collection_path}/{self._doc_id}"

    async def get(self) -> DocumentSnapshot:
        async with self._backend._get_connection() as conn:
            cursor = await conn.execute(
                "SELECT data FROM documents WHERE collection_path = ? AND doc_id = ?",
                (self._collection_path, self._doc_id)
            )
            row = await cursor.fetchone()

            if row:
                data = json.loads(row[0])
                snapshot = DocumentSnapshot(
                    _id=self._doc_id,
                    _exists=True,
                    _data=data,
                    _path=self.path
                )
            else:
                snapshot = DocumentSnapshot(
                    _id=self._doc_id,
                    _exists=False,
                    _data=None,
                    _path=self.path
                )

            snapshot._reference = self
            return snapshot

    async def set(self, data: Dict[str, Any], merge: bool = False) -> None:
        resolved_data = _resolve_timestamps(data)
        current_time = time.time()

        async with self._backend._get_connection() as conn:
            if merge:
                # Get existing data and merge
                cursor = await conn.execute(
                    "SELECT data FROM documents WHERE collection_path = ? AND doc_id = ?",
                    (self._collection_path, self._doc_id)
                )
                row = await cursor.fetchone()

                if row:
                    existing_data = json.loads(row[0])
                    resolved_data = _deep_merge(existing_data, resolved_data)

            # Upsert
            full_id = f"{self._collection_path}/{self._doc_id}"
            await conn.execute(
                """
                INSERT INTO documents (id, collection_path, doc_id, data, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    data = excluded.data,
                    updated_at = excluded.updated_at
                """,
                (full_id, self._collection_path, self._doc_id,
                 json.dumps(resolved_data), current_time, current_time)
            )
            await conn.commit()

    async def update(self, data: Dict[str, Any]) -> None:
        resolved_data = _resolve_timestamps(data)

        async with self._backend._get_connection() as conn:
            cursor = await conn.execute(
                "SELECT data FROM documents WHERE collection_path = ? AND doc_id = ?",
                (self._collection_path, self._doc_id)
            )
            row = await cursor.fetchone()

            if not row:
                raise Exception(f"Document {self.path} does not exist")

            existing_data = json.loads(row[0])
            merged_data = _deep_merge(existing_data, resolved_data)

            await conn.execute(
                "UPDATE documents SET data = ?, updated_at = ? WHERE collection_path = ? AND doc_id = ?",
                (json.dumps(merged_data), time.time(), self._collection_path, self._doc_id)
            )
            await conn.commit()

    async def delete(self) -> None:
        async with self._backend._get_connection() as conn:
            await conn.execute(
                "DELETE FROM documents WHERE collection_path = ? AND doc_id = ?",
                (self._collection_path, self._doc_id)
            )
            await conn.commit()

    def collection(self, name: str) -> 'SQLiteCollectionReference':
        subcollection_path = f"{self.path}/{name}"
        return SQLiteCollectionReference(self._backend, subcollection_path)


class SQLiteCollectionReference(CollectionReference):
    """SQLite implementation of CollectionReference."""

    def __init__(self, backend: 'SQLiteBackend', collection_path: str):
        self._backend = backend
        self._collection_path = collection_path

    @property
    def path(self) -> str:
        return self._collection_path

    def document(self, doc_id: Optional[str] = None) -> SQLiteDocumentReference:
        if doc_id is None:
            doc_id = str(uuid.uuid4())
        return SQLiteDocumentReference(self._backend, self._collection_path, doc_id)

    async def add(self, data: Dict[str, Any]) -> Tuple[Any, SQLiteDocumentReference]:
        doc_id = str(uuid.uuid4())
        doc_ref = SQLiteDocumentReference(self._backend, self._collection_path, doc_id)
        await doc_ref.set(data)
        return (time.time(), doc_ref)

    def where(self, field: str, op: str, value: Any) -> SQLiteQuery:
        return SQLiteQuery(self._backend, self._collection_path).where(field, op, value)

    def order_by(self, field: str, direction: str = 'ASCENDING') -> SQLiteQuery:
        return SQLiteQuery(self._backend, self._collection_path).order_by(field, direction)

    def limit(self, count: int) -> SQLiteQuery:
        return SQLiteQuery(self._backend, self._collection_path).limit(count)

    async def stream(self) -> AsyncIterator[DocumentSnapshot]:
        async with self._backend._get_connection() as conn:
            cursor = await conn.execute(
                "SELECT doc_id, data FROM documents WHERE collection_path = ?",
                (self._collection_path,)
            )
            async for row in cursor:
                doc_id, data_json = row
                data = json.loads(data_json)
                snapshot = DocumentSnapshot(
                    _id=doc_id,
                    _exists=True,
                    _data=data,
                    _path=f"{self._collection_path}/{doc_id}"
                )
                snapshot._reference = SQLiteDocumentReference(
                    self._backend, self._collection_path, doc_id
                )
                yield snapshot

    async def list_documents(self) -> List[SQLiteDocumentReference]:
        refs = []
        async with self._backend._get_connection() as conn:
            cursor = await conn.execute(
                "SELECT doc_id FROM documents WHERE collection_path = ?",
                (self._collection_path,)
            )
            async for row in cursor:
                refs.append(SQLiteDocumentReference(
                    self._backend, self._collection_path, row[0]
                ))
        return refs


class SQLiteBackend(StorageBackendInterface):
    """
    SQLite storage backend implementation.

    Uses a single documents table to store all documents as JSON,
    with path-based hierarchy for subcollections.
    """

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._connection: Optional[aiosqlite.Connection] = None
        self._initialized = False

    def _get_connection(self):
        """Get a connection context manager."""
        return aiosqlite.connect(self._db_path)

    async def initialize(self) -> None:
        """Initialize the database schema."""
        # Ensure directory exists
        db_dir = os.path.dirname(self._db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        async with self._get_connection() as conn:
            # Create documents table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    collection_path TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    data TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(collection_path, doc_id)
                )
            """)

            # Create indexes
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_collection_path ON documents(collection_path)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_doc_id ON documents(doc_id)"
            )

            # Create sync metadata table for hybrid mode
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sync_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL,
                    last_synced_at REAL,
                    sync_status TEXT,
                    sync_error TEXT,
                    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
                )
            """)

            await conn.commit()

        self._initialized = True
        logger.info(f"SQLite backend initialized at {self._db_path}")

    async def close(self) -> None:
        """Close the database connection."""
        self._initialized = False
        logger.info("SQLite backend closed")

    def collection(self, name: str) -> SQLiteCollectionReference:
        return SQLiteCollectionReference(self, name)

    def server_timestamp(self) -> ServerTimestampSentinel:
        return SERVER_TIMESTAMP

    @property
    def is_available(self) -> bool:
        return self._initialized

    @property
    def backend_type(self) -> str:
        return 'sqlite'
