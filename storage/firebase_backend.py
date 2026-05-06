# storage/firebase_backend.py
"""
Firebase/Firestore backend wrapper.

Wraps Firestore classes to match the abstract storage interface,
maintaining full compatibility with current behavior.
"""

import asyncio
import logging
import uuid
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

logger = logging.getLogger(__name__)


class FirebaseQuery(Query):
    """Firebase implementation of Query interface."""

    def __init__(self, firestore_query):
        self._query = firestore_query

    def where(self, field: str, op: str, value: Any) -> 'FirebaseQuery':
        # Map operators to Firestore format
        op_map = {
            '==': '==',
            '!=': '!=',
            '<': '<',
            '<=': '<=',
            '>': '>',
            '>=': '>=',
            'in': 'in',
            'array-contains': 'array_contains',
        }
        firestore_op = op_map.get(op, op)
        return FirebaseQuery(self._query.where(field, firestore_op, value))

    def order_by(self, field: str, direction: str = 'ASCENDING') -> 'FirebaseQuery':
        from google.cloud.firestore_v1 import Query as FSQuery
        dir_map = {
            'ASCENDING': FSQuery.ASCENDING,
            'DESCENDING': FSQuery.DESCENDING,
        }
        return FirebaseQuery(
            self._query.order_by(field, direction=dir_map.get(direction, FSQuery.ASCENDING))
        )

    def limit(self, count: int) -> 'FirebaseQuery':
        return FirebaseQuery(self._query.limit(count))

    async def get(self) -> List[DocumentSnapshot]:
        docs = await asyncio.to_thread(self._query.get)
        return [self._wrap_snapshot(doc) for doc in docs]

    async def stream(self) -> AsyncIterator[DocumentSnapshot]:
        docs = await asyncio.to_thread(lambda: list(self._query.stream()))
        for doc in docs:
            yield self._wrap_snapshot(doc)

    def _wrap_snapshot(self, firestore_snapshot) -> DocumentSnapshot:
        snapshot = DocumentSnapshot(
            _id=firestore_snapshot.id,
            _exists=firestore_snapshot.exists,
            _data=firestore_snapshot.to_dict() if firestore_snapshot.exists else None,
            _path=firestore_snapshot.reference.path
        )
        snapshot._reference = FirebaseDocumentReference(firestore_snapshot.reference)
        return snapshot


class FirebaseDocumentReference(DocumentReference):
    """Firebase implementation of DocumentReference."""

    def __init__(self, firestore_ref):
        self._ref = firestore_ref

    @property
    def id(self) -> str:
        return self._ref.id

    @property
    def path(self) -> str:
        return self._ref.path

    async def get(self) -> DocumentSnapshot:
        doc = await asyncio.to_thread(self._ref.get)
        snapshot = DocumentSnapshot(
            _id=doc.id,
            _exists=doc.exists,
            _data=doc.to_dict() if doc.exists else None,
            _path=self._ref.path
        )
        snapshot._reference = self
        return snapshot

    async def set(self, data: Dict[str, Any], merge: bool = False) -> None:
        # Convert SERVER_TIMESTAMP sentinels
        from firebase_admin import firestore as fs
        resolved_data = self._resolve_timestamps(data, fs.SERVER_TIMESTAMP)
        await asyncio.to_thread(self._ref.set, resolved_data, merge=merge)

    async def update(self, data: Dict[str, Any]) -> None:
        from firebase_admin import firestore as fs
        resolved_data = self._resolve_timestamps(data, fs.SERVER_TIMESTAMP)
        await asyncio.to_thread(self._ref.update, resolved_data)

    async def delete(self) -> None:
        await asyncio.to_thread(self._ref.delete)

    def collection(self, name: str) -> 'FirebaseCollectionReference':
        return FirebaseCollectionReference(self._ref.collection(name))

    def _resolve_timestamps(self, data: Dict[str, Any], firestore_timestamp) -> Dict[str, Any]:
        """Convert SERVER_TIMESTAMP sentinels to Firestore timestamps."""
        resolved = {}
        for key, value in data.items():
            if isinstance(value, ServerTimestampSentinel):
                resolved[key] = firestore_timestamp
            elif isinstance(value, dict):
                if value.get("__sentinel__") == "SERVER_TIMESTAMP":
                    resolved[key] = firestore_timestamp
                else:
                    resolved[key] = self._resolve_timestamps(value, firestore_timestamp)
            elif isinstance(value, list):
                resolved[key] = [
                    self._resolve_timestamps(item, firestore_timestamp)
                    if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                resolved[key] = value
        return resolved


class FirebaseCollectionReference(CollectionReference):
    """Firebase implementation of CollectionReference."""

    def __init__(self, firestore_collection):
        self._collection = firestore_collection

    @property
    def path(self) -> str:
        return self._collection.path if hasattr(self._collection, 'path') else str(self._collection.id)

    def document(self, doc_id: Optional[str] = None) -> FirebaseDocumentReference:
        if doc_id is None:
            doc_id = str(uuid.uuid4())
        return FirebaseDocumentReference(self._collection.document(doc_id))

    async def add(self, data: Dict[str, Any]) -> Tuple[Any, FirebaseDocumentReference]:
        from firebase_admin import firestore as fs
        doc_ref = FirebaseDocumentReference(self._collection.document())
        resolved_data = doc_ref._resolve_timestamps(data, fs.SERVER_TIMESTAMP)
        result = await asyncio.to_thread(self._collection.add, resolved_data)
        # Result is (timestamp, doc_ref)
        return (result[0], FirebaseDocumentReference(result[1]))

    def where(self, field: str, op: str, value: Any) -> FirebaseQuery:
        op_map = {
            '==': '==',
            '!=': '!=',
            '<': '<',
            '<=': '<=',
            '>': '>',
            '>=': '>=',
            'in': 'in',
            'array-contains': 'array_contains',
        }
        firestore_op = op_map.get(op, op)
        return FirebaseQuery(self._collection.where(field, firestore_op, value))

    def order_by(self, field: str, direction: str = 'ASCENDING') -> FirebaseQuery:
        from google.cloud.firestore_v1 import Query as FSQuery
        dir_map = {
            'ASCENDING': FSQuery.ASCENDING,
            'DESCENDING': FSQuery.DESCENDING,
        }
        return FirebaseQuery(
            self._collection.order_by(field, direction=dir_map.get(direction, FSQuery.ASCENDING))
        )

    def limit(self, count: int) -> FirebaseQuery:
        return FirebaseQuery(self._collection.limit(count))

    async def stream(self) -> AsyncIterator[DocumentSnapshot]:
        docs = await asyncio.to_thread(lambda: list(self._collection.stream()))
        for doc in docs:
            snapshot = DocumentSnapshot(
                _id=doc.id,
                _exists=doc.exists,
                _data=doc.to_dict() if doc.exists else None,
                _path=doc.reference.path
            )
            snapshot._reference = FirebaseDocumentReference(doc.reference)
            yield snapshot

    async def list_documents(self) -> List[FirebaseDocumentReference]:
        docs = await asyncio.to_thread(lambda: list(self._collection.list_documents()))
        return [FirebaseDocumentReference(doc) for doc in docs]


class FirebaseBackend(StorageBackendInterface):
    """
    Firebase/Firestore storage backend implementation.

    Wraps the existing Firestore client to provide a unified interface.
    """

    def __init__(self, firestore_client=None, credentials_path: Optional[str] = None,
                 storage_bucket: Optional[str] = None):
        self._db = firestore_client
        self._credentials_path = credentials_path
        self._storage_bucket = storage_bucket
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize Firebase if not already done."""
        if self._db is not None:
            self._initialized = True
            logger.info("Firebase backend initialized with existing client")
            return

        try:
            import firebase_admin
            from firebase_admin import credentials, firestore

            if not firebase_admin._apps:
                if self._credentials_path:
                    cred = credentials.Certificate(self._credentials_path)
                    init_options = {}
                    if self._storage_bucket:
                        init_options['storageBucket'] = self._storage_bucket
                    firebase_admin.initialize_app(cred, init_options)
                else:
                    # Try to use default credentials
                    firebase_admin.initialize_app()

            self._db = firestore.client()
            self._initialized = True
            logger.info("Firebase backend initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Firebase backend: {e}")
            self._initialized = False
            raise

    async def close(self) -> None:
        """Close Firebase connection."""
        # Firebase Admin SDK doesn't require explicit closing
        self._initialized = False
        logger.info("Firebase backend closed")

    def collection(self, name: str) -> FirebaseCollectionReference:
        if not self._db:
            raise RuntimeError("Firebase backend not initialized")
        return FirebaseCollectionReference(self._db.collection(name))

    def server_timestamp(self) -> ServerTimestampSentinel:
        return SERVER_TIMESTAMP

    @property
    def is_available(self) -> bool:
        return self._initialized and self._db is not None

    @property
    def backend_type(self) -> str:
        return 'firebase'

    @property
    def native_client(self):
        """Access the underlying Firestore client for advanced operations."""
        return self._db
