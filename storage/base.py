# storage/base.py
"""
Abstract interfaces defining the storage contract.

These interfaces mirror the Firestore API to enable seamless switching
between Firebase and SQLite backends.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, AsyncIterator, Union, Tuple
from dataclasses import dataclass
import time


class ServerTimestampSentinel:
    """Sentinel value representing server timestamp, resolved on write."""
    def __repr__(self):
        return "SERVER_TIMESTAMP"

    def to_dict(self) -> Dict[str, str]:
        """Convert to dict representation for JSON storage."""
        return {"__sentinel__": "SERVER_TIMESTAMP"}


# Global sentinel instance
SERVER_TIMESTAMP = ServerTimestampSentinel()


@dataclass
class DocumentSnapshot:
    """
    Unified document snapshot with exists check and data access.
    Mirrors Firestore's DocumentSnapshot.
    """
    _id: str
    _exists: bool
    _data: Optional[Dict[str, Any]]
    _path: str = ""

    @property
    def id(self) -> str:
        """Document ID."""
        return self._id

    @property
    def exists(self) -> bool:
        """Whether the document exists."""
        return self._exists

    @property
    def reference(self) -> 'DocumentReference':
        """Reference to this document (set by backend)."""
        # This is set by the backend implementation
        return getattr(self, '_reference', None)

    def to_dict(self) -> Optional[Dict[str, Any]]:
        """Get document data as dictionary."""
        return self._data if self._exists else None

    def get(self, field: str, default: Any = None) -> Any:
        """Get a specific field from the document."""
        if not self._exists or self._data is None:
            return default
        return self._data.get(field, default)


class Query(ABC):
    """
    Query builder with filters, ordering, and limits.
    Mirrors Firestore's Query interface.
    """

    @abstractmethod
    def where(self, field: str, op: str, value: Any) -> 'Query':
        """
        Add a filter condition.

        Args:
            field: Field name to filter on
            op: Operator ('==', '!=', '<', '<=', '>', '>=', 'in', 'array-contains')
            value: Value to compare against

        Returns:
            New Query with filter applied
        """
        pass

    @abstractmethod
    def order_by(self, field: str, direction: str = 'ASCENDING') -> 'Query':
        """
        Order results by field.

        Args:
            field: Field name to order by
            direction: 'ASCENDING' or 'DESCENDING'

        Returns:
            New Query with ordering applied
        """
        pass

    @abstractmethod
    def limit(self, count: int) -> 'Query':
        """
        Limit number of results.

        Args:
            count: Maximum number of documents to return

        Returns:
            New Query with limit applied
        """
        pass

    @abstractmethod
    async def get(self) -> List[DocumentSnapshot]:
        """
        Execute the query and return results.

        Returns:
            List of DocumentSnapshots matching the query
        """
        pass

    @abstractmethod
    async def stream(self) -> AsyncIterator[DocumentSnapshot]:
        """
        Stream query results.

        Yields:
            DocumentSnapshots matching the query
        """
        pass


class DocumentReference(ABC):
    """
    Reference to a single document.
    Mirrors Firestore's DocumentReference.
    """

    @property
    @abstractmethod
    def id(self) -> str:
        """Document ID."""
        pass

    @property
    @abstractmethod
    def path(self) -> str:
        """Full document path."""
        pass

    @abstractmethod
    async def get(self) -> DocumentSnapshot:
        """
        Get the document.

        Returns:
            DocumentSnapshot with exists check and data
        """
        pass

    @abstractmethod
    async def set(self, data: Dict[str, Any], merge: bool = False) -> None:
        """
        Set document data.

        Args:
            data: Document data to set
            merge: If True, merge with existing data; if False, overwrite
        """
        pass

    @abstractmethod
    async def update(self, data: Dict[str, Any]) -> None:
        """
        Update specific fields in the document.

        Args:
            data: Fields to update

        Raises:
            Exception if document doesn't exist
        """
        pass

    @abstractmethod
    async def delete(self) -> None:
        """Delete the document."""
        pass

    @abstractmethod
    def collection(self, name: str) -> 'CollectionReference':
        """
        Get a subcollection reference.

        Args:
            name: Subcollection name

        Returns:
            CollectionReference for the subcollection
        """
        pass


class CollectionReference(ABC):
    """
    Reference to a collection of documents.
    Mirrors Firestore's CollectionReference.
    """

    @property
    @abstractmethod
    def path(self) -> str:
        """Full collection path."""
        pass

    @abstractmethod
    def document(self, doc_id: Optional[str] = None) -> DocumentReference:
        """
        Get a document reference within this collection.

        Args:
            doc_id: Document ID. If None, auto-generate one.

        Returns:
            DocumentReference for the document
        """
        pass

    @abstractmethod
    async def add(self, data: Dict[str, Any]) -> Tuple[Any, DocumentReference]:
        """
        Add a new document with auto-generated ID.

        Args:
            data: Document data

        Returns:
            Tuple of (write_time, DocumentReference)
        """
        pass

    @abstractmethod
    def where(self, field: str, op: str, value: Any) -> Query:
        """
        Create a query with a filter condition.

        Args:
            field: Field name to filter on
            op: Operator ('==', '!=', '<', '<=', '>', '>=', 'in', 'array-contains')
            value: Value to compare against

        Returns:
            Query with filter applied
        """
        pass

    @abstractmethod
    def order_by(self, field: str, direction: str = 'ASCENDING') -> Query:
        """
        Create a query ordered by field.

        Args:
            field: Field name to order by
            direction: 'ASCENDING' or 'DESCENDING'

        Returns:
            Query with ordering applied
        """
        pass

    @abstractmethod
    def limit(self, count: int) -> Query:
        """
        Create a query with result limit.

        Args:
            count: Maximum number of documents

        Returns:
            Query with limit applied
        """
        pass

    @abstractmethod
    async def stream(self) -> AsyncIterator[DocumentSnapshot]:
        """
        Stream all documents in the collection.

        Yields:
            DocumentSnapshots for each document
        """
        pass

    @abstractmethod
    async def list_documents(self) -> List[DocumentReference]:
        """
        List all document references in the collection.

        Returns:
            List of DocumentReferences
        """
        pass


class StorageBackendInterface(ABC):
    """
    Main storage backend interface.
    Provides collection access and lifecycle management.
    """

    @abstractmethod
    def collection(self, name: str) -> CollectionReference:
        """
        Get a collection reference.

        Args:
            name: Collection name or path

        Returns:
            CollectionReference for the collection
        """
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the storage backend.
        Called at application startup.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        Close the storage backend.
        Called at application shutdown.
        """
        pass

    @abstractmethod
    def server_timestamp(self) -> ServerTimestampSentinel:
        """
        Get a server timestamp sentinel.
        Resolved to actual timestamp on write.

        Returns:
            ServerTimestampSentinel instance
        """
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the backend is available and operational.

        Returns:
            True if backend is ready for operations
        """
        pass

    @property
    @abstractmethod
    def backend_type(self) -> str:
        """
        Get the backend type identifier.

        Returns:
            'firebase', 'sqlite', or 'hybrid'
        """
        pass
