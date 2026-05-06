# storage/__init__.py
"""
Storage Abstraction Layer for Artha AI

Provides automatic fallback from Firebase/Firestore to local SQLite database,
ensuring the system remains functional even when Firebase is unavailable.

Usage:
    from storage import get_storage, init_storage, close_storage

    # Initialize at startup
    await init_storage()

    # Get storage instance
    storage = get_storage()

    # Use like Firestore
    doc_ref = storage.collection('users').document('user123')
    await doc_ref.set({'name': 'John'})
    doc = await doc_ref.get()

    # Cleanup at shutdown
    await close_storage()
"""

from .base import (
    StorageBackendInterface,
    CollectionReference,
    DocumentReference,
    DocumentSnapshot,
    Query,
    ServerTimestampSentinel,
    SERVER_TIMESTAMP,
)

from .storage_config import (
    get_storage_backend,
    init_storage,
    close_storage,
    get_storage,
    StorageConfig,
)

__all__ = [
    # Interfaces
    'StorageBackendInterface',
    'CollectionReference',
    'DocumentReference',
    'DocumentSnapshot',
    'Query',
    'ServerTimestampSentinel',
    'SERVER_TIMESTAMP',
    # Factory and lifecycle
    'get_storage_backend',
    'init_storage',
    'close_storage',
    'get_storage',
    'StorageConfig',
]
