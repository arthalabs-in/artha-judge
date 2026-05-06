# storage/storage_config.py
"""
Storage configuration and factory.

Provides factory functions and configuration for creating storage backends
based on environment variables.
"""

import os
import logging
from typing import Optional
from dataclasses import dataclass

from .base import StorageBackendInterface, SERVER_TIMESTAMP
from .sqlite_backend import SQLiteBackend
from .firebase_backend import FirebaseBackend
from .hybrid_backend import HybridBackend
from .sync_manager import SyncManager

logger = logging.getLogger(__name__)

# Global storage instance
_storage_instance: Optional[StorageBackendInterface] = None
_sync_manager: Optional[SyncManager] = None


@dataclass
class StorageConfig:
    """Configuration for storage backends."""
    backend_type: str = 'firebase'  # 'firebase', 'sqlite', or 'hybrid'
    sqlite_db_path: str = ''
    firebase_credentials_path: Optional[str] = None
    firebase_storage_bucket: Optional[str] = None
    sync_interval: float = 60.0  # For hybrid mode

    @classmethod
    def from_env(cls) -> 'StorageConfig':
        """Create configuration from environment variables."""
        # Expand user paths
        default_sqlite_path = os.path.expanduser('~/ai_app_data/artha.db')

        return cls(
            backend_type=os.getenv('STORAGE_BACKEND', 'firebase').lower(),
            sqlite_db_path=os.getenv('SQLITE_DB_PATH', default_sqlite_path),
            firebase_credentials_path=os.getenv('FIREBASE_ADMIN_SDK_PATH'),
            firebase_storage_bucket=os.getenv('FIREBASE_STORAGE_BUCKET'),
            sync_interval=float(os.getenv('SYNC_INTERVAL', '60')),
        )


def get_storage_backend(config: Optional[StorageConfig] = None) -> StorageBackendInterface:
    """
    Factory function to create the appropriate storage backend.

    Args:
        config: Storage configuration. If None, reads from environment.

    Returns:
        StorageBackendInterface implementation
    """
    if config is None:
        config = StorageConfig.from_env()

    logger.info(f"Creating storage backend: {config.backend_type}")

    if config.backend_type == 'sqlite':
        return SQLiteBackend(config.sqlite_db_path)

    elif config.backend_type == 'firebase':
        return FirebaseBackend(
            credentials_path=config.firebase_credentials_path,
            storage_bucket=config.firebase_storage_bucket
        )

    elif config.backend_type == 'hybrid':
        sqlite_backend = SQLiteBackend(config.sqlite_db_path)
        firebase_backend = FirebaseBackend(
            credentials_path=config.firebase_credentials_path,
            storage_bucket=config.firebase_storage_bucket
        )
        return HybridBackend(sqlite_backend, firebase_backend)

    else:
        raise ValueError(f"Unknown storage backend type: {config.backend_type}")


async def init_storage(config: Optional[StorageConfig] = None) -> StorageBackendInterface:
    """
    Initialize the global storage instance.

    Should be called during application startup.

    Args:
        config: Storage configuration. If None, reads from environment.

    Returns:
        Initialized StorageBackendInterface
    """
    global _storage_instance, _sync_manager

    if _storage_instance is not None:
        logger.warning("Storage already initialized, returning existing instance")
        return _storage_instance

    if config is None:
        config = StorageConfig.from_env()

    _storage_instance = get_storage_backend(config)
    await _storage_instance.initialize()

    # Start sync manager for hybrid mode
    if config.backend_type == 'hybrid' and isinstance(_storage_instance, HybridBackend):
        _sync_manager = SyncManager(
            sqlite_path=config.sqlite_db_path,
            firebase_backend=_storage_instance.firebase_backend,
            sync_interval=config.sync_interval
        )
        await _sync_manager.start()

    logger.info(f"Storage initialized: {_storage_instance.backend_type}")
    return _storage_instance


async def close_storage() -> None:
    """
    Close the global storage instance.

    Should be called during application shutdown.
    """
    global _storage_instance, _sync_manager

    if _sync_manager is not None:
        await _sync_manager.stop()
        _sync_manager = None

    if _storage_instance is not None:
        await _storage_instance.close()
        _storage_instance = None

    logger.info("Storage closed")


def get_storage() -> StorageBackendInterface:
    """
    Get the global storage instance.

    Returns:
        StorageBackendInterface instance

    Raises:
        RuntimeError: If storage has not been initialized
    """
    if _storage_instance is None:
        raise RuntimeError(
            "Storage not initialized. Call init_storage() first, "
            "or use get_storage_backend() for a standalone instance."
        )
    return _storage_instance


def get_sync_manager() -> Optional[SyncManager]:
    """
    Get the global sync manager (if running in hybrid mode).

    Returns:
        SyncManager instance or None
    """
    return _sync_manager


def server_timestamp():
    """
    Get a server timestamp sentinel.

    Convenience function that doesn't require a storage instance.
    """
    return SERVER_TIMESTAMP
