from __future__ import annotations

import json
import os
import sqlite3
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_CACHE_DB_PATH = "user_data/judgment_cache/hash_cache.db"


class JudgmentHashCache:
    def __init__(self, db_path: str | os.PathLike[str] | None = None):
        self.db_path = Path(db_path or os.getenv("JUDGMENT_CACHE_DB_PATH", DEFAULT_CACHE_DB_PATH))

    def get(self, *, user_id: str, document_hash: str) -> dict[str, Any] | None:
        if not user_id or not document_hash or not self.db_path.exists():
            return None
        with sqlite3.connect(self.db_path) as connection:
            self._ensure_schema(connection)
            row = connection.execute(
                """
                SELECT payload
                FROM judgment_hash_cache
                WHERE user_id = ? AND document_hash = ?
                """,
                (user_id, document_hash),
            ).fetchone()
        if not row:
            return None
        try:
            payload = json.loads(row[0])
        except (TypeError, json.JSONDecodeError):
            return None
        if not payload.get("extraction") or not payload.get("action_items"):
            return None
        return payload

    def put(self, *, user_id: str, document_hash: str, record: dict[str, Any]) -> None:
        if not user_id or not document_hash or not record.get("extraction") or not record.get("action_items"):
            return
        payload = deepcopy(record)
        now = datetime.now(UTC).isoformat()
        payload.setdefault("source_metadata", {})
        payload["source_metadata"] = {
            **(payload.get("source_metadata") or {}),
            "hash_cache_saved_at": now,
            "hash_cache_source_record_id": record.get("record_id"),
        }
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as connection:
            self._ensure_schema(connection)
            connection.execute(
                """
                INSERT INTO judgment_hash_cache
                    (user_id, document_hash, payload, source_record_id, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id, document_hash) DO UPDATE SET
                    payload = excluded.payload,
                    source_record_id = excluded.source_record_id,
                    updated_at = excluded.updated_at
                """,
                (user_id, document_hash, encoded, str(record.get("record_id") or ""), now),
            )

    @staticmethod
    def _ensure_schema(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS judgment_hash_cache (
                user_id TEXT NOT NULL,
                document_hash TEXT NOT NULL,
                payload TEXT NOT NULL,
                source_record_id TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, document_hash)
            )
            """
        )

