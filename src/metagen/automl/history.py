from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from metagen.utils.io import ensure_dir


@dataclass(frozen=True)
class RunRecord:
    """Represents a stored AutoML run."""

    spec_hash: str
    blueprint: dict[str, Any]
    metrics: dict[str, Any]
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "spec_hash": self.spec_hash,
            "blueprint": dict(self.blueprint),
            "metrics": dict(self.metrics),
            "timestamp": self.timestamp,
        }


class HistoryDatabase:
    """SQLite-backed history store for AutoML runs."""

    def __init__(self, root_dir: Path | None = None, db_name: str = "history.db") -> None:
        self.root_dir = ensure_dir(root_dir or Path.cwd() / ".metagen")
        self.db_path = self.root_dir / db_name
        self._init_db()

    def save_run(
        self,
        spec_hash: str,
        blueprint: dict[str, Any],
        metrics: dict[str, Any],
        *,
        timestamp: str | None = None,
    ) -> None:
        """Persist a run to the database."""
        if not spec_hash:
            raise ValueError("spec_hash must be provided")

        resolved_ts = timestamp or datetime.now(UTC).isoformat()
        blueprint_json = json.dumps(blueprint)
        metrics_json = json.dumps(metrics)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (spec_hash, blueprint_json, metrics_json, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (spec_hash, blueprint_json, metrics_json, resolved_ts),
            )

    def load_runs(self, spec_hash: str | None = None, limit: int | None = None) -> list[RunRecord]:
        """Load runs from the database, optionally filtered by spec hash."""
        if limit is not None and limit < 1:
            raise ValueError("limit must be >= 1")

        query = "SELECT spec_hash, blueprint_json, metrics_json, timestamp FROM runs"
        params: list[Any] = []
        if spec_hash is not None:
            query += " WHERE spec_hash = ?"
            params.append(spec_hash)
        query += " ORDER BY timestamp DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        records: list[RunRecord] = []
        for row in rows:
            spec_hash_value, blueprint_json, metrics_json, timestamp_value = row
            records.append(
                RunRecord(
                    spec_hash=spec_hash_value,
                    blueprint=json.loads(blueprint_json),
                    metrics=json.loads(metrics_json),
                    timestamp=timestamp_value,
                )
            )
        return records

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    spec_hash TEXT NOT NULL,
                    blueprint_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
                """
            )
