from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

MEMORY_TYPES = {"working", "episodic", "semantic", "procedural"}
TRUST_LEVELS = {"untrusted", "candidate", "verified"}


class MemoryError(ValueError):
    """Raised when memory lifecycle or provenance rules are violated."""


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


class MemoryStore:
    def __init__(self, path: Path | str = "artifacts/memory/memory.sqlite") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS memory_items (
                id TEXT PRIMARY KEY,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                source_id TEXT NOT NULL,
                source_kind TEXT NOT NULL,
                created_at TEXT NOT NULL,
                trust_level TEXT NOT NULL,
                validation_evidence_id TEXT,
                retention_until TEXT,
                embedding_ref TEXT,
                conflict_key TEXT,
                metadata_json TEXT NOT NULL,
                deleted_at TEXT
            );
            CREATE INDEX IF NOT EXISTS memory_type_idx ON memory_items(memory_type);
            CREATE INDEX IF NOT EXISTS memory_hash_idx ON memory_items(content_hash);
            CREATE TABLE IF NOT EXISTS memory_links (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relationship TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(source_id, target_id, relationship)
            );
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def add(
        self,
        memory_type: str,
        content: str,
        *,
        source_id: str,
        source_kind: str,
        retention_until: str | None = None,
        embedding_ref: str | None = None,
        conflict_key: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        if memory_type not in MEMORY_TYPES:
            raise MemoryError(f"unsupported memory type: {memory_type}")
        if not content or not source_id or not source_kind:
            raise MemoryError("content, source_id, and source_kind are required")
        item_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        duplicate = self.connection.execute(
            "SELECT * FROM memory_items WHERE content_hash=? AND deleted_at IS NULL",
            (item_hash,),
        ).fetchone()
        if duplicate:
            return {"status": "duplicate", **self._row(duplicate)}
        if source_kind == "model_output":
            trust_level = "untrusted"
        else:
            trust_level = "candidate"
        conflict = None
        if conflict_key:
            conflict = self.connection.execute(
                "SELECT id FROM memory_items WHERE conflict_key=? AND deleted_at IS NULL "
                "AND content_hash!=?",
                (conflict_key, item_hash),
            ).fetchone()
        item_id = hashlib.sha256(
            f"{memory_type}:{source_id}:{item_hash}".encode()
        ).hexdigest()[:24]
        self.connection.execute(
            "INSERT INTO memory_items VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                item_id,
                memory_type,
                content,
                item_hash,
                source_id,
                source_kind,
                _now(),
                trust_level,
                None,
                retention_until,
                embedding_ref,
                conflict_key,
                json.dumps(metadata or {}, sort_keys=True),
                None,
            ),
        )
        if conflict:
            self.link(item_id, conflict["id"], "conflicts_with")
            status = "conflict"
        else:
            status = "created"
        self.connection.commit()
        return {"status": status, **self.get(item_id)}

    def validate(self, item_id: str, evidence_id: str) -> dict:
        if not evidence_id:
            raise MemoryError("validation evidence ID is required")
        item = self.get(item_id)
        if item["deleted_at"]:
            raise MemoryError("cannot validate deleted memory")
        self.connection.execute(
            "UPDATE memory_items SET trust_level='verified', validation_evidence_id=? WHERE id=?",
            (evidence_id, item_id),
        )
        self.connection.commit()
        return self.get(item_id)

    def get(self, item_id: str) -> dict:
        row = self.connection.execute(
            "SELECT * FROM memory_items WHERE id=?", (item_id,)
        ).fetchone()
        if not row:
            raise MemoryError(f"memory item not found: {item_id}")
        return self._row(row)

    def inspect(self, memory_type: str | None = None) -> list[dict]:
        if memory_type and memory_type not in MEMORY_TYPES:
            raise MemoryError(f"unsupported memory type: {memory_type}")
        if memory_type:
            rows = self.connection.execute(
                "SELECT * FROM memory_items WHERE memory_type=? AND deleted_at IS NULL "
                "ORDER BY created_at, id",
                (memory_type,),
            ).fetchall()
        else:
            rows = self.connection.execute(
                "SELECT * FROM memory_items WHERE deleted_at IS NULL ORDER BY created_at, id"
            ).fetchall()
        return [self._row(row) for row in rows]

    def link(self, source_id: str, target_id: str, relationship: str) -> None:
        if not relationship:
            raise MemoryError("relationship is required")
        self.connection.execute(
            "INSERT OR IGNORE INTO memory_links VALUES (?, ?, ?, ?)",
            (source_id, target_id, relationship, _now()),
        )
        self.connection.commit()

    def delete(self, item_id: str) -> dict:
        self.get(item_id)
        self.connection.execute(
            "UPDATE memory_items SET deleted_at=? WHERE id=?", (_now(), item_id)
        )
        self.connection.commit()
        return self.get(item_id)

    def purge_expired(self, now: str | None = None) -> int:
        cutoff = now or _now()
        cursor = self.connection.execute(
            "UPDATE memory_items SET deleted_at=? WHERE deleted_at IS NULL "
            "AND retention_until IS NOT NULL AND retention_until <= ?",
            (_now(), cutoff),
        )
        self.connection.commit()
        return cursor.rowcount

    @staticmethod
    def _row(row: sqlite3.Row) -> dict:
        value = dict(row)
        value["metadata"] = json.loads(value.pop("metadata_json"))
        return value


def self_test() -> dict:
    store = MemoryStore(":memory:")
    try:
        memory_types = ["working", "episodic", "semantic", "procedural"]
        created = [
            store.add(
                memory_type,
                f"{memory_type} content",
                source_id=f"source-{memory_type}",
                source_kind="human_note",
                embedding_ref=f"embedding:{memory_type}",
            )
            for memory_type in memory_types
        ]
        candidate = store.add(
            "semantic",
            "model proposed fact",
            source_id="model-1",
            source_kind="model_output",
            conflict_key="fact-1",
        )
        untrusted_before_validation = candidate["trust_level"] == "untrusted"
        verified = store.validate(candidate["id"], "verification:exact:1")
        duplicate = store.add(
            "semantic", "model proposed fact", source_id="model-2", source_kind="model_output"
        )
        conflict = store.add(
            "semantic",
            "conflicting fact",
            source_id="human-2",
            source_kind="human_note",
            conflict_key="fact-1",
        )
        store.link(created[0]["id"], verified["id"], "supports")
        expired = store.add(
            "episodic",
            "temporary event",
            source_id="event-1",
            source_kind="human_note",
            retention_until="2000-01-01T00:00:00+00:00",
        )
        purged = store.purge_expired("2026-09-14T00:00:00+00:00")
        round_trip = store.get(created[0]["id"])
        return {
            "ok": (
                len(created) == 4
                and untrusted_before_validation
                and verified["trust_level"] == "verified"
                and duplicate["status"] == "duplicate"
                and conflict["status"] == "conflict"
                and purged == 1
                and round_trip["embedding_ref"] == "embedding:working"
                and expired["id"] != verified["id"]
            ),
            "memory_types": memory_types,
            "untrusted_model_candidate": untrusted_before_validation,
            "verified_with_evidence": verified["validation_evidence_id"],
            "duplicate_status": duplicate["status"],
            "conflict_status": conflict["status"],
            "expired_tombstones": purged,
            "provenance_round_trip": round_trip["source_id"],
        }
    finally:
        store.close()
