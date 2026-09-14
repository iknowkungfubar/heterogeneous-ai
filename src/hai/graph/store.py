from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

TRUST_LEVELS = {"untrusted", "candidate", "verified"}
TRUST_RANK = {"untrusted": 0, "candidate": 1, "verified": 2}


class GraphError(ValueError):
    """Raised when graph data or a graph operation is invalid."""


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _trust_level(source_kind: str) -> str:
    return "untrusted" if source_kind == "model_output" else "candidate"


class GraphStore:
    """A small SQLite graph with explicit provenance and conservative traversal."""

    def __init__(self, path: Path | str = "artifacts/graphs/knowledge.sqlite") -> None:
        self.db_path = str(path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                source_id TEXT NOT NULL,
                source_kind TEXT NOT NULL,
                created_at TEXT NOT NULL,
                trust_level TEXT NOT NULL,
                validation_evidence_id TEXT,
                metadata_json TEXT NOT NULL,
                deleted_at TEXT
            );
            CREATE INDEX IF NOT EXISTS entity_label_idx ON entities(label);
            CREATE INDEX IF NOT EXISTS entity_trust_idx ON entities(trust_level);
            CREATE TABLE IF NOT EXISTS relations (
                id TEXT PRIMARY KEY,
                source_entity_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                target_entity_id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                source_kind TEXT NOT NULL,
                created_at TEXT NOT NULL,
                trust_level TEXT NOT NULL,
                validation_evidence_id TEXT,
                conflict_key TEXT,
                status TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                deleted_at TEXT,
                FOREIGN KEY(source_entity_id) REFERENCES entities(id),
                FOREIGN KEY(target_entity_id) REFERENCES entities(id)
            );
            CREATE INDEX IF NOT EXISTS relation_source_idx
                ON relations(source_entity_id, relation_type, trust_level);
            CREATE INDEX IF NOT EXISTS relation_target_idx
                ON relations(target_entity_id, relation_type, trust_level);
            CREATE INDEX IF NOT EXISTS relation_conflict_idx ON relations(conflict_key);
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    @staticmethod
    def _validate_text(*values: str) -> None:
        if any(not value or not value.strip() for value in values):
            raise GraphError("labels, types, IDs, and relation names are required")

    def add_entity(
        self,
        label: str,
        entity_type: str,
        *,
        source_id: str,
        source_kind: str,
        metadata: dict | None = None,
    ) -> dict:
        self._validate_text(label, entity_type, source_id, source_kind)
        entity_id = hashlib.sha256(
            f"{entity_type}:{label.casefold()}".encode()
        ).hexdigest()[:24]
        existing = self.connection.execute(
            "SELECT * FROM entities WHERE id=? AND deleted_at IS NULL", (entity_id,)
        ).fetchone()
        if existing:
            return {**self._entity_row(existing), "status": "duplicate"}
        self.connection.execute(
            "INSERT INTO entities VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                entity_id,
                label,
                entity_type,
                source_id,
                source_kind,
                _now(),
                _trust_level(source_kind),
                None,
                json.dumps(metadata or {}, sort_keys=True),
                None,
            ),
        )
        self.connection.commit()
        return {"status": "created", **self.get_entity(entity_id)}

    def add_relation(
        self,
        source_entity_id: str,
        relation_type: str,
        target_entity_id: str,
        *,
        source_id: str,
        source_kind: str,
        conflict_key: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        self._validate_text(
            source_entity_id, relation_type, target_entity_id, source_id, source_kind
        )
        self.get_entity(source_entity_id)
        self.get_entity(target_entity_id)
        relation_id = hashlib.sha256(
            f"{source_entity_id}:{relation_type}:{target_entity_id}:{source_id}".encode()
        ).hexdigest()[:24]
        existing = self.connection.execute(
            "SELECT * FROM relations WHERE id=? AND deleted_at IS NULL", (relation_id,)
        ).fetchone()
        if existing:
            return {**self._relation_row(existing), "status": "duplicate"}
        conflict = None
        if conflict_key:
            conflict = self.connection.execute(
                "SELECT id FROM relations WHERE conflict_key=? AND target_entity_id!=? "
                "AND deleted_at IS NULL ORDER BY id LIMIT 1",
                (conflict_key, target_entity_id),
            ).fetchone()
        status = "conflict" if conflict else "active"
        self.connection.execute(
            "INSERT INTO relations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                relation_id,
                source_entity_id,
                relation_type,
                target_entity_id,
                source_id,
                source_kind,
                _now(),
                _trust_level(source_kind),
                None,
                conflict_key,
                status,
                json.dumps(metadata or {}, sort_keys=True),
                None,
            ),
        )
        self.connection.commit()
        return {"status": status, **self.get_relation(relation_id)}

    def validate_entity(self, entity_id: str, evidence_id: str) -> dict:
        self._validate_text(entity_id, evidence_id)
        entity = self.get_entity(entity_id)
        if entity["deleted_at"]:
            raise GraphError("cannot validate deleted entity")
        self.connection.execute(
            "UPDATE entities SET trust_level='verified', validation_evidence_id=? WHERE id=?",
            (evidence_id, entity_id),
        )
        self.connection.commit()
        return self.get_entity(entity_id)

    def validate_relation(self, relation_id: str, evidence_id: str) -> dict:
        self._validate_text(relation_id, evidence_id)
        relation = self.get_relation(relation_id)
        if relation["deleted_at"]:
            raise GraphError("cannot validate deleted relation")
        self.connection.execute(
            "UPDATE relations SET trust_level='verified', validation_evidence_id=? WHERE id=?",
            (evidence_id, relation_id),
        )
        self.connection.commit()
        return self.get_relation(relation_id)

    def get_entity(self, entity_id: str) -> dict:
        row = self.connection.execute("SELECT * FROM entities WHERE id=?", (entity_id,)).fetchone()
        if not row:
            raise GraphError(f"entity not found: {entity_id}")
        return self._entity_row(row)

    def get_relation(self, relation_id: str) -> dict:
        row = self.connection.execute(
            "SELECT * FROM relations WHERE id=?", (relation_id,)
        ).fetchone()
        if not row:
            raise GraphError(f"relation not found: {relation_id}")
        return self._relation_row(row)

    def find_entities(self, label: str) -> list[dict]:
        rows = self.connection.execute(
            "SELECT * FROM entities WHERE label=? AND deleted_at IS NULL ORDER BY id", (label,)
        ).fetchall()
        return [self._entity_row(row) for row in rows]

    def neighbors(
        self,
        entity_id: str,
        *,
        relation_type: str | None = None,
        min_trust: str = "verified",
    ) -> list[dict]:
        self._check_trust(min_trust)
        self.get_entity(entity_id)
        query = (
            "SELECT r.*, e.label AS target_label FROM relations r "
            "JOIN entities e ON e.id=r.target_entity_id "
            "WHERE r.source_entity_id=? AND r.deleted_at IS NULL AND e.deleted_at IS NULL "
            "AND r.trust_level IN ({})"
        ).format(",".join("?" for _ in range(len(TRUST_LEVELS) - TRUST_RANK[min_trust])))
        args: list[object] = [
            entity_id,
            *sorted(TRUST_RANK, key=TRUST_RANK.get)[TRUST_RANK[min_trust] :],
        ]
        if relation_type:
            query += " AND r.relation_type=?"
            args.append(relation_type)
        query += " ORDER BY r.relation_type, r.target_entity_id, r.id"
        return [
            self._relation_row(row, include_target=True)
            for row in self.connection.execute(query, args)
        ]

    def path(
        self,
        start_entity_id: str,
        goal_entity_id: str,
        *,
        max_hops: int = 4,
        min_trust: str = "verified",
    ) -> dict | None:
        self._check_trust(min_trust)
        if max_hops < 0:
            raise GraphError("max_hops must be non-negative")
        self.get_entity(start_entity_id)
        self.get_entity(goal_entity_id)
        queue = deque([(start_entity_id, [start_entity_id], [])])
        visited = {start_entity_id}
        while queue:
            current, entities, relations = queue.popleft()
            if current == goal_entity_id:
                return {"entities": entities, "relations": relations, "hops": len(relations)}
            if len(relations) >= max_hops:
                continue
            for relation in self.neighbors(current, min_trust=min_trust):
                target = relation["target_entity_id"]
                if target in visited:
                    continue
                visited.add(target)
                queue.append((target, [*entities, target], [*relations, relation]))
        return None

    def import_fixture(self, path: Path | str) -> dict:
        imported = {"entities": 0, "relations": 0, "conflicts": 0}
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise GraphError(f"invalid fixture JSON at line {line_number}") from exc
                if record.get("kind") == "entity":
                    result = self.add_entity(
                        record["label"], record.get("entity_type", "concept"),
                        source_id=record["source_id"], source_kind=record["source_kind"],
                        metadata=record.get("metadata"),
                    )
                    imported["entities"] += result["status"] == "created"
                elif record.get("kind") == "relation":
                    source_entity_id = record.get("source_entity_id")
                    target_entity_id = record.get("target_entity_id")
                    if not source_entity_id and record.get("source_label"):
                        matches = self.find_entities(record["source_label"])
                        if len(matches) != 1:
                            raise GraphError(
                                f"fixture source label is not unique at line {line_number}"
                            )
                        source_entity_id = matches[0]["id"]
                    if not target_entity_id and record.get("target_label"):
                        matches = self.find_entities(record["target_label"])
                        if len(matches) != 1:
                            raise GraphError(
                                f"fixture target label is not unique at line {line_number}"
                            )
                        target_entity_id = matches[0]["id"]
                    result = self.add_relation(
                        source_entity_id,
                        record["relation_type"],
                        target_entity_id,
                        source_id=record["source_id"], source_kind=record["source_kind"],
                        conflict_key=record.get("conflict_key"), metadata=record.get("metadata"),
                    )
                    imported["relations"] += result["status"] in {"active", "conflict"}
                    imported["conflicts"] += result["status"] == "conflict"
                else:
                    raise GraphError(f"unsupported fixture record at line {line_number}")
        return imported

    def ingest_memory(self, item: dict) -> dict:
        """Create an entity from a MemoryStore item without upgrading its trust."""
        result = self.add_entity(
            item["content"], item["memory_type"],
            source_id=item["source_id"], source_kind=item["source_kind"],
            metadata={"memory_id": item["id"], "content_hash": item["content_hash"]},
        )
        if item.get("trust_level") == "verified" and result["trust_level"] != "verified":
            result = self.validate_entity(
                result["id"], item.get("validation_evidence_id") or "memory-validation"
            )
        return result

    @staticmethod
    def _check_trust(min_trust: str) -> None:
        if min_trust not in TRUST_LEVELS:
            raise GraphError(f"unsupported trust level: {min_trust}")

    @staticmethod
    def _entity_row(row: sqlite3.Row) -> dict:
        value = dict(row)
        value["metadata"] = json.loads(value.pop("metadata_json"))
        return value

    @staticmethod
    def _relation_row(row: sqlite3.Row, include_target: bool = False) -> dict:
        value = dict(row)
        if not include_target:
            value.pop("target_label", None)
        value["metadata"] = json.loads(value.pop("metadata_json"))
        return value


def self_test() -> dict:
    store = GraphStore(":memory:")
    try:
        alice = store.add_entity("Alice", "person", source_id="note-1", source_kind="human_note")
        bob = store.add_entity("Bob", "person", source_id="note-1", source_kind="human_note")
        carol = store.add_entity("Carol", "person", source_id="note-2", source_kind="human_note")
        for entity in (alice, bob, carol):
            store.validate_entity(entity["id"], f"evidence-{entity['label'].lower()}")
        knows_ab = store.add_relation(
            alice["id"], "knows", bob["id"], source_id="note-1", source_kind="human_note",
            conflict_key="alice-knows-target", metadata={"quote": "Alice knows Bob"},
        )
        knows_bc = store.add_relation(
            bob["id"], "knows", carol["id"], source_id="note-2", source_kind="human_note",
        )
        model_edge = store.add_relation(
            alice["id"], "knows", carol["id"], source_id="model-1", source_kind="model_output",
            conflict_key="alice-knows-target",
        )
        store.validate_relation(knows_ab["id"], "evidence-knows-ab")
        store.validate_relation(knows_bc["id"], "evidence-knows-bc")
        verified_neighbors = store.neighbors(alice["id"])
        all_neighbors = store.neighbors(alice["id"], min_trust="untrusted")
        route = store.path(alice["id"], carol["id"], max_hops=2)
        duplicate = store.add_relation(
            alice["id"], "knows", bob["id"], source_id="note-1", source_kind="human_note",
            conflict_key="alice-knows-target",
        )
        checks = {
            "verified_neighbors_exclude_untrusted": len(verified_neighbors) == 1,
            "all_neighbors_include_untrusted": len(all_neighbors) == 2,
            "deterministic_verified_path": route is not None and route["hops"] == 2,
            "provenance_round_trip": verified_neighbors[0]["source_id"] == "note-1"
            and verified_neighbors[0]["validation_evidence_id"] == "evidence-knows-ab",
            "conflict_explicit": model_edge["status"] == "conflict"
            and model_edge["trust_level"] == "untrusted",
            "duplicate_is_idempotent": duplicate["status"] == "duplicate",
        }
        return {
            "ok": all(checks.values()),
            "checks": checks,
            "entities": 3,
            "relations": 3,
            "verified_relation_count": len(verified_neighbors),
            "untrusted_relation_count": len(all_neighbors) - len(verified_neighbors),
            "path": route,
        }
    finally:
        store.close()
