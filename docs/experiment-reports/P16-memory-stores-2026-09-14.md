# P16 memory stores — 2026-09-14

## Hypothesis

Explicit working, episodic, semantic, and procedural stores can preserve
provenance while enforcing candidate validation, duplicate/conflict handling,
and retention controls.

## Design and provenance

- Experiment: `P16-MEMORY-20260914`
- Parent: `P15-RETRIEVAL-20260914`
- Store: SQLite at ignored local path `artifacts/memory/memory.sqlite`
- Implementation: `src/hai/memory/store.py`
- Memory types: `working`, `episodic`, `semantic`, `procedural`
- Lifecycle states: candidate/untrusted → verified with required evidence ID;
  deletion is a timestamped tombstone.

Each item stores content hash, source ID/kind, creation timestamp, trust level,
validation evidence ID, retention deadline, optional embedding reference,
conflict key, metadata, and deletion timestamp. Relationship links are stored
separately and survive item round trips.

## Commands and results

```text
docker compose run --rm hai python -m ruff check src tests
All checks passed!

docker compose run --rm hai python -m pytest -m 'not slow' -q
28 passed

hai memory self-test
ok=true
memory_types=[working, episodic, semantic, procedural]
model candidate trust before validation=untrusted
validated evidence=verification:exact:1
duplicate_status=duplicate
conflict_status=conflict
expired_tombstones=1

hai memory init
SQLite schema initialized successfully.

hai memory inspect --type episodic
Queryable type-specific inspection returned non-deleted items only.
```

The self-test also records a relationship link and verifies embedding/source
provenance after a round trip. The live self-test was logged as MLflow run
`8af2e30813e64a77b1b960e0c95b770e`.

## Gate evaluation

- Memory types distinct/queryable: **pass**. Four explicit types share a
  versioned schema and type-filtered inspection API.
- Untrusted input cannot silently become trusted memory: **pass**. Model output
  is forced to `untrusted`; verification evidence is required before the
  `verified` state.
- Provenance survives round trips: **pass**. Source ID, content hash,
  embedding reference, metadata, validation evidence, and retention fields are
  persisted and returned.

## Conclusion and next action

P16 passes. The memory layer is conservative by construction: conflicts remain
visible, duplicates do not create new facts, expired records become tombstones,
and model output is never auto-trusted. Advance to P17 for the provenance-aware
knowledge graph.
