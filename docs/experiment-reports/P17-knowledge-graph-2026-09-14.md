# P17 knowledge graph — 2026-09-14

## Hypothesis

A deterministic SQLite graph can expose entity lookup, trusted relation
queries, provenance-aware traversal, and explicit contradictory edges without
silently promoting model output to factual knowledge.

## Design and provenance

- Experiment: `P17-KNOWLEDGE-GRAPH-20260914`
- Parent: `P16-MEMORY-20260914`
- Store: SQLite at ignored local path `artifacts/graphs/knowledge.sqlite`
- Implementation: `src/hai/graph/store.py`
- Operator interface: `hai graph init`, `import-fixture`, `query`, `path`, and
  `self-test`
- Fixture: `tests/fixtures/graph.jsonl`

Entities and relations persist source ID, source kind, creation time, trust
level, validation evidence ID, metadata, and deletion state. Human/fixture
inputs begin as `candidate`; model output begins as `untrusted`. Only an
explicit evidence ID can promote an item to `verified`. Contradictory relation
keys remain stored with `status: conflict`.

## Commands and results

```text
docker compose run --rm hai python -m ruff check src tests
All checks passed!

docker compose run --rm hai python -m pytest -m 'not slow' -q
31 passed in 3.31s

docker compose run --rm hai python scripts/check-phase.py
First dependency-ready phase: P17 — Knowledge graph (state=ready)

hai graph self-test
ok=true
verified_neighbors_exclude_untrusted=true
all_neighbors_include_untrusted=true
deterministic_verified_path=true
provenance_round_trip=true
conflict_explicit=true
duplicate_is_idempotent=true
```

The live self-test was recorded in MLflow as run
`9e62a03f762f475789ece707fb7893e1`.

The fixture smoke path also passed: two entities and one `capital_of` relation
were imported, and `graph query --subject Sacramento --predicate capital_of
--min-trust candidate` returned exactly one relation with source and target
provenance.

## Gate evaluation

- Graph operations deterministic/tested: **pass**. Stable IDs, sorted
  neighbors, bounded breadth-first traversal, duplicate idempotency, and 31
  passing non-slow tests are recorded above.
- Factual edges support provenance/trust: **pass**. Source identity, source
  kind, validation evidence, metadata, and trust filtering are persisted and
  round-tripped; default queries require `verified` relations.
- Conflicts remain explicit: **pass**. A contradictory target with the same
  conflict key is retained as `status: conflict` and does not enter the
  default verified path.

## Conclusion and next action

P17 passes. The project can advance to the graph-neural-network specialist in
P18; the graph store remains the deterministic, provenance-aware reference
layer for that later learned component.
