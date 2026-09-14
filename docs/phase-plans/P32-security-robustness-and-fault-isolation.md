# P32 — Security, robustness, and fault isolation

**Goal:** Harden tool boundaries, datasets, memory, model artifacts, inputs, dependencies, and component failure behavior.

**Dependencies:** `P31`  
**Authoritative parent:** `MASTER_RUNBOOK.md`  
**State source:** `docs/status/phase-state.yaml`

## What the novice should learn in this phase

Threat modeling, prompt injection, memory poisoning, artifact integrity, tool sandboxing, and fault isolation.

You do not need to master later phases yet. Before executing a command, be able to explain what it is intended to prove.

## Inputs / prerequisites

- All dependencies are `passed` with evidence.
- Working-tree/configuration state is understood and reproducible.
- This is the first dependency-ready incomplete phase unless the human explicitly authorizes another scope.
- Required data/models/artifacts have known provenance.

## Human approval boundary

Normal repository-local edits, tests, and short smoke runs may proceed. Obtain explicit human approval before host-level changes, destructive operations, large downloads, long GPU runs, paid/external services, private-data use, or publication/upload.

## Execution plan

1. Implement allowlisted structured tools; no model/RL arbitrary shell.
2. Harden memory ingestion against untrusted content.
3. Validate/checksum artifacts.
4. Test malformed inputs, injection-like retrieved content, corrupt media, timeouts, expert crashes, unavailable stores.
5. Document network/secret/dependency boundaries.

## Operator commands

```bash
hai security self-test
hai robustness fault-inject --suite component-failures
hai artifacts verify-hashes
```

Commands shown here are the intended stable interface. If a command does not exist yet, implementing and testing the smallest correct version is part of this phase before asking the novice to use it.

## Evidence to capture

Record in `docs/experiment-reports/` or the relevant experiment run directory:

- phase/experiment ID and date;
- Git commit/working-tree state;
- commands executed;
- configs and seeds;
- relevant environment/data/model/tokenizer hashes;
- metrics/pass-fail outputs;
- deviations/troubleshooting;
- conclusion and next action.

Never commit credentials/private data or giant raw logs.

## Acceptance gate

- [ ] Security tests demonstrate core boundaries.
- [ ] Single component failures degrade gracefully.

A checkbox requires direct evidence. Do not infer success from a neighboring test.

## Failure handling

1. Leave the phase `ready` or mark it `blocked`; never `passed`.
2. Preserve failing command, relevant logs, config, and environment fingerprint.
3. Diagnose from the lowest layer upward.
4. Change one important variable at a time when practical.
5. Record a negative hypothesis result rather than manipulating evaluation.

## Completion procedure

When all gate items pass:

1. Re-run relevant unit/integration/smoke/regression checks.
2. Record evidence paths in `docs/status/phase-state.yaml`.
3. Set `P32` to `passed`.
4. Set `P33` to `ready` only when its dependencies are passed.
5. Commit evidence/state change if authorized.
6. Apply the next phase's own approval boundary before expensive actions.
