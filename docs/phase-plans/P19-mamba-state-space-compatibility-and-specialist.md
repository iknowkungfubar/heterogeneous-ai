# P19 — Mamba/state-space compatibility and specialist

**Goal:** Validate AMD/ROCm support, train an SSM specialist, and compare it against the GRU.

**Dependencies:** `P18`  
**Authoritative parent:** `MASTER_RUNBOOK.md`  
**State source:** `docs/status/phase-state.yaml`

## What the novice should learn in this phase

State-space models, hardware-specific kernels, compatibility spikes, and throughput/memory comparison.

You do not need to master later phases yet. Before executing a command, be able to explain what it is intended to prove.

## Inputs / prerequisites

- All dependencies are `passed` with evidence.
- Working-tree/configuration state is understood and reproducible.
- This is the first dependency-ready incomplete phase unless the human explicitly authorizes another scope.
- Required data/models/artifacts have known provenance.

## Human approval boundary

Normal repository-local edits, tests, and short smoke runs may proceed. Obtain explicit human approval before host-level changes, destructive operations, large downloads, long GPU runs, paid/external services, private-data use, or publication/upload.

## Execution plan

1. Record exact upstream Mamba version/commit/install mode.
2. Test import, CPU forward, GPU forward/backward, short train, checkpoint/resume.
3. Do not require CUDA-only optimized extensions; record actual AMD path.
4. Train a comparable SSM only after compatibility passes.
5. Compare quality, complementarity, throughput, and VRAM with GRU.

## Operator commands

```bash
hai mamba compatibility-check
hai mamba smoke-train --steps 100
hai mamba benchmark --against GRU-0001
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

- [ ] AMD GPU training is stable or phase is honestly blocked.
- [ ] SSM replaces GRU only when evidence justifies it.

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
3. Set `P19` to `passed`.
4. Set `P20` to `ready` only when its dependencies are passed.
5. Commit evidence/state change if authorized.
6. Apply the next phase's own approval boundary before expensive actions.
