# P00 — Repository bootstrap

**Goal:** Create the governed repository, documentation set, status files, and reproducible project skeleton.

**Dependencies:** None  
**Authoritative parent:** `MASTER_RUNBOOK.md`  
**State source:** `docs/status/phase-state.yaml`

## What the novice should learn in this phase

Git repository structure, tracked vs generated files, why governance comes before training.

You do not need to master later phases yet. Before executing a command, be able to explain what it is intended to prove.

## Inputs / prerequisites

- All dependencies are `passed` with evidence.
- Working-tree/configuration state is understood and reproducible.
- This is the first dependency-ready incomplete phase unless the human explicitly authorizes another scope.
- Required data/models/artifacts have known provenance.

## Human approval boundary

Normal repository-local edits, tests, and short smoke runs may proceed. Obtain explicit human approval before host-level changes, destructive operations, large downloads, long GPU runs, paid/external services, private-data use, or publication/upload.

## Execution plan

1. Copy this scaffold into the target repository without removing governance files.
2. Inspect `.gitignore`; verify datasets/checkpoints/secrets are excluded while manifests/configs remain tracked.
3. Run `python scripts/check-phase.py` and confirm P00 is first dependency-ready.
4. Run `git diff --check` before the first commit.
5. Create the initial signed-off commit if commit authorization is available.

## Operator commands

```bash
mkdir -p ~/iknowkungfubar/heterogeneous-ai
cd ~/iknowkungfubar/heterogeneous-ai
git init -b main
git status
python scripts/check-phase.py
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

- [ ] Repository has the documented tree.
- [ ] Governance/status/task graph files parse and reference valid phase plans.
- [ ] No obvious secrets or large generated artifacts are tracked.
- [ ] Git reports the intended branch and coherent initial state.

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
3. Set `P00` to `passed`.
4. Set `P01` to `ready` only when its dependencies are passed.
5. Commit evidence/state change if authorized.
6. Apply the next phase's own approval boundary before expensive actions.
