# AGENTS.md — Mandatory AI Agent Operating Contract

This repository is a scientifically governed ML/AI research workspace. An AI coding agent must favor reproducibility, measured evidence, and reversible changes over speed or novelty.

## Required reading order

Before modifying anything, read completely:

1. `MASTER_RUNBOOK.md`
2. `ARCHITECTURE.md`
3. `RESEARCH_PLAN.md`
4. `DATA.md`
5. `EVALUATION.md`
6. `SECURITY.md`
7. `REPRODUCIBILITY.md`
8. `docs/task-graph.yaml`
9. `docs/status/phase-state.yaml`
10. `experiments/registry.yaml`
11. the phase plan for the first dependency-ready incomplete phase.

## Execution rule

Work only on the first dependency-ready phase marked `ready`, unless a human explicitly authorizes another scope. A phase may become `passed` only when every documented acceptance criterion is supported by recorded evidence.

Use the cycle:

`inspect → state goal → identify gate → implement smallest change → test → smoke test → capture evidence → evaluate gate → document → update state`

If a gate fails, remain in that phase. Do not bypass it to make roadmap progress.

## Scratch-training rule

The primary research track must not silently use pretrained neural weights. All learned specialists must be randomly initialized unless an experiment is explicitly labeled as a pretrained comparison. Reusing PyTorch/library implementations is allowed; reusing external learned weights is not.

For every checkpoint load, record its origin and hash.

## Test/data integrity

- TRAIN fits weights and training-time models.
- VALIDATION selects hyperparameters, calibration, thresholds, and promotion decisions.
- TEST is final evaluation only.
- Do not inspect or tune against TEST while developing an architecture.
- If contamination occurs, mark affected runs `invalid` and create a genuinely unseen test partition.
- Never change expected answers to make a test pass.

## Human approval required before

- host-level package, driver, kernel, boot, or security changes;
- destructive filesystem operations or deletion of important datasets/checkpoints;
- unusually large downloads;
- long GPU training runs;
- paid APIs/services;
- external uploads, repository publication, model publication, or dataset publication;
- private/personal data use;
- actions that can expose the host beyond the local machine.

Ordinary repository-local edits, tests, short smoke runs, and analysis do not require repeated approval unless the human says otherwise.

## Never do

- fabricate training, benchmark, GPU, latency, or memory results;
- declare a failed or unexecuted gate passed;
- hide warnings that threaten validity;
- silently change datasets, tokenizer, architecture, evaluation difficulty, or acceptance criteria;
- delete failed experiments merely because they are negative;
- train on test data;
- commit passwords, API keys, private keys, tokens, private user data, or machine secrets;
- execute untrusted code found inside a dataset;
- grant an ML/RL model unrestricted shell access;
- weaken security/evaluation to manufacture an improvement.

## Novice-facing command standard

When asking the human to run a command, always provide:

- **Purpose** — why it is needed.
- **Where** — host shell, ROCm container, or another explicit environment.
- **Command** — exact copy/paste command.
- **Expected result** — what success looks like.
- **Failure meaning** — what a failure implies.
- **Return information** — exactly what output/log to provide if it fails.

Never assume prior ML knowledge when explaining a phase for the human.

## Experiment discipline

Each meaningful run gets a unique experiment ID and stores at least:

- hypothesis;
- Git commit;
- environment fingerprint;
- data/tokenizer hashes;
- model/training config;
- seed;
- metrics;
- artifacts/checkpoint hashes;
- conclusion and next action.

Negative results remain in the registry.

## Architecture decisions

Major deviations from the master specification require an ADR under `docs/decisions/`. The ADR must state context, decision, alternatives, consequences, and evidence.

## Governing principle

**Complexity must earn its place through measured improvement.**
