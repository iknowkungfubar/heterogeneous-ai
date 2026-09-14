# Start Here

This repository is intentionally large because it describes the full project. **Do not try to execute the entire roadmap.** The phase system exists so a novice and an AI coding agent have exactly one active scope at a time.

## First session

1. Put these files in the repository root.
2. Read `README.md`, then `AGENTS.md`, then `MASTER_RUNBOOK.md`.
3. Run:

```bash
python scripts/verify-scaffold.py
python scripts/check-phase.py
```

Expected first phase:

```text
P00 — Repository bootstrap
```

4. Open the P00 plan under `docs/phase-plans/` and execute it exactly.
5. Do **not** begin GPU setup or training until P00 passes and P01 becomes ready.

## If you are using an AI coding agent

Give it this instruction:

> Read `AGENTS.md`, `MASTER_RUNBOOK.md`, `ARCHITECTURE.md`, `RESEARCH_PLAN.md`, `DATA.md`, `EVALUATION.md`, `SECURITY.md`, `REPRODUCIBILITY.md`, `docs/task-graph.yaml`, `docs/status/phase-state.yaml`, and `experiments/registry.yaml`. Run `python scripts/verify-scaffold.py` and `python scripts/check-phase.py`. Work only on the first dependency-ready incomplete phase. Follow that phase's plan and acceptance gate. Do not start a long training run, modify the host, make a large download, perform destructive actions, use private data, or publish/upload artifacts without explicit human approval. Never use pretrained neural weights in the scratch-training path or fabricate experiment evidence.

## Key distinction

Many commands in later phase plans are **stable target commands** that the agent will implement during that phase. Their presence does not imply that the finished model system already exists in this bootstrap scaffold.
