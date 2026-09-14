# P01 — Host, GPU, Docker, and ROCm environment

**Goal:** Verify the AMD GPU compute path and create a pinned ROCm/PyTorch container environment.

**Dependencies:** `P00`  
**Authoritative parent:** `MASTER_RUNBOOK.md`  
**State source:** `docs/status/phase-state.yaml`

## What the novice should learn in this phase

Host kernel/device access vs container userspace, `/dev/kfd`, `/dev/dri`, ROCm, and why PyTorch uses `torch.cuda` APIs under ROCm.

You do not need to master later phases yet. Before executing a command, be able to explain what it is intended to prove.

## Inputs / prerequisites

- All dependencies are `passed` with evidence.
- Working-tree/configuration state is understood and reproducible.
- This is the first dependency-ready incomplete phase unless the human explicitly authorizes another scope.
- Required data/models/artifacts have known provenance.

## Human approval boundary

Normal repository-local edits, tests, and short smoke runs may proceed. Obtain explicit human approval before host-level changes, destructive operations, large downloads, long GPU runs, paid/external services, private-data use, or publication/upload.

## Execution plan

1. Confirm the AMD GPU appears in `lspci` and `/dev/kfd` plus DRI render nodes exist.
2. Install/enable Docker on the host only with human approval if needed.
3. Pull/build the pinned image and launch with GPU device mapping.
4. Confirm PyTorch reports HIP and a GPU device; do not replace the image PyTorch with generic PyPI torch.
5. Capture environment evidence and generate the concrete requirements freeze from the validated container.
6. If upgrading the image/version, create an ADR and repeat P01.

## Operator commands

**Host:**
```bash
./scripts/health-check.sh
docker run --rm hello-world
docker pull rocm/pytorch:rocm10.0_ubuntu24.04_py3.12_pytorch_release_2.12.0
./scripts/enter-rocm.sh
```
**Inside container:**
```bash
python -m hai.cli.main env-check
./scripts/capture-environment.sh
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

- [ ] Docker operates normally.
- [ ] GPU device files are accessible.
- [ ] PyTorch reports `torch.cuda.is_available() == True` under ROCm.
- [ ] A GPU tensor operation succeeds.
- [ ] Environment/freeze evidence is recorded.

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
3. Set `P01` to `passed`.
4. Set `P02` to `ready` only when its dependencies are passed.
5. Commit evidence/state change if authorized.
6. Apply the next phase's own approval boundary before expensive actions.
