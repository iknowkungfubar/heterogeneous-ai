# Experiment / Phase Evidence: P01-host-preflight-2026-09-13

## Hypothesis / goal

The host prerequisites for the pinned ROCm/PyTorch environment are present without changing the host.

## Git/environment

- Date: 2026-09-13
- Branch: `main`, tracking `origin/main`
- HEAD: `07ad41e7bced3c4cace98e844cd89c93b79bb7cb`
- No host, package, image, dataset, or model changes were performed.

## Data/tokenizer/model configs and hashes

The target container configuration is `configs/environments/rocm.yaml`; no image was pulled or built.

## Commands

```text
./scripts/health-check.sh
docker info --format 'Server={{.ServerVersion}} Rootless={{.SecurityOptions}}'
docker compose config --quiet
```

## Results

- Host: Linux `7.2.4-arch1-2`, x86_64.
- PCI: AMD Radeon RX 7900 XT/7900 XTX/7900 GRE/7900M device visible at `03:00.0`.
- `/dev/kfd`: present and accessible.
- `/dev/dri/renderD128`: present and accessible.
- Docker client: `29.8.0`; server: `29.7.2`.
- Compose configuration: valid.

## Failures/deviations

No preflight failures. The pinned ROCm image was not pulled or run because that action may require a large download and explicit human approval.

## Interpretation

The host-side prerequisites are ready. The remaining P01 gate requires container userspace validation and must not be inferred from host visibility alone.

## Gate decision

BLOCKED

## Next action

After approval, pull/build the pinned image, run the Compose container, execute `python -m hai.cli.main env-check`, run the GPU tensor smoke test, capture the environment freeze, and evaluate the full P01 acceptance gate.
