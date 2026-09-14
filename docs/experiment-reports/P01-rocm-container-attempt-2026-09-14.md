# Experiment / Phase Evidence: P01-rocm-container-attempt-2026-09-14

## Hypothesis / goal

The pinned ROCm/PyTorch image and repository runtime can expose the AMD GPU through Docker with reproducible environment evidence.

## Git/environment

- Date: 2026-09-14
- Branch: `main`, synchronized with `origin/main` before this local evidence update
- Base image digest: `sha256:55bf8baa2a513b1c05bd256119fbc57a6ca64170e6cf5fe519b1cdd0c458cfd9`
- No host packages, drivers, kernel, or security settings were changed.

## Commands

```text
docker pull rocm/pytorch:rocm10.0_ubuntu24.04_py3.12_pytorch_release_2.12.0
docker compose build --pull=false
docker run ... python -c 'import torch; ...'
docker run ... bash scripts/capture-environment.sh
docker run ... bash -lc 'python -m hai.cli.main env-check && python -m hai.training.linear_smoke --steps 500'
docker run ... python -m hai.training.linear_smoke --steps 2000
docker run ... bash -lc 'pytest -m "not gpu and not slow"'
```

## Results

- Image pull passed and returned the digest recorded above.
- Base-image PyTorch check passed: `torch=2.12.0+rocm10.0.0`, `hip=7.15.26333`, CUDA API available, one device.
- GPU tensor operation passed: `tensor([2.], device='cuda:0')`.
- `hai env-check` passed and identified `AMD Radeon RX 7900 GRE`.
- Base-image non-GPU tests passed: 2 passed, 1 GPU test deselected.
- Environment evidence was written to ignored `artifacts/environment/` with `SHA256SUMS`.
- Linear smoke at 500 steps did not converge: weight `2.630606`, bias `1.971550`, loss `4.60278893`.
- Linear smoke at 2000 steps converged: weight `2.991279`, bias `1.992060`, loss `0.00260179`.

## Failures/deviations

The repository image build did not complete because the build container could not resolve `pypi.org` while running the Dockerfile pip-upgrade step. Repeated DNS retries were stopped. The pulled base image and direct mounted-source validation remain usable, but the full Compose-built project image is not yet validated.

## Interpretation

The host/device/container GPU path is healthy. P01's full acceptance gate remains incomplete because the repository image build and freeze were not completed. The 500-step smoke failure belongs to the later P02 smoke gate; 2000 steps converged.

## Gate decision

BLOCKED

## Next action

Diagnose container DNS/package-index access or make the Docker build consume validated cached/pinned dependencies without weakening integrity checks. Then rerun the repository image build, environment check, capture, and P01 acceptance gate.
