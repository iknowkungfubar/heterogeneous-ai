# Experiment / Phase Evidence: P01-rocm-container-complete-2026-09-14

## Hypothesis / goal

The pinned ROCm/PyTorch environment can be built and run with the AMD GPU visible, while the committed Compose runtime retains default IPC and seccomp isolation.

## Git/environment

- Date: 2026-09-14
- Branch: `main`
- Base image digest: `sha256:55bf8baa2a513b1c05bd256119fbc57a6ca64170e6cf5fe519b1cdd0c458cfd9`
- Built image digest: `sha256:6c88727db55ddd2de6ac5be7c2bfeaef51766da8202b0f39a84d2cbe062aa0f1`
- No host package, driver, kernel, or security changes were made.

## Commands

```text
docker build --network=host --pull=false -t heterogeneous-ai-hai:p01-dns-test -f Dockerfile.rocm .
docker tag heterogeneous-ai-hai:p01-dns-test heterogeneous-ai-hai:latest
docker compose run --rm hai bash -lc 'python -m hai.cli.main env-check && python -m hai.training.linear_smoke --steps 2000'
docker compose run --rm hai bash scripts/capture-environment.sh
sha256sum artifacts/environment/*
```

## Results

- Dockerfile build passed using the explicit build-time host network option; no network weakening was committed to Compose.
- The resulting image was run through the committed Compose service.
- `hai env-check` passed: PyTorch `2.12.0+rocm10.0.0`, HIP `7.15.26333`, CUDA API available, one `AMD Radeon RX 7900 GRE` device.
- GPU tensor operation passed in the pulled base image.
- 2000-step GPU linear smoke passed: weight `2.987903`, bias `1.992482`, loss `0.00494115`.
- Environment evidence capture passed and wrote `artifacts/environment/SHA256SUMS` plus Torch, pip, GPU, and ROCm reports.
- Earlier host preflight confirmed `/dev/kfd`, `/dev/dri/renderD128`, and Docker availability.

## Failures/deviations

The default Compose build command could not resolve `pypi.org` through BuildKit's default network. The successful build therefore used an explicit command-line build network override for validation; the committed runtime keeps default IPC and seccomp isolation. The dependency set also selected an NVIDIA NCCL wheel during unconstrained installation; the P01 lock/freeze remains evidence to address before relying on the environment for research claims.

## Interpretation

All P01 acceptance criteria have direct evidence. The container environment is ready for P02, with the build-network invocation and dependency-resolution caveat recorded for reproducibility.

## Gate decision

PASS

## Next action

Advance P01 to `passed`, make P02 `ready`, and execute only P02's PyTorch fundamentals and GPU training smoke plan.
