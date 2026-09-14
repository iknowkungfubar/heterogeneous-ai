# Experiment / Phase Evidence: P02-GPU-20260914

## Hypothesis / goal

PyTorch can execute GPU forward/backward passes, update parameters, maintain finite loss, report peak memory, and save/reload a checkpoint in the validated ROCm container.

## Git/environment

- Date: 2026-09-14
- Git base before this P02 change: `6d7808e`
- Working tree: dirty only with this P02 implementation/evidence work
- Container image: `heterogeneous-ai-hai@sha256:6c88727db55ddd2de6ac5be7c2bfeaef51766da8202b0f39a84d2cbe062aa0f1`
- Base ROCm image: `rocm/pytorch:rocm10.0_ubuntu24.04_py3.12_pytorch_release_2.12.0@sha256:55bf8baa2a513b1c05bd256119fbc57a6ca64170e6cf5fe519b1cdd0c458cfd9`

## Data/tokenizer/model configs and hashes

- No dataset or tokenizer used.
- Model: one-layer `torch.nn.Linear(1, 1)` initialized with seed `1337`.
- Training: AdamW, learning rate `0.01`, 2000 steps, target `y=3x+2` over 10,000 GPU samples.
- Environment evidence: `artifacts/environment/SHA256SUMS` (ignored generated artifact).

## Commands

- `docker compose run --rm hai bash scripts/gpu-smoke.sh`
- `docker compose run --rm hai pytest -m gpu tests/smoke/test_gpu_linear.py -q`
- `docker compose run --rm hai pytest -m 'not slow' -q`

## Results

- `hai env-check`: passed; PyTorch `2.12.0+rocm10.0.0`, HIP `7.15.26333`, one `AMD Radeon RX 7900 GRE`.
- GPU smoke: passed; weight `2.980123`, bias `1.995303`, loss `0.01321094`.
- Peak allocated GPU memory: `159630336` bytes.
- Checkpoint save/reload round-trip: `True`.
- GPU smoke tests: `2 passed`.
- Full non-slow container test suite: `4 passed`.
- No NaN/Inf was observed; training explicitly rejects non-finite loss.

## Failures/deviations

The original 500-step command and 300-step test threshold did not converge reliably on this ROCm/GPU combination. The smoke command and test now use 2000 steps, with measured convergence preserved in this report. No evaluation or expected answer was weakened.

## Interpretation

All P02 acceptance criteria have direct evidence: GPU forward/backward, optimizer parameter updates, finite decreasing loss, peak memory reporting, and checkpoint save/reload.

## Gate decision

PASS

## Next action

Set P02 to `passed`, set P03 to `ready`, and begin data governance only after reviewing P03's approval boundary.
