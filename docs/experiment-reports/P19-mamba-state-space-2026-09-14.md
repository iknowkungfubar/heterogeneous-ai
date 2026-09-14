# P19 Mamba/state-space compatibility and specialist — 2026-09-14

## Hypothesis

A dependency-free PyTorch state-space recurrence can execute stably on the
validated AMD/ROCm path and provide a useful quality/throughput tradeoff
against the frozen GRU specialist.

## Compatibility and provenance

- Experiment: `P19-SSM-20260914`
- Parent: `P18-GNN-20260914`
- Configuration: `configs/models/ssm.yaml`
- Implementation: `src/hai/models/state_space.py`
- Initialization: random, seed `1337`
- Task hash: `19320dce27dc00cd00f95dd0006e062dd64c7529f5be824d3b012988ddafadcf`
- Config SHA-256: `ac3e3b262fdc3b54fd29c92a8ac658d3c96b854d1d802f82fae06ea08da02513`

The optional upstream `mamba-ssm` package was not installed and no external
download was performed. The measured specialist is explicitly a
`plain_pytorch_diagonal_state_space` implementation, so the result does not
claim compatibility with optimized CUDA-only Mamba extensions.

## Compatibility command

```text
hai mamba compatibility-check --config configs/models/ssm.yaml
torch=2.12.0+rocm10.0.0
hip=7.15.26333
device=cuda
cpu_forward=true
cpu_finite=true
gpu.available=true
gpu.forward_backward=true
gpu.peak_memory_bytes=160651776
upstream_package=mamba-ssm
upstream_version=null
optimized_extension_available=false
```

Compatibility MLflow run: `4980d891b493454dbd40a488465ce376`.

## Smoke training and checkpoint/resume

```text
hai mamba smoke-train --config configs/models/ssm.yaml --steps 100
device=cuda
parameter_count=2233
start_loss=2.059492349624634
final_loss=0.17117001116275787
validation_loss=0.1676044911146164
validation_perplexity=1.1824687719345093
validation_accuracy=1.0
checkpoint_round_trip=true
resumed_step=100
peak_memory_bytes=160782336
checkpoint_sha256=15e6f6f8169c5e12dc869d51e86f3deb9d1dc80613aca6d0cfa2c1a7ba750616
```

Smoke MLflow run: `d4245442b9e441beb22b58ea95460e60`.

## GRU comparison

The comparison used the same deterministic task, seed, sequence length,
training steps, and AMD device for both models:

| Metric | Plain SSM | GRU-0001 baseline |
|---|---:|---:|
| Parameters | 2,233 | 12,456 |
| Validation loss | 0.167604 | 0.006603 |
| Validation accuracy | 1.0 | 1.0 |
| Validation perplexity | 1.182469 | 1.006625 |
| Tokens/second | 136,708 | 33,157 |
| Peak memory bytes | 160,782,336 | 166,419,968 |

The SSM was faster, smaller, and lower-memory, but its validation loss was
`+0.161001` worse and its accuracy was not better. It therefore does not
replace the GRU. Benchmark MLflow run: `af715c139f84488ca53e6439aca512cf`.

## Gate evaluation

- AMD GPU training stable: **pass**. CPU forward, AMD GPU
  forward/backward, finite loss, 100-step training, and checkpoint round-trip
  all passed in the pinned ROCm container.
- SSM replaces GRU only when evidence justifies it: **pass with no
  replacement**. The SSM is retained as a documented efficiency/compatibility
  comparison, while GRU remains the quality-selected sequence specialist.

Repository verification also passed: Ruff completed successfully and all 35
non-slow tests passed.

## Conclusion and next action

P19 passes as a compatibility-gated negative replacement result. The project
keeps the plain SSM available for future efficiency studies, does not claim
optimized upstream Mamba support, and advances to P20 vision specialization.
