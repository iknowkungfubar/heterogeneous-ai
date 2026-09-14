# Heterogeneous AI Research Platform

[![CI](https://github.com/iknowkungfubar/heterogeneous-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/iknowkungfubar/heterogeneous-ai/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A local-first research platform for testing whether a collection of small, diverse specialists can outperform a more homogeneous language-model baseline at comparable practical compute.

The project combines scratch-trained neural models with deterministic reasoning, explicit memory, retrieval, provenance, verification, calibrated uncertainty, and constrained orchestration. It is deliberately evidence-driven: a component remains only when measured experiments show that it earns its complexity.

## Current status

This repository is an early-stage, governed research scaffold. Work proceeds through dependency-ordered phases, beginning with repository bootstrap and environment validation. Later architecture entries describe the target research program; they are not claims that every listed subsystem is already implemented.

The primary research track is trained from random initialization. External pretrained neural checkpoints are not silently used. Negative results, failed experiments, and rejected components remain part of the scientific record.

## Research question

> Can architecturally diverse specialist models plus deterministic reasoning, explicit memory, verification, calibrated uncertainty, and learned routing improve quality, coverage, and efficiency over a substantially more homogeneous baseline?

Quality is evaluated together with coverage, latency, active experts, memory, and compute. A negative result is a valid result.

## Target architecture

```text
input and modality manager
            │
      typed representations
            │
       executive router
            │
  specialists and deterministic tools
            │
 evidence + provenance + calibration
            │
 verification → consensus → answer or abstain
            │
       traceable experience/memory
```

The planned specialist set includes a scratch-trained Transformer, an independent sequence model, symbolic reasoning, retrieval and explicit memory, graph reasoning, vision, audio, multimodal alignment, planning, and constrained expert orchestration. Each phase has its own acceptance gate; later work does not bypass an unmet gate.

## Reproducible project contract

The repository is governed by these documents:

- [`MASTER_RUNBOOK.md`](MASTER_RUNBOOK.md) — authoritative specification and lifecycle.
- [`AGENTS.md`](AGENTS.md) — operating contract for coding agents and contributors.
- [`docs/status/phase-state.yaml`](docs/status/phase-state.yaml) — machine-readable progress state.
- [`docs/task-graph.yaml`](docs/task-graph.yaml) — phase dependencies and deliverables.
- [`docs/phase-plans/`](docs/phase-plans/) — executable plans for P00–P35.
- [`experiments/registry.yaml`](experiments/registry.yaml) — experiment record contract.

The execution rule is simple: inspect the current state, work only on the first dependency-ready phase, capture evidence, and do not mark a gate passed without direct evidence.

## Quick start

```bash
git clone https://github.com/iknowkungfubar/heterogeneous-ai.git
cd heterogeneous-ai

python scripts/verify-scaffold.py
python scripts/check-phase.py
```

These checks validate the repository scaffold and identify the first phase that is eligible to run. The ROCm/PyTorch environment is containerized; do not install the full research stack globally on the host. Review [`START_HERE.md`](START_HERE.md) before beginning work.

## Local experiment UI

The repository includes an opt-in, loopback-only MLflow UI for viewing training
runs, live loss curves, parameters, and checkpoint artifacts. Start it after
building the validated image:

```bash
make mlflow-up
make mlflow-url
# open http://127.0.0.1:5000
```

Run training with tracking enabled through the Compose service network:

```bash
MLFLOW_TRACKING_URI=http://mlflow:5000 \
  docker compose run --rm hai python -m hai.cli.main train \
  --model configs/models/transformer-smoke.yaml \
  --tokenizer artifacts/tokenizers/bpe-8192-v1/tokenizer.json \
  --dataset-manifest data/manifests/tinystories-smoke.yaml \
  --max-steps 30 --experiment tracked-smoke
```

Compose training containers automatically use the local MLflow service. Host-side
training commands can opt in with `MLFLOW_TRACKING_URI=http://127.0.0.1:5000`.
MLflow data stays in the ignored local `artifacts/mlflow/` directory and the
service binds only to the local machine; it is not a public endpoint.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/hai/` | Python package and specialist boundaries |
| `tests/` | Unit, integration, smoke, regression, and acceptance tests |
| `configs/` | Environment, data, model, training, routing, and evaluation configuration |
| `scripts/` | Scaffold, environment, experiment, and verification commands |
| `docs/` | Architecture, phase plans, decisions, cards, and evidence templates |
| `experiments/` | Registry and immutable run-record boundary |
| `data/` | Data-layout placeholders and manifests; raw data is not committed |
| `artifacts/` | Generated-artifact layout; checkpoints and large outputs are ignored |

## Data and model integrity

- TRAIN, VALIDATION, and TEST have separate roles; test data is not used for tuning.
- Datasets, licenses, provenance, tokenizers, configurations, seeds, metrics, and checkpoint hashes are recorded for meaningful runs.
- The primary track starts neural weights randomly and records any checkpoint origin and hash.
- Generated checkpoints, raw datasets, secrets, and local service state are excluded from Git.

## Security and responsible contribution

Read [`SECURITY.md`](SECURITY.md) before adding data, tools, services, or model capabilities. Neural and reinforcement-learning components do not receive unrestricted host-shell access; tools are allowlisted and structured. Do not commit credentials, private data, downloaded checkpoints, or unreviewed datasets.

Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) and the relevant phase plan before opening a pull request. Keep changes narrow, add tests, preserve provenance, and document architecture deviations with an ADR.

## Hardware and runtime

The initial reference target is a Linux consumer system with an AMD Radeon RX 7900 GRE-class GPU and approximately 16 GB VRAM. The supported research environment is the pinned ROCm/PyTorch container described in [`configs/environments/rocm.yaml`](configs/environments/rocm.yaml). GPU and host setup require explicit validation before later training phases.

## License

Repository code and documentation are licensed under the MIT License. Dataset, model, and third-party dependency licenses remain applicable and must be documented in their respective cards.
