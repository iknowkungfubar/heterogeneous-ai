# Heterogeneous AI Research Platform

A local-first research project for building and evaluating a **heterogeneous multimodal AI system from scratch** on consumer hardware. Instead of relying on one large language model, the system combines small specialist neural architectures with deterministic reasoning, explicit memory, retrieval, graphs, perception, world modeling, verification, calibrated uncertainty, and learned orchestration.

## Start here

1. Read `MASTER_RUNBOOK.md` — the authoritative specification.
2. Read `AGENTS.md` — mandatory rules for AI coding agents.
3. Read `docs/status/phase-state.yaml` — the current execution state.
4. Read the first `ready` phase under `docs/phase-plans/`.
5. Execute only that phase until its acceptance gate passes.

For a new clone, the first phase is `P00`.

## Research hypothesis

> Can small, architecturally diverse specialist models plus deterministic reasoning, explicit memory, verification, calibrated uncertainty, and learned routing outperform a substantially more homogeneous language-model baseline at similar practical active compute?

A negative result is valid. Components must earn their place through measured contribution.

## Final target

The mature system can include:

- scratch-trained Transformer language specialist;
- GRU and/or Mamba/SSM sequence specialist;
- deterministic symbolic solver;
- scratch-trained embedding/retrieval model;
- explicit working, episodic, semantic, relational, and procedural memory;
- provenance-aware knowledge graph and GNN specialist;
- scratch-trained vision and audio specialists;
- multimodal alignment and fusion;
- world model and deterministic planner;
- learned task router and constrained compute-aware RL orchestration;
- independent verification, confidence calibration, consensus, and abstention;
- local CLI/API, experiment registry, model/data cards, regression tests, and full ablation reports.

## Repository map

- `MASTER_RUNBOOK.md` — master specification and lifecycle.
- `docs/phase-plans/` — executable plans for P00 through P35.
- `docs/task-graph.yaml` — dependency graph and deliverables.
- `docs/status/phase-state.yaml` — mutable progress state.
- `configs/` — environment, data, model, training, routing, and evaluation configs.
- `experiments/` — experiment registry and immutable run records.
- `src/hai/` — implementation package.
- `tests/` — unit, integration, smoke, regression, and acceptance tests.
- `artifacts/` — local generated outputs; large artifacts are ignored by Git.

## Important boundary

This project uses software libraries such as PyTorch, but the primary research track does **not** use pretrained neural weights. A model created from a library configuration with random initialization is allowed; an external pretrained checkpoint is not.

## Hardware reference

The initial reference machine is a Linux consumer PC with one AMD Radeon RX 7900 GRE-class GPU (16 GB VRAM), approximately 32 GB system RAM, and an 8-core-class CPU. The runtime is containerized so the host desktop distribution does not become the ML dependency environment.

## License

MIT for repository code and documentation. Individual datasets and third-party dependencies retain their own licenses and must be tracked in data/model cards.
