# P11 learned task router — 2026-09-14

## Hypothesis

An interpretable, TRAIN-only router can select an applicable specialist while
preserving downstream utility and exposing an auditable feature/decision trace.

## Scope and provenance

- Experiment: `P11-ROUTER-20260914`
- Parent: `P10-PROTOCOL-20260914`
- Benchmark: `core-objective-v1`
- Benchmark manifest: `data/manifests/core-objective-v1.yaml`
- TRAIN hash: `8b8484f0e30aa0570889da2770d2ad566771e18789fb971b9598bf6b36a664be`
- VALIDATION hash: `bcb0032e877d411f31a1500e8d49ade054edd1e7595365c83def5323eac02e11`
- Router config: `configs/routing/default.yaml`
- Seed: `1337`
- Working tree before implementation: clean except pre-existing `.codegraph/` and `.serena/`
- Training data: 84 records from TRAIN only; TEST was not used for selection.

The router uses a depth-3 decision tree over task type, prompt length,
surface markers, and symbolic applicability. Artifacts are written to the
ignored path `artifacts/routers/router-v1/`; the metadata includes the feature
names, classes, threshold, known categories, and `sklearn` decision trace.

## Commands and results

```text
docker compose run --rm hai python -m ruff check src tests
All checks passed!

docker compose run --rm hai python -m pytest -m 'not slow' -q
19 passed

hai router build-dataset --benchmark core-objective-v1
records=84, labels={symbolic: 84}

hai router train --config configs/routing/default.yaml --benchmark core-objective-v1 --split train
records=84, classes=[symbolic], decision_trace='|--- class: 0'

hai router evaluate --config configs/routing/default.yaml --benchmark core-objective-v1 --split validation
samples=84, accuracy=0.2857142857142857, always_symbolic_accuracy=0.2857142857142857
abstained=0, selected_expert_calls=84

hai router route --config configs/routing/default.yaml --category arithmetic --prompt '3 + 4 * 2'
selected_expert=symbolic, confidence=1.0

hai router route --config configs/routing/default.yaml --category unknown --prompt 'opaque request'
selected_expert=abstain, confidence=0.0
```

## Gate evaluation

- Allowed training data: **pass**. Dataset construction reads only the
  benchmark TRAIN split and training rejects any other split.
- Utility versus simple baseline: **pass**. Router utility equals the
  always-symbolic baseline on VALIDATION; no improvement is claimed.
- Inspectable traces: **pass**. Feature dictionaries, confidence, selected
  expert/abstention, known-category guard, and the serialized decision-tree
  trace are available.

## Conclusion and next action

The router contract is implemented and reproducible, but this run is a
negative efficiency result: the current protocol exposes only one structured
expert, so the learned model has one class, makes all 84 validation calls, and
cannot outperform the always-symbolic baseline. The router is not promoted as
an efficiency improvement. P12 may proceed to add verification; later router
work should be re-evaluated after additional capable specialists exist.
