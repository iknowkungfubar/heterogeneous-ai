# P18 graph neural-network specialist — 2026-09-14

## Hypothesis

A small scratch-trained message-passing network can add relational node
classification value beyond deterministic graph algorithms and feature-only or
random baselines.

## Design and provenance

- Experiment: `P18-GNN-20260914`
- Parent: `P17-KNOWLEDGE-GRAPH-20260914`
- Configuration: `configs/models/gnn.yaml`
- Implementation: `src/hai/models/gnn.py`
- Graph: deterministic synthetic homophilic graph, 90 nodes, 3 classes, 8
  random node features, seed `1337`
- Split: 54 TRAIN nodes, 18 VALIDATION nodes, 18 held-out TEST nodes
- Graph hash: `2bbe8387acd0a927f12d39600753545b32d30a3d5c5e9198c21105a73d16678a`
- Config SHA-256: `f2f3020c44db5a03a33f078cc1b204d7dc2786bf28d14cad490a17febfd06bb3`

The GNN uses plain PyTorch message passing with separate self and neighbor
projections. It is randomly initialized and trained only against TRAIN node
labels. Validation labels are used only for comparison. The deterministic
graph baseline uses labeled TRAIN neighbors, while the MLP sees node features
without adjacency.

## Commands and results

```text
docker compose run --rm hai python -m hai.cli.main gnn train --config configs/models/gnn.yaml
initialization=random
seed=1337
parameter_count=339
train_nodes=54
validation_nodes=18
test_nodes=18
start_loss=1.2013249397277832
final_loss=0.003538471180945635
checkpoint_sha256=ef04e11ceafdf8c009a29d9867d7725dcddeb607584ecfbe62c6101742432416

docker compose run --rm hai python -m hai.cli.main compare graph-baseline gnn-v1 --split validation
gnn_accuracy=0.5
graph_baseline_accuracy=1.0
mlp_accuracy=0.2777777910232544
random_baseline_accuracy=0.5
best_baseline_accuracy=1.0
gnn_delta_vs_best_baseline=-0.5
retained=false
```

MLflow runs:

- training: `605a06524bf6498d84026dee8a2975b7`
- evaluation: `6fae8501dfbc487cbbc1e1fc36fef473`
- comparison: `f927b53a061c445f98e3e35467722e84`

Repository verification:

```text
docker compose run --rm hai python -m ruff check src tests
All checks passed!

docker compose run --rm hai python -m pytest -m 'not slow' -q
33 passed in 4.11s
```

## Gate evaluation

- GNN reproducible/scratch-trained: **pass**. Random initialization, fixed
  seed, graph hash, checkpoint hash, and deterministic round-trip tests pass.
- Contribution beyond graph baselines quantified: **pass**. On VALIDATION the
  GNN scored 0.50 versus 1.00 for the deterministic graph baseline; its delta
  versus the strongest baseline was -0.50.
- No-value result may retire it: **pass**. The GNN is not retained as a
  promoted specialist because it added no value over the graph algorithm.

## Conclusion and next action

P18 is a negative result. The graph algorithm is currently the stronger and
more interpretable relational component, so `gnn-v1` remains a reproducible
comparison artifact rather than a production specialist. Advance to P19 with
the GNN result preserved for future architecture comparisons.
