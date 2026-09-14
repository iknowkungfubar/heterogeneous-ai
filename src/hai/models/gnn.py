from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import torch
import yaml
from torch import Tensor, nn

from hai.data.pipeline import sha256_file


class GNNEvaluationError(ValueError):
    """Raised when a GNN experiment violates its reproducibility contract."""


def load_gnn_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("id") != "gnn-v1":
        raise GNNEvaluationError("GNN config must declare gnn-v1")
    if config.get("initialization") != "random":
        raise GNNEvaluationError("GNN must explicitly request random initialization")
    required = (
        "seed", "nodes", "classes", "feature_dim", "hidden_dim", "training_steps",
        "learning_rate", "train_fraction", "validation_fraction",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise GNNEvaluationError(f"GNN config missing: {', '.join(missing)}")
    return config


def _seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _graph(config: dict) -> dict[str, Tensor | str]:
    """Build a deterministic homophilic graph and its fixed node splits."""
    generator = torch.Generator().manual_seed(config["seed"])
    nodes = config["nodes"]
    classes = config["classes"]
    labels = torch.arange(nodes, dtype=torch.long) % classes
    features = torch.randn(nodes, config["feature_dim"], generator=generator)
    random_values = torch.rand(nodes, nodes, generator=generator)
    adjacency = torch.zeros(nodes, nodes, dtype=torch.float32)
    for left in range(nodes):
        for right in range(left + 1, nodes):
            probability = (
                config["same_class_edge_probability"]
                if labels[left] == labels[right]
                else config["different_class_edge_probability"]
            )
            if random_values[left, right] < probability:
                adjacency[left, right] = 1.0
                adjacency[right, left] = 1.0
    degree = adjacency.sum(dim=1, keepdim=True)
    normalized = adjacency / degree.clamp_min(1.0)
    permutation = torch.randperm(nodes, generator=generator)
    train_count = int(nodes * config["train_fraction"])
    validation_count = int(nodes * config["validation_fraction"])
    train = permutation[:train_count]
    validation = permutation[train_count : train_count + validation_count]
    test = permutation[train_count + validation_count :]
    payload = {
        "labels": labels.tolist(),
        "features": features.tolist(),
        "adjacency": adjacency.tolist(),
        "train": train.tolist(),
        "validation": validation.tolist(),
        "test": test.tolist(),
    }
    graph_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "features": features,
        "labels": labels,
        "adjacency": adjacency,
        "normalized_adjacency": normalized,
        "train": train,
        "validation": validation,
        "test": test,
        "graph_hash": graph_hash,
    }


class MessagePassingGNN(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, classes: int) -> None:
        super().__init__()
        self.self_projection = nn.Linear(feature_dim, hidden_dim)
        self.neighbor_projection = nn.Linear(feature_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, classes)

    def forward(self, features: Tensor, normalized_adjacency: Tensor) -> Tensor:
        hidden = torch.relu(
            self.self_projection(features)
            + self.neighbor_projection(normalized_adjacency @ features)
        )
        return self.output(hidden)


class FeatureOnlyMLP(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, classes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, classes)
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)


def _accuracy(logits: Tensor, labels: Tensor, indices: Tensor) -> float:
    if len(indices) == 0:
        return 0.0
    return float(logits[indices].argmax(dim=1).eq(labels[indices]).float().mean())


def _train_model(
    model: nn.Module,
    inputs: tuple[Tensor, ...],
    labels: Tensor,
    train: Tensor,
    config: dict,
) -> list[float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    losses = []
    model.train()
    for _ in range(config["training_steps"]):
        logits = model(*inputs)
        loss = nn.functional.cross_entropy(logits[train], labels[train])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def _majority(labels: Tensor, indices: Tensor, classes: int) -> int:
    counts = torch.bincount(labels[indices], minlength=classes)
    return int(counts.argmax())


def graph_baseline(graph: dict, split: str, classes: int) -> float:
    indices = graph[split]
    train = set(int(value) for value in graph["train"])
    labels = graph["labels"]
    adjacency = graph["adjacency"]
    predictions = []
    fallback = _majority(labels, graph["train"], classes)
    for node in indices.tolist():
        neighbors = [
            neighbor
            for neighbor in adjacency[node].nonzero().flatten().tolist()
            if neighbor in train
        ]
        if not neighbors:
            predictions.append(fallback)
            continue
        predictions.append(_majority(labels, torch.tensor(neighbors), classes))
    return float(torch.tensor(predictions).eq(labels[indices]).float().mean())


def random_baseline(graph: dict, split: str, classes: int, seed: int) -> float:
    generator = torch.Generator().manual_seed(seed + 1)
    predictions = torch.randint(classes, (len(graph[split]),), generator=generator)
    return float(predictions.eq(graph["labels"][graph[split]]).float().mean())


def _fit_baseline_mlp(graph: dict, config: dict) -> tuple[float, list[float]]:
    _seed(config["seed"])
    model = FeatureOnlyMLP(config["feature_dim"], config["hidden_dim"], config["classes"])
    losses = _train_model(model, (graph["features"],), graph["labels"], graph["train"], config)
    model.eval()
    with torch.no_grad():
        accuracy = _accuracy(model(graph["features"]), graph["labels"], graph["validation"])
    return accuracy, losses


def train_gnn(config_path: Path, root: Path) -> dict:
    config = load_gnn_config(config_path)
    _seed(config["seed"])
    graph = _graph(config)
    model = MessagePassingGNN(config["feature_dim"], config["hidden_dim"], config["classes"])
    losses = _train_model(
        model,
        (graph["features"], graph["normalized_adjacency"]),
        graph["labels"],
        graph["train"],
        config,
    )
    output = root / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "model.pt"
    torch.save(
        {
            "config": config,
            "graph_hash": graph["graph_hash"],
            "state_dict": model.state_dict(),
            "losses": losses,
        },
        checkpoint_path,
    )
    metadata = {
        "gnn_id": config["id"],
        "initialization": config["initialization"],
        "seed": config["seed"],
        "graph_hash": graph["graph_hash"],
        "train_nodes": len(graph["train"]),
        "validation_nodes": len(graph["validation"]),
        "test_nodes": len(graph["test"]),
        "training_steps": config["training_steps"],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "start_loss": losses[0],
        "final_loss": losses[-1],
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metadata


def _load(config_path: Path, root: Path) -> tuple[dict, dict, MessagePassingGNN]:
    config = load_gnn_config(config_path)
    checkpoint_path = root / config["output_dir"] / "model.pt"
    if not checkpoint_path.is_file():
        raise GNNEvaluationError(f"GNN checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    graph = _graph(config)
    if checkpoint["graph_hash"] != graph["graph_hash"]:
        raise GNNEvaluationError("GNN checkpoint graph hash does not match config")
    model = MessagePassingGNN(config["feature_dim"], config["hidden_dim"], config["classes"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return config, graph, model


def evaluate_gnn(config_path: Path, root: Path, split: str = "validation") -> dict:
    if split not in {"validation", "test"}:
        raise GNNEvaluationError("GNN evaluation split must be validation or test")
    config, graph, model = _load(config_path, root)
    with torch.no_grad():
        gnn_logits = model(graph["features"], graph["normalized_adjacency"])
    gnn_accuracy = _accuracy(gnn_logits, graph["labels"], graph[split])
    mlp_accuracy, _ = _fit_baseline_mlp(graph, config)
    graph_accuracy = graph_baseline(graph, split, config["classes"])
    random_accuracy = random_baseline(graph, split, config["classes"], config["seed"])
    best_baseline = max(mlp_accuracy, graph_accuracy, random_accuracy)
    return {
        "split": split,
        "graph_hash": graph["graph_hash"],
        "gnn_accuracy": gnn_accuracy,
        "mlp_accuracy": mlp_accuracy,
        "graph_baseline_accuracy": graph_accuracy,
        "random_baseline_accuracy": random_accuracy,
        "best_baseline_accuracy": best_baseline,
        "gnn_delta_vs_best_baseline": gnn_accuracy - best_baseline,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint_sha256": sha256_file(root / config["output_dir"] / "model.pt"),
    }


def compare_gnn(config_path: Path, root: Path, split: str = "validation") -> dict:
    result = evaluate_gnn(config_path, root, split)
    result["retained"] = result["gnn_delta_vs_best_baseline"] > 0.0
    result["conclusion"] = (
        "GNN adds validation accuracy beyond the strongest baseline."
        if result["retained"]
        else "GNN adds no measured validation accuracy beyond the strongest baseline."
    )
    return result
