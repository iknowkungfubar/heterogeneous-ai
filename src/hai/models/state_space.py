from __future__ import annotations

import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import torch
import yaml
from torch import Tensor, nn

from hai.data.pipeline import sha256_file


class StateSpaceError(ValueError):
    """Raised when the state-space compatibility or training contract fails."""


class PlainStateSpaceModel(nn.Module):
    """A dependency-free diagonal state-space recurrence for AMD compatibility."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.embedding = nn.Embedding(config["vocab_size"], config["embedding_size"])
        self.input_projection = nn.Linear(config["embedding_size"], config["hidden_size"])
        self.output = nn.Linear(config["hidden_size"], config["vocab_size"])
        self.decay_logit = nn.Parameter(torch.tensor(float(config["initial_decay_logit"])))

    def forward(self, tokens: Tensor) -> Tensor:
        state = torch.zeros(
            tokens.shape[0], self.input_projection.out_features, device=tokens.device
        )
        decay = torch.sigmoid(self.decay_logit)
        outputs = []
        for step in range(tokens.shape[1]):
            proposal = torch.tanh(self.input_projection(self.embedding(tokens[:, step])))
            state = decay * state + (1.0 - decay) * proposal
            outputs.append(self.output(state))
        return torch.stack(outputs, dim=1)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class GRUBaseline(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.embedding = nn.Embedding(config["vocab_size"], config["embedding_size"])
        self.gru = nn.GRU(
            config["embedding_size"], config["hidden_size"], batch_first=True
        )
        self.output = nn.Linear(config["hidden_size"], config["vocab_size"])

    def forward(self, tokens: Tensor) -> Tensor:
        hidden, _ = self.gru(self.embedding(tokens))
        return self.output(hidden)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def load_state_space_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("id") != "ssm-v1":
        raise StateSpaceError("state-space config must declare ssm-v1")
    if config.get("initialization") != "random":
        raise StateSpaceError("state-space config must explicitly request random initialization")
    return config


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _task_batch(config: dict, seed: int, device: torch.device) -> tuple[Tensor, Tensor]:
    starts = torch.arange(config["batch_size"], device=device, dtype=torch.long)
    starts = (starts + seed) % config["vocab_size"]
    positions = torch.arange(config["sequence_length"] + 1, device=device)
    sequence = (starts[:, None] + positions[None, :]) % config["vocab_size"]
    return sequence[:, :-1], sequence[:, 1:]


def _task_hash(config: dict) -> str:
    task = {
        "vocab_size": config["vocab_size"],
        "sequence_length": config["sequence_length"],
        "train_seed": config["train_seed"],
        "validation_seed": config["validation_seed"],
    }
    return hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()


def compatibility_check(config_path: Path, root: Path) -> dict:
    config = load_state_space_config(config_path)
    device = _device()
    model = PlainStateSpaceModel(config).to(device)
    inputs, targets = _task_batch(config, config["train_seed"], device)
    with torch.no_grad():
        cpu_model = PlainStateSpaceModel(config)
        cpu_inputs, _ = _task_batch(config, config["train_seed"], torch.device("cpu"))
        cpu_logits = cpu_model(cpu_inputs)
    gpu_result = {"available": False, "forward_backward": False, "peak_memory_bytes": 0}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
        logits = model(inputs)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, config["vocab_size"]), targets.reshape(-1)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        gpu_result = {
            "available": True,
            "forward_backward": bool(torch.isfinite(loss).item()),
            "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
        }
    try:
        mamba_version = importlib.metadata.version("mamba-ssm")
        mamba_available = True
    except importlib.metadata.PackageNotFoundError:
        mamba_version = None
        mamba_available = False
    return {
        "implementation": "plain_pytorch_diagonal_state_space",
        "upstream_package": "mamba-ssm",
        "upstream_version": mamba_version,
        "upstream_install_mode": "not-installed; no external download performed"
        if not mamba_available
        else "environment package",
        "optimized_extension_available": mamba_available,
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "device": str(device),
        "cpu_forward": list(cpu_logits.shape)
        == [config["batch_size"], config["sequence_length"], config["vocab_size"]],
        "cpu_finite": bool(torch.isfinite(cpu_logits).all().item()),
        "gpu": gpu_result,
        "task_hash": _task_hash(config),
        "config_sha256": sha256_file(config_path),
        "root": str(root),
    }


def _step(
    model: nn.Module, optimizer: torch.optim.Optimizer, inputs: Tensor, targets: Tensor
) -> float:
    logits = model(inputs)
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach().cpu())


def _evaluate(model: nn.Module, config: dict, device: torch.device) -> dict:
    model.eval()
    inputs, targets = _task_batch(config, config["validation_seed"], device)
    with torch.no_grad():
        logits = model(inputs)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, config["vocab_size"]), targets.reshape(-1)
        )
    return {
        "validation_loss": float(loss.cpu()),
        "validation_perplexity": float(torch.exp(loss).cpu()),
        "validation_accuracy": float(logits.argmax(dim=-1).eq(targets).float().mean().cpu()),
    }


def _checkpoint_payload(
    model: nn.Module, optimizer: torch.optim.Optimizer, step: int, config: dict
) -> dict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": config,
        "task_hash": _task_hash(config),
    }


def _train_model(model: nn.Module, config: dict, steps: int, device: torch.device) -> dict:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    losses = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(steps):
        inputs, targets = _task_batch(config, config["train_seed"] + step, device)
        losses.append(_step(model, optimizer, inputs, targets))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    metrics = _evaluate(model, config, device)
    metrics.update(
        {
            "start_loss": losses[0],
            "final_loss": losses[-1],
            "steps": steps,
            "tokens_per_second": (
                steps * config["batch_size"] * config["sequence_length"] / max(elapsed, 1e-9)
            ),
            "parameter_count": model.parameter_count(),
            "peak_memory_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
            ),
        }
    )
    return metrics, optimizer


def smoke_train(config_path: Path, root: Path, steps: int) -> dict:
    config = load_state_space_config(config_path)
    if steps <= 0:
        raise StateSpaceError("smoke-train steps must be positive")
    _seed(config["seed"])
    device = _device()
    model = PlainStateSpaceModel(config)
    metrics, optimizer = _train_model(model, config, steps, device)
    output = root / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.pt"
    torch.save(_checkpoint_payload(model, optimizer, steps, config), checkpoint_path)
    reloaded = PlainStateSpaceModel(config).to(device)
    reloaded_optimizer = torch.optim.AdamW(reloaded.parameters(), lr=config["learning_rate"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    reloaded.load_state_dict(checkpoint["model"])
    reloaded_optimizer.load_state_dict(checkpoint["optimizer"])
    probe, _ = _task_batch(config, config["validation_seed"], device)
    with torch.no_grad():
        checkpoint_round_trip = bool(torch.equal(model(probe), reloaded(probe)))
    result = {
        "experiment": config["id"],
        "device": str(device),
        "initialization": config["initialization"],
        "task_hash": _task_hash(config),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_round_trip": checkpoint_round_trip,
        "resumed_step": int(checkpoint["step"]),
        **metrics,
    }
    (output / "metadata.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def benchmark_against_gru(config_path: Path, root: Path, against: str, steps: int) -> dict:
    if against != "GRU-0001":
        raise StateSpaceError("only the frozen GRU-0001 comparison is supported")
    config = load_state_space_config(config_path)
    device = _device()
    _seed(config["seed"])
    ssm_metrics, _ = _train_model(PlainStateSpaceModel(config), config, steps, device)
    _seed(config["seed"])
    gru_metrics, _ = _train_model(GRUBaseline(config), config, steps, device)
    return {
        "against": against,
        "device": str(device),
        "task_hash": _task_hash(config),
        "ssm": ssm_metrics,
        "gru": gru_metrics,
        "ssm_validation_loss_delta": (
            ssm_metrics["validation_loss"] - gru_metrics["validation_loss"]
        ),
        "ssm_validation_accuracy_delta": (
            ssm_metrics["validation_accuracy"] - gru_metrics["validation_accuracy"]
        ),
        "ssm_throughput_delta": (
            ssm_metrics["tokens_per_second"] - gru_metrics["tokens_per_second"]
        ),
        "ssm_replaces_gru": ssm_metrics["validation_loss"] < gru_metrics["validation_loss"],
    }
