from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import torch
import yaml
from torch import Tensor, nn

from hai.data.pipeline import sha256_file


class AudioError(ValueError):
    """Raised when the procedural audio experiment contract is invalid."""


def load_audio_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("id") != "audio-v1":
        raise AudioError("audio config must declare audio-v1")
    if config.get("initialization") != "random":
        raise AudioError("audio model must explicitly request random initialization")
    required = (
        "seed", "samples", "sample_length", "sample_rate", "classes", "training_steps",
        "batch_size", "learning_rate", "train_fraction", "validation_fraction", "output_dir",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise AudioError(f"audio config missing: {', '.join(missing)}")
    return config


def _seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dataset(config: dict) -> dict:
    generator = torch.Generator().manual_seed(config["seed"])
    labels = torch.arange(config["samples"], dtype=torch.long) % config["classes"]
    time = torch.arange(config["sample_length"], dtype=torch.float32) / config["sample_rate"]
    base_frequencies = torch.tensor(config["base_frequencies"], dtype=torch.float32)
    phases = torch.rand(config["samples"], generator=generator) * 6.283185307
    amplitudes = 0.75 + torch.rand(config["samples"], generator=generator) * 0.2
    waves = []
    for index in range(config["samples"]):
        frequency = base_frequencies[labels[index]]
        wave = amplitudes[index] * torch.sin(2 * torch.pi * frequency * time + phases[index])
        noise = torch.randn(config["sample_length"], generator=generator) * config["noise_level"]
        waves.append((wave + noise).clamp(-1.0, 1.0))
    waveforms = torch.stack(waves).unsqueeze(1)
    permutation = torch.randperm(config["samples"], generator=generator)
    train_count = int(config["samples"] * config["train_fraction"])
    validation_count = int(config["samples"] * config["validation_fraction"])
    splits = {
        "train": permutation[:train_count],
        "validation": permutation[train_count : train_count + validation_count],
        "test": permutation[train_count + validation_count :],
    }
    payload = {
        "waveforms": waveforms.tolist(),
        "labels": labels.tolist(),
        "splits": {key: value.tolist() for key, value in splits.items()},
    }
    data_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"waveforms": waveforms, "labels": labels, "splits": splits, "data_hash": data_hash}


class TinyAudioCNN(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, stride=2, padding=4),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(32, config["classes"])

    def forward(self, waveforms: Tensor) -> Tensor:
        return self.classifier(self.features(waveforms).squeeze(-1))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def data_verify(config_path: Path) -> dict:
    config = load_audio_config(config_path)
    data = _dataset(config)
    return {
        "dataset": "procedural-tones-v1",
        "external_download": False,
        "license": "original procedural waveforms; no third-party audio artifacts",
        "data_hash": data["data_hash"],
        "sample_rate": config["sample_rate"],
        "sample_length": config["sample_length"],
        "train_samples": len(data["splits"]["train"]),
        "validation_samples": len(data["splits"]["validation"]),
        "test_samples": len(data["splits"]["test"]),
        "config_sha256": sha256_file(config_path),
    }


def _evaluate(model: TinyAudioCNN, data: dict, indices: Tensor, device: torch.device) -> dict:
    model.eval()
    waveforms = data["waveforms"][indices].to(device)
    labels = data["labels"][indices].to(device)
    with torch.no_grad():
        logits = model(waveforms)
        loss = nn.functional.cross_entropy(logits, labels)
    return {
        "loss": float(loss.cpu()),
        "accuracy": float(logits.argmax(dim=1).eq(labels).float().mean().cpu()),
    }


def train_audio(config_path: Path, root: Path) -> dict:
    config = load_audio_config(config_path)
    _seed(config["seed"])
    data = _dataset(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyAudioCNN(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    train_indices = data["splits"]["train"]
    losses = []
    for step in range(config["training_steps"]):
        start = (step * config["batch_size"]) % len(train_indices)
        indices = train_indices[start : start + config["batch_size"]]
        if len(indices) < config["batch_size"]:
            indices = torch.cat((indices, train_indices[: config["batch_size"] - len(indices)]))
        waveforms = data["waveforms"][indices].to(device)
        labels = data["labels"][indices].to(device)
        loss = nn.functional.cross_entropy(model(waveforms), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    output = root / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.pt"
    torch.save(
        {"config": config, "data_hash": data["data_hash"], "state_dict": model.state_dict()},
        checkpoint_path,
    )
    validation = _evaluate(model, data, data["splits"]["validation"], device)
    return {
        "experiment": config["id"],
        "device": str(device),
        "initialization": config["initialization"],
        "data_hash": data["data_hash"],
        "parameter_count": model.parameter_count(),
        "training_steps": config["training_steps"],
        "start_loss": losses[0],
        "final_loss": losses[-1],
        "validation_loss": validation["loss"],
        "validation_accuracy": validation["accuracy"],
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }


def evaluate_audio(config_path: Path, root: Path, split: str = "validation") -> dict:
    if split not in {"validation", "test"}:
        raise AudioError("audio evaluation split must be validation or test")
    config = load_audio_config(config_path)
    data = _dataset(config)
    checkpoint_path = root / config["output_dir"] / "checkpoint.pt"
    if not checkpoint_path.is_file():
        raise AudioError(f"audio checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["data_hash"] != data["data_hash"]:
        raise AudioError("audio checkpoint data hash does not match config")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyAudioCNN(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    metrics = _evaluate(model, data, data["splits"][split], device)
    majority = int(torch.bincount(data["labels"][data["splits"][split]]).argmax())
    majority_accuracy = float(
        data["labels"][data["splits"][split]].eq(majority).float().mean()
    )
    generator = torch.Generator().manual_seed(config["seed"])
    random_predictions = torch.randint(
        config["classes"], (len(data["splits"][split]),), generator=generator
    )
    return {
        "split": split,
        "data_hash": data["data_hash"],
        "parameter_count": model.parameter_count(),
        "loss": metrics["loss"],
        "accuracy": metrics["accuracy"],
        "majority_baseline_accuracy": majority_accuracy,
        "random_baseline_accuracy": float(
            random_predictions.eq(data["labels"][data["splits"][split]]).float().mean()
        ),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
