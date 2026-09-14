from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

import torch
import yaml
from torch import Tensor, nn

from hai.data.pipeline import sha256_file


class VisionError(ValueError):
    """Raised when the standalone vision experiment contract is invalid."""


def load_vision_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("id") != "vision-v1":
        raise VisionError("vision config must declare vision-v1")
    if config.get("initialization") != "random":
        raise VisionError("vision model must explicitly request random initialization")
    required = (
        "seed", "image_size", "classes", "samples", "patch_size", "embedding_dim",
        "training_steps", "batch_size", "learning_rate", "train_fraction",
        "validation_fraction", "output_dir",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise VisionError(f"vision config missing: {', '.join(missing)}")
    return config


def _seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _draw_shape(image: Tensor, label: int, offset: int) -> None:
    center = image.shape[-1] // 2 + offset
    if label == 0:
        image[0, max(0, center - 1) : min(image.shape[-1], center + 1), 2:-2] = 1.0
    elif label == 1:
        image[0, 2:-2, max(0, center - 1) : min(image.shape[-1], center + 1)] = 1.0
    elif label == 2:
        for diagonal in range(2, image.shape[-1] - 2):
            row = diagonal + offset
            if 0 <= row < image.shape[-1]:
                image[0, row, diagonal] = 1.0
                if diagonal + 1 < image.shape[-1]:
                    image[0, row, diagonal + 1] = 1.0
    else:
        image[0, max(1, center - 3) : min(image.shape[-1], center + 3), max(1, center - 3)] = 1.0
        image[
            0,
            max(1, center - 3) : min(image.shape[-1], center + 3),
            min(image.shape[-1] - 1, center + 3),
        ] = 1.0
        image[
            0,
            max(1, center - 3),
            max(1, center - 3) : min(image.shape[-1], center + 4),
        ] = 1.0
        image[
            0,
            min(image.shape[-1] - 1, center + 2),
            max(1, center - 3) : min(image.shape[-1], center + 4),
        ] = 1.0


def _dataset(config: dict) -> dict[str, Tensor | str]:
    generator = torch.Generator().manual_seed(config["seed"])
    images = torch.zeros(config["samples"], 1, config["image_size"], config["image_size"])
    labels = torch.arange(config["samples"], dtype=torch.long) % config["classes"]
    offsets = torch.randint(-1, 2, (config["samples"],), generator=generator)
    noise = torch.rand(images.shape, generator=generator) * 0.08
    for index in range(config["samples"]):
        _draw_shape(images[index], int(labels[index]), int(offsets[index]))
    images = (images + noise).clamp(0.0, 1.0)
    permutation = torch.randperm(config["samples"], generator=generator)
    train_count = int(config["samples"] * config["train_fraction"])
    validation_count = int(config["samples"] * config["validation_fraction"])
    train = permutation[:train_count]
    validation = permutation[train_count : train_count + validation_count]
    test = permutation[train_count + validation_count :]
    payload = {
        "images": images.tolist(),
        "labels": labels.tolist(),
        "train": train.tolist(),
        "validation": validation.tolist(),
        "test": test.tolist(),
    }
    dataset_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "images": images,
        "labels": labels,
        "train": train,
        "validation": validation,
        "test": test,
        "dataset_hash": dataset_hash,
    }


class TinyVisionTransformer(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        patches_per_side = config["image_size"] // config["patch_size"]
        patch_count = patches_per_side * patches_per_side
        self.patch_embedding = nn.Conv2d(
            1, config["embedding_dim"], config["patch_size"], config["patch_size"]
        )
        self.class_token = nn.Parameter(torch.zeros(1, 1, config["embedding_dim"]))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, patch_count + 1, config["embedding_dim"])
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config["embedding_dim"],
            nhead=config["attention_heads"],
            dim_feedforward=config["feedforward_dim"],
            dropout=0.0,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config["layers"])
        self.norm = nn.LayerNorm(config["embedding_dim"])
        self.classifier = nn.Linear(config["embedding_dim"], config["classes"])

    def forward(self, images: Tensor) -> Tensor:
        patches = self.patch_embedding(images).flatten(2).transpose(1, 2)
        class_tokens = self.class_token.expand(images.shape[0], -1, -1)
        hidden = torch.cat((class_tokens, patches), dim=1) + self.position_embedding
        return self.classifier(self.norm(self.encoder(hidden)[:, 0]))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def data_verify(config_path: Path) -> dict:
    config = load_vision_config(config_path)
    data = _dataset(config)
    return {
        "dataset": "procedural-shapes-v1",
        "external_download": False,
        "license": "original procedural data; no third-party image artifacts",
        "dataset_hash": data["dataset_hash"],
        "samples": config["samples"],
        "train_samples": len(data["train"]),
        "validation_samples": len(data["validation"]),
        "test_samples": len(data["test"]),
        "class_count": config["classes"],
        "image_shape": list(data["images"].shape[1:]),
        "config_sha256": sha256_file(config_path),
    }


def _evaluate(
    model: TinyVisionTransformer, data: dict, indices: Tensor, device: torch.device
) -> dict:
    model.eval()
    images = data["images"][indices].to(device)
    labels = data["labels"][indices].to(device)
    with torch.no_grad():
        logits = model(images)
        loss = nn.functional.cross_entropy(logits, labels)
    return {
        "loss": float(loss.cpu()),
        "accuracy": float(logits.argmax(dim=1).eq(labels).float().mean().cpu()),
    }


def train_vision(config_path: Path, root: Path) -> dict:
    config = load_vision_config(config_path)
    _seed(config["seed"])
    data = _dataset(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyVisionTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    losses = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model.train()
    for step in range(config["training_steps"]):
        selection = data["train"][(step * config["batch_size"]) % len(data["train"]) :]
        if len(selection) < config["batch_size"]:
            selection = torch.cat(
                (selection, data["train"][: config["batch_size"] - len(selection)])
            )
        images = data["images"][selection].to(device)
        labels = data["labels"][selection].to(device)
        logits = model(images)
        loss = nn.functional.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    validation = _evaluate(model, data, data["validation"], device)
    output = root / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.pt"
    torch.save(
        {"config": config, "dataset_hash": data["dataset_hash"], "state_dict": model.state_dict()},
        checkpoint_path,
    )
    return {
        "experiment": config["id"],
        "initialization": config["initialization"],
        "device": str(device),
        "dataset_hash": data["dataset_hash"],
        "train_samples": len(data["train"]),
        "validation_samples": len(data["validation"]),
        "test_samples": len(data["test"]),
        "parameter_count": model.parameter_count(),
        "start_loss": losses[0],
        "final_loss": losses[-1],
        "validation_loss": validation["loss"],
        "validation_accuracy": validation["accuracy"],
        "training_steps": config["training_steps"],
        "images_per_second": (
            config["training_steps"] * config["batch_size"] / max(elapsed, 1e-9)
        ),
        "peak_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }


def evaluate_vision(config_path: Path, root: Path, split: str = "validation") -> dict:
    if split not in {"validation", "test"}:
        raise VisionError("vision evaluation split must be validation or test")
    config = load_vision_config(config_path)
    data = _dataset(config)
    checkpoint_path = root / config["output_dir"] / "checkpoint.pt"
    if not checkpoint_path.is_file():
        raise VisionError(f"vision checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["dataset_hash"] != data["dataset_hash"]:
        raise VisionError("vision checkpoint dataset hash does not match config")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyVisionTransformer(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    metrics = _evaluate(model, data, data[split], device)
    return {
        "split": split,
        "dataset_hash": data["dataset_hash"],
        "parameter_count": model.parameter_count(),
        "loss": metrics["loss"],
        "accuracy": metrics["accuracy"],
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
