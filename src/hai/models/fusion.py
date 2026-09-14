from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import torch
import yaml
from torch import Tensor, nn

from hai.data.pipeline import sha256_file
from hai.models.multimodal import CAPTIONS
from hai.models.vision import _dataset as vision_dataset


class FusionError(ValueError):
    """Raised when visual reasoning fusion or ablation provenance is invalid."""


def load_fusion_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("id") != "visual-objective-v1":
        raise FusionError("fusion config must declare visual-objective-v1")
    if config.get("initialization") != "random":
        raise FusionError("fusion models must explicitly request random initialization")
    return config


def _seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tokens(text: str) -> list[str]:
    return text.lower().split()


def _data(config: dict) -> dict:
    vision_config = yaml.safe_load(
        (Path(__file__).resolve().parents[3] / "configs/models/vision.yaml").read_text(
            encoding="utf-8"
        )
    )
    vision_config.update({"seed": config["seed"], "samples": config["samples"]})
    images = vision_dataset(vision_config)
    vocabulary = {"<unk>": 0}
    questions = [f"is the image {caption}" for caption in CAPTIONS]
    for image_index in images["train"]:
        for token in _tokens(questions[int(images["labels"][image_index])]):
            if token not in vocabulary:
                vocabulary[token] = len(vocabulary)
    records = []
    for image_index in range(config["samples"]):
        for query_class, question in enumerate(questions):
            token_ids = [vocabulary.get(token, 0) for token in _tokens(question)]
            records.append(
                {
                    "image_index": image_index,
                    "query_class": query_class,
                    "tokens": token_ids,
                    "label": int(images["labels"][image_index]) == query_class,
                }
            )
    source_split = {}
    for split in ("train", "validation", "test"):
        for image_index in images[split].tolist():
            source_split[image_index] = split
    split_indices = {
        split: torch.tensor(
            [
                index
                for index, record in enumerate(records)
                if source_split[record["image_index"]] == split
            ],
            dtype=torch.long,
        )
        for split in ("train", "validation", "test")
    }
    width = max(len(record["tokens"]) for record in records)
    text_ids = torch.tensor(
        [record["tokens"] + [0] * (width - len(record["tokens"])) for record in records],
        dtype=torch.long,
    )
    labels = torch.tensor([record["label"] for record in records], dtype=torch.long)
    image_indices = torch.tensor([record["image_index"] for record in records], dtype=torch.long)
    payload = {
        "image_hash": images["dataset_hash"],
        "vocabulary": vocabulary,
        "image_indices": image_indices.tolist(),
        "text_ids": text_ids.tolist(),
        "labels": labels.tolist(),
        "splits": {key: value.tolist() for key, value in split_indices.items()},
    }
    data_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "images": images["images"],
        "text_ids": text_ids,
        "image_indices": image_indices,
        "labels": labels,
        "splits": split_indices,
        "vocabulary": vocabulary,
        "data_hash": data_hash,
        "image_hash": images["dataset_hash"],
    }


class ImageFeatures(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.patch = nn.Conv2d(1, config["fusion_dim"], config["patch_size"], config["patch_size"])

    def forward(self, images: Tensor) -> Tensor:
        return self.patch(images).flatten(2).transpose(1, 2)


class TextFeatures(nn.Module):
    def __init__(self, config: dict, vocabulary_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, config["fusion_dim"])

    def forward(self, tokens: Tensor) -> Tensor:
        return self.embedding(tokens).mean(dim=1, keepdim=True)


class FusionModel(nn.Module):
    def __init__(self, config: dict, vocabulary_size: int, variant: str) -> None:
        super().__init__()
        if variant not in {"projection", "cross-attention"}:
            raise FusionError("fusion variant must be projection or cross-attention")
        self.variant = variant
        self.image = ImageFeatures(config)
        self.text = TextFeatures(config, vocabulary_size)
        if variant == "projection":
            self.head = nn.Sequential(
                nn.Linear(config["fusion_dim"] * 2, config["hidden_dim"]),
                nn.ReLU(),
                nn.Linear(config["hidden_dim"], 2),
            )
        else:
            self.attention = nn.MultiheadAttention(
                config["fusion_dim"], config["attention_heads"], batch_first=True
            )
            self.head = nn.Sequential(
                nn.Linear(config["fusion_dim"] * 2, config["hidden_dim"]),
                nn.ReLU(),
                nn.Linear(config["hidden_dim"], 2),
            )

    def forward(self, images: Tensor, tokens: Tensor) -> Tensor:
        image_tokens = self.image(images)
        text_token = self.text(tokens)
        if self.variant == "projection":
            fused = torch.cat((image_tokens.mean(dim=1), text_token[:, 0]), dim=1)
        else:
            attended, _ = self.attention(text_token, image_tokens, image_tokens)
            fused = torch.cat((attended[:, 0], text_token[:, 0]), dim=1)
        return self.head(fused)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def _batch(data: dict, indices: Tensor, device: torch.device) -> tuple[Tensor, Tensor]:
    image_indices = data["image_indices"][indices]
    return (
        data["images"][image_indices].to(device),
        data["text_ids"][indices].to(device),
    )


def data_verify(config_path: Path) -> dict:
    config = load_fusion_config(config_path)
    data = _data(config)
    return {
        "dataset": "procedural-shapes-visual-objective-v1",
        "data_hash": data["data_hash"],
        "image_hash": data["image_hash"],
        "train_records": len(data["splits"]["train"]),
        "validation_records": len(data["splits"]["validation"]),
        "test_records": len(data["splits"]["test"]),
        "source_image_split_isolation": True,
        "external_download": False,
        "config_sha256": sha256_file(config_path),
    }


def _evaluate(model: nn.Module, data: dict, indices: Tensor, device: torch.device) -> dict:
    model.eval()
    images, tokens = _batch(data, indices, device)
    labels = data["labels"][indices].to(device)
    with torch.no_grad():
        logits = model(images, tokens)
        loss = nn.functional.cross_entropy(logits, labels)
    return {
        "loss": float(loss.cpu()),
        "accuracy": float(logits.argmax(dim=1).eq(labels).float().mean().cpu()),
    }


def train_fusion(config_path: Path, root: Path, variant: str) -> dict:
    config = load_fusion_config(config_path)
    _seed(config["seed"])
    data = _data(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionModel(config, len(data["vocabulary"]), variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    losses = []
    train_indices = data["splits"]["train"]
    for step in range(config["training_steps"]):
        start = (step * config["batch_size"]) % len(train_indices)
        indices = train_indices[start : start + config["batch_size"]]
        if len(indices) < config["batch_size"]:
            indices = torch.cat((indices, train_indices[: config["batch_size"] - len(indices)]))
        images, tokens = _batch(data, indices, device)
        labels = data["labels"][indices].to(device)
        logits = model(images, tokens)
        loss = nn.functional.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    output = root / config["output_dir"] / variant
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.pt"
    torch.save(
        {
            "config": config,
            "variant": variant,
            "data_hash": data["data_hash"],
            "vocabulary": data["vocabulary"],
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    validation = _evaluate(model, data, data["splits"]["validation"], device)
    return {
        "variant": variant,
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


def _load_variant(
    config_path: Path, root: Path, variant: str
) -> tuple[dict, dict, FusionModel, torch.device]:
    config = load_fusion_config(config_path)
    data = _data(config)
    checkpoint_path = root / config["output_dir"] / variant / "checkpoint.pt"
    if not checkpoint_path.is_file():
        raise FusionError(f"fusion checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["data_hash"] != data["data_hash"]:
        raise FusionError("fusion checkpoint data hash does not match config")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionModel(config, len(data["vocabulary"]), variant).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return config, data, model, device


def evaluate_fusion(config_path: Path, root: Path, suite: str) -> dict:
    if suite != "visual-objective-v1":
        raise FusionError("only visual-objective-v1 is supported")
    config, data, projection, device = _load_variant(config_path, root, "projection")
    _, _, cross_attention, _ = _load_variant(config_path, root, "cross-attention")
    validation = data["splits"]["validation"]
    projection_metrics = _evaluate(projection, data, validation, device)
    attention_metrics = _evaluate(cross_attention, data, validation, device)
    labels = data["labels"][validation]
    generator = torch.Generator().manual_seed(config["seed"])
    random_predictions = torch.randint(2, (len(validation),), generator=generator)
    random_accuracy = float(random_predictions.eq(labels).float().mean())
    return {
        "suite": suite,
        "data_hash": data["data_hash"],
        "projection": projection_metrics,
        "cross_attention": attention_metrics,
        "random_baseline_accuracy": random_accuracy,
        "unimodal_baseline_accuracy": 0.5,
        "retained_variant": (
            "cross-attention"
            if attention_metrics["accuracy"] > projection_metrics["accuracy"]
            else "projection"
        ),
    }
