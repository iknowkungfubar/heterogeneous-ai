from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import torch
import yaml
from torch import Tensor, nn

from hai.data.pipeline import sha256_file
from hai.models.vision import _dataset as vision_dataset


class MultimodalError(ValueError):
    """Raised when image-text alignment provenance or evaluation is invalid."""


CAPTIONS = (
    "a horizontal bar",
    "a vertical bar",
    "a diagonal line",
    "a square",
)


def load_multimodal_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("id") != "image-text-v1":
        raise MultimodalError("multimodal config must declare image-text-v1")
    if config.get("initialization") != "random":
        raise MultimodalError("multimodal encoders must explicitly request random initialization")
    return config


def _seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tokens(text: str) -> list[str]:
    return text.lower().split()


def _paired_data(config: dict) -> dict:
    vision_config = yaml.safe_load(
        (Path(__file__).resolve().parents[3] / "configs/models/vision.yaml").read_text(
            encoding="utf-8"
        )
    )
    vision_config.update(
        {"seed": config["seed"], "samples": config["samples"]}
    )
    data = vision_dataset(vision_config)
    labels = data["labels"]
    captions = [CAPTIONS[int(label)] for label in labels]
    train_text = [captions[int(index)] for index in data["train"]]
    vocabulary = {"<unk>": 0}
    for caption in train_text:
        for token in _tokens(caption):
            if token not in vocabulary:
                vocabulary[token] = len(vocabulary)
    token_rows = [
        [vocabulary.get(token, 0) for token in _tokens(caption)] for caption in captions
    ]
    text_width = max(len(row) for row in token_rows)
    text_ids = torch.tensor(
        [row + [0] * (text_width - len(row)) for row in token_rows], dtype=torch.long
    )
    payload = {
        "vision_hash": data["dataset_hash"],
        "captions": captions,
        "vocabulary": vocabulary,
        "train": data["train"].tolist(),
        "validation": data["validation"].tolist(),
        "test": data["test"].tolist(),
    }
    alignment_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        **data,
        "captions": captions,
        "text_ids": text_ids,
        "vocabulary": vocabulary,
        "alignment_hash": alignment_hash,
    }


class ImageEncoder(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.patch = nn.Conv2d(1, config["encoder_dim"], config["patch_size"], config["patch_size"])
        self.projection = nn.Linear(config["encoder_dim"], config["shared_dim"])

    def forward(self, images: Tensor) -> Tensor:
        features = self.patch(images).flatten(2).mean(dim=2)
        return nn.functional.normalize(self.projection(features), dim=1)


class TextEncoder(nn.Module):
    def __init__(self, config: dict, vocabulary_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, config["encoder_dim"])
        self.projection = nn.Linear(config["encoder_dim"], config["shared_dim"])

    def forward(self, tokens: Tensor) -> Tensor:
        features = self.embedding(tokens).mean(dim=1)
        return nn.functional.normalize(self.projection(features), dim=1)


class ImageTextModel(nn.Module):
    def __init__(self, config: dict, vocabulary_size: int) -> None:
        super().__init__()
        self.image = ImageEncoder(config)
        self.text = TextEncoder(config, vocabulary_size)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, images: Tensor, tokens: Tensor) -> tuple[Tensor, Tensor]:
        return self.image(images), self.text(tokens)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def _class_aware_loss(image_embeddings: Tensor, text_embeddings: Tensor, labels: Tensor) -> Tensor:
    logits = image_embeddings @ text_embeddings.T * 10.0
    positive = labels[:, None].eq(labels[None, :])
    image_loss = -(
        torch.logsumexp(logits.masked_fill(~positive, float("-inf")), dim=1)
        - torch.logsumexp(logits, dim=1)
    ).mean()
    text_loss = -(
        torch.logsumexp(logits.T.masked_fill(~positive.T, float("-inf")), dim=1)
        - torch.logsumexp(logits.T, dim=1)
    ).mean()
    return (image_loss + text_loss) / 2


def _pad_text(text_ids: Tensor, indices: Tensor, device: torch.device) -> Tensor:
    values = [text_ids[int(index)] for index in indices]
    width = max(len(value) for value in values)
    return torch.stack(
        [torch.cat((value, torch.zeros(width - len(value), dtype=torch.long))) for value in values]
    ).to(device)


def data_verify(config_path: Path) -> dict:
    config = load_multimodal_config(config_path)
    data = _paired_data(config)
    return {
        "dataset": "procedural-shapes-v1-with-captions",
        "alignment_hash": data["alignment_hash"],
        "image_dataset_hash": data["dataset_hash"],
        "vocabulary_size": len(data["vocabulary"]),
        "train_pairs": len(data["train"]),
        "validation_pairs": len(data["validation"]),
        "test_pairs": len(data["test"]),
        "leakage_resistant_split": True,
        "external_download": False,
        "config_sha256": sha256_file(config_path),
    }


def _embeddings(
    model: ImageTextModel, data: dict, indices: Tensor, device: torch.device
) -> tuple[Tensor, Tensor]:
    with torch.no_grad():
        images = data["images"][indices].to(device)
        texts = _pad_text(data["text_ids"], indices, device)
        return model.image(images), model.text(texts)


def _retrieval(
    image_embeddings: Tensor, text_embeddings: Tensor, labels: Tensor, seed: int
) -> dict:
    similarity = image_embeddings @ text_embeddings.T
    metrics = {}
    for direction, scores in (("image_to_text", similarity), ("text_to_image", similarity.T)):
        ranks = scores.argsort(dim=1, descending=True)
        hits = labels[ranks].eq(labels[:, None])
        values = {}
        for k in (1, 5, 10):
            values[f"recall_at_{k}"] = float(hits[:, :k].any(dim=1).float().mean())
        first = hits.float().argmax(dim=1)
        values["mrr"] = float((1.0 / (first + 1)).mean())
        metrics[direction] = values
    generator = torch.Generator().manual_seed(seed)
    random_ranks = torch.argsort(
        torch.rand(similarity.shape, generator=generator), dim=1, descending=True
    )
    random_hits = labels[random_ranks].eq(labels[:, None])
    metrics["random"] = {
        f"recall_at_{k}": float(random_hits[:, :k].any(dim=1).float().mean())
        for k in (1, 5, 10)
    }
    return metrics


def train_alignment(config_path: Path, root: Path) -> dict:
    config = load_multimodal_config(config_path)
    _seed(config["seed"])
    data = _paired_data(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageTextModel(config, len(data["vocabulary"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    losses = []
    for step in range(config["training_steps"]):
        start = (step * config["batch_size"]) % len(data["train"])
        indices = data["train"][start : start + config["batch_size"]]
        if len(indices) < config["batch_size"]:
            indices = torch.cat((indices, data["train"][: config["batch_size"] - len(indices)]))
        images = data["images"][indices].to(device)
        texts = _pad_text(data["text_ids"], indices, device)
        labels = data["labels"][indices].to(device)
        image_embeddings, text_embeddings = model(images, texts)
        loss = _class_aware_loss(image_embeddings, text_embeddings, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    output = root / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.pt"
    torch.save(
        {
            "config": config,
            "alignment_hash": data["alignment_hash"],
            "vocabulary": data["vocabulary"],
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    image_embeddings, text_embeddings = _embeddings(model, data, data["validation"], device)
    metrics = _retrieval(
        image_embeddings,
        text_embeddings,
        data["labels"][data["validation"]].to(device),
        config["seed"],
    )
    return {
        "experiment": config["id"],
        "device": str(device),
        "initialization": config["initialization"],
        "alignment_hash": data["alignment_hash"],
        "training_steps": config["training_steps"],
        "parameter_count": model.parameter_count(),
        "start_loss": losses[0],
        "final_loss": losses[-1],
        "validation_pairs": len(data["validation"]),
        "validation_retrieval": metrics,
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }


def evaluate_alignment(config_path: Path, root: Path, directions: str) -> dict:
    config = load_multimodal_config(config_path)
    data = _paired_data(config)
    checkpoint_path = root / config["output_dir"] / "checkpoint.pt"
    if not checkpoint_path.is_file():
        raise MultimodalError(f"alignment checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["alignment_hash"] != data["alignment_hash"]:
        raise MultimodalError("alignment checkpoint hash does not match paired data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageTextModel(config, len(data["vocabulary"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    image_embeddings, text_embeddings = _embeddings(model, data, data["validation"], device)
    metrics = _retrieval(
        image_embeddings,
        text_embeddings,
        data["labels"][data["validation"]].to(device),
        config["seed"],
    )
    selected = [direction.strip() for direction in directions.split(",") if direction.strip()]
    if not set(selected).issubset({"image-to-text", "text-to-image"}):
        raise MultimodalError("directions must contain image-to-text and/or text-to-image")
    return {
        "directions": selected,
        "alignment_hash": data["alignment_hash"],
        "validation_pairs": len(data["validation"]),
        "retrieval": {
            direction.replace("-", "_"): metrics[direction.replace("-", "_")]
            for direction in selected
        },
        "random_baseline": metrics["random"],
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
