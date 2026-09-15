from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import torch
import yaml
from torch import Tensor, nn

from hai.data.pipeline import sha256_file
from hai.models.audio import TinyAudioCNN
from hai.models.audio import _dataset as audio_dataset


class SpeechError(ValueError):
    """Raised when audio-text alignment or ASR provenance is invalid."""


TRANSCRIPTS = ("low", "midlow", "midhigh", "high")


def _seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _speech_data(config: dict) -> dict:
    base = audio_dataset(config)
    group_size = config["source_group_size"]
    groups = config["samples"] // group_size
    generator = torch.Generator().manual_seed(config["seed"] + 17)
    group_order = torch.randperm(groups, generator=generator)
    train_groups = int(groups * config["train_fraction"])
    validation_groups = int(groups * config["validation_fraction"])
    group_splits = {
        "train": group_order[:train_groups],
        "validation": group_order[train_groups : train_groups + validation_groups],
        "test": group_order[train_groups + validation_groups :],
    }
    splits = {
        split: torch.tensor(
            [
                group * group_size + offset
                for group in groups_for_split
                for offset in range(group_size)
            ],
            dtype=torch.long,
        )
        for split, groups_for_split in group_splits.items()
    }
    source_ids = torch.arange(config["samples"], dtype=torch.long) // group_size
    transcript_ids = base["labels"].clone()
    payload = {
        "audio_hash": base["data_hash"],
        "source_ids": source_ids.tolist(),
        "transcript_ids": transcript_ids.tolist(),
        "splits": {key: value.tolist() for key, value in splits.items()},
    }
    data_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "waveforms": base["waveforms"],
        "transcript_ids": transcript_ids,
        "source_ids": source_ids,
        "splits": splits,
        "data_hash": data_hash,
        "audio_hash": base["data_hash"],
    }


def _config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("id") != "audio-v1":
        raise SpeechError("speech commands require the audio-v1 config")
    if config.get("initialization") != "random":
        raise SpeechError("speech models must explicitly request random initialization")
    return config


class AudioTextAlignment(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.audio = TinyAudioCNN(config).features
        self.audio_projection = nn.Linear(32, config["alignment_dim"])
        self.text = nn.Embedding(config["classes"], config["alignment_dim"])

    def forward(self, waveforms: Tensor, transcript_ids: Tensor) -> tuple[Tensor, Tensor]:
        audio = nn.functional.normalize(
            self.audio_projection(self.audio(waveforms).squeeze(-1)), dim=1
        )
        text = nn.functional.normalize(self.text(transcript_ids), dim=1)
        return audio, text


def _alignment_loss(audio: Tensor, text: Tensor, labels: Tensor) -> Tensor:
    logits = audio @ text.T * 10.0
    positive = labels[:, None].eq(labels[None, :])
    first = -(
        torch.logsumexp(logits.masked_fill(~positive, float("-inf")), dim=1)
        - torch.logsumexp(logits, dim=1)
    ).mean()
    second = -(
        torch.logsumexp(logits.T.masked_fill(~positive.T, float("-inf")), dim=1)
        - torch.logsumexp(logits.T, dim=1)
    ).mean()
    return (first + second) / 2


def data_verify(config_path: Path) -> dict:
    config = _config(config_path)
    data = _speech_data(config)
    train_sources = set(data["source_ids"][data["splits"]["train"]].tolist())
    validation_sources = set(data["source_ids"][data["splits"]["validation"]].tolist())
    return {
        "dataset": "procedural-tones-speech-pairs-v1",
        "data_hash": data["data_hash"],
        "audio_hash": data["audio_hash"],
        "train_records": len(data["splits"]["train"]),
        "validation_records": len(data["splits"]["validation"]),
        "test_records": len(data["splits"]["test"]),
        "source_overlap": bool(train_sources & validation_sources),
        "external_download": False,
        "config_sha256": sha256_file(config_path),
    }


def train_alignment(config_path: Path, root: Path) -> dict:
    config = _config(config_path)
    _seed(config["seed"])
    data = _speech_data(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioTextAlignment(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    train_indices = data["splits"]["train"]
    losses = []
    for step in range(config["alignment_steps"]):
        start = (step * config["batch_size"]) % len(train_indices)
        indices = train_indices[start : start + config["batch_size"]]
        if len(indices) < config["batch_size"]:
            indices = torch.cat((indices, train_indices[: config["batch_size"] - len(indices)]))
        waveforms = data["waveforms"][indices].to(device)
        labels = data["transcript_ids"][indices].to(device)
        audio, text = model(waveforms, labels)
        loss = _alignment_loss(audio, text, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    output = root / config["output_dir"] / "alignment"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.pt"
    torch.save({"data_hash": data["data_hash"], "state_dict": model.state_dict()}, checkpoint_path)
    validation = data["splits"]["validation"]
    with torch.no_grad():
        audio, text = model(
            data["waveforms"][validation].to(device),
            data["transcript_ids"][validation].to(device),
        )
        scores = audio @ text.T
        ranks = scores.argsort(dim=1, descending=True)
        hits = data["transcript_ids"][validation].to(device)[ranks].eq(
            data["transcript_ids"][validation].to(device)[:, None]
        )
    return {
        "device": str(device),
        "initialization": config["initialization"],
        "data_hash": data["data_hash"],
        "training_steps": config["alignment_steps"],
        "start_loss": losses[0],
        "final_loss": losses[-1],
        "validation_recall_at_1": float(hits[:, :1].any(dim=1).float().mean()),
        "random_recall_at_1": 1.0 / config["classes"],
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }


def train_asr(config_path: Path, root: Path) -> dict:
    config = _config(config_path)
    _seed(config["seed"])
    data = _speech_data(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyAudioCNN(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    train_indices = data["splits"]["train"]
    losses = []
    for step in range(config["asr_steps"]):
        start = (step * config["batch_size"]) % len(train_indices)
        indices = train_indices[start : start + config["batch_size"]]
        if len(indices) < config["batch_size"]:
            indices = torch.cat((indices, train_indices[: config["batch_size"] - len(indices)]))
        waveforms = data["waveforms"][indices].to(device)
        labels = data["transcript_ids"][indices].to(device)
        loss = nn.functional.cross_entropy(model(waveforms), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    output = root / config["output_dir"] / "asr"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.pt"
    torch.save({"data_hash": data["data_hash"], "state_dict": model.state_dict()}, checkpoint_path)
    return {
        "device": str(device),
        "initialization": config["initialization"],
        "data_hash": data["data_hash"],
        "training_steps": config["asr_steps"],
        "start_loss": losses[0],
        "final_loss": losses[-1],
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }


def _wer(reference: list[str], hypothesis: list[str]) -> float:
    distance = [[0] * (len(hypothesis) + 1) for _ in range(len(reference) + 1)]
    for row in range(len(reference) + 1):
        distance[row][0] = row
    for column in range(len(hypothesis) + 1):
        distance[0][column] = column
    for row in range(1, len(reference) + 1):
        for column in range(1, len(hypothesis) + 1):
            distance[row][column] = min(
                distance[row - 1][column] + 1,
                distance[row][column - 1] + 1,
                distance[row - 1][column - 1] + (reference[row - 1] != hypothesis[column - 1]),
            )
    return distance[-1][-1] / max(1, len(reference))


def evaluate_alignment(config_path: Path, root: Path) -> dict:
    config = _config(config_path)
    data = _speech_data(config)
    checkpoint_path = root / config["output_dir"] / "alignment" / "checkpoint.pt"
    if not checkpoint_path.is_file():
        raise SpeechError(f"alignment checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["data_hash"] != data["data_hash"]:
        raise SpeechError("alignment checkpoint data hash does not match config")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioTextAlignment(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    indices = data["splits"]["validation"]
    with torch.no_grad():
        audio, text = model(
            data["waveforms"][indices].to(device), data["transcript_ids"][indices].to(device)
        )
        retrieved_indices = (audio @ text.T).argmax(dim=1)
    labels = data["transcript_ids"][indices].to(device)
    retrieved_labels = labels[retrieved_indices]
    return {
        "validation_recall_at_1": float(retrieved_labels.eq(labels).float().mean()),
        "class_aware_alignment_accuracy": float(retrieved_labels.eq(labels).float().mean()),
        "random_recall_at_1": 1.0 / config["classes"],
        "data_hash": data["data_hash"],
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }


def evaluate_asr(config_path: Path, root: Path) -> dict:
    config = _config(config_path)
    data = _speech_data(config)
    checkpoint_path = root / config["output_dir"] / "asr" / "checkpoint.pt"
    if not checkpoint_path.is_file():
        raise SpeechError(f"ASR checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["data_hash"] != data["data_hash"]:
        raise SpeechError("ASR checkpoint data hash does not match config")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyAudioCNN(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    indices = data["splits"]["validation"]
    with torch.no_grad():
        predictions = model(data["waveforms"][indices].to(device)).argmax(dim=1).cpu()
    references = [TRANSCRIPTS[int(label)] for label in data["transcript_ids"][indices]]
    hypotheses = [TRANSCRIPTS[int(label)] for label in predictions]
    return {
        "split": "validation",
        "wer": _wer(references, hypotheses),
        "exact_transcript_accuracy": (
            sum(ref == hyp for ref, hyp in zip(references, hypotheses, strict=True))
            / len(references)
        ),
        "held_out_source_groups": len(set(data["source_ids"][indices].tolist())),
        "source_overlap_with_train": bool(
            set(data["source_ids"][indices].tolist())
            & set(data["source_ids"][data["splits"]["train"]].tolist())
        ),
        "data_hash": data["data_hash"],
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
