from __future__ import annotations

import json
import random
from pathlib import Path

import torch
import yaml
from torch import Tensor, nn

from hai.data.pipeline import _repo_path, sha256_file
from hai.models.transformer_smoke import _batch, _split_tokens, _tracking_run


class GRUGovernanceError(ValueError):
    """Raised when the independent sequence specialist violates its contract."""


def load_gru_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("initialization") != "random":
        raise GRUGovernanceError("GRU config must explicitly request random initialization")
    required = ("embedding_size", "hidden_size", "num_layers", "context_length", "seed")
    missing = [key for key in required if key not in config]
    if missing:
        raise GRUGovernanceError(f"GRU config missing: {', '.join(missing)}")
    return config


class GRULanguageModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config["vocab_size"], config["embedding_size"])
        self.gru = nn.GRU(
            input_size=config["embedding_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            batch_first=True,
        )
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"])

    def forward(self, tokens: Tensor) -> Tensor:
        hidden, _ = self.gru(self.embedding(tokens))
        return self.lm_head(hidden)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def _load_inputs(config_path: Path, tokenizer_path: Path, manifest_path: Path):
    config = load_gru_config(config_path)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise GRUGovernanceError("dataset manifest must be a YAML mapping")
    root = config_path.resolve().parents[2]
    train = manifest["processed"]["train"]
    validation = manifest["processed"]["validation"]
    train_path = _repo_path(root, train["path"], "train")
    validation_path = _repo_path(root, validation["path"], "validation")
    if (
        sha256_file(train_path) != train["sha256"]
        or sha256_file(validation_path) != validation["sha256"]
    ):
        raise GRUGovernanceError("dataset split checksum verification failed")
    config["vocab_size"] = len(
        json.loads(tokenizer_path.read_text(encoding="utf-8"))["model"]["vocab"]
    )
    return config, manifest, train_path, validation_path


def _save_checkpoint(
    path: Path, model: GRULanguageModel, optimizer: torch.optim.Optimizer, step: int, config: dict
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": config,
            "rng": torch.get_rng_state(),
        },
        path,
    )


def train_gru(
    config_path: Path,
    tokenizer_path: Path,
    manifest_path: Path,
    output: Path,
    max_steps: int,
    resume: Path | None = None,
) -> dict:
    config, manifest, train_path, _ = _load_inputs(config_path, tokenizer_path, manifest_path)
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRULanguageModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    start_step = 0
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
    blocks = _split_tokens(train_path, tokenizer_path, config["context_length"])
    if not blocks:
        raise GRUGovernanceError("train split produced no token blocks")
    model.train()
    losses: list[float] = []
    with _tracking_run(output, config, manifest, tokenizer_path) as tracker:
        for step in range(start_step, max_steps):
            inputs, targets = _batch(blocks, config["micro_batch_size"], step, device)
            loss = nn.functional.cross_entropy(
                model(inputs).reshape(-1, config["vocab_size"]), targets.reshape(-1)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"])
            optimizer.step()
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            if tracker is not None:
                tracker.log_metric("train_loss", loss_value, step=step + 1)
        checkpoint_path = output / "checkpoint.pt"
        _save_checkpoint(checkpoint_path, model, optimizer, max_steps, config)
        if tracker is not None:
            tracker.log_metric("parameter_count", model.parameter_count(), step=max_steps)
            tracker.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
    return {
        "experiment": output.name,
        "device": str(device),
        "parameter_count": model.parameter_count(),
        "start_loss": losses[0],
        "final_loss": losses[-1],
        "steps": max_steps,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }


def _load_gru_checkpoint(
    config_path: Path, checkpoint_path: Path, device: torch.device
) -> GRULanguageModel:
    config = load_gru_config(config_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config.update({"vocab_size": checkpoint["config"]["vocab_size"]})
    model = GRULanguageModel(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def evaluate_gru(
    config_path: Path, tokenizer_path: Path, manifest_path: Path, checkpoint_path: Path
) -> dict:
    config, _, _, validation_path = _load_inputs(config_path, tokenizer_path, manifest_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_gru_checkpoint(config_path, checkpoint_path, device)
    blocks = _split_tokens(validation_path, tokenizer_path, config["context_length"])
    inputs, targets = _batch(blocks, min(config["micro_batch_size"], len(blocks)), 0, device)
    with torch.no_grad():
        logits = model(inputs)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, config["vocab_size"]), targets.reshape(-1)
        )
    return {
        "split": "validation",
        "loss": float(loss.cpu()),
        "perplexity": float(torch.exp(loss).cpu()),
        "parameter_count": model.parameter_count(),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }


def compare_errors(
    transformer_config_path: Path,
    gru_config_path: Path,
    tokenizer_path: Path,
    manifest_path: Path,
    transformer_checkpoint: Path,
    gru_checkpoint: Path,
    max_blocks: int = 128,
) -> dict:
    from hai.models.transformer_smoke import TinyCausalDecoder, load_model_config

    transformer_config = load_model_config(transformer_config_path)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    root = transformer_config_path.resolve().parents[2]
    validation = manifest["processed"]["validation"]
    validation_path = _repo_path(root, validation["path"], "validation")
    tokenizer_config = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    vocab_size = len(tokenizer_config["model"]["vocab"])
    transformer_config["vocab_size"] = vocab_size
    gru_config, _, _, _ = _load_inputs(gru_config_path, tokenizer_path, manifest_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer = TinyCausalDecoder(transformer_config).to(device)
    transformer.load_state_dict(
        torch.load(transformer_checkpoint, map_location=device, weights_only=False)["model"]
    )
    transformer.eval()
    gru = _load_gru_checkpoint(gru_config_path, gru_checkpoint, device)
    blocks = _split_tokens(validation_path, tokenizer_path, gru_config["context_length"])
    counts = {"both_correct": 0, "transformer_only": 0, "gru_only": 0, "both_wrong": 0}
    for step in range(min(max_blocks, len(blocks))):
        inputs, targets = _batch(blocks, 1, step, device)
        with torch.no_grad():
            transformer_correct = transformer(inputs).argmax(dim=-1).eq(targets)
            gru_correct = gru(inputs).argmax(dim=-1).eq(targets)
        counts["both_correct"] += int((transformer_correct & gru_correct).sum())
        counts["transformer_only"] += int((transformer_correct & ~gru_correct).sum())
        counts["gru_only"] += int((~transformer_correct & gru_correct).sum())
        counts["both_wrong"] += int((~transformer_correct & ~gru_correct).sum())
    total = sum(counts.values())
    complementarity = (counts["transformer_only"] + counts["gru_only"]) / total if total else 0.0
    return {
        "blocks": min(max_blocks, len(blocks)),
        "tokens": total,
        "counts": counts,
        "complementarity_rate": complementarity,
    }
