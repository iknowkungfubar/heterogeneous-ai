from __future__ import annotations

import json
import os
import random
from contextlib import contextmanager
from pathlib import Path

import torch
import yaml
from torch import Tensor, nn

from hai.data.pipeline import _repo_path, sha256_file


class TransformerGovernanceError(ValueError):
    """Raised when the smoke model provenance or runtime contract is invalid."""


def load_model_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("initialization") != "random":
        raise TransformerGovernanceError(
            "model config must explicitly request random initialization"
        )
    return config


class TinyCausalDecoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        width = config["hidden_size"]
        self.token_embedding = nn.Embedding(config["vocab_size"], width)
        self.position_embedding = nn.Embedding(config["context_length"], width)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=config["num_attention_heads"],
            dim_feedforward=config["intermediate_size"],
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=config["num_hidden_layers"])
        self.final_norm = nn.LayerNorm(width)
        self.lm_head = nn.Linear(width, config["vocab_size"], bias=False)
        if config.get("tie_word_embeddings"):
            self.lm_head.weight = self.token_embedding.weight

    def forward(self, tokens: Tensor) -> Tensor:
        _, length = tokens.shape
        positions = torch.arange(length, device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        mask = torch.triu(
            torch.ones(length, length, device=tokens.device, dtype=torch.bool), diagonal=1
        )
        return self.lm_head(self.final_norm(self.decoder(hidden, mask=mask)))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def _split_tokens(path: Path, tokenizer_path: Path, context_length: int) -> list[Tensor]:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos = tokenizer.token_to_id("<eos>")
    if eos is None:
        raise TransformerGovernanceError("tokenizer has no <eos> token")
    token_ids: list[int] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            token_ids.extend(tokenizer.encode(record["text"]).ids)
            token_ids.append(eos)
    width = context_length + 1
    return [
        torch.tensor(token_ids[offset : offset + width], dtype=torch.long)
        for offset in range(0, len(token_ids) - width + 1, context_length)
    ]


def _batch(
    blocks: list[Tensor], batch_size: int, step: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    selected = [blocks[(step * batch_size + index) % len(blocks)] for index in range(batch_size)]
    batch = torch.stack(selected).to(device)
    return batch[:, :-1], batch[:, 1:]


def _save_checkpoint(
    path: Path, model: TinyCausalDecoder, optimizer: torch.optim.Optimizer, step: int, config: dict
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


@contextmanager
def _tracking_run(output: Path, config: dict, manifest: dict, tokenizer_path: Path):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        yield None
        return
    try:
        import mlflow
    except ImportError as exc:
        raise TransformerGovernanceError(
            "MLFLOW_TRACKING_URI is set but mlflow is not installed"
        ) from exc
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "heterogeneous-ai"))
    with mlflow.start_run(run_name=output.name):
        mlflow.log_params(
            {
                "model_id": config.get("id", output.name),
                "seed": config["seed"],
                "hidden_size": config["hidden_size"],
                "layer_count": config.get("num_hidden_layers", config.get("num_layers")),
                "context_length": config["context_length"],
                "micro_batch_size": config["micro_batch_size"],
                "learning_rate": config["learning_rate"],
                "dataset_id": manifest.get("dataset_id", "unknown"),
                "tokenizer_sha256": sha256_file(tokenizer_path),
            }
        )
        mlflow.set_tags(
            {
                "initialization": config.get("initialization", "unknown"),
                "train_split_only": "true",
            }
        )
        yield mlflow


def train_model(
    config_path: Path,
    tokenizer_path: Path,
    manifest_path: Path,
    output: Path,
    max_steps: int,
    resume: Path | None = None,
) -> dict:
    config = load_model_config(config_path)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    train = manifest["processed"]["train"]
    validation = manifest["processed"]["validation"]
    root = config_path.resolve().parents[2]
    train_path = _repo_path(root, train["path"], "train")
    validation_path = _repo_path(root, validation["path"], "validation")
    if (
        sha256_file(train_path) != train["sha256"]
        or sha256_file(validation_path) != validation["sha256"]
    ):
        raise TransformerGovernanceError("dataset split checksum verification failed")
    config["vocab_size"] = len(
        json.loads(tokenizer_path.read_text(encoding="utf-8"))["model"]["vocab"]
    )
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyCausalDecoder(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    start_step = 0
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
    train_blocks = _split_tokens(train_path, tokenizer_path, config["context_length"])
    if not train_blocks:
        raise TransformerGovernanceError("train split produced no token blocks")
    model.train()
    losses = []
    with _tracking_run(output, config, manifest, tokenizer_path) as tracker:
        for step in range(start_step, max_steps):
            inputs, targets = _batch(train_blocks, config["micro_batch_size"], step, device)
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


def evaluate_model(
    config_path: Path, tokenizer_path: Path, manifest_path: Path, checkpoint_path: Path
) -> dict:
    config = load_model_config(config_path)
    config["vocab_size"] = len(
        json.loads(tokenizer_path.read_text(encoding="utf-8"))["model"]["vocab"]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = TinyCausalDecoder(config).to(device)
    model.load_state_dict(checkpoint["model"])
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    root = config_path.resolve().parents[2]
    validation = manifest["processed"]["validation"]
    blocks = _split_tokens(
        _repo_path(root, validation["path"], "validation"), tokenizer_path, config["context_length"]
    )
    model.eval()
    with torch.no_grad():
        inputs, targets = _batch(blocks, min(config["micro_batch_size"], len(blocks)), 0, device)
        loss = nn.functional.cross_entropy(
            model(inputs).reshape(-1, config["vocab_size"]), targets.reshape(-1)
        )
    return {
        "split": "validation",
        "loss": float(loss.cpu()),
        "perplexity": float(torch.exp(loss).cpu()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }


def generate_text(
    config_path: Path,
    tokenizer_path: Path,
    checkpoint_path: Path,
    prompt: str,
    max_new_tokens: int = 32,
) -> dict:
    config = load_model_config(config_path)
    config["vocab_size"] = len(
        json.loads(tokenizer_path.read_text(encoding="utf-8"))["model"]["vocab"]
    )
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = TinyCausalDecoder(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    tokens = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = tokens[:, -config["context_length"] :]
            next_token = model(context)[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens = torch.cat((tokens, next_token), dim=1)
    return {
        "prompt": prompt,
        "text": tokenizer.decode(tokens[0].tolist(), skip_special_tokens=False),
    }
