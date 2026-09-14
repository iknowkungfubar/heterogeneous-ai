from __future__ import annotations

import json
from pathlib import Path

import yaml

from hai.data.pipeline import _repo_path, sha256_file


class TokenizerGovernanceError(ValueError):
    """Raised when a tokenizer violates the project artifact contract."""


def _load_tokenizer_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TokenizerGovernanceError("tokenizer config must be a YAML mapping")
    required = ("id", "type", "vocab_size", "train_split_only", "special_tokens", "output_dir")
    missing = [key for key in required if key not in config]
    if missing:
        raise TokenizerGovernanceError(f"tokenizer config missing: {', '.join(missing)}")
    if config["type"] != "bpe" or config["train_split_only"] is not True:
        raise TokenizerGovernanceError("only train_split_only BPE tokenizers are supported")
    if not isinstance(config["vocab_size"], int) or config["vocab_size"] < 16:
        raise TokenizerGovernanceError("vocab_size must be an integer of at least 16")
    tokens = config["special_tokens"]
    if not isinstance(tokens, list) or len(tokens) != len(set(tokens)):
        raise TokenizerGovernanceError("special_tokens must be a list of unique strings")
    if not all(isinstance(token, str) and token for token in tokens):
        raise TokenizerGovernanceError("special_tokens must contain non-empty strings")
    return config


def _manifest_train_path(manifest_path: Path, root: Path) -> tuple[dict, Path]:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TokenizerGovernanceError("dataset manifest must be a YAML mapping")
    train = manifest.get("processed", {}).get("train")
    if not isinstance(train, dict):
        raise TokenizerGovernanceError("dataset manifest has no processed train split")
    relative = train.get("path")
    if not isinstance(relative, str) or any(
        part in relative.lower() for part in ("test", "validation")
    ):
        raise TokenizerGovernanceError("tokenizer train input must be the train split only")
    train_path = _repo_path(root, relative, "tokenizer train")
    if not train_path.is_file() or sha256_file(train_path) != train.get("sha256"):
        raise TokenizerGovernanceError("train split checksum verification failed")
    return manifest, train_path


def _artifact_paths(config: dict, root: Path) -> tuple[Path, Path]:
    output = Path(config["output_dir"])
    if output.is_absolute() or ".." in output.parts:
        raise TokenizerGovernanceError("tokenizer output_dir must stay inside the repository")
    directory = root / output
    return directory / "tokenizer.json", directory / "metadata.yaml"


def _records(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TokenizerGovernanceError(
                    f"invalid train JSONL at line {line_number}"
                ) from exc
            if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                raise TokenizerGovernanceError(f"train record {line_number} has no text")
            yield record["text"]


def train_tokenizer(config_path: Path, manifest_path: Path, root: Path) -> dict:
    config = _load_tokenizer_config(config_path)
    manifest, train_path = _manifest_train_path(manifest_path, root)
    tokenizer_path, metadata_path = _artifact_paths(config, root)
    if tokenizer_path.exists() or metadata_path.exists():
        raise TokenizerGovernanceError("refusing to overwrite an existing tokenizer artifact")
    try:
        from tokenizers import Tokenizer
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.trainers import BpeTrainer
    except ImportError as exc:
        raise TokenizerGovernanceError("tokenizers is required for BPE training") from exc

    tokenizer = Tokenizer(BPE(unk_token="<unk>", byte_fallback=True))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=config["vocab_size"],
        special_tokens=config["special_tokens"],
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(_records(train_path), trainer=trainer)
    directory = tokenizer_path.parent
    directory.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokenizer_path))
    token_ids = {token: tokenizer.token_to_id(token) for token in config["special_tokens"]}
    if any(value is None for value in token_ids.values()):
        raise TokenizerGovernanceError("a configured special token was not included")
    metadata = {
        "tokenizer_id": config["id"],
        "type": "bpe",
        "vocab_size_configured": config["vocab_size"],
        "vocab_size_actual": tokenizer.get_vocab_size(),
        "special_tokens": token_ids,
        "dataset_manifest": str(manifest_path.resolve().relative_to(root.resolve())),
        "dataset_id": manifest.get("dataset_id"),
        "train_split": {
            "path": str(train_path.resolve().relative_to(root.resolve())),
            "sha256": sha256_file(train_path),
            "records": manifest["processed"]["train"].get("records"),
        },
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "pretrained_vocabulary": False,
    }
    metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    return metadata


def inspect_tokenizer(tokenizer_dir: Path, text: str) -> dict:
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise TokenizerGovernanceError(f"tokenizer artifact does not exist: {tokenizer_path}")
    try:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    except Exception as exc:
        raise TokenizerGovernanceError(f"could not load tokenizer: {exc}") from exc
    encoded = tokenizer.encode(text)
    return {
        "text": text,
        "ids": encoded.ids,
        "tokens": encoded.tokens,
        "decoded": tokenizer.decode(encoded.ids, skip_special_tokens=False),
    }


def verify_tokenizer(tokenizer_dir: Path, root: Path) -> dict:
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    metadata_path = tokenizer_dir / "metadata.yaml"
    if not tokenizer_path.is_file() or not metadata_path.is_file():
        raise TokenizerGovernanceError("tokenizer.json and metadata.yaml are both required")
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("pretrained_vocabulary") is not False:
        raise TokenizerGovernanceError("tokenizer provenance does not prove scratch initialization")
    if sha256_file(tokenizer_path) != metadata.get("tokenizer_sha256"):
        raise TokenizerGovernanceError("tokenizer artifact checksum verification failed")
    train_path = _repo_path(root, metadata.get("train_split", {}).get("path"), "tokenizer train")
    if sha256_file(train_path) != metadata.get("train_split", {}).get("sha256"):
        raise TokenizerGovernanceError("tokenizer train-source checksum verification failed")
    try:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    except Exception as exc:
        raise TokenizerGovernanceError(f"could not reload tokenizer: {exc}") from exc
    special_tokens = metadata.get("special_tokens", {})
    round_trips = {}
    for text in (
        "",
        "Hello, world! 123",
        "naïve café — 東京",
        "<pad> <unk> <bos> <eos>",
        "x" * 2048,
    ):
        encoded = tokenizer.encode(text)
        round_trips[text] = tokenizer.decode(encoded.ids, skip_special_tokens=False) == text
    if not all(round_trips.values()):
        failed = [text for text, passed in round_trips.items() if not passed]
        raise TokenizerGovernanceError(f"encode/decode round-trip failed for {len(failed)} cases")
    return {
        "ok": True,
        "tokenizer_sha256": metadata["tokenizer_sha256"],
        "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": special_tokens,
        "round_trip_cases": len(round_trips),
        "round_trip_loaded": all(round_trips.values()),
    }
