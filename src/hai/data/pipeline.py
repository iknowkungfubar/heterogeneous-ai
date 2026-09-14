from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import yaml


class DataGovernanceError(ValueError):
    """Raised when a dataset violates the repository data contract."""


def load_config(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise DataGovernanceError("dataset config must be a YAML mapping")
    return data


def inspect_config(path: Path) -> dict:
    config = load_config(path)
    required = ("id", "purpose", "manifest")
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise DataGovernanceError(f"dataset config missing required fields: {', '.join(missing)}")
    if config.get("license"):
        config["license_status"] = "declared"
    elif config.get("license_must_be_recorded_at_execution"):
        config["license_status"] = "required-before-fetch"
    elif config.get("license_must_be_recorded_before_download"):
        config["license_status"] = "required-before-fetch"
    else:
        config["license_status"] = "not-declared"
    return config


def _validated_config(path: Path) -> dict:
    config = inspect_config(path)
    if config.get("source") != "roneneldan/TinyStories":
        raise DataGovernanceError("only the approved TinyStories source is enabled")
    if not config.get("source_revision"):
        raise DataGovernanceError("dataset config must pin source_revision")
    if not config.get("license") or not config.get("license_source"):
        raise DataGovernanceError("dataset config must record license and license_source")
    max_examples = config.get("max_train_examples")
    if not isinstance(max_examples, int) or not 0 < max_examples <= 100_000:
        raise DataGovernanceError("max_train_examples must be an integer from 1 through 100000")
    return config


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member_name(name: str) -> PurePosixPath:
    if "\x00" in name:
        raise DataGovernanceError("archive member contains NUL")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise DataGovernanceError(f"unsafe archive member path: {name}")
    return path


def validate_archive(path: Path) -> list[str]:
    names: list[str] = []
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as archive:
            for member in archive.getmembers():
                _safe_member_name(member.name)
                if member.issym() or member.islnk():
                    raise DataGovernanceError(f"archive links are not allowed: {member.name}")
                names.append(member.name)
        return names
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                _safe_member_name(member.filename)
                names.append(member.filename)
        return names
    raise DataGovernanceError(f"unsupported archive format: {path}")


def deterministic_split(
    records: Iterable[dict],
    seed: int,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> dict[str, list[dict]]:
    ordered = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or "id" not in record:
            raise DataGovernanceError(f"record {index} must be a mapping with an id")
        key = f"{seed}:{record['id']}".encode()
        ordered.append((hashlib.sha256(key).hexdigest(), record))
    ordered.sort(key=lambda item: item[0])
    total = len(ordered)
    train_count = int(total * train_fraction)
    validation_count = int(total * validation_fraction)
    return {
        "train": [record for _, record in ordered[:train_count]],
        "validation": [
            record for _, record in ordered[train_count : train_count + validation_count]
        ],
        "test": [record for _, record in ordered[train_count + validation_count :]],
    }


def write_jsonl(records: Iterable[dict], path: Path) -> str:
    if path.exists():
        raise DataGovernanceError(f"refusing to overwrite existing dataset file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    return sha256_file(path)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise DataGovernanceError(f"dataset file does not exist: {path}")
    records: list[dict] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataGovernanceError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(record, dict) or not isinstance(record.get("id"), str):
                raise DataGovernanceError(f"record at {path}:{line_number} lacks a string id")
            records.append(record)
    return records


def _manifest_path(config: dict, root: Path) -> Path:
    manifest = Path(config["manifest"])
    if manifest.is_absolute() or ".." in manifest.parts:
        raise DataGovernanceError("manifest path must be relative to the repository")
    return root / manifest


def _repo_path(root: Path, value: object, field: str) -> Path:
    if not isinstance(value, str):
        raise DataGovernanceError(f"manifest {field} path must be a string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise DataGovernanceError(f"manifest {field} path escapes the repository")
    return root / path


def fetch_dataset(config_path: Path, root: Path) -> dict:
    config = _validated_config(config_path)
    raw_path = root / "data" / "raw" / config["id"] / "records.jsonl"
    if raw_path.exists():
        raise DataGovernanceError(f"refusing to overwrite existing raw file: {raw_path}")
    parquet_api = (
        "https://huggingface.co/api/datasets/"
        f"{config['source']}/parquet/default/train"
    )
    try:
        with urllib.request.urlopen(parquet_api, timeout=30) as response:
            parquet_urls = json.load(response)
    except Exception as exc:  # network/library errors need a governance-level message
        raise DataGovernanceError(f"approved dataset metadata fetch failed: {exc}") from exc
    if not isinstance(parquet_urls, list) or not parquet_urls:
        raise DataGovernanceError("approved dataset returned no Parquet shards")
    parquet_url = parquet_urls[0]
    if not isinstance(parquet_url, str) or not parquet_url.startswith(
        "https://huggingface.co/api/datasets/roneneldan/TinyStories/parquet/"
    ):
        raise DataGovernanceError("dataset API returned an unexpected shard URL")

    def normalized_records(parquet_path: Path) -> Iterable[dict]:
        try:
            import pyarrow.parquet as parquet
            table = parquet.read_table(parquet_path, columns=["text"])
        except Exception as exc:
            raise DataGovernanceError(f"approved Parquet shard could not be read: {exc}") from exc
        accepted = 0
        for example in table.to_pylist():
            text = example.get("text") if isinstance(example, dict) else None
            if not isinstance(text, str) or not text.strip():
                continue
            if accepted >= config["max_train_examples"]:
                break
            yield {"id": f"{config['id']}-train-{accepted:06d}", "text": text}
            accepted += 1

    with tempfile.NamedTemporaryFile(prefix="tinystories-", suffix=".parquet") as download:
        try:
            with urllib.request.urlopen(parquet_url, timeout=120) as response:
                shutil.copyfileobj(response, download)
            download.flush()
        except Exception as exc:
            raise DataGovernanceError(f"approved dataset shard download failed: {exc}") from exc
        records = list(normalized_records(Path(download.name)))
    if not records:
        raise DataGovernanceError("approved dataset returned no usable text records")
    raw_hash = write_jsonl(records, raw_path)
    os.chmod(raw_path, 0o444)
    manifest = {
        "dataset_id": config["id"],
        "purpose": config["purpose"],
        "source": config["source"],
        "source_revision": config["source_revision"],
        "source_artifact": parquet_url,
        "license": config["license"],
        "license_source": config["license_source"],
        "retrieved_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "split_seed": config["split_seed"],
        "raw": {
            "path": str(raw_path.relative_to(root)),
            "sha256": raw_hash,
            "records": len(records),
        },
        "processed": {},
        "tokenizer_training_uses_test_data": bool(config["tokenizer_training_uses_test_data"]),
    }
    manifest_path = _manifest_path(config, root)
    if manifest_path.exists():
        raise DataGovernanceError(f"refusing to overwrite existing manifest: {manifest_path}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return manifest


def prepare_dataset(config_path: Path, root: Path) -> dict:
    config = _validated_config(config_path)
    raw_path = root / "data" / "raw" / config["id"] / "records.jsonl"
    records = _read_jsonl(raw_path)
    splits = deterministic_split(records, seed=config["split_seed"])
    processed_dir = root / "data" / "processed" / config["id"]
    processed: dict[str, dict] = {}
    for split_name, split_records in splits.items():
        split_path = processed_dir / f"{split_name}.jsonl"
        split_hash = write_jsonl(split_records, split_path)
        processed[split_name] = {
            "path": str(split_path.relative_to(root)),
            "sha256": split_hash,
            "records": len(split_records),
        }
    manifest_path = _manifest_path(config, root)
    if not manifest_path.is_file():
        raise DataGovernanceError(f"manifest does not exist: {manifest_path}; fetch first")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("raw", {}).get("sha256") != sha256_file(
        raw_path
    ):
        raise DataGovernanceError("raw checksum does not match acquisition manifest")
    manifest["processed"] = processed
    manifest["split_counts"] = {name: len(values) for name, values in splits.items()}
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return manifest


def verify_dataset(config_path: Path, root: Path) -> dict:
    config = _validated_config(config_path)
    manifest_path = _manifest_path(config, root)
    if not manifest_path.is_file():
        raise DataGovernanceError(f"manifest does not exist: {manifest_path}")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise DataGovernanceError("dataset manifest must be a YAML mapping")
    raw = manifest.get("raw", {})
    raw_path = _repo_path(root, raw.get("path"), "raw")
    if not raw_path.is_file() or sha256_file(raw_path) != raw.get("sha256"):
        raise DataGovernanceError("raw dataset checksum verification failed")
    if raw_path.stat().st_mode & 0o222:
        raise DataGovernanceError("raw dataset must be read-only")
    processed = manifest.get("processed", {})
    if set(processed) != {"train", "validation", "test"}:
        raise DataGovernanceError("manifest must contain all processed splits")
    ids: set[str] = set()
    counts: dict[str, int] = {}
    for split_name, details in processed.items():
        split_path = _repo_path(root, details.get("path"), split_name)
        if sha256_file(split_path) != details["sha256"]:
            raise DataGovernanceError(f"processed checksum verification failed: {split_name}")
        split_records = _read_jsonl(split_path)
        split_ids = {record["id"] for record in split_records}
        if ids & split_ids:
            raise DataGovernanceError(f"split overlap detected: {split_name}")
        ids.update(split_ids)
        counts[split_name] = len(split_records)
    return {
        "ok": True,
        "raw_records": raw.get("records"),
        "split_counts": counts,
        "manifest": str(manifest_path.relative_to(root)),
    }


def copy_local_source(source: Path, destination: Path) -> str:
    if not source.is_file():
        raise DataGovernanceError(f"local source does not exist: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise DataGovernanceError(f"refusing to overwrite existing raw file: {destination}")
    shutil.copyfile(source, destination)
    os.chmod(destination, 0o444)
    return sha256_file(destination)
