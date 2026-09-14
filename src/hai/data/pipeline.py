from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import zipfile
from collections.abc import Iterable
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    return sha256_file(path)


def copy_local_source(source: Path, destination: Path) -> str:
    if not source.is_file():
        raise DataGovernanceError(f"local source does not exist: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise DataGovernanceError(f"refusing to overwrite existing raw file: {destination}")
    shutil.copyfile(source, destination)
    os.chmod(destination, 0o444)
    return sha256_file(destination)
