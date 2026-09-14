import os
from pathlib import Path

import pytest
import yaml

from hai.data.pipeline import (
    DataGovernanceError,
    deterministic_split,
    prepare_dataset,
    sha256_file,
    validate_archive,
    verify_dataset,
    write_jsonl,
)


def test_deterministic_split_is_stable_and_disjoint() -> None:
    records = [{"id": f"record-{index}", "text": str(index)} for index in range(20)]
    first = deterministic_split(records, seed=1337)
    second = deterministic_split(list(reversed(records)), seed=1337)

    assert first == second
    assert sum(len(part) for part in first.values()) == len(records)
    assert not set(record["id"] for record in first["train"]) & {
        record["id"] for record in first["validation"]
    }
    assert not set(record["id"] for record in first["validation"]) & {
        record["id"] for record in first["test"]
    }


def test_archive_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar"
    import tarfile

    with tarfile.open(archive, "w") as stream:
        info = tarfile.TarInfo("../escape.txt")
        info.size = 0
        stream.addfile(info)

    with pytest.raises(DataGovernanceError, match="unsafe archive"):
        validate_archive(archive)


def test_prepare_and_verify_dataset_manifest(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """id: tinystories-smoke-v1
purpose: test
source: roneneldan/TinyStories
license: CDLA-Sharing-1.0
license_source: https://huggingface.co/datasets/roneneldan/TinyStories
source_revision: main
max_train_examples: 100
split_seed: 1337
tokenizer_training_uses_test_data: false
manifest: data/manifests/test.yaml
""",
        encoding="utf-8",
    )
    raw = tmp_path / "data/raw/tinystories-smoke-v1/records.jsonl"
    write_jsonl(({"id": f"r-{i}", "text": "story"} for i in range(20)), raw)
    os.chmod(raw, 0o444)
    manifest = {
        "raw": {
            "path": "data/raw/tinystories-smoke-v1/records.jsonl",
            "sha256": sha256_file(raw),
            "records": 20,
        }
    }
    manifest_path = tmp_path / "data/manifests/test.yaml"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    prepared = prepare_dataset(config, tmp_path)
    assert prepared["split_counts"] == {"train": 16, "validation": 2, "test": 2}
    assert verify_dataset(config, tmp_path)["ok"]


def test_verify_rejects_manifest_path_escape(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """id: tinystories-smoke-v1
purpose: test
source: roneneldan/TinyStories
license: CDLA-Sharing-1.0
license_source: https://huggingface.co/datasets/roneneldan/TinyStories
source_revision: main
max_train_examples: 100
split_seed: 1337
tokenizer_training_uses_test_data: false
manifest: data/manifests/test.yaml
""",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "data/manifests/test.yaml"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(yaml.safe_dump({"raw": {"path": "/etc/passwd"}}), encoding="utf-8")
    with pytest.raises(DataGovernanceError, match="escapes"):
        verify_dataset(config, tmp_path)
