from pathlib import Path

import pytest

from hai.data.pipeline import DataGovernanceError, deterministic_split, validate_archive


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
