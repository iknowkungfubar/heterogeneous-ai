from pathlib import Path

import pytest

pytest.importorskip("torch")

from hai.models.vision import _dataset, data_verify, load_vision_config, train_vision


def test_procedural_vision_data_is_deterministic():
    config = load_vision_config(Path("configs/models/vision.yaml"))
    first = _dataset(config)
    second = _dataset(config)

    assert first["dataset_hash"] == second["dataset_hash"]
    assert len(first["train"]) == 288
    assert len(first["validation"]) == 96
    assert len(first["test"]) == 96
    assert data_verify(Path("configs/models/vision.yaml"))["external_download"] is False


def test_vision_smoke_writes_hash_bound_checkpoint(tmp_path):
    import yaml

    config = load_vision_config(Path("configs/models/vision.yaml"))
    config.update(
        {"samples": 40, "training_steps": 1, "batch_size": 8, "output_dir": "vision-test"}
    )
    config_path = tmp_path / "vision.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    result = train_vision(config_path, tmp_path)

    assert result["initialization"] == "random"
    assert result["checkpoint_sha256"]
    assert result["dataset_hash"]
