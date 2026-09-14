from pathlib import Path

import pytest

pytest.importorskip("torch")

from hai.models.fusion import _data, load_fusion_config, train_fusion


def test_visual_objective_data_is_split_by_source_image():
    config = load_fusion_config(Path("configs/models/fusion.yaml"))
    data = _data(config)

    assert data["data_hash"] == _data(config)["data_hash"]
    train_images = set(data["image_indices"][data["splits"]["train"]].tolist())
    validation_images = set(data["image_indices"][data["splits"]["validation"]].tolist())
    assert train_images.isdisjoint(validation_images)
    assert len(data["splits"]["train"]) == 1152
    assert len(data["splits"]["validation"]) == 384


def test_projection_fusion_writes_hash_bound_checkpoint(tmp_path):
    import yaml

    config = load_fusion_config(Path("configs/models/fusion.yaml"))
    config.update({"samples": 40, "training_steps": 1, "batch_size": 8})
    config_path = tmp_path / "fusion.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = train_fusion(config_path, tmp_path, "projection")

    assert result["initialization"] == "random"
    assert result["data_hash"]
    assert result["checkpoint_sha256"]
