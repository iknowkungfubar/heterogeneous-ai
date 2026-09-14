from pathlib import Path

import pytest

pytest.importorskip("torch")

from hai.models.multimodal import _paired_data, load_multimodal_config, train_alignment


def test_paired_image_text_data_is_deterministic():
    config = load_multimodal_config(Path("configs/models/multimodal.yaml"))

    first = _paired_data(config)
    second = _paired_data(config)

    assert first["alignment_hash"] == second["alignment_hash"]
    assert len(first["vocabulary"]) == 8
    assert len(first["train"]) == 288
    assert len(first["validation"]) == 96
    assert len(first["test"]) == 96


def test_alignment_training_writes_hash_bound_checkpoint(tmp_path):
    import yaml

    config = load_multimodal_config(Path("configs/models/multimodal.yaml"))
    config.update({"samples": 40, "training_steps": 1, "batch_size": 8})
    config_path = tmp_path / "multimodal.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = train_alignment(config_path, tmp_path)

    assert result["initialization"] == "random"
    assert result["alignment_hash"]
    assert result["checkpoint_sha256"]
