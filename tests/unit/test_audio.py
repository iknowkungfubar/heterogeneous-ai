from pathlib import Path

import pytest

pytest.importorskip("torch")

from hai.models.audio import _dataset, load_audio_config, train_audio


def test_procedural_audio_data_is_deterministic():
    config = load_audio_config(Path("configs/models/audio.yaml"))
    first = _dataset(config)
    second = _dataset(config)

    assert first["data_hash"] == second["data_hash"]
    assert first["waveforms"].shape == (480, 1, 512)
    assert len(first["splits"]["train"]) == 288
    assert len(first["splits"]["validation"]) == 96
    assert len(first["splits"]["test"]) == 96


def test_audio_training_writes_hash_bound_checkpoint(tmp_path):
    import yaml

    config = load_audio_config(Path("configs/models/audio.yaml"))
    config.update({"samples": 40, "training_steps": 1, "batch_size": 8})
    config_path = tmp_path / "audio.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = train_audio(config_path, tmp_path)

    assert result["initialization"] == "random"
    assert result["data_hash"]
    assert result["checkpoint_sha256"]
