from pathlib import Path

import pytest

pytest.importorskip("torch")

from hai.models.speech import _config, _speech_data, train_asr


def test_speech_data_has_source_safe_splits():
    config = _config(Path("configs/models/audio.yaml"))
    data = _speech_data(config)
    train_sources = set(data["source_ids"][data["splits"]["train"]].tolist())
    validation_sources = set(data["source_ids"][data["splits"]["validation"]].tolist())

    assert data["data_hash"] == _speech_data(config)["data_hash"]
    assert train_sources.isdisjoint(validation_sources)
    assert len(data["splits"]["train"]) == 288
    assert len(data["splits"]["validation"]) == 96


def test_asr_training_writes_hash_bound_checkpoint(tmp_path):
    import yaml

    config = _config(Path("configs/models/audio.yaml"))
    config.update({"samples": 40, "alignment_steps": 1, "asr_steps": 1, "batch_size": 8})
    config_path = tmp_path / "audio.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = train_asr(config_path, tmp_path)

    assert result["initialization"] == "random"
    assert result["data_hash"]
    assert result["checkpoint_sha256"]
