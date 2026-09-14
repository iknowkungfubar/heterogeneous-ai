from pathlib import Path

import pytest

pytest.importorskip("torch")

from hai.models.state_space import (
    _task_hash,
    compatibility_check,
    load_state_space_config,
    smoke_train,
)


def test_state_space_compatibility_has_explicit_install_provenance():
    config_path = Path("configs/models/ssm.yaml")
    config = load_state_space_config(config_path)
    result = compatibility_check(config_path, Path.cwd())

    assert result["implementation"] == "plain_pytorch_diagonal_state_space"
    assert result["task_hash"] == _task_hash(config)
    assert result["config_sha256"]
    assert result["cpu_forward"] is True
    assert result["cpu_finite"] is True
    assert result["upstream_package"] == "mamba-ssm"


def test_state_space_smoke_checkpoint_round_trip(tmp_path):
    config = load_state_space_config(Path("configs/models/ssm.yaml"))
    config_path = tmp_path / "ssm.yaml"
    config["output_dir"] = "artifacts/ssm-test"
    import yaml

    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    result = smoke_train(config_path, tmp_path, steps=2)

    assert result["initialization"] == "random"
    assert result["checkpoint_round_trip"] is True
    assert result["resumed_step"] == 2
