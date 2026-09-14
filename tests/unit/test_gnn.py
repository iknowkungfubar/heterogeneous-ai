from pathlib import Path

import yaml

from hai.models.gnn import _graph, evaluate_gnn, graph_baseline, load_gnn_config, train_gnn


def test_graph_generation_is_deterministic():
    config = load_gnn_config(Path("configs/models/gnn.yaml"))

    first = _graph(config)
    second = _graph(config)

    assert first["graph_hash"] == second["graph_hash"]
    assert first["graph_hash"] == "2bbe8387acd0a927f12d39600753545b32d30a3d5c5e9198c21105a73d16678a"
    assert first["train"].tolist() == second["train"].tolist()
    assert graph_baseline(first, "validation", config["classes"]) == 1.0


def test_scratch_gnn_checkpoint_round_trip(tmp_path):
    config = yaml.safe_load(Path("configs/models/gnn.yaml").read_text(encoding="utf-8"))
    config.update({"training_steps": 3, "output_dir": "gnn-checkpoint"})
    config_path = tmp_path / "gnn.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    metadata = train_gnn(config_path, tmp_path)
    result = evaluate_gnn(config_path, tmp_path, "validation")

    assert metadata["initialization"] == "random"
    assert metadata["checkpoint_sha256"] == result["checkpoint_sha256"]
    assert metadata["graph_hash"] == result["graph_hash"]
    assert result["split"] == "validation"
