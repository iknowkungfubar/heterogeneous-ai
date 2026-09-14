from pathlib import Path

from hai.routing.router import extract_features, load_router_config


def test_router_features_are_inspectable() -> None:
    features = extract_features(
        {"category": "linear_equations", "prompt": "3*x + 4 = 19"}
    )

    assert features == {
        "task_type": "linear_equations",
        "input_length": 12.0,
        "contains_numbers": 1.0,
        "contains_equation": 1.0,
        "contains_graph_structure": 0.0,
        "symbolic_applicable": 1.0,
    }


def test_router_config_has_fail_closed_threshold() -> None:
    config = load_router_config(Path("configs/routing/default.yaml"))

    assert config["id"] == "router-v1"
    assert config["confidence_threshold"] == 0.75
