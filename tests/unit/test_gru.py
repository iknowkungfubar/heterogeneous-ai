from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from hai.models.gru import GRULanguageModel, load_gru_config  # noqa: E402


def test_gru_is_randomly_initialized_and_shapes_logits() -> None:
    config = load_gru_config(Path("configs/models/gru.yaml"))
    config["vocab_size"] = 64
    model = GRULanguageModel(config)
    tokens = torch.zeros((2, 8), dtype=torch.long)
    logits = model(tokens)
    assert logits.shape == (2, 8, 64)
    assert model.parameter_count() > 0
    assert config["initialization"] == "random"
