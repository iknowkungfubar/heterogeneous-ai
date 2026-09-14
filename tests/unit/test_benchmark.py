from pathlib import Path

import pytest

from hai.evaluation.benchmark import (
    BenchmarkError,
    evaluate_benchmark,
    generate_benchmark,
    verify_benchmark,
)


def test_benchmark_generation_is_reproducible_and_isolated(tmp_path: Path) -> None:
    config = tmp_path / "benchmark.yaml"
    config.write_text(
        """id: fixture
split_seed: 1337
samples_per_category: 2
categories:
  arithmetic: true
  linear_equations: true
""",
        encoding="utf-8",
    )
    first = generate_benchmark(config, tmp_path)
    assert verify_benchmark(config, tmp_path)["ok"]
    assert evaluate_benchmark(config, tmp_path, "symbolic")["accuracy"] == 1.0
    assert first["splits"]["validation"]["records"] == 4
    with pytest.raises(BenchmarkError, match="overwrite"):
        generate_benchmark(config, tmp_path)


def test_non_symbolic_expert_is_explicitly_not_applicable(tmp_path: Path) -> None:
    config = tmp_path / "benchmark.yaml"
    config.write_text(
        """id: fixture
split_seed: 1337
samples_per_category: 1
categories:
  arithmetic: true
""",
        encoding="utf-8",
    )
    generate_benchmark(config, tmp_path)
    result = evaluate_benchmark(config, tmp_path, "transformer-baseline")
    assert result["status"] == "not_applicable"
