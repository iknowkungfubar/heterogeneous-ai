from pathlib import Path

import pytest

from hai.calibration.engine import CalibrationError, calibrate, calibration_metrics
from hai.evaluation.benchmark import generate_benchmark


def test_calibration_metrics_report_perfect_reliability() -> None:
    result = calibration_metrics([0.0, 1.0], [0, 1])

    assert result["ece"] == 0.0
    assert result["brier"] == 0.0


def test_calibration_uses_validation_only(tmp_path: Path) -> None:
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
    generate_benchmark(config, tmp_path)

    result = calibrate(config, tmp_path)

    assert result["split"] == "validation"
    assert set(result["methods"]) == {"temperature", "logistic", "isotonic"}
    with pytest.raises(CalibrationError, match="VALIDATION"):
        calibrate(config, tmp_path, "test")
