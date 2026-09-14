from hai.evaluation.ablation import _bootstrap_interval, _percentile


def test_ablation_statistics_are_deterministic() -> None:
    assert _percentile([0.0, 0.5, 1.0], 0.5) == 0.5
    first = _bootstrap_interval([0, 1, 1, 1], [1337], 20)
    second = _bootstrap_interval([0, 1, 1, 1], [1337], 20)

    assert first == second
    assert 0.0 <= first[0] <= first[1] <= 1.0
