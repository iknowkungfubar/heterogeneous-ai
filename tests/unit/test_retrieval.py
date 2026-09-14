from hai.retrieval.engine import _lexical, _ranking_metrics


def test_lexical_and_ranking_metrics_are_deterministic() -> None:
    assert _lexical("red fox", "a red fox jumps") == 1.0
    records = [{"id": "doc-1"}, {"id": "doc-2"}]
    metrics = _ranking_metrics(
        [[0, 1], [1, 0]], records, records, "validation-sha", "validation"
    )

    assert metrics["recall_at_1"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["provenance_sample"]["source_sha256"] == "validation-sha"
