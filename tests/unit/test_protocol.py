from hai.experts.protocol import core_experts


def test_all_core_experts_emit_versioned_results_with_provenance() -> None:
    results = [
        adapter.answer(f"task-{expert_id}", "3*x + 4 = 19")
        for expert_id, adapter in core_experts().items()
    ]
    assert {result.schema_version for result in results} == {"1.0"}
    assert all(result.model_version for result in results)
    assert all(result.evidence for result in results)
    assert all(result.to_dict()["schema_version"] == "1.0" for result in results)
