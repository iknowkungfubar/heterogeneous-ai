from hai.common.schema import ExpertResult


def test_expert_result_minimal() -> None:
    result = ExpertResult(task_id="t1", expert_id="symbolic-v1", answer="5", verified=True)
    assert result.answer == "5"
    assert result.verified is True
