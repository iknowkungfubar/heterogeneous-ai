import pytest

from hai.common.schema import ExpertResult


def test_expert_result_minimal() -> None:
    result = ExpertResult(
        schema_version="1.0",
        task_id="t1",
        expert_id="symbolic-v1",
        answer="5",
        verified=True,
        verification_type="exact",
    )
    assert result.answer == "5"
    assert result.verified is True


def test_expert_result_serializes_and_reloads() -> None:
    result = ExpertResult(
        schema_version="1.0",
        task_id="t2",
        expert_id="symbolic-v1",
        answer="5",
        model_version="symbolic-1",
        raw_score=1.0,
        evidence=[],
    )
    assert ExpertResult.from_dict(result.to_dict()).to_dict() == result.to_dict()


def test_expert_result_rejects_malformed_confidence() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        ExpertResult(
            schema_version="1.0", task_id="t3", expert_id="x", answer=None, raw_confidence=2.0
        )
