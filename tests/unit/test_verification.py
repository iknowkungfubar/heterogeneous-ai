from hai.common.schema import ExpertResult
from hai.verification.layer import verified_consensus, verify_candidate, verify_expert_result


def test_verifier_accepts_independent_exact_and_substitution_checks() -> None:
    arithmetic = verify_candidate("a", "arithmetic", "3 + 4 * 2", "11")
    equation = verify_candidate("e", "linear_equations", "3*x + 4 = 19", "5")

    assert arithmetic.passed and arithmetic.strength == "exact"
    assert equation.passed and equation.strength == "exact"


def test_verifier_rejects_plausible_wrong_candidates() -> None:
    wrong_arithmetic = verify_candidate("a", "arithmetic", "3 + 4 * 2", "12")
    wrong_path = verify_candidate("g", "graph_paths", "path a to c via b", ["a", "c"])

    assert wrong_arithmetic.status == "rejected"
    assert wrong_path.status == "rejected"


def test_verified_consensus_requires_independent_verification() -> None:
    result = ExpertResult(
        schema_version="1.0", task_id="a", expert_id="symbolic", answer="11"
    )
    check = verify_expert_result(
        {"category": "arithmetic", "prompt": "3 + 4 * 2"}, result
    )

    assert verified_consensus([(result, check)]) == {
        "status": "verified",
        "answer": "11",
        "strength": "exact",
    }
