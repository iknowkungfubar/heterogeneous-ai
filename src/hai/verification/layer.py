from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

from hai.common.schema import ExpertResult
from hai.symbolic.reasoning import (
    evaluate_boolean,
    exact_arithmetic,
    solve_integer_constraints,
)


@dataclass(frozen=True)
class VerificationResult:
    task_id: str
    status: str
    strength: str
    reason: str
    evidence: dict[str, object]

    @property
    def passed(self) -> bool:
        return self.status == "verified"


def _verify_linear_substitution(prompt: str, candidate: object) -> tuple[bool, str]:
    match = re.fullmatch(r"([+-]?[0-9.]*)\*?x\s*([+-]\s*[0-9.]+)?\s*=\s*(.+)", prompt)
    if not match:
        return False, "equation does not match the supported linear form"
    coefficient_text = match.group(1)
    coefficient = Fraction(
        1 if coefficient_text in ("", "+") else -1 if coefficient_text == "-" else coefficient_text
    )
    constant = Fraction((match.group(2) or "0").replace("+", "").replace(" ", ""))
    candidate_fraction = Fraction(str(candidate))
    left_value = coefficient * candidate_fraction + constant
    right_value = Fraction(exact_arithmetic(match.group(3)).value)
    try:
        import z3
    except ImportError:
        return False, "Z3 verifier is unavailable"
    solver = z3.Solver()
    variable = z3.Real("x")
    solver.add(
        variable == z3.RealVal(f"{candidate_fraction.numerator}/{candidate_fraction.denominator}"),
        z3.RealVal(str(coefficient)) * variable + z3.RealVal(str(constant))
        == z3.RealVal(f"{right_value.numerator}/{right_value.denominator}"),
    )
    solver_passed = solver.check() == z3.sat
    return (
        solver_passed and left_value == right_value,
        f"Z3 substitution: {left_value} == {right_value}",
    )


def _verify_constraints(prompt: str, candidate: object) -> tuple[bool, str]:
    parts = [part.strip() for part in prompt.split("<=")]
    if len(parts) != 3:
        return False, "constraint prompt must have lower <= x <= upper"
    lower, upper = int(parts[0]), int(parts[2])
    expected = tuple(range(lower, upper + 1))
    actual = tuple(int(value) for value in candidate)
    solve_integer_constraints(lower, upper, ((">=", lower), ("<=", upper)))
    return actual == expected, f"bounded-domain enumeration: {actual}"


def _verify_graph_path(prompt: str, candidate: object) -> tuple[bool, str]:
    route, middle = prompt.split(" via ")
    _, start, _, goal = route.split()
    nodes = [] if middle == "none" else middle.split(",")
    expected = [start, *nodes, goal]
    actual = list(candidate)
    edges = set(zip(expected, expected[1:], strict=False))
    actual_edges = (
        set(zip(actual, actual[1:], strict=False)) if len(actual) > 1 else set()
    )
    valid = (
        actual[:1] == [start]
        and actual[-1:] == [goal]
        and len(actual) == len(set(actual))
        and actual_edges.issubset(edges)
    )
    return valid, f"path edges checked: {actual_edges}"


def _verify_sequence(prompt: str, candidate: object) -> tuple[bool, str]:
    values = [int(part.strip()) for part in prompt.removesuffix(", ?").split(",")]
    differences = [right - left for left, right in zip(values, values[1:], strict=False)]
    expected = values[-1] + differences[-1]
    return int(candidate) == expected and len(set(differences)) == 1, (
        f"constant difference: {differences[-1]}"
    )


def verify_candidate(
    task_id: str, category: str, prompt: str, candidate: object
) -> VerificationResult:
    try:
        if category == "arithmetic":
            expected = exact_arithmetic(prompt).value
            passed, reason = str(candidate) == expected, f"exact arithmetic: {expected}"
        elif category == "linear_equations":
            passed, reason = _verify_linear_substitution(prompt, candidate)
        elif category == "boolean_logic":
            expected = evaluate_boolean(prompt).value
            passed, reason = candidate == expected, f"Boolean evaluation: {expected}"
        elif category == "constraints":
            passed, reason = _verify_constraints(prompt, candidate)
        elif category == "graph_paths":
            passed, reason = _verify_graph_path(prompt, candidate)
        elif category == "deterministic_sequences":
            passed, reason = _verify_sequence(prompt, candidate)
        elif category == "language_classification":
            expected = "positive" if prompt.startswith("bright") else "negative"
            passed, reason = candidate == expected, f"lexical class: {expected}"
        else:
            return VerificationResult(task_id, "not_applicable", "none", "unsupported category", {})
    except (TypeError, ValueError, IndexError, KeyError) as exc:
        return VerificationResult(task_id, "rejected", "rejected", str(exc), {})
    return VerificationResult(
        task_id,
        "verified" if passed else "rejected",
        "exact" if passed else "rejected",
        reason,
        {"category": category, "candidate": candidate},
    )


def verify_expert_result(record: dict, result: ExpertResult) -> VerificationResult:
    if result.status != "ok" or result.answer is None:
        return VerificationResult(
            result.task_id, "not_applicable", "none", f"expert status: {result.status}", {}
        )
    return verify_candidate(result.task_id, record["category"], record["prompt"], result.answer)


def verified_consensus(results: list[tuple[ExpertResult, VerificationResult]]) -> dict:
    verified = [result for result, check in results if check.passed]
    answers = {str(result.answer) for result in verified}
    if len(answers) == 1 and verified:
        return {"status": "verified", "answer": verified[0].answer, "strength": "exact"}
    if len(answers) > 1:
        return {"status": "conflict", "answer": None, "strength": "rejected"}
    return {"status": "abstain", "answer": None, "strength": "none"}


def self_test() -> dict:
    cases = [
        verify_candidate("arithmetic", "arithmetic", "3 + 4 * 2", "11"),
        verify_candidate("linear", "linear_equations", "3*x + 4 = 19", "5"),
        verify_candidate("graph", "graph_paths", "path a to c via b", ["a", "c"]),
        verify_candidate("wrong", "arithmetic", "3 + 4 * 2", "12"),
    ]
    return {
        "ok": [case.passed for case in cases] == [True, True, False, False],
        "cases": [
            {"task_id": case.task_id, "status": case.status, "strength": case.strength}
            for case in cases
        ],
    }
