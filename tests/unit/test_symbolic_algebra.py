import pytest

from hai.symbolic.algebra import solve_linear_equation
from hai.symbolic.reasoning import (
    SymbolicError,
    convert_unit,
    date_add_days,
    deterministic_bfs,
    evaluate_boolean,
    exact_arithmetic,
    solve_integer_constraints,
    solve_linear_equation_text,
)


def test_solve_linear_equation() -> None:
    assert solve_linear_equation(3, 4, 19) == 5


def test_symbolic_adapters_return_exact_values_and_traces() -> None:
    assert exact_arithmetic("1 + 2 * 3").value == "7"
    assert solve_linear_equation_text("3*x + 4 = 19").value == "5"
    assert evaluate_boolean("not False and True").value is True
    assert solve_integer_constraints(0, 10, ((">", 2), ("<", 5))).value == (3, 4)
    assert deterministic_bfs(
        {"a": ("c", "b"), "b": ("d",), "c": ("d",), "d": ()}, "a", "d"
    ).value == ("a", "b", "d")
    assert convert_unit(2, "km", "m").value == "2000"
    assert date_add_days("2026-01-01", 31).value == "2026-02-01"


def test_symbolic_adapters_reject_unsafe_or_ambiguous_inputs() -> None:
    for operation, value in (
        (exact_arithmetic, "__import__('os').system('id')"),
        (evaluate_boolean, "True or os.system('id')"),
    ):
        with pytest.raises(SymbolicError):
            operation(value)
    with pytest.raises(SymbolicError):
        solve_linear_equation_text("x = x")
