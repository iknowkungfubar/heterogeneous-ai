from hai.symbolic.algebra import solve_linear_equation


def test_solve_linear_equation() -> None:
    assert solve_linear_equation(3, 4, 19) == 5
