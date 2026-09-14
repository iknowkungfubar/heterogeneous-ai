from __future__ import annotations


def solve_linear_equation(a: int | float, b: int | float, c: int | float) -> float:
    """Solve a*x + b = c. Kept tiny for the P08 starting unit test."""
    if a == 0:
        raise ValueError("a must be non-zero")
    return (c - b) / a
