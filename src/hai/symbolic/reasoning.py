from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass
from datetime import date, timedelta
from fractions import Fraction
from typing import TypeVar


class SymbolicError(ValueError):
    """Raised when a symbolic request is unsupported or ambiguous."""


T = TypeVar("T")


@dataclass(frozen=True)
class ReasoningResult[T]:
    value: T
    trace: tuple[str, ...]


def _fraction(node: ast.AST) -> Fraction:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return Fraction(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _fraction(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left, right = _fraction(node.left), _fraction(node.right)
        operations = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
        }
        operation = next((fn for kind, fn in operations.items() if isinstance(node.op, kind)), None)
        if operation is None:
            raise SymbolicError("only +, -, *, and / are supported")
        if isinstance(node.op, ast.Div) and right == 0:
            raise SymbolicError("division by zero")
        return operation(left, right)
    raise SymbolicError("arithmetic expression contains an unsupported node")


def exact_arithmetic(expression: str) -> ReasoningResult[str]:
    if not re.fullmatch(r"[0-9+*/().\s-]+", expression):
        raise SymbolicError("arithmetic input contains unsupported characters")
    try:
        value = _fraction(ast.parse(expression, mode="eval").body)
    except (SyntaxError, ZeroDivisionError) as exc:
        raise SymbolicError("invalid arithmetic expression") from exc
    return ReasoningResult(str(value), (f"parse: {expression}", f"exact result: {value}"))


def solve_linear_equation_text(expression: str) -> ReasoningResult[str]:
    if not re.fullmatch(r"[0-9xX+*/().=\s-]+", expression):
        raise SymbolicError("equation contains unsupported characters")
    if expression.count("=") != 1 or expression.lower().count("x") != 1:
        raise SymbolicError("expected exactly one x and one equality")
    left, right = (part.strip() for part in expression.split("="))
    variable = Fraction(1)
    constant = Fraction(0)
    match = re.fullmatch(r"([+-]?[0-9.]*)\*?[xX]\s*([+-]\s*[0-9.]+)?", left.replace(" ", ""))
    if match:
        coefficient = (
            match.group(1) in ("", "+")
            and Fraction(1)
            or (Fraction(-1) if match.group(1) == "-" else Fraction(match.group(1)))
        )
        variable, constant = coefficient, Fraction(0)
        if match.group(2):
            constant = Fraction(match.group(2).replace("+", "").replace(" ", ""))
    else:
        raise SymbolicError("supported equation form is a*x + b = c")
    target = _fraction(ast.parse(right, mode="eval").body)
    if variable == 0:
        raise SymbolicError("equation has no unique solution")
    solution = (target - constant) / variable
    return ReasoningResult(
        str(solution),
        (f"coefficient: {variable}", f"constant: {constant}", f"solution: {solution}"),
    )


def evaluate_boolean(expression: str) -> ReasoningResult[bool]:
    if not re.fullmatch(r"[A-Za-z()\s]+", expression):
        raise SymbolicError("Boolean input contains unsupported characters")

    def visit(node: ast.AST) -> bool:
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not visit(node.operand)
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            values = [visit(item) for item in node.values]
            return all(values) if isinstance(node.op, ast.And) else any(values)
        raise SymbolicError("Boolean expressions support only True, False, and, or, not")

    normalized = re.sub(r"\btrue\b", "True", expression, flags=re.IGNORECASE)
    normalized = re.sub(r"\bfalse\b", "False", normalized, flags=re.IGNORECASE)
    try:
        value = visit(ast.parse(normalized, mode="eval").body)
    except SyntaxError as exc:
        raise SymbolicError("invalid Boolean expression") from exc
    return ReasoningResult(value, (f"parse: {expression}", f"Boolean result: {value}"))


def solve_integer_constraints(
    lower: int, upper: int, constraints: tuple[tuple[str, int], ...]
) -> ReasoningResult[tuple[int, ...]]:
    if lower > upper:
        raise SymbolicError("constraint bounds are inverted")
    operations = {
        "<": lambda value, bound: value < bound,
        "<=": lambda value, bound: value <= bound,
        ">": lambda value, bound: value > bound,
        ">=": lambda value, bound: value >= bound,
        "==": lambda value, bound: value == bound,
    }
    if any(operator_name not in operations for operator_name, _ in constraints):
        raise SymbolicError("unsupported constraint operator")
    values = tuple(
        value
        for value in range(lower, upper + 1)
        if all(operations[op](value, bound) for op, bound in constraints)
    )
    return ReasoningResult(
        values,
        (f"domain: [{lower}, {upper}]", f"constraints: {constraints}", f"solutions: {values}"),
    )


def deterministic_bfs(
    graph: dict[str, tuple[str, ...]], start: str, goal: str
) -> ReasoningResult[tuple[str, ...]]:
    if start not in graph or goal not in graph:
        raise SymbolicError("graph start and goal must be declared nodes")
    queue: list[tuple[str, ...]] = [(start,)]
    visited = {start}
    while queue:
        path = queue.pop(0)
        node = path[-1]
        if node == goal:
            return ReasoningResult(
                path, (f"BFS start: {start}", f"BFS goal: {goal}", f"path: {path}")
            )
        for neighbor in sorted(graph[node]):
            if neighbor not in visited and neighbor in graph:
                visited.add(neighbor)
                queue.append((*path, neighbor))
    raise SymbolicError("no path exists")


_UNIT_FACTORS = {
    "m": Fraction(1),
    "km": Fraction(1000),
    "cm": Fraction(1, 100),
    "s": Fraction(1),
    "min": Fraction(60),
    "h": Fraction(3600),
    "g": Fraction(1),
    "kg": Fraction(1000),
}
_UNIT_GROUP = {
    "m": "length",
    "km": "length",
    "cm": "length",
    "s": "time",
    "min": "time",
    "h": "time",
    "g": "mass",
    "kg": "mass",
}


def convert_unit(value: int | float, source: str, target: str) -> ReasoningResult[str]:
    if (
        source not in _UNIT_FACTORS
        or target not in _UNIT_FACTORS
        or _UNIT_GROUP[source] != _UNIT_GROUP[target]
    ):
        raise SymbolicError("unsupported or incompatible units")
    converted = Fraction(str(value)) * _UNIT_FACTORS[source] / _UNIT_FACTORS[target]
    return ReasoningResult(
        str(converted), (f"convert {value} {source} to {target}", f"result: {converted} {target}")
    )


def date_add_days(value: str, days: int) -> ReasoningResult[str]:
    try:
        result = date.fromisoformat(value) + timedelta(days=days)
    except ValueError as exc:
        raise SymbolicError("date must use ISO format YYYY-MM-DD") from exc
    return ReasoningResult(
        result.isoformat(), (f"date: {value}", f"add days: {days}", f"result: {result.isoformat()}")
    )
