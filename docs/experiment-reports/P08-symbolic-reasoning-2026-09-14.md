# Experiment / Phase Evidence: P08-symbolic-reasoning-2026-09-14

## Hypothesis / goal

Typed deterministic symbolic adapters can solve a useful initial set of exact
reasoning tasks, return auditable traces, and reject unsupported or ambiguous
inputs without executing arbitrary code.

## Implemented interfaces

- Exact arithmetic with integer/fraction results.
- Restricted linear equations in the form `a*x + b = c`.
- Boolean expressions using only `True`, `False`, `and`, `or`, and `not`.
- Bounded integer constraints with explicit operators.
- Deterministic breadth-first graph traversal with sorted neighbors.
- Compatible unit conversion for length, time, and mass units.
- ISO-date plus/minus-day operations.

All interfaces return a typed value and an immutable operation trace. Unsupported
syntax, unsafe characters, incompatible units, missing graph nodes, impossible
paths, and non-unique equations fail with `SymbolicError`.

## Direct verification

- `hai symbolic solve '3*x + 4 = 19'`: exact result `5`, structured trace.
- `hai symbolic solve '1 + 2 * 3'`: exact result `7`, structured trace.
- `hai symbolic self-test`: `2/2` cases passed.
- Unit suite: `12 passed`.
- Lint: passed.
- Adversarial tests reject Python import/system-call payloads and ambiguous `x = x`.

## Gate decision

PASS

## Limitations and negative findings

This is a deliberately bounded typed adapter surface, not a general natural
language theorem prover. Free-form symbolic parsing, unsupported operators, and
ambiguous equations are rejected. Broader logic, unit dimensional analysis,
constraint languages, and benchmark coverage belong in later phases.

## Next action

Advance to P09 objective benchmark framework and use these deterministic
operations as explicit benchmark components.
