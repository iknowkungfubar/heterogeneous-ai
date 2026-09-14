# P12 verification layer — 2026-09-14

## Hypothesis

Deterministic checks independent of model confidence can verify structured
answers, reject plausible wrong candidates, and expose verification strength to
consensus.

## Scope and provenance

- Experiment: `P12-VERIFICATION-20260914`
- Parent: `P11-ROUTER-20260914`
- Benchmark: `core-objective-v1`
- Manifest hash: `1cdfbf16d8ac3efd278023931053ab34519bd634e58b74a5c1b7a78c26cb2bb`
- Validation split: 84 records across seven categories
- Implementation: `src/hai/verification/layer.py`
- Working-tree state before implementation: clean except pre-existing `.codegraph/` and `.serena/`

## Verification design

The verifier returns an explicit status and strength rather than trusting raw
confidence or an expert's self-reported flag:

- exact arithmetic and Boolean evaluation;
- Z3-backed linear-equation substitution;
- independently enumerated bounded constraints;
- graph start/goal, uniqueness, and edge checks;
- constant-difference sequence checks;
- deterministic lexical classification;
- `not_applicable` for unsupported categories and `rejected` for failed checks.

`verified_consensus` accepts an answer only when at least one independently
verified result agrees; conflicting verified answers produce `conflict`, and
unverified results produce `abstain`.

## Commands and results

```text
docker compose run --rm hai python -m ruff check src tests
All checks passed!

docker compose run --rm hai python -m pytest -m 'not slow' -q
22 passed

hai verify self-test
ok=true; correct arithmetic and Z3 substitution accepted; graph shortcut and
wrong arithmetic candidates rejected

hai benchmark verify --config configs/benchmarks/core.yaml
ok=true; train=84, validation=84, test=84; all split/category checks passed

hai benchmark evaluate --config configs/benchmarks/core.yaml \
  --system router-plus-verifier --split validation
samples=84; verified=84; rejected=0; verification_rate=1.0
```

## Gate evaluation

- Deterministically verifiable tasks independently checked: **pass**. All 84
  validation records passed independent verification.
- Plausible wrong candidates rejected: **pass**. Self-test rejected a wrong
  arithmetic result and a graph shortcut that reached the goal without using
  the declared path.
- Verification strength flows into consensus: **pass**. The consensus helper
  accepts only independently verified results and emits conflict/abstain
  states otherwise.

## Conclusion and next action

P12 passes. Verification is now a separate evidence-bearing layer with
fail-closed rejection and explicit strength semantics. Advance to P13 for
calibration, consensus, and abstention policy over router/expert outputs.
