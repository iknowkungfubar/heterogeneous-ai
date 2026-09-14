# P13 calibration, consensus, and abstention — 2026-09-14

## Hypothesis

Validation-only calibration plus evidence-aware consensus can report reliable
confidence, preserve independently verified answers, and abstain on unsupported
tasks with explicit coverage.

## Scope and provenance

- Experiment: `P13-CALIBRATION-20260914`
- Parent: `P12-VERIFICATION-20260914`
- Benchmark: `core-objective-v1`
- Manifest hash: `1cdfbf16d8ac3efd278023931053ab34519bd634e58b74a5c1b7a78c26cb2bb`
- Calibration split: VALIDATION only, 84 records
- Implementation: `src/hai/calibration/engine.py`
- Artifacts: ignored local JSON reports under `artifacts/calibration/`

## Calibration and consensus policy

Calibration compares temperature, logistic, and isotonic transforms. The raw
protocol confidence is evaluated with ECE and Brier score; no TEST data is used
for fitting or threshold selection. Consensus requires a passing independent
verification result, while unsupported/error outputs abstain. Accuracy-versus-
coverage points are emitted for thresholds 0.50, 0.75, and 0.90.

## Commands and results

```text
docker compose run --rm hai python -m ruff check src tests
All checks passed!

docker compose run --rm hai python -m pytest -m 'not slow' -q
24 passed

hai calibrate --split validation
samples=84; positive_labels=24
raw ECE=0.0; raw Brier=0.0
temperature ECE≈0.0, Brier≈0.0
logistic ECE=0.1016818985, Brier=0.0110400140
isotonic ECE=0.0, Brier=0.0

hai consensus evaluate --split validation
accepted=24; abstained=60; coverage=0.2857142857
selective_accuracy=1.0; incorrect_accepted=0

hai evaluation reliability-report --split validation
reliability JSON written to artifacts/calibration/reliability-v1.json
```

## Gate evaluation

- Calibration metrics reported: **pass**. ECE/Brier are recorded for raw,
  temperature, logistic, and isotonic methods with reliability bins.
- Consensus prioritizes verified evidence: **pass**. Only independently
  verified symbolic outputs are accepted; unsupported outputs do not become
  consensus answers because their raw confidence is absent/zero.
- Abstention and coverage reported: **pass**. 60/84 validation requests
  abstain, yielding 28.57% coverage at 100% selective accuracy.

## Conclusion and next action

P13 passes. The current specialist pool is intentionally conservative: the
system accepts 24 verified arithmetic/equation answers and abstains on 60
unsupported outputs. This is a coverage limitation, not a fabricated quality
claim. Advance to P14 for the core heterogeneous ablation and milestone-freeze
analysis.
