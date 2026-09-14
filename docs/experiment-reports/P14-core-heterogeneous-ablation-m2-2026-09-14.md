# P14 core heterogeneous ablation and M2 freeze — 2026-09-14

## Hypothesis

The frozen heterogeneous core will improve structured TEST utility over the
Transformer baseline without unnecessary expert calls.

## Freeze and provenance

- Experiment: `P14-CORE-M2-20260914`
- Parent: `P13-CALIBRATION-20260914`
- Suite: `configs/ablation/core-m2.yaml`
- Benchmark: `core-objective-v1`
- TEST manifest hash: `e55a5a92ecdd992da73d6a3c307d0c98d356abd964085989b856d8f6a6d331a8`
- TEST records: 84 across seven categories
- Frozen validation threshold: `0.75`
- Bootstrap seeds: `1337, 2027, 31415`; 1,000 iterations per seed
- TEST was evaluated only after the suite/configuration freeze.

## Ablation results

| Variant | Status | Task accuracy | Coverage | Selective accuracy | Experts/request | 95% bootstrap CI |
|---|---|---:|---:|---:|---:|---:|
| Transformer baseline | not applicable | — | — | — | — | — |
| GRU sequence | not applicable | — | — | — | — | — |
| Symbolic-only | scored | 0.2857 | 0.2857 | 1.0000 | 1 | [0.1905, 0.3810] |
| Heterogeneous core | scored | 0.2857 | 0.2857 | 1.0000 | 1 | [0.1905, 0.3810] |
| All experts | scored | 0.2857 | 0.2857 | 1.0000 | 3 | [0.1905, 0.3810] |

Resource measurements were captured in the JSON artifact at
`artifacts/ablations/core-m2.json`. The all-experts variant increased calls
threefold without changing correctness or coverage.

## Commands and results

```text
docker compose run --rm hai python -m ruff check src tests
All checks passed!

docker compose run --rm hai python -m pytest -m 'not slow' -q
25 passed

hai ablation run --suite core-m2 --split test
TEST evaluation completed for all five frozen variants.

hai report milestone M2
M2 report loaded from the immutable local ablation artifact.
```

## Gate evaluation

- Core TEST stayed isolated until freeze: **pass**. The suite fixed the
  threshold, variants, seeds, and bootstrap procedure before TEST execution.
- Ablation/resource results complete: **pass**. Supported variants include
  task accuracy, coverage, selective accuracy, call count, latency, and
  bootstrap intervals; unsupported baselines are explicitly marked.
- Evidence-based M2 conclusion: **pass**. The core does not yet beat a learned
  baseline because the learned baselines cannot be scored on this structured
  benchmark; within the supported protocol, extra experts add cost but no
  measured value.

## M2 conclusion and next action

M2 is frozen as a negative result for the current structured core. The root
cause is interface coverage, not evidence that Transformer/GRU modeling is
inferior: neither learned specialist emits structured benchmark answers in the
frozen protocol. Do not expand the heterogeneous core by adding more routing
complexity until a fair learned structured-output adapter exists. Advance to
P15 only under that limitation and provenance.
