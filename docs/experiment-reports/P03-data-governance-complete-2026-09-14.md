# Experiment / Phase Evidence: P03-data-governance-complete-2026-09-14

## Hypothesis / goal

The approved TinyStories smoke source can be acquired through a constrained
Parquet path, normalized into immutable raw JSONL, and deterministically split
without leakage.

## Git/environment

- Date: 2026-09-14
- Pre-change checkpoint: `36bb28a`
- Runtime: `heterogeneous-ai-hai` ROCm image, PyArrow 25.0.1
- Acquisition used the approved external-download authorization.
- No private or personal data was supplied by the operator.

## Commands and direct evidence

- `hai data inspect-config --config configs/datasets/tinystories-smoke.yaml`: passed.
- `hai data fetch --config configs/datasets/tinystories-smoke.yaml`: passed; 100,000 records.
- `hai data prepare --config configs/datasets/tinystories-smoke.yaml`: passed.
- `hai data verify --config configs/datasets/tinystories-smoke.yaml`: passed with `ok: true`.
- Isolated repeat preparation using the same raw file and seed reproduced all
  three processed SHA-256 values exactly.
- Raw file mode: `0444`.
- Unit/lint baseline: `ruff` passed; `7 passed` non-slow tests.

## Canonical manifest results

Manifest: `data/manifests/tinystories-smoke.yaml`

- Raw: 100,000 records, SHA-256
  `5ea77ac3c9fadc57e1db0c82fbb6709bef209dd9213eb8fe9423c13cca96ac1c`.
- Train: 80,000 records, SHA-256
  `1613a0868eda0f1ba05ed6bb3680165566515f643e430f98abf62d106dd16785`.
- Validation: 10,000 records, SHA-256
  `9d87a6a79461390c40acd8ed95e652c9fe4c403ec35521aecd0f7e3f55da9b21`.
- Test: 10,000 records, SHA-256
  `139823eb6f9d97b8b02364d72b8898c70bc119f16e6a1b8f84cdec59d997f2e6`.
- Split seed: `1337`; tokenizer training uses test data: `false`.

## Gate decision

PASS

## Residual limitation

The manifest records the upstream revision as `main` and the exact Parquet API
artifact URL, but not an upstream immutable commit or source-artifact SHA-256.
This remains a supply-chain follow-up before a production dataset release; the
local canonical raw and processed artifacts are independently checksummed.

## Next action

Advance to P04 tokenizer-from-scratch work. Preserve this dataset split and do
not tune tokenizer or model decisions against the held-out test split.
