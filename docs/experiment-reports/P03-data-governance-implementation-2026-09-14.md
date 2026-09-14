# Experiment / Phase Evidence: P03-data-governance-implementation-2026-09-14

## Hypothesis / goal

The repository can inspect dataset configuration, reject unsafe archive paths, create deterministic disjoint splits, and support immutable local raw-file copying before external data is admitted as canonical.

## Git/environment

- Date: 2026-09-14
- Base checkpoint: `6d7808e`
- No external dataset was downloaded.
- No private or personal data was used.

## Data/tokenizer/model configs and hashes

- Config inspected: `configs/datasets/tinystories-smoke.yaml`.
- Declared source: `roneneldan/TinyStories`, revision `main`.
- Declared license: `CDLA-Sharing-1.0`, verified from the dataset card before recording metadata.
- No raw-data checksum exists yet because no external bytes were fetched.

## Implementation

- Added `src/hai/data/pipeline.py` with safe YAML config inspection, SHA-256 hashing, archive member validation, deterministic hash-based splitting, JSONL writing, and non-overwriting local raw copying.
- Added `hai data inspect-config --config ...` to the CLI.
- Added unit coverage for deterministic disjoint splits and archive traversal rejection.

## Verification

- `ruff check ...`: passed.
- Non-GPU unit suite: `4 passed`, `2 deselected`.
- `python -m hai.cli.main data inspect-config --config configs/datasets/tinystories-smoke.yaml`: passed; license metadata displayed.
- `git diff --check`: passed.

## Failures/deviations

The phase gate is not complete: no real raw dataset was fetched, no canonical manifest with file checksums exists, and no processed train/validation/test artifacts were produced. The referenced dataset is large, so acquisition remains a separate approval-bound action.

## Gate decision

BLOCKED

## Next action

After approval, fetch the declared dataset revision, record exact files/sizes/SHA-256 values in a manifest and dataset card, run deterministic processing/splitting, and verify repeated manifests match. Do not mark P03 passed before those direct artifacts exist.
