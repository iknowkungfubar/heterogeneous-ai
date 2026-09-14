# Experiment / Phase Evidence: P09-objective-benchmark-2026-09-14

## Hypothesis / goal

A deterministic benchmark generator can produce programmatic ground truth,
isolated train/validation/test partitions, and one common expert-evaluation
harness without silently scoring unsupported experts.

## Benchmark definition

- Benchmark: `core-objective-v1`, generator version `1`.
- Seed: `1337`.
- Categories: arithmetic, linear equations, Boolean logic, constraints, graph
  paths, deterministic sequences, and language classification.
- Samples: 12 per category per split; 84 records per split.
- Generated files are ignored local artifacts; the manifest is tracked at
  `data/manifests/core-objective-v1.yaml`.

## Checksums and isolation

- TRAIN SHA-256: `8b8484f0e30aa0570889da2770d2ad566771e18789fb971b9598bf6b36a664be`.
- VALIDATION SHA-256: `bcb0032e877d411f31a1500e8d49ade054edd1e7595365c83def5323eac02e11`.
- TEST SHA-256: `e55a5a92ecdd992da73d6a3c307d0c98d356abd964085989b856d8f6a6d331a8`.
- Manifest SHA-256: `1cdfbf16d8ac3efd278023931053ab34519bd634eb58b74a5c1b7a78c26cb2bb`.
- Verification confirmed 252 unique IDs with no cross-split overlap.

## Evaluation

- Symbolic expert through the common harness: 84/84 validation correct, accuracy
  `1.0`, with per-category 12/12 results.
- Transformer baseline: `not_applicable`; no structured-output adapter exists.
- GRU specialist: `not_applicable`; no structured-output adapter exists.

The learned-expert result is intentionally a status, not a fabricated score.
Their natural-language checkpoints were trained for next-token prediction and
cannot be fairly evaluated on exact structured answers without a separate
adapter and prompt/decoding protocol.

## Verification

- `benchmark generate`: passed.
- `benchmark verify`: passed with all split/category counts.
- `benchmark evaluate --expert symbolic --split validation`: passed at 1.0.
- Generator overwrite refusal and deterministic fixture tests: passed.
- Adversarial/unsupported expert handling: passed.
- Lint: passed.
- Unit suite: `14 passed`.

## Gate decision

PASS

## Limitations and negative findings

The suite is small and synthetic, so it is a correctness harness rather than a
general capability benchmark. Learned experts remain unscored until a governed
structured-output adapter exists. Test scoring is implemented but intentionally
not used for development decisions in this phase.

## Next action

Advance to P10 unified expert protocol and provenance.
