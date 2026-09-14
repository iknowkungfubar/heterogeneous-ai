# Experiment / Phase Evidence: P04-tokenizer-from-scratch-2026-09-14

## Hypothesis / goal

A project-owned BPE tokenizer can be trained from the governed TRAIN split only,
reloaded, and used for exact encode/decode round trips without pretrained
vocabulary or weights.

## Git/environment

- Date: 2026-09-14
- Base checkpoint: `c3dbdcd`
- Runtime: `heterogeneous-ai-hai` ROCm image, `tokenizers` from the project environment.
- Training used no pretrained vocabulary, weights, or external tokenizer files.

## Commands and direct evidence

- `hai tokenizer train --config configs/tokenizers/bpe-8192.yaml --dataset-manifest data/manifests/tinystories-smoke.yaml`: passed.
- `hai tokenizer inspect --tokenizer artifacts/tokenizers/bpe-8192-v1 --text 'The dog walked home. 123 naïve café — 東京'`: passed with exact Unicode-preserving decode.
- `hai tokenizer verify --tokenizer artifacts/tokenizers/bpe-8192-v1`: passed; `ok: true`, five exact round-trip cases.
- `ruff check src tests`: passed.
- Non-slow unit suite: `9 passed`.

## Artifact/provenance

- Artifact directory: `artifacts/tokenizers/bpe-8192-v1` (ignored local artifact).
- Configured and actual vocabulary size: `8192` / `8192`.
- Special token IDs: `<pad>=0`, `<unk>=1`, `<bos>=2`, `<eos>=3`.
- Tokenizer SHA-256: `5186f3747fbd6f944ad4e62b09a624c6f2019734260057e348dcdf096da9d4d2`.
- TRAIN source SHA-256: `1613a0868eda0f1ba05ed6bb3680165566515f643e430f98abf62d106dd16785`.
- TRAIN records used: `80000`.
- Metadata records `pretrained_vocabulary: false` and the source manifest/path/hash.

## Token-length distribution

Measured over all 80,000 TRAIN records after reload:

- Minimum: 52 tokens
- Median: 186 tokens
- P95: 438 tokens
- Mean: 215.0456 tokens
- Maximum: 1,174 tokens

## Gate decision

PASS

## Next action

Advance to P05 tiny Transformer pipeline smoke model. Keep this tokenizer artifact
and TRAIN/validation/TEST split immutable for downstream comparisons.
