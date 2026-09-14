# Data Governance

## Rules

- Every dataset needs a manifest and data card before it becomes a canonical training source.
- Record origin, version/date, license/terms, checksums, sizes, processing, split method, and known limitations.
- Raw downloaded data is immutable. Processing creates new `interim` or `processed` artifacts.
- Keep training, validation, and test responsibilities separate.
- Train the tokenizer on training data only.
- Do not execute code embedded in datasets.
- Reject unsafe archives/path traversal and unexpected file types during ingestion.
- Do not place personal/private user information into a public research dataset.

## Directory lifecycle

```text
data/raw        immutable source copies
data/interim    normalized/intermediate results
data/processed  model-ready datasets
data/benchmarks benchmark splits/generators
data/multimodal paired media metadata
data/manifests  tracked provenance + checksums
```

Large data directories are ignored by Git; manifests are tracked.

## Recommended staged text data

1. A small educational text corpus for engineering smoke tests.
2. A constrained BabyLM-style training budget for the first serious from-scratch baseline.
3. Larger or broader corpora only after the baseline/evaluation system is stable.

## Manifest schema

Each manifest should include:

```yaml
name:
source:
license:
version:
retrieved_at:
checksums: {}
raw_files: []
processing_version:
split_seed:
train_count:
validation_count:
test_count:
word_count:
token_count:
known_limitations: []
```

## Test contamination response

If test data influences architecture/hyperparameter choices, mark affected experiments `invalid`; retire that test partition and create a truly unseen replacement before claiming final performance.
