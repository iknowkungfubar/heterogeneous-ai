# Dataset Card: tinystories-smoke-v1

## Purpose

Engineering smoke testing for the scratch-trained research track. This dataset is
not a final benchmark or a claim of broad language-model quality.

## Source/version/retrieval date

- Source: `roneneldan/TinyStories`
- Declared source revision: `main`
- Retrieved: `2026-09-14T06:37:52+00:00`
- Source artifact: [TinyStories Parquet shard](https://huggingface.co/api/datasets/roneneldan/TinyStories/parquet/default/train/0.parquet)
- Acquisition method: allowlisted Hugging Face Parquet API, one shard, first 100,000 usable text records.

## License/terms

The dataset card declares `CDLA-Sharing-1.0`; the license source is the
[TinyStories dataset card](https://huggingface.co/datasets/roneneldan/TinyStories).
Downstream users must review the source terms before redistribution.

## Files and SHA-256

The raw and processed files are intentionally ignored from Git because they are
large local data artifacts. Their checksums are recorded in
`data/manifests/tinystories-smoke.yaml`:

- Raw JSONL: `5ea77ac3c9fadc57e1db0c82fbb6709bef209dd9213eb8fe9423c13cca96ac1c`
- Train JSONL: `1613a0868eda0f1ba05ed6bb3680165566515f643e430f98abf62d106dd16785`
- Validation JSONL: `9d87a6a79461390c40acd8ed95e652c9fe4c403ec35521aecd0f7e3f55da9b21`
- Test JSONL: `139823eb6f9d97b8b02364d72b8898c70bc119f16e6a1b8f84cdec59d997f2e6`

## Content description

Records contain a stable repository-generated `id` and source `text` field.
Blank or non-text records are excluded from the canonical normalized input.

## Processing pipeline/version

The pipeline is implemented in `src/hai/data/pipeline.py`. Raw JSONL is written
once and changed to mode `0444`; preparation refuses to overwrite existing
outputs and records every processed-file checksum.

## Split method/seed

Records are ordered by `SHA256(f"{seed}:{id}")` with seed `1337`, then assigned
to train/validation/test at 80%/10%/10%. Test data is not used for tokenizer
training or promotion decisions.

## Counts and sizes

| Artifact | Records | Approximate bytes |
| --- | ---: | ---: |
| Raw | 100,000 | 95,327,528 |
| Train | 80,000 | 76,229,013 |
| Validation | 10,000 | 9,572,653 |
| Test | 10,000 | 9,525,862 |

## Known limitations/biases

This is a bounded smoke subset from a synthetic-story corpus, not a
representative evaluation set. The source revision is recorded as `main`; a
future release should replace that mutable reference with an immutable upstream
revision identifier and artifact checksum.

## Privacy/safety review

No private or personal data was supplied by the operator. The repository does
not treat that statement as a guarantee about upstream content; users should
review the upstream dataset and license before broader use.
