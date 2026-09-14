# Experiment / Phase Evidence: P07-gru-sequence-specialist-2026-09-14

## Hypothesis / goal

A randomly initialized GRU trained on the same governed corpus and tokenizer as
the Transformer will make measurably different next-token errors.

## Configuration and provenance

- Experiment ID: `GRU-0001`.
- Model config: `configs/models/gru.yaml`.
- Dataset manifest: `data/manifests/tinystories-smoke.yaml`.
- Tokenizer: `bpe-8192-v1`.
- Seed: `1337`.
- Device: CUDA in the validated ROCm Compose image.
- Parameter count: `11,879,424`.
- No pretrained weights or vocabulary were used.

## Training and held-out evaluation

- Steps: `100`.
- Training loss: `9.010451316833496` to `5.95500373840332`.
- VALIDATION loss: `6.0092668533325195`.
- VALIDATION perplexity: `407.1846923828125`.
- Checkpoint SHA-256:
  `1ff7c9166c171776eaa3370607d3316feea5ffb9e667f594205b8551fb8839eb`.

## Paired error analysis

The Transformer checkpoint `TFRM-0001` and GRU checkpoint `GRU-0001` were scored
on the same first 64 VALIDATION token blocks, totaling 32,768 predictions:

- Both correct: `439`.
- Transformer-only correct: `4,828`.
- GRU-only correct: `1,948`.
- Both wrong: `25,553`.
- Exact-one-correct complementarity: `20.6787%`.

## Negative finding

Most predictions were wrong for both specialists. The result supports measurable
error diversity, but does not establish production usefulness or a fair accuracy
comparison because the runs used different parameter counts and short budgets.

## Verification

- GRU model creation and parameter count: passed.
- CUDA training and checkpoint: passed.
- VALIDATION evaluation: passed.
- Paired error comparison: passed.
- Repository lint and tests: pending final gate.

## Gate decision

PASS — independent specialist and complementarity evidence recorded.

## Next action

Advance to P08 deterministic symbolic reasoning after committing and merging this
phase evidence.
