# Model Card: transformer-baseline-v1

## Purpose

First meaningful scratch-trained Transformer baseline for pipeline and
validation-reference work. This is a smoke-scale baseline on TinyStories, not a
BabyLM benchmark result.

## Architecture and exact parameter count

Eight causal Transformer layers, hidden size 512, feed-forward size 1536, eight
attention heads, context length 512, tied input/output embeddings.

Exact parameter count: **25,478,144**.

## Initialization

All model weights were randomly initialized. No pretrained model or vocabulary
weights were loaded.

## Training data

Manifest: `data/manifests/tinystories-smoke.yaml`.
TRAIN split SHA-256:
`1613a0868eda0f1ba05ed6bb3680165566515f643e430f98abf62d106dd16785`.
VALIDATION and TEST remained separate and were not used for optimization.

The BabyLM dataset in the original plan was not used; see ADR-0004.

## Tokenizer / input representation

Project-owned BPE tokenizer `bpe-8192-v1`, trained from TRAIN only. Its SHA-256
is recorded in the P04 model evidence.

## Training configuration

Experiment: `TFRM-0001`; seed `1337`; AdamW; learning rate `0.0003`; weight decay
`0.1`; gradient clip norm `1.0`; CUDA; 300 smoke steps.

## Hardware/software

Validated ROCm Compose image, PyTorch `2.12.0+rocm10.0.0`, device `cuda`.
Five-step preflight peak allocated GPU memory was `1,140,398,080` bytes at
`0.338` steps/second.

## Evaluation

VALIDATION loss: `7.340579509735107`.
VALIDATION perplexity: `1541.605224609375`.
Checkpoint-generated text was recorded as a deterministic smoke output, not a
quality claim.

## Strengths

Reproducible random initialization, governed data and tokenizer provenance,
held-out validation, and a complete checkpoint/generation path.

## Known limitations/failure modes

The run is short and the output remains repetitive. The dataset is a bounded
synthetic-story smoke subset, so metrics must not be compared with BabyLM or
treated as evidence of production language quality.

## Safety/data considerations

No private data was supplied. Review upstream TinyStories terms and content
before redistribution or broader use.

## Checkpoint artifact and SHA-256

Local ignored artifact: `artifacts/checkpoints/TFRM-0001/checkpoint.pt`.

SHA-256:
`f0a245bf2e63da308b25b1108dd6e90259a37315c4cdf51f7e010f1e036fca03`.
