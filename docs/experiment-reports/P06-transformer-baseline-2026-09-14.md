# Experiment / Phase Evidence: P06-transformer-baseline-2026-09-14

## Hypothesis / goal

The 25–35M parameter Transformer can train from scratch on the governed smoke
dataset, produce a reproducible checkpoint, and run held-out validation and
generation on CUDA.

## Scope decision

ADR-0004 records the approved use of `tinystories-smoke-v1` because the planned
BabyLM manifest is not present. This result is explicitly smoke-scale and is not
a BabyLM comparison.

## Preflight

- Exact model parameter count: `25,478,144`.
- Device: CUDA in the validated ROCm Compose image.
- Context length: 512; micro-batch: 4.
- Five-step loss: `190.32144165039062` to `61.02756118774414`.
- Peak allocated GPU memory: `1,140,398,080` bytes.
- Throughput: `0.3383365399` steps/second.

## Baseline run

- Experiment ID: `TFRM-0001`.
- Seed: `1337`.
- Steps: `300`.
- Start loss: `190.32144165039062`.
- Final training loss: `7.3690185546875`.
- Checkpoint: `artifacts/checkpoints/TFRM-0001/checkpoint.pt`.
- Checkpoint SHA-256:
  `f0a245bf2e63da308b25b1108dd6e90259a37315c4cdf51f7e010f1e036fca03`.

## Held-out evaluation and generation

- Split: VALIDATION only.
- Loss: `7.340579509735107`.
- Perplexity: `1541.605224609375`.
- Prompt: `Once upon a time`.
- Generation completed with fixed greedy decoding.

## Verification

- `hai model create`: passed with exact parameter count.
- `hai train`: passed on CUDA.
- `hai evaluate`: passed on VALIDATION.
- `hai generate`: passed.
- Final repository lint/unit checks and phase-state validation: pending final gate.

## Limitations

The short 300-step run and bounded TinyStories data are sufficient for a
reproducible baseline checkpoint, not a quality or scaling claim. The model
card preserves this limitation.

## Gate decision

PASS — smoke-scale baseline freeze.

## Next action

Advance to P07 only after this evidence, model card, experiment registry, and
phase state are committed, CI-checked, merged, and synchronized.
