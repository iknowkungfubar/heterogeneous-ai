# Experiment / Phase Evidence: P05-tiny-transformer-smoke-2026-09-14

## Hypothesis / goal

A tiny randomly initialized causal decoder can consume the frozen project
tokenizer, reduce causal next-token loss, checkpoint/resume in a fresh process,
evaluate on VALIDATION, and generate text without pretrained weights.

## Git/environment

- Date: 2026-09-14
- Base checkpoint: `5890ae9`
- Runtime: validated ROCm Compose image; PyTorch `2.12.0+rocm10.0.0`; device `cuda`.
- Model config: `configs/models/transformer-smoke.yaml`.
- Tokenizer artifact: `bpe-8192-v1`, SHA-256 recorded in the P04 report.
- Data manifest: `data/manifests/tinystories-smoke.yaml`.

## Commands and direct evidence

- `hai model create ...`: passed; random initialization, exact parameter count `1,395,968`.
- `hai train ... --max-steps 30 --experiment P05-gpu-30`: passed on CUDA; loss reduced from `82.07455444335938` to `33.86113739013672`.
- Fresh process resume from `P05-gpu-30/checkpoint.pt` to step 40: passed; resumed loss `33.14984893798828`, final loss `28.93526840209961`.
- `hai evaluate ... --checkpoint P05-gpu-resume/checkpoint.pt`: passed on VALIDATION; loss `26.39118003845215`, perplexity `289430208512.0`.
- `hai generate ... --prompt 'Once upon a time'`: passed with fixed greedy decoding and a deterministic sample.
- `ruff check src tests`: passed.
- Non-slow unit suite: `9 passed`.

## Model/checkpoint evidence

- Architecture: 2-layer causal Transformer encoder stack with causal mask,
  hidden size 128, four attention heads, feed-forward size 384, tied input/output
  embeddings, context length 128.
- Checkpoint SHA-256: `f6adce8b231ab31d83c8491870e5670c21e803fcfa022655c62dd812bde62572`.
- Checkpoint contains model state, optimizer state, step, config, and RNG state.
- No pretrained weights or vocabulary entered the path.

## Interpretation and limitations

The smoke proves plumbing and resume semantics, not language quality. The short
run produces high perplexity and repetitive text, so no model-quality claim is
made and the result must not be used as a benchmark baseline.

## Gate decision

PASS

## Next action

Advance to P06 scratch-trained Transformer baseline with a properly designed
training/evaluation budget and pre-registered comparison criteria.
