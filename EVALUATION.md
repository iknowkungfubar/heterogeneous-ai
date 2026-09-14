# Evaluation Policy

## Evaluation hierarchy

- Unit tests verify implementation behavior.
- Integration tests verify component contracts.
- Smoke tests verify miniature training/inference paths.
- Validation evaluation supports tuning/promotion decisions.
- Test evaluation supports final scientific claims.
- Acceptance tests verify the complete released system.

## Domain metrics

Use metrics appropriate to the task rather than one universal score:

- language/classification: accuracy, exact match, F1, loss/perplexity when meaningful;
- calibration: ECE, Brier score, reliability plots;
- retrieval: Recall@K, MRR, nDCG;
- graph: task accuracy plus deterministic-baseline comparison;
- vision: classification/retrieval/grounded task metrics;
- audio: classification plus WER for speech;
- world model: one-step transition accuracy/error and multi-step rollout error;
- planning: goal success, path quality/cost, planning latency;
- orchestration: task accuracy, coverage, expert calls, routing accuracy, latency, active parameters.

## Always report

- sample count;
- dataset/split version and hash;
- checkpoint/model version;
- seed(s);
- confidence interval or variability for consequential comparisons where feasible;
- resource/latency measurements for efficiency claims.

## Ablation rule

A promoted component must survive a component-removal test. If removing it has no meaningful negative effect, reconsider its place in the final system.

## Abstention

Report accuracy and coverage together. A system that achieves high accuracy only by refusing nearly everything is not equivalent to a high-coverage system.

## Fixed qualitative probes

Generation/demo samples may be retained for human inspection, but qualitative impressions never replace objective evaluation.
