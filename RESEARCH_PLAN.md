# Research Plan

## Primary hypothesis

A heterogeneous specialist system can improve **verified correctness, calibration, and efficiency** relative to a small standalone language-model baseline under consumer-compute constraints.

## Primary comparisons

At minimum compare:

1. Transformer only.
2. Alternative sequence model only.
3. Symbolic only where applicable.
4. Naive neural ensemble.
5. Transformer + symbolic.
6. All core specialists with static rules.
7. Learned router.
8. Router + verifier.
9. Router + verifier + calibration + consensus + abstention.
10. Later memory, graph, SSM, vision/audio, world-model, and RL additions through ablation.

## Research questions

- Do structurally diverse specialists make sufficiently different errors to justify orchestration?
- How much of the gain comes from deterministic verification rather than neural diversity?
- Can calibrated abstention increase high-confidence accuracy without collapsing coverage?
- Does explicit retrieval outperform attempting to encode factual updates in model weights?
- Does a learned GNN add anything over deterministic graph traversal for targeted tasks?
- Is an AMD-compatible Mamba/SSM specialist more efficient or complementary than GRU?
- Can scratch-trained visual/audio encoders be aligned well enough at consumer scale to improve objective multimodal tasks?
- Can a small world model provide useful planning predictions before rollout error becomes prohibitive?
- Can compute-aware routing reduce active parameters/expert calls without losing verified accuracy?

## Experimental rules

- Fix evaluation before architecture promotion.
- Use independent test data.
- Run multiple seeds when a claimed difference is small or consequential.
- Record negative findings.
- Run component-removal ablations.
- Report compute/resource cost alongside accuracy.

## Key metrics

- task accuracy / exact match / F1 as appropriate;
- verified-answer accuracy;
- coverage and abstention rate;
- Expected Calibration Error and Brier score;
- retrieval Recall@K/MRR/nDCG;
- WER for speech;
- world-model transition and rollout error;
- planning success/path quality;
- router selection accuracy and experts invoked/request;
- latency, peak VRAM/RAM, active parameters, tokens/sec;
- component contribution under ablation.

## Scientific success

The final conclusion may be **yes**, **partially**, or **no**. A negative result that is reproducible and well measured is a valid outcome.
