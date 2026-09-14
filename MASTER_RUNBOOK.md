# MASTER_RUNBOOK.md — Heterogeneous Multimodal AI From Scratch

**Authority:** This is the master project specification. `docs/phase-plans/*.md` are executable derivations of this document. When a phase plan conflicts with this file, this file wins unless a later approved ADR explicitly supersedes the relevant requirement.

## 1. Mission

Build a reproducible local-first heterogeneous multimodal AI research system from random neural initialization on consumer hardware. Specialists use different computational approaches and cooperate through stable interfaces, learned routing, independent verification, calibrated uncertainty, consensus, explicit memory, and abstention.

The project is not defined as successful merely because all fashionable model families are present. **Complexity must earn its place through measured improvement.** Failed hypotheses remain documented results.

## 2. Final architecture

```text
User / Environment
        │
Input + modality manager
        │
Typed representations
        │
Executive router
        │
 ┌──────┼────────┬─────────┬─────────┬──────────┐
 │      │        │         │         │          │
Text   SSM      GNN     Symbolic  Retrieval   Perception
LM   sequence  graph      exact     memory    vision/audio
 │      │        │         │         │          │
 └──────┴────────┴─────────┼─────────┴──────────┘
                          │
               candidate ExpertResults
                          │
                       Verifier
                          │
          evidence + provenance + calibration
                          │
                     Consensus
                          │
                   uncertainty gate
                    /            \
               answer            abstain
                    \            /
                    trace + result
                          │
                controlled experience
                          │
           memory / governed learning loop
```

The mature system includes scratch-trained language, sequence/SSM, embedding/retrieval, graph, vision, audio, and world-model components; deterministic symbolic algorithms; explicit memory and knowledge graph; planning; compute-aware orchestration; verifier/calibration/consensus/abstention; and a local service/CLI.

## 3. Definition of “from scratch”

Primary-track neural weights begin from random initialization. The tokenizer is project-trained. No external learned checkpoint may enter the primary scratch path. Library implementations are allowed. Project-produced checkpoints may be resumed. A separate clearly labeled pretrained comparison track may be added only after the scratch system is mature.

## 4. Hardware/runtime strategy

Reference target: one Linux consumer GPU with approximately 16 GB VRAM, 32 GB RAM, and a modern desktop CPU. The reference AMD RX 7900 GRE is `gfx1100`; the environment is containerized. Do not globally install the full ML Python stack on the host.

The pinned initial container is configured in `configs/environments/rocm.yaml`. The image is deliberately treated as configuration because ROCm/PyTorch versions evolve. Version changes require an ADR plus P01 smoke revalidation.

Specialists train sequentially. Joint end-to-end training is not a prerequisite. At inference, the model manager may lazily load specialists according to memory/latency tradeoffs.

## 5. Scientific integrity rules

- TRAIN: fit model/router weights.
- VALIDATION: select hyperparameters, calibration, thresholds, and promotion decisions.
- TEST: final evaluation only.
- Test contamination invalidates the affected experiment.
- Negative results stay in the experiment registry.
- Major experiments record code/environment/data/tokenizer/config/seed/checkpoint hashes and metrics.
- Important small performance differences require repeated seeds/statistical uncertainty where feasible.
- A component must be removed in ablation to demonstrate contribution.
- Efficiency claims report latency, active experts/parameters, and memory/compute alongside quality.

## 6. Evidence hierarchy

When resolving candidate answers, use the strongest applicable support. The default ordering is:

1. formal proof or deterministic exact verification;
2. deterministic execution/test evidence;
3. externally retrieved provenance-backed evidence;
4. independently verified learned evidence;
5. calibrated specialist reliability;
6. model agreement;
7. unsupported raw model confidence.

Agreement alone is never proof.

## 7. Memory architecture

Keep memory types distinct:

- working — active request/context state;
- episodic — events, actions, observations, outcomes;
- semantic — retrievable facts/documents;
- relational — knowledge graph;
- procedural — callable skills/processes;
- neural state — learned recurrent/SSM state where relevant.

New factual knowledge should enter explicit memory before weight updates. Memory records carry provenance and trust state. Unverified model output is never silently promoted to trusted knowledge.

## 8. Expert protocol

All specialists implement a stable `ExpertResult` contract containing task/expert IDs, answer/prediction, raw/calibrated scores, evidence references, verification state, latency/resource metadata, and extensible metadata. Cross-modal representations similarly carry modality/model/provenance metadata.

## 9. Verification, confidence, and abstention

Use deterministic verification whenever the task permits it. Neural scores must be calibrated on validation data before being described as probabilities of correctness. The system must support abstention for unsupported domains, low calibrated confidence, strong unresolved disagreement, failed verification, and out-of-distribution inputs.

Report quality and coverage together.

## 10. World model and planning boundary

World-model outputs are predictions, never observations. Track rollout error with horizon. Deterministic search algorithms establish the planning baseline before RL. RL is initially constrained to orchestration actions and never receives arbitrary shell access.

## 11. Performance/lifecycle

Correctness precedes optimization. Profile before applying batching, compilation, caching, lazy loading, or quantization. Canonical research checkpoints remain preserved; optimized inference variants are separate artifacts and must pass regression evaluation.

## 12. Security boundary

Neural/RL components request allowlisted structured tools through policy code; they do not execute arbitrary host commands. Validate dataset archives/types/sizes, checksum promoted artifacts, bind local services to loopback by default, keep secrets out of Git/prompts/datasets, and fault-isolate specialist failures.

## 13. Phase graph

| Phase | Name | Depends on |
|---|---|---|
| `P00` | Repository bootstrap | — |
| `P01` | Host, GPU, Docker, and ROCm environment | P00 |
| `P02` | PyTorch fundamentals and GPU training smoke tests | P01 |
| `P03` | Data governance and dataset pipeline | P02 |
| `P04` | Tokenizer from scratch | P03 |
| `P05` | Tiny Transformer pipeline smoke model | P04 |
| `P06` | Scratch-trained Transformer baseline | P05 |
| `P07` | Independent sequence specialist | P06 |
| `P08` | Deterministic symbolic reasoning specialist | P07 |
| `P09` | Objective benchmark framework | P08 |
| `P10` | Unified expert protocol and provenance | P09 |
| `P11` | Learned task router | P10 |
| `P12` | Verification layer | P11 |
| `P13` | Calibration, consensus, and abstention | P12 |
| `P14` | Core heterogeneous ablation study and M2 freeze | P13 |
| `P15` | Scratch-trained semantic retrieval | P14 |
| `P16` | Working, episodic, semantic, and procedural memory | P15 |
| `P17` | Knowledge graph | P16 |
| `P18` | Graph neural-network specialist | P17 |
| `P19` | Mamba/state-space compatibility and specialist | P18 |
| `P20` | Vision specialist from scratch | P19 |
| `P21` | Image-text representation alignment | P20 |
| `P22` | Multimodal fusion and visual reasoning | P21 |
| `P23` | Audio specialist from scratch | P22 |
| `P24` | Audio-text alignment and speech | P23 |
| `P25` | Unified multimodal router and expert orchestration | P24 |
| `P26` | World model | P25 |
| `P27` | Planning and counterfactual reasoning | P26 |
| `P28` | Compute-aware reinforcement learning orchestration | P27 |
| `P29` | Continual learning and regression protection | P28 |
| `P30` | Performance optimization and model lifecycle | P29 |
| `P31` | Local CLI and inference service | P30 |
| `P32` | Security, robustness, and fault isolation | P31 |
| `P33` | Comprehensive benchmark and reliability suite | P32 |
| `P34` | Final ablation, scaling, and scientific analysis | P33 |
| `P35` | Reproducible release and final acceptance | P34 |

`docs/task-graph.yaml` is the machine-readable dependency graph. `docs/status/phase-state.yaml` is the mutable state. Each phase has a detailed executable plan in `docs/phase-plans/`.

## 14. Milestones

### M0 — Infrastructure (P00–P02)
Repository, ROCm/PyTorch environment, GPU/training smoke tests.

### M1 — Scratch language baseline (P03–P06)
Governed data, project tokenizer, smoke model, frozen meaningful Transformer baseline.

### M1.5 — Diverse core intelligence (P07–P10)
Independent neural sequence specialist, exact symbolic specialist, objective benchmarks, common protocol.

### M2 — Heterogeneous reasoning (P11–P14)
Router, verifier, calibration, consensus, abstention, and ablation demonstrating whether the core hypothesis has merit. **Do not start perception/world-model expansion before M2 is evaluated.**

### M3 — Explicit knowledge (P15–P18)
Scratch retrieval, memory, knowledge graph, and GNN.

### M4 — Improved sequence architecture (P19)
Mamba/SSM compatibility and measured GRU-vs-SSM comparison.

### M5 — Vision/multimodality (P20–P22)
Scratch vision, image-text alignment, multimodal fusion.

### M6 — Audio (P23–P25)
Scratch audio, audio-text alignment/speech, unified multimodal routing.

### M7 — Predictive intelligence (P26–P27)
World model, planning, and counterfactual reasoning.

### M8 — Learned orchestration and lifelong controls (P28–P30)
Constrained RL routing, continual learning, and performance/model lifecycle.

### M9 — Productized local research system (P31–P35)
Local interface, security/robustness, comprehensive evaluation, final ablations, reproducible release.

## 15. Training run safety

Before any long run verify: correct branch/commit, experiment config and ID, data/tokenizer hashes, free disk, GPU health, checkpoint directory/resume test, logging, and evaluation setup. A human must approve a long run.

OOM recovery order: lower micro-batch → increase accumulation → activation checkpointing → reduce context → investigate leaks → only then change architecture. A changed architecture is a new experiment.

NaN recovery: preserve logs → reproduce small → inspect inputs/labels/loss/LR/gradients → test clipping/precision changes one variable at a time.

## 16. Definition of component done

A specialist is complete only when it has implementation, tests, reproducible training/config, checkpoint, model card, held-out evaluation, documented limitations/failures, common interface support, latency/resource measurements, and artifact hashes.

## 17. Definition of project done

Final acceptance requires all retained canonical subsystems to work together; scratch-trained core neural models and tokenizer; governed data; retrieval/memory/graph; multimodal perception/alignment; world model/planning; orchestration/verification/calibration/abstention; continual-learning regression controls; robust fault behavior; local CLI/API; full benchmark and ablation suites; model/data/system cards; clean-environment release test; and a final report that answers the research hypothesis without predetermined conclusions.

## 18. Human learning model

At each phase the novice needs to understand only: what is being built, why it exists, its inputs/outputs, how it learns/operates, how it is measured, and what failure looks like. Learn → build smallest version → inspect → measure → repair → freeze → advance.

## 19. Governing question

Before adding any technique ask: **What measured weakness does this solve, what is the smallest experiment, what is the baseline, what counts as success/failure, what complexity does it add, and can it be removed if it fails?**
