# Architecture

## System concept

The system is a modular cognitive architecture. Specialists are trained independently where practical, exposed through a common protocol, and orchestrated by routing, verification, consensus, and uncertainty logic.

```text
Input Manager
    │
    ├── Text encoder / language path
    ├── Vision encoder
    ├── Audio encoder
    └── Structured/graph/state adapters
    │
Shared/typed representations
    │
Executive Router
    │
    ├── Transformer language expert
    ├── GRU or Mamba/SSM sequence expert
    ├── Symbolic solver
    ├── Retriever
    ├── Knowledge graph / GNN
    ├── Vision expert
    ├── Audio expert
    └── World model / planner
    │
Candidate ExpertResult objects
    │
Verifier
    │
Calibration + evidence ranking
    │
Consensus / abstention
    │
Final result + trace + provenance
```

## Architectural principles

1. **Typed boundaries.** Specialists communicate with stable schemas rather than sharing implementation details.
2. **Independent evidence.** Agreement is useful but is not proof; deterministic or externally grounded evidence outranks unsupported neural agreement.
3. **Externalized knowledge.** New facts should usually enter explicit memory/retrieval rather than forcing weight updates.
4. **Separate prediction from observation.** World-model simulations and neural guesses carry provenance distinct from observed/retrieved facts.
5. **Train sequentially on consumer hardware.** Specialists need not be resident or train jointly.
6. **Lazy inference.** The router should invoke the minimum useful specialist set.
7. **Replaceability.** GRU can remain if Mamba is unstable; a component can be retired if ablation does not justify it.
8. **Verification first.** Where formal or executable verification exists, use it.

## Core data structures

### ExpertResult

Every expert returns, at minimum:

- task ID;
- expert ID/version;
- answer or structured prediction;
- raw score if available;
- calibrated confidence if available;
- evidence/provenance references;
- verification status/type;
- latency and resource metadata;
- additional typed metadata.

### Representation

Cross-modal components exchange a typed representation containing:

- vector/tensor reference;
- modality;
- originating model/version;
- confidence/quality metadata;
- provenance;
- optional spatial/temporal alignment metadata.

### MemoryRecord

Memory entries contain:

- immutable ID;
- memory type;
- content or artifact reference;
- timestamp/version;
- source/provenance;
- trust/confidence state;
- embedding/graph links;
- retention metadata.

## Long-term component topology

### Language
A scratch-trained small causal Transformer provides natural-language modeling and a common language interface.

### Sequence
A GRU establishes architectural diversity first. A later Mamba/SSM phase tests whether state-space sequence modeling improves complementarity and efficiency on the actual AMD stack.

### Symbolic
SymPy/Z3/algorithms provide exact reasoning for supported formal domains.

### Retrieval and memory
A scratch-trained dual encoder plus lexical retrieval supplies explicit evidence. Working, episodic, semantic, relational, and procedural memories remain distinct.

### Graph
A provenance-aware knowledge graph supports deterministic traversal; a GNN is retained only if it contributes beyond algorithms.

### Vision/audio
Scratch-trained unimodal encoders are validated independently before cross-modal alignment and fusion.

### World model/planner
A predictive model learns state transitions in controlled environments. Deterministic planners use the model for search and counterfactual evaluation.

### Orchestration
An interpretable supervised router is the baseline. Constrained RL can later optimize expert-call cost, but does not get arbitrary host capabilities.

## Resource strategy

Canonical checkpoints remain full-quality research artifacts. Inference variants may use quantization/caching/lazy loading only after regression comparison. Model lifecycle management must make loaded/resident specialists explicit.
