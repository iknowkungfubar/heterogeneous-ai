# ADR-0002: Establish sequence diversity with GRU before Mamba/SSM

## Context
The project needs a non-Transformer sequence specialist, but optimized SSM/Mamba execution can have hardware/toolchain-specific behavior on consumer AMD GPUs.

## Decision
Implement and evaluate a plain-PyTorch GRU first. Treat Mamba/SSM as a later compatibility-gated experiment. Do not make the whole project depend on Mamba availability.

## Consequences
The research can test heterogeneous neural errors early. Later SSM work has a clean GRU baseline and can be rejected without destabilizing the architecture.
