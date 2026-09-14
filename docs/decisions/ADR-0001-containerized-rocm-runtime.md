# ADR-0001: Use a containerized ROCm/PyTorch runtime

## Context
The reference host is a rolling Linux desktop distribution while ML framework/ROCm compatibility changes over time. Host-global Python and framework installation would make experiments harder to reproduce and easier to break.

## Decision
Keep the host responsible for its AMD kernel/device path and container runtime. Run the project ML userspace in a pinned AMD ROCm/PyTorch container. Treat the image as versioned configuration.

## Alternatives
- Install ROCm/PyTorch globally on the host.
- Run a full VM with GPU passthrough.
- Use cloud GPU as the primary environment.

## Consequences
The project gains cleaner reproducibility and dependency isolation. The host kernel/device layer still must support the GPU. Version upgrades require a smoke revalidation.
