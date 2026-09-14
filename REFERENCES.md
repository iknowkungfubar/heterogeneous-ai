# Technical References

This file records upstream references that inform environment/implementation decisions. Re-check upstream documentation when deliberately upgrading a dependency.

- AMD ROCm PyTorch install/container documentation: https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html
- AMD ROCm GPU specifications (RX 7900 GRE / gfx1100): https://rocm.docs.amd.com/en/docs-10.0.0/reference/gpu-specs.html
- AMD ROCm container GPU device guidance: https://rocm.docs.amd.com/_/downloads/install-on-linux/en/latest/pdf/
- State Spaces Mamba: https://github.com/state-spaces/mamba
- BabyLM 2026: https://babylm.github.io/
- Hugging Face Tokenizers: https://huggingface.co/docs/tokenizers/
- PyTorch: https://pytorch.org/docs/
- MLflow: https://mlflow.org/docs/latest/

The repository pins runtime versions through configuration rather than assuming `latest`.
