# Reproducibility

A result is not canonical unless another clean environment can reconstruct its configuration and load/produce the expected artifacts.

## Every serious run records

- experiment ID and hypothesis;
- Git commit and dirty/clean status;
- container image identifier/digest when available;
- Python/PyTorch/ROCm versions;
- dependency lock/freeze;
- GPU/CPU/RAM information;
- dataset manifest/hash;
- tokenizer hash;
- model and training config;
- random seed;
- checkpoint hash;
- metrics and evaluation command;
- wall-clock/resource information;
- conclusion.

## Random seeds

Use an explicit seed for all development experiments. Important architecture comparisons should be repeated with multiple seeds when practical, e.g. `1337`, `2027`, and `31415`.

## Canonical artifacts

Never overwrite canonical checkpoints. Use experiment-derived paths such as:

`artifacts/checkpoints/TFRM-0007/step-010000/`

## Release test

A final release requires a fresh clone/container test that can:

1. build the environment;
2. run lint/unit tests;
3. run GPU and miniature training smoke tests on compatible hardware;
4. load canonical checkpoints;
5. run representative evaluation;
6. start the local CLI/API;
7. reproduce documented hashes/config references.
