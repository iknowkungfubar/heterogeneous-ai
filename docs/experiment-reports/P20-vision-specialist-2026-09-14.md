# P20 vision specialist from scratch — 2026-09-14

## Hypothesis

A small randomly initialized vision encoder can learn a standalone image
classification task from deterministic image tensors before any multimodal
fusion is attempted.

## Dataset and provenance

- Experiment: `P20-VISION-20260914`
- Parent: `P19-SSM-20260914`
- Dataset card: `docs/dataset-cards/procedural-shapes-v1.md`
- Configuration: `configs/models/vision.yaml`
- Implementation: `src/hai/models/vision.py`
- Dataset: `procedural-shapes-v1`, generated locally; no external download
- Dataset hash: `ce68e799bd3bef9064badbf3852739a795fa85903db6f0ce650d39f0e263c6e4`
- Config SHA-256: `fefe6d01d55974b5d2d25f20cdcb790c22346fc0eb1a1e81913b60c3c8c89ad9`
- Shape: 480 grayscale 16x16 images, four classes
- Splits: 288 TRAIN, 96 VALIDATION, 96 TEST

The model is a 9,860-parameter patch-embedding/Transformer encoder with random
initialization. TRAIN images are used for optimization; the recorded result
uses VALIDATION only. TEST remains reserved for a later frozen evaluation.

## Commands and results

```text
hai vision data-verify --config configs/models/vision.yaml
external_download=false
class_count=4
train_samples=288
validation_samples=96
test_samples=96

hai vision train --config configs/models/vision.yaml
device=cuda
initialization=random
parameter_count=9860
training_steps=80
start_loss=1.6190468072891235
final_loss=0.006880323868244886
validation_loss=0.0065082586370408535
validation_accuracy=1.0
images_per_second=858.3098884517267
peak_memory_bytes=169977344
checkpoint_sha256=5d2af341dd67376a65afff6c222dde40aae07e2f1b9cd8968e6cfa079d707bf2

hai vision evaluate --config configs/models/vision.yaml --split validation
accuracy=1.0
loss=0.0065082586370408535
checkpoint_sha256=5d2af341dd67376a65afff6c222dde40aae07e2f1b9cd8968e6cfa079d707bf2
```

MLflow runs:

- data verification: `21a8112aaaad4d5a84b800457b1df687`
- training: `88e10f7c344f4808ab377b8041b297fc`
- evaluation: `e4a0a4ff62554e918b8dcb3e6087908e`

Ruff passed and the full non-slow suite passed with 37 tests. The pinned
ROCm container emitted an MIOpen database readability warning for
`gfx110050.HIP.fdb.txt`; the GPU training, evaluation, and checkpoint hash
round-trip still completed successfully.

## Gate evaluation

- Standalone vision works before fusion: **pass**. Data verification,
  scratch training, held-out VALIDATION evaluation, and provenance-bound
  checkpoint reload all completed before any multimodal work.
- Model is scratch-trained and held-out evaluated: **pass**. The model uses
  random initialization and reached 1.0 VALIDATION accuracy without reading
  TEST labels.

## Conclusion and next action

P20 passes. The repository now has a measured standalone visual specialist
and a documented procedural dataset boundary. Advance to P21 image-text
representation alignment.
