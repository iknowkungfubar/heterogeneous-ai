# P21 image-text representation alignment — 2026-09-14

## Hypothesis

Scratch image and text encoders can learn a shared representation space on
paired procedural shape/caption data and exceed seeded random retrieval in both
directions.

## Data and provenance

- Experiment: `P21-IMAGE-TEXT-20260914`
- Parent: `P20-VISION-20260914`
- Configuration: `configs/models/multimodal.yaml`
- Implementation: `src/hai/models/multimodal.py`
- Image source: governed `procedural-shapes-v1`; no external download
- Image dataset hash: `ce68e799bd3bef9064badbf3852739a795fa85903db6f0ce650d39f0e263c6e4`
- Alignment hash: `853cbe143d23b50913a313f02921689e7fcead89789a0cfee0e35307c43a4125`
- Config SHA-256: `88e5338cb8e0382279674aa9b88979aae222f890df37b07823577e17f2161ff5`
- Pairs: 288 TRAIN, 96 VALIDATION, 96 TEST
- Caption vocabulary: 8 tokens, learned from TRAIN captions only

Captions are class-level descriptions such as “a horizontal bar.” Retrieval
hits are therefore class-aware: any same-class caption/image is a valid
semantic hit. This avoids counting repeated captions as false negatives while
still requiring the learned embedding to identify the visual class.

## Commands and results

```text
hai multimodal data-verify --config configs/models/multimodal.yaml
external_download=false
leakage_resistant_split=true
train_pairs=288
validation_pairs=96
test_pairs=96

hai multimodal train-image-text-alignment --config configs/models/multimodal.yaml
device=cuda
initialization=random
parameter_count=2913
training_steps=80
start_loss=1.4816081523895264
final_loss=0.19744589924812317
checkpoint_sha256=cde10af6f04b1b325d0df4059aac26e6a7d0c64cb81f46f368050f66c36c1895

hai multimodal evaluate-retrieval --directions image-to-text,text-to-image
image_to_text_recall_at_1=0.8229166865
text_to_image_recall_at_1=1.0
random_recall_at_1=0.3125
image_to_text_mrr=0.8266119957
text_to_image_mrr=1.0
```

MLflow runs:

- data verification: `0bb876ed50c14ce6a24e6db6efc30ec1`
- training: `1a9cb5720e504b9ea6f4688dc31a3773`
- evaluation: `1318dfdb720343a7a0951780cb484232`

Ruff passed and all 39 non-slow tests passed. The pinned ROCm container
reported the same MIOpen database readability warning observed in P20; it did
not prevent training or retrieval evaluation.

## Gate evaluation

- Bidirectional retrieval exceeds random baseline: **pass**. Both directions
  exceed random recall@1: image-to-text `0.823 > 0.313`, text-to-image
  `1.0 > 0.313`.
- Alignment artifacts/metrics reproducible: **pass**. The alignment hash,
  split sizes, TRAIN-only vocabulary, checkpoint hash, and deterministic tests
  are recorded above.

## Conclusion and next action

P21 passes. Scratch image/text encoders learned a shared class-level space on
the governed paired task. Advance to P22 multimodal fusion and visual
reasoning; retain the class-aware metric definition and leakage boundaries.
