# P22 multimodal fusion and visual reasoning — 2026-09-14

## Hypothesis

A fused image/question model can answer objective visual yes/no questions
better than unimodal and random baselines, while a cross-attention variant
should only be retained if it improves over a simpler projection fusion.

## Data and provenance

- Experiment: `P22-VISUAL-OBJECTIVE-20260914`
- Parent: `P21-IMAGE-TEXT-20260914`
- Configuration: `configs/models/fusion.yaml`
- Implementation: `src/hai/models/fusion.py`
- Dataset: deterministic procedural shape images paired with four candidate
  questions per image
- Data hash: `bb68da9d45b70d3a5c98f8f50bc2100e2a42dd80b3d1ee3f86034568908ca089`
- Source image hash: `ce68e799bd3bef9064badbf3852739a795fa85903db6f0ce650d39f0e263c6e4`
- Config SHA-256: `88a71fb70437b2ec7c7bf1238f436e51e706218f6c9387efc34f9b15bc10e85b`
- Splits: 1,152 TRAIN, 384 VALIDATION, 384 TEST records

All question variants for a source image remain in that image’s split, so
validation images cannot leak into TRAIN through another question variant.

## Commands and results

```text
hai multimodal fusion-data-verify --config configs/models/fusion.yaml
source_image_split_isolation=true
external_download=false
train_records=1152
validation_records=384
test_records=384

hai multimodal train-fusion --variant projection --config configs/models/fusion.yaml
parameter_count=4114
start_loss=0.7510514855384827
final_loss=0.5604122281074524
validation_accuracy=0.75

hai multimodal train-fusion --variant cross-attention --config configs/models/fusion.yaml
parameter_count=8338
start_loss=0.7111332416534424
final_loss=0.5588645935058594
validation_accuracy=0.75

hai multimodal evaluate --suite visual-objective-v1 --config configs/models/fusion.yaml
projection_accuracy=0.75
cross_attention_accuracy=0.75
unimodal_baseline_accuracy=0.5
random_baseline_accuracy=0.4921875
retained_variant=projection
```

MLflow runs:

- data verification: `be32d9b897d94a07b4f0418802b12182`
- projection training: `5d7ae35b8f5a4d5994d77f82cfff5a73`
- cross-attention training: `749847326fc147db851fd8996cc5f958`
- ablation evaluation: `ac84e79bc94b4fc2bcbdd92208d0c6ba`

Ruff passed and all 41 non-slow tests passed. The pinned ROCm container again
reported the MIOpen database readability warning for `gfx110050.HIP.fdb.txt`;
both GPU fusion variants trained and evaluated successfully.

## Gate evaluation

- Multimodal reasoning beats unimodal/random baselines: **pass**. Both fusion
  variants reached `0.75`, exceeding the `0.50` unimodal and `0.492` random
  VALIDATION baselines.
- Retained fusion is supported by ablation: **pass**. Cross-attention added
  no accuracy improvement over projection while using roughly twice the
  parameters, so the simpler projection variant is retained.

## Conclusion and next action

P22 passes. Projection fusion is the retained visual reasoning design;
cross-attention remains a measured ablation rather than production complexity.
Advance to P23 audio specialization.
