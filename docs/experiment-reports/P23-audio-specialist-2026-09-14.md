# P23 audio specialist from scratch — 2026-09-14

## Hypothesis

A small randomly initialized waveform CNN can learn standalone sound
classification on held-out data and beat majority/random baselines before any
speech or audio-text alignment work.

## Dataset and provenance

- Experiment: `P23-AUDIO-20260914`
- Parent: `P22-VISUAL-OBJECTIVE-20260914`
- Dataset card: `docs/dataset-cards/procedural-tones-v1.md`
- Configuration: `configs/models/audio.yaml`
- Implementation: `src/hai/models/audio.py`
- Dataset: local procedural waveforms; no external download
- Data hash: `738e9da21cce61647c8f7e7ab7e1f85d1e553363732ed415131a5e815dff7863`
- Config SHA-256: `96a599fb7b8292172c32d44ba12d2d01ed1c27f10fc07291c85bbacafa52d790`
- Format: 480 mono waveforms, 8 kHz, 512 samples, four frequency classes
- Splits: 288 TRAIN, 96 VALIDATION, 96 TEST

The model consumes waveform tensors directly and uses random initialization.
No pretrained audio weights, speech data, or external feature extractor is
used.

## Commands and results

```text
hai audio data-verify --config configs/models/audio.yaml
external_download=false
sample_rate=8000
sample_length=512
train_samples=288
validation_samples=96
test_samples=96

hai audio train --config configs/models/audio.yaml --task sound-classification
device=cuda
parameter_count=3908
training_steps=80
start_loss=1.3969742059707642
final_loss=0.015687011182308197
validation_loss=0.01357134710997343
validation_accuracy=1.0
checkpoint_sha256=1b08676b79d9e4cd80cf04b733782985a02502065b0a1ec1931f7558cbb0cb52

hai audio evaluate --config configs/models/audio.yaml --split validation
accuracy=1.0
majority_baseline_accuracy=0.2604166567
random_baseline_accuracy=0.1770833284
```

MLflow runs:

- data verification: `1661790ae0f5497cbe3f823129c873fa`
- training: `e56fd409cb164fd0ad40f17f39e1c411`
- evaluation: `af1a5241d4024e50913994fa1973f451`

Ruff passed and all 43 non-slow tests passed. The pinned ROCm container
reported the known MIOpen database readability warning for
`gfx110050.HIP.fdb.txt`; GPU training, evaluation, and checkpoint hash
validation completed successfully.

## Gate evaluation

- Standalone audio learning is stable and beats baseline: **pass**. The
  scratch CNN reached `1.0` VALIDATION accuracy, exceeding both majority
  (`0.260`) and seeded random (`0.177`) baselines.

## Conclusion and next action

P23 passes. The repository now has a measured standalone waveform specialist;
advance to P24 audio-text alignment and speech.
