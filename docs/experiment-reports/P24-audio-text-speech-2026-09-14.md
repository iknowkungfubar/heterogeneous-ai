# P24 audio-text alignment and speech — 2026-09-14

## Hypothesis

Scratch audio and transcript representations can align measurably, and a
small randomly initialized audio recognition head can report held-out WER on
source-safe paired data.

## Data and provenance

- Experiment: `P24-AUDIO-TEXT-20260914`
- Parent: `P23-AUDIO-20260914`
- Dataset card: `docs/dataset-cards/procedural-tones-speech-pairs-v1.md`
- Implementation: `src/hai/models/speech.py`
- Audio source hash: `738e9da21cce61647c8f7e7ab7e1f85d1e553363732ed415131a5e815dff7863`
- Paired data hash: `8b001c853f6379767f359bb6dc71eea07419723320930abede39dd82036a8edd`
- Config SHA-256: `fd17d00fedef1657f98ab38462214b894e165f635f20a86611638d165d764433`
- Splits: 288 TRAIN, 96 VALIDATION, 96 TEST records
- Source-safe validation groups: 24, with no TRAIN source overlap

## Commands and results

```text
hai audio speech-data-verify --config configs/models/audio.yaml
source_overlap=false
external_download=false
train_records=288
validation_records=96
test_records=96

hai audio train-alignment --config configs/models/audio.yaml
device=cuda
initialization=random
training_steps=60
start_loss=2.4824957847595215
final_loss=0.0008835271000862122
validation_recall_at_1=1.0
random_recall_at_1=0.25
checkpoint_sha256=ff5a4a88e88fb2fa22e439b1193231d6b8421a8ab1a78cfbf1a4af72ae3739eb

hai audio train-asr --config configs/models/audio.yaml
device=cuda
initialization=random
training_steps=60
start_loss=1.3867261409759521
final_loss=0.020057588815689087

hai audio evaluate-asr --config configs/models/audio.yaml --metric wer
wer=0.0
exact_transcript_accuracy=1.0
held_out_source_groups=24
source_overlap_with_train=false
checkpoint_sha256=82b312f43965d300bb31d953d17c7db042164883a46da44661726ebd1355ecfb
```

MLflow runs:

- data verification: `20086dd7f73f42f0b2e76ed0cc647d38`
- alignment training: `1e6dda2a64fe42aab0d7781912aeeed5`
- alignment evaluation: `e7dc1b90a44943cdaf8d193f6bbefa0b`
- ASR training: `6778634550c94167b491154334b73436`
- ASR evaluation: `b202c7186cc84b3ba69bd3115ceaf60f`

Ruff passed and all 45 non-slow tests passed. The pinned ROCm container
reported the known MIOpen database readability warning; GPU alignment, ASR,
and held-out evaluation completed successfully.

## Gate evaluation

- Audio-text alignment measurable: **pass**. Class-aware validation recall@1
  was `1.0`, above random `0.25`, with hash-bound paired data.
- ASR WER reported on held-out sources: **pass**. Validation WER was `0.0`
  with 24 held-out source groups and no source overlap with TRAIN.

## Conclusion and next action

P24 passes. The repository now has a measurable audio-text alignment path and
a bounded scratch ASR/WER path. Advance to P25 unified multimodal routing.
