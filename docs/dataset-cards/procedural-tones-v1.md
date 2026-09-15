# Dataset Card: procedural-tones-v1

## Purpose

Standalone P23 audio-specialist gate for controlled sound classification. This
is not a speech-recognition benchmark or a claim of real-world acoustic
coverage.

## Provenance and license

Waveforms are generated locally by `src/hai/models/audio.py` from
`configs/models/audio.yaml` and seed `1337`. No external audio files,
downloads, voices, or pretrained audio weights are used. The generated samples
are original repository-local artifacts.

## Data and preprocessing

The dataset contains 480 mono waveforms sampled at 8 kHz, each 512 samples
long. Four classes use base frequencies 220, 440, 660, and 880 Hz with
deterministic phase, amplitude, and bounded Gaussian noise variation. The
model consumes waveform tensors directly; no external feature extractor is
required.

## Splits

The fixed split is 288 TRAIN, 96 VALIDATION, and 96 TEST examples. TRAIN is
used for optimization, VALIDATION for development evaluation, and TEST remains
reserved for a future frozen evaluation.

## Reproducibility and limitations

`hai audio data-verify` reports the data/configuration hashes. This synthetic
task validates waveform decoding, a scratch CNN, and held-out evaluation. It
does not measure speech, speaker variation, environmental noise, or music.
