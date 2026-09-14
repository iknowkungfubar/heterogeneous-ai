# Dataset Card: procedural-shapes-v1

## Purpose

Standalone vision-specialist gate for P20. This is a deterministic procedural
image task, not a claim about broad real-world visual recognition.

## Provenance and license

Images are generated locally by `src/hai/models/vision.py` from configuration
and seed `1337`. No external image files or downloads are used. The generated
examples are original repository-local artifacts and are not published as a
separate dataset.

## Data and labels

The dataset contains 480 grayscale 16x16 images across four shape classes:
horizontal bar, vertical bar, diagonal, and square. Each image receives small
deterministic pixel noise and a bounded deterministic offset. Labels are
generated from the sample index and the split permutation is seed-bound.

## Splits

The fixed split is 288 TRAIN, 96 VALIDATION, and 96 TEST images. TRAIN is used
for optimization; VALIDATION is used for development evaluation; TEST remains
reserved for final evaluation after the standalone gate is frozen.

## Reproducibility

The configuration is `configs/models/vision.yaml`. `hai vision data-verify`
reports the generated dataset hash and configuration hash. Any mismatch causes
checkpoint evaluation to fail closed.

## Limitations

Procedural shapes do not measure robustness to natural image variation,
real-world semantics, or distribution shift. The task exists to validate the
standalone image tensor/model/evaluation path before multimodal fusion.
