# ADR-0004: Use the governed TinyStories split for the first baseline

## Context

P06 requires a scratch-trained 25–35M parameter Transformer baseline, but the
planned `TFRM-0001` experiment references `babylm-strict-small.yaml` and no
BabyLM manifest or approved local artifact exists. P03 produced a governed,
checksummed TinyStories split with train/validation/test roles and P05 proved the
end-to-end model path on that split.

## Decision

Run the first baseline against the existing `tinystories-smoke-v1` manifest and
label the result as a smoke-scale baseline. Keep the model architecture and
validation/test separation unchanged. Do not claim the result represents the
BabyLM strict-small regime.

## Alternatives considered

1. Download BabyLM now: deferred because its source, license, revision, and
   manifest are not yet governed in this repository.
2. Mark P06 blocked: rejected because a complete governed dataset and approved
   model path already support a useful baseline plumbing checkpoint.
3. Use the TEST split for tuning: rejected by the repository data contract.

## Consequences

The baseline is reproducible and can validate training/checkpoint/resource
plumbing immediately, but its metrics are not comparable to a future BabyLM
baseline. A future BabyLM experiment must use a new manifest, experiment ID,
and separately recorded provenance.

## Evidence

- `data/manifests/tinystories-smoke.yaml`
- `docs/experiment-reports/P03-data-governance-complete-2026-09-14.md`
- `docs/experiment-reports/P05-tiny-transformer-smoke-2026-09-14.md`
