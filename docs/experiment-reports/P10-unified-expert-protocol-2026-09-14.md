# Experiment / Phase Evidence: P10-unified-expert-protocol-2026-09-14

## Hypothesis / goal

The symbolic, Transformer, and GRU expert surfaces can emit one versioned,
serializable result contract containing answers, confidence/score fields,
latency, evidence, verification metadata, status, and model provenance.

## Protocol

`ExpertResult` schema version `1.0` now validates required task/expert/model
identifiers, confidence ranges, non-negative latency, verification metadata, and
explicit statuses: `ok`, `not_applicable`, `error`, or `abstain`.
`EvidenceRef` validates source identity and serializes safely. Both contracts
round-trip through JSON-compatible dictionaries.

## Core adapters

- `symbolic`: exact arithmetic/linear-equation adapter with operation traces,
  exact verification, raw score, confidence, and measured latency.
- `transformer-baseline`: explicit `not_applicable` result until a structured
  answer decoder exists; includes model version and reason evidence.
- `gru-sequence`: same explicit status and provenance treatment.

All adapters implement the same `ExpertAdapter` protocol and are returned by
`core_experts()`; orchestration does not inspect model internals.

## Verification

- `hai expert self-test --all`: all three adapters serialized successfully with
  model versions and evidence.
- Unit suite: `16 passed`.
- Lint: passed.
- Malformed confidence and missing verification metadata are rejected.
- Symbolic error responses preserve structured failure evidence.

## Gate decision

PASS

## Limitations and negative findings

The learned experts intentionally emit `not_applicable` for structured benchmark
prompts because no safe answer decoder has been implemented. This is explicit
provenance, not a benchmark score or confidence claim.

## Next action

Advance to P11 learned task router.
