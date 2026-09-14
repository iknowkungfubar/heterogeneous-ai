from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class EvidenceRef:
    kind: str
    source_id: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExpertResult:
    task_id: str
    expert_id: str
    answer: str | None
    raw_confidence: float | None = None
    calibrated_confidence: float | None = None
    evidence: list[EvidenceRef] = field(default_factory=list)
    verified: bool = False
    verification_type: str | None = None
    latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Representation:
    modality: str
    model_id: str
    vector_ref: str | None = None
    confidence: float | None = None
    provenance: list[EvidenceRef] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
