from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class EvidenceRef:
    kind: str
    source_id: str
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kind or not self.source_id:
            raise ValueError("evidence kind and source_id are required")
        if not isinstance(self.detail, dict):
            raise TypeError("evidence detail must be a mapping")

    def to_dict(self) -> dict[str, Any]:
        result = {"kind": self.kind, "source_id": self.source_id, "detail": self.detail}
        json.dumps(result)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceRef:
        if not isinstance(data, dict):
            raise TypeError("evidence must be a mapping")
        return cls(kind=data["kind"], source_id=data["source_id"], detail=data.get("detail", {}))


@dataclass(slots=True)
class ExpertResult:
    schema_version: str
    task_id: str
    expert_id: str
    answer: str | None
    model_version: str = "unknown"
    raw_score: float | None = None
    raw_confidence: float | None = None
    calibrated_confidence: float | None = None
    evidence: list[EvidenceRef] = field(default_factory=list)
    verified: bool = False
    verification_type: str | None = None
    latency_ms: float | None = None
    status: str = "ok"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError("unsupported expert result schema version")
        if not self.task_id or not self.expert_id or not self.model_version:
            raise ValueError("task_id, expert_id, and model_version are required")
        if self.status not in {"ok", "not_applicable", "error", "abstain"}:
            raise ValueError("invalid expert result status")
        for name in ("raw_score", "raw_confidence", "calibrated_confidence"):
            value = getattr(self, name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.latency_ms is not None and self.latency_ms < 0:
            raise ValueError("latency_ms cannot be negative")
        if self.verified and not self.verification_type:
            raise ValueError("verification_type is required when verified is true")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a mapping")

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "expert_id": self.expert_id,
            "answer": self.answer,
            "model_version": self.model_version,
            "raw_score": self.raw_score,
            "raw_confidence": self.raw_confidence,
            "calibrated_confidence": self.calibrated_confidence,
            "evidence": [item.to_dict() for item in self.evidence],
            "verified": self.verified,
            "verification_type": self.verification_type,
            "latency_ms": self.latency_ms,
            "status": self.status,
            "metadata": self.metadata,
        }
        json.dumps(result)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExpertResult:
        if not isinstance(data, dict):
            raise TypeError("expert result must be a mapping")
        return cls(
            schema_version=data["schema_version"],
            task_id=data["task_id"],
            expert_id=data["expert_id"],
            answer=data.get("answer"),
            model_version=data.get("model_version", "unknown"),
            raw_score=data.get("raw_score"),
            raw_confidence=data.get("raw_confidence"),
            calibrated_confidence=data.get("calibrated_confidence"),
            evidence=[EvidenceRef.from_dict(item) for item in data.get("evidence", [])],
            verified=data.get("verified", False),
            verification_type=data.get("verification_type"),
            latency_ms=data.get("latency_ms"),
            status=data.get("status", "ok"),
            metadata=data.get("metadata", {}),
        )


@dataclass(slots=True)
class Representation:
    modality: str
    model_id: str
    vector_ref: str | None = None
    confidence: float | None = None
    provenance: list[EvidenceRef] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "modality": self.modality,
            "model_id": self.model_id,
            "vector_ref": self.vector_ref,
            "confidence": self.confidence,
            "provenance": [item.to_dict() for item in self.provenance],
            "metadata": self.metadata,
        }
        json.dumps(result)
        return result
