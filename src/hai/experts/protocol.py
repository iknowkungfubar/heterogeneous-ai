from __future__ import annotations

import time
from typing import Protocol

from hai.common.schema import EvidenceRef, ExpertResult
from hai.symbolic.reasoning import SymbolicError, exact_arithmetic, solve_linear_equation_text


class ExpertAdapter(Protocol):
    expert_id: str
    model_version: str

    def answer(self, task_id: str, prompt: str) -> ExpertResult: ...


class SymbolicExpert:
    expert_id = "symbolic"
    model_version = "symbolic-1.0"

    def answer(self, task_id: str, prompt: str) -> ExpertResult:
        started = time.perf_counter()
        try:
            result = (
                solve_linear_equation_text(prompt) if "=" in prompt else exact_arithmetic(prompt)
            )
        except SymbolicError as exc:
            return ExpertResult(
                schema_version="1.0",
                task_id=task_id,
                expert_id=self.expert_id,
                model_version=self.model_version,
                answer=None,
                evidence=[EvidenceRef("error", task_id, {"message": str(exc)})],
                latency_ms=(time.perf_counter() - started) * 1000,
                status="error",
            )
        return ExpertResult(
            schema_version="1.0",
            task_id=task_id,
            expert_id=self.expert_id,
            model_version=self.model_version,
            answer=result.value,
            raw_score=1.0,
            raw_confidence=1.0,
            evidence=[EvidenceRef("symbolic_trace", task_id, {"trace": result.trace})],
            verified=True,
            verification_type="exact_symbolic",
            latency_ms=(time.perf_counter() - started) * 1000,
            metadata={"operation_trace": list(result.trace)},
        )


class NotApplicableExpert:
    def __init__(self, expert_id: str, model_version: str, reason: str):
        self.expert_id = expert_id
        self.model_version = model_version
        self.reason = reason

    def answer(self, task_id: str, prompt: str) -> ExpertResult:
        del prompt
        return ExpertResult(
            schema_version="1.0",
            task_id=task_id,
            expert_id=self.expert_id,
            model_version=self.model_version,
            answer=None,
            evidence=[EvidenceRef("not_applicable", task_id, {"reason": self.reason})],
            status="not_applicable",
            metadata={"reason": self.reason},
        )


def core_experts() -> dict[str, ExpertAdapter]:
    return {
        "symbolic": SymbolicExpert(),
        "transformer-baseline": NotApplicableExpert(
            "transformer-baseline", "transformer-baseline-v1", "no structured-output decoder"
        ),
        "gru-sequence": NotApplicableExpert(
            "gru-sequence", "gru-sequence-v1", "no structured-output decoder"
        ),
    }
