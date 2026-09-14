from __future__ import annotations

import json
import math
from pathlib import Path

import yaml

from hai.common.schema import ExpertResult
from hai.evaluation.benchmark import _manifest_path, _read_records, load_benchmark_config
from hai.experts.protocol import core_experts
from hai.verification.layer import VerificationResult, verified_consensus, verify_expert_result


class CalibrationError(ValueError):
    """Raised when calibration inputs or methods violate the P13 contract."""


def _clip(value: float) -> float:
    return min(max(value, 1e-6), 1.0 - 1e-6)


def _logit(value: float) -> float:
    clipped = _clip(value)
    return math.log(clipped / (1.0 - clipped))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _isotonic(scores: list[float], labels: list[int]) -> list[float]:
    blocks: list[list[float | int]] = []
    for score, label in sorted(zip(scores, labels, strict=True)):
        blocks.append([score, score, 1, label])
        while len(blocks) > 1 and blocks[-2][3] / blocks[-2][2] > blocks[-1][3] / blocks[-1][2]:
            right = blocks.pop()
            left = blocks.pop()
            blocks.append([left[0], right[1], left[2] + right[2], left[3] + right[3]])
    calibrated = []
    for score in scores:
        block = next(block for block in blocks if block[0] <= score <= block[1])
        calibrated.append(float(block[3] / block[2]))
    return calibrated


def _logistic(scores: list[float], labels: list[int]) -> list[float]:
    weight, bias = 0.0, 0.0
    for _ in range(500):
        gradient_weight = gradient_bias = 0.0
        for score, label in zip(scores, labels, strict=True):
            error = _sigmoid(weight * score + bias) - label
            gradient_weight += error * score
            gradient_bias += error
        weight -= 0.1 * gradient_weight / len(scores)
        bias -= 0.1 * gradient_bias / len(scores)
    return [_sigmoid(weight * score + bias) for score in scores]


def _temperature(scores: list[float], labels: list[int]) -> list[float]:
    candidates = [0.25 + index * 0.05 for index in range(76)]
    best = min(
        candidates,
        key=lambda temperature: sum(
            (label - _sigmoid(_logit(score) / temperature)) ** 2
            for score, label in zip(scores, labels, strict=True)
        ),
    )
    return [_sigmoid(_logit(score) / best) for score in scores]


def calibration_metrics(scores: list[float], labels: list[int], bins: int = 10) -> dict:
    if not scores or len(scores) != len(labels):
        raise CalibrationError("scores and labels must be non-empty and equal length")
    brier = sum((score - label) ** 2 for score, label in zip(scores, labels, strict=True)) / len(
        scores
    )
    ece = 0.0
    reliability = []
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        members = [
            (score, label)
            for score, label in zip(scores, labels, strict=True)
            if lower <= score < upper or (index == bins - 1 and score == upper)
        ]
        if members:
            mean_score = sum(score for score, _ in members) / len(members)
            accuracy = sum(label for _, label in members) / len(members)
            ece += len(members) / len(scores) * abs(mean_score - accuracy)
            reliability.append(
                {
                    "bin": index,
                    "count": len(members),
                    "confidence": mean_score,
                    "accuracy": accuracy,
                }
            )
    return {"ece": ece, "brier": brier, "reliability": reliability}


def _validation_outputs(
    config_path: Path, root: Path, split: str
) -> list[tuple[dict, ExpertResult, VerificationResult, int]]:
    if split != "validation":
        raise CalibrationError("calibration is restricted to the VALIDATION split")
    config = load_benchmark_config(config_path)
    manifest = yaml.safe_load(_manifest_path(root, config).read_text(encoding="utf-8"))
    records = _read_records(root, manifest, split)
    expert = core_experts()["symbolic"]
    outputs = []
    for record in records:
        result = expert.answer(record["id"], record["prompt"])
        verification = verify_expert_result(record, result)
        label = int(verification.passed and result.answer == record["answer"])
        outputs.append((record, result, verification, label))
    return outputs


def calibrate(
    config_path: Path, root: Path, split: str = "validation", method: str = "all"
) -> dict:
    outputs = _validation_outputs(config_path, root, split)
    scores = [float(result.raw_confidence or 0.0) for _, result, _, _ in outputs]
    labels = [label for _, _, _, label in outputs]
    methods = ("temperature", "logistic", "isotonic") if method == "all" else (method,)
    report = {
        "split": split,
        "samples": len(outputs),
        "positive_labels": sum(labels),
        "raw": calibration_metrics(scores, labels),
        "methods": {},
    }
    for selected in methods:
        if selected == "temperature":
            calibrated = _temperature(scores, labels)
        elif selected == "logistic":
            calibrated = _logistic(scores, labels)
        elif selected == "isotonic":
            calibrated = _isotonic(scores, labels)
        else:
            raise CalibrationError(f"unsupported calibration method: {selected}")
        report["methods"][selected] = calibration_metrics(calibrated, labels)
    return report


def evaluate_consensus(config_path: Path, root: Path, split: str = "validation") -> dict:
    outputs = _validation_outputs(config_path, root, split)
    accepted = abstained = incorrect = 0
    traces = []
    for record, result, verification, _ in outputs:
        decision = verified_consensus([(result, verification)])
        if decision["status"] == "verified":
            accepted += 1
            incorrect += int(decision["answer"] != record["answer"])
        else:
            abstained += 1
        traces.append(
            {
                "task_id": record["id"],
                "decision": decision["status"],
                "verification": verification.status,
                "confidence": result.raw_confidence,
            }
        )
    accuracy_vs_coverage = []
    for threshold in (0.5, 0.75, 0.9):
        selected = [
            (result, verification)
            for _, result, verification, _ in outputs
            if float(result.raw_confidence or 0.0) >= threshold
        ]
        verified_selected = [result for result, check in selected if check.passed]
        accuracy = (
            sum(
                result.answer == record["answer"]
                for record, result, _, _ in outputs
                if result in verified_selected
            )
            / len(verified_selected)
            if verified_selected
            else 0.0
        )
        accuracy_vs_coverage.append(
            {
                "threshold": threshold,
                "coverage": len(verified_selected) / len(outputs) if outputs else 0.0,
                "selective_accuracy": accuracy,
            }
        )
    return {
        "split": split,
        "samples": len(outputs),
        "accepted": accepted,
        "abstained": abstained,
        "coverage": accepted / len(outputs) if outputs else 0.0,
        "selective_accuracy": (accepted - incorrect) / accepted if accepted else 0.0,
        "incorrect_accepted": incorrect,
        "accuracy_vs_coverage": accuracy_vs_coverage,
        "decision_trace": traces[:5],
    }


def write_report(report: dict, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return {"path": str(path), **report}
