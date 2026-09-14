from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

import yaml

from hai.evaluation.benchmark import _manifest_path, _read_records, load_benchmark_config
from hai.experts.protocol import core_experts
from hai.verification.layer import verified_consensus, verify_expert_result


class AblationError(ValueError):
    """Raised when an ablation suite is not frozen or cannot be evaluated."""


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _bootstrap_interval(
    values: list[int], seeds: list[int], iterations: int
) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    estimates = []
    for seed in seeds:
        rng = random.Random(seed)
        for _ in range(iterations):
            sample = [values[rng.randrange(len(values))] for _ in values]
            estimates.append(sum(sample) / len(sample))
    return _percentile(estimates, 0.025), _percentile(estimates, 0.975)


def _evaluate_supported_variant(records: list[dict], variant: str) -> dict:
    experts = core_experts()
    selected = experts["symbolic"]
    correct = []
    accepted = 0
    abstained = 0
    latencies = []
    traces = []
    for record in records:
        results = []
        adapters = [selected] if variant != "all-experts" else list(experts.values())
        for expert in adapters:
            result = expert.answer(record["id"], record["prompt"])
            verification = verify_expert_result(record, result)
            results.append((result, verification))
            if result.latency_ms is not None:
                latencies.append(result.latency_ms)
        decision = verified_consensus(results)
        if decision["status"] == "verified":
            accepted += 1
            is_correct = int(decision["answer"] == record["answer"])
            correct.append(is_correct)
        else:
            abstained += 1
            correct.append(0)
        traces.append(
            {
                "task_id": record["id"],
                "decision": decision["status"],
                "experts_called": len(adapters),
            }
        )
    total = len(records)
    return {
        "status": "scored",
        "samples": total,
        "correct": sum(correct),
        "task_accuracy": sum(correct) / total if total else 0.0,
        "coverage": accepted / total if total else 0.0,
        "selective_accuracy": sum(correct) / accepted if accepted else 0.0,
        "abstained": abstained,
        "experts_per_request": 3 if variant == "all-experts" else 1,
        "median_latency_ms": statistics.median(latencies) if latencies else None,
        "p95_latency_ms": _percentile(latencies, 0.95) if latencies else None,
        "correct_samples": correct,
        "decision_trace": traces[:5],
    }


def run_ablation(config_path: Path, root: Path, split: str = "test") -> dict:
    suite = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(suite, dict) or suite.get("id") != "core-m2":
        raise AblationError("ablation config must declare core-m2")
    if split != "test":
        raise AblationError("core-m2 ablation is frozen for TEST evaluation only")
    benchmark_config = load_benchmark_config(
        root / "configs" / "benchmarks" / "core.yaml"
    )
    manifest = yaml.safe_load(_manifest_path(root, benchmark_config).read_text(encoding="utf-8"))
    records = _read_records(root, manifest, split)
    results = {}
    for variant in suite["variants"]:
        if variant in {"transformer-baseline", "gru-sequence"}:
            results[variant] = {
                "status": "not_applicable",
                "reason": "no structured-output adapter in the frozen protocol",
            }
            continue
        result = _evaluate_supported_variant(records, variant)
        result["bootstrap_task_accuracy_95ci"] = _bootstrap_interval(
            result.pop("correct_samples"), suite["bootstrap_seeds"], suite["bootstrap_iterations"]
        )
        results[variant] = result
    return {
        "suite": suite["id"],
        "benchmark": suite["benchmark"],
        "split": split,
        "frozen_validation_threshold": suite["frozen_validation_threshold"],
        "bootstrap": {
            "seeds": suite["bootstrap_seeds"],
            "iterations": suite["bootstrap_iterations"],
            "interval": "percentile-95",
        },
        "results": results,
        "conclusion": (
            "negative-core-value: learned structured baselines are not applicable "
            "and symbolic coverage is limited"
        ),
    }


def milestone_report(path: Path) -> dict:
    if not path.is_file():
        raise AblationError(f"ablation report does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
