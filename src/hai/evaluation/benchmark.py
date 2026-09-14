from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import yaml

from hai.data.pipeline import _repo_path, sha256_file, write_jsonl
from hai.symbolic.reasoning import (
    deterministic_bfs,
    evaluate_boolean,
    exact_arithmetic,
    solve_integer_constraints,
    solve_linear_equation_text,
)


class BenchmarkError(ValueError):
    """Raised when benchmark generation or verification fails closed."""


def load_benchmark_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("categories"), dict):
        raise BenchmarkError("benchmark config must contain categories")
    enabled = [name for name, active in config["categories"].items() if active]
    if not enabled or not isinstance(config.get("split_seed"), int):
        raise BenchmarkError("benchmark config needs enabled categories and integer split_seed")
    config["samples_per_category"] = int(config.get("samples_per_category", 12))
    if config["samples_per_category"] < 1 or config["samples_per_category"] > 1000:
        raise BenchmarkError("samples_per_category must be between 1 and 1000")
    return config


def _record(
    category: str, split: str, index: int, prompt: str, answer: object, difficulty: str
) -> dict:
    return {
        "id": f"core-objective-v1:{split}:{category}:{index:04d}",
        "category": category,
        "difficulty": difficulty,
        "prompt": prompt,
        "answer": answer,
    }


def _make_record(category: str, split: str, index: int, rng: random.Random) -> dict:
    if category == "arithmetic":
        a, b, c = rng.randint(1, 20), rng.randint(1, 20), rng.randint(1, 9)
        return _record(category, split, index, f"{a} + {b} * {c}", str(a + b * c), "easy")
    if category == "linear_equations":
        a, x, b = rng.randint(1, 9), rng.randint(-9, 9), rng.randint(-9, 9)
        return _record(category, split, index, f"{a}*x {b:+d} = {a * x + b}", str(x), "medium")
    if category == "boolean_logic":
        left, right = bool(rng.randint(0, 1)), bool(rng.randint(0, 1))
        prompt = f"{str(left)} and not {str(right)}"
        return _record(category, split, index, prompt, left and not right, "easy")
    if category == "constraints":
        minimum = rng.randint(1, 15)
        maximum = rng.randint(minimum, 20)
        answer = list(range(minimum, maximum + 1))
        return _record(category, split, index, f"{minimum} <= x <= {maximum}", answer, "medium")
    if category == "graph_paths":
        length = rng.randint(2, 5)
        path = [f"n{index}_{part}" for part in range(length)]
        middle = ",".join(path[1:-1]) or "none"
        return _record(
            category, split, index, f"path {path[0]} to {path[-1]} via {middle}", path, "medium"
        )
    if category == "deterministic_sequences":
        start, step = rng.randint(0, 20), rng.randint(1, 9)
        sequence = [start + step * offset for offset in range(4)]
        return _record(
            category, split, index, ", ".join(map(str, sequence)) + ", ?", start + 4 * step, "easy"
        )
    if category == "language_classification":
        positive = rng.randint(0, 1) == 1
        text = "bright kind helpful" if positive else "dark harsh harmful"
        return _record(category, split, index, text, "positive" if positive else "negative", "easy")
    raise BenchmarkError(f"unsupported benchmark category: {category}")


def _manifest_path(root: Path, config: dict) -> Path:
    return root / "data" / "manifests" / f"{config.get('id', 'core-objective-v1')}.yaml"


def generate_benchmark(config_path: Path, root: Path) -> dict:
    config = load_benchmark_config(config_path)
    benchmark_id = config.get("id", "core-objective-v1")
    output_dir = root / "data" / "benchmarks" / benchmark_id
    manifest_path = root / "data" / "manifests" / f"{benchmark_id}.yaml"
    if output_dir.exists() or manifest_path.exists():
        raise BenchmarkError("refusing to overwrite an existing benchmark")
    enabled = [name for name, active in config["categories"].items() if active]
    files: dict[str, dict] = {}
    for split_index, split in enumerate(("train", "validation", "test")):
        records = []
        for category_index, category in enumerate(enabled):
            seed = config["split_seed"] + split_index * 100_000 + category_index * 1_000
            rng = random.Random(seed)
            records.extend(
                _make_record(category, split, index, rng)
                for index in range(config["samples_per_category"])
            )
        path = output_dir / f"{split}.jsonl"
        file_hash = write_jsonl(records, path)
        files[split] = {
            "path": str(path.relative_to(root)),
            "sha256": file_hash,
            "records": len(records),
        }
    manifest = {
        "benchmark_id": benchmark_id,
        "version": 1,
        "generator": "hai.evaluation.benchmark",
        "split_seed": config["split_seed"],
        "categories": enabled,
        "samples_per_category": config["samples_per_category"],
        "splits": files,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return manifest


def _read_records(root: Path, manifest: dict, split: str) -> list[dict]:
    details = manifest["splits"].get(split)
    if not isinstance(details, dict):
        raise BenchmarkError(f"manifest has no {split} split")
    path = _repo_path(root, details.get("path"), f"benchmark {split}")
    if sha256_file(path) != details.get("sha256"):
        raise BenchmarkError(f"benchmark {split} checksum verification failed")
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def verify_benchmark(config_path: Path, root: Path) -> dict:
    config = load_benchmark_config(config_path)
    manifest_path = _manifest_path(root, config)
    if not manifest_path.is_file():
        raise BenchmarkError("benchmark manifest does not exist; generate first")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise BenchmarkError("benchmark manifest must be a YAML mapping")
    split_records = {
        split: _read_records(root, manifest, split) for split in ("train", "validation", "test")
    }
    ids = [record["id"] for records in split_records.values() for record in records]
    if len(ids) != len(set(ids)):
        raise BenchmarkError("benchmark IDs overlap across splits")
    expected_categories = set(manifest["categories"])
    for split, records in split_records.items():
        if {record.get("category") for record in records} != expected_categories:
            raise BenchmarkError(f"{split} does not contain every configured category")
    return {
        "ok": True,
        "benchmark_id": manifest["benchmark_id"],
        "split_counts": {split: len(records) for split, records in split_records.items()},
        "category_counts": dict(
            Counter(record["category"] for record in split_records["validation"])
        ),
        "manifest_sha256": sha256_file(manifest_path),
    }


def _symbolic_answer(record: dict) -> object:
    category, prompt = record["category"], record["prompt"]
    if category == "arithmetic":
        return exact_arithmetic(prompt).value
    if category == "linear_equations":
        return solve_linear_equation_text(prompt).value
    if category == "boolean_logic":
        return evaluate_boolean(prompt).value
    if category == "constraints":
        minimum, maximum = (int(part.strip()) for part in prompt.split("<=")[::2])
        return list(solve_integer_constraints(0, 20, ((">=", minimum), ("<=", maximum))).value)
    if category == "graph_paths":
        route, middle = prompt.split(" via ")
        _, start, _, goal = route.split()
        nodes = [] if middle == "none" else middle.split(",")
        path = [start, *nodes, goal]
        graph = {node: (path[index + 1],) for index, node in enumerate(path[:-1])}
        graph[path[-1]] = ()
        return list(deterministic_bfs(graph, start, goal).value)
    if category == "deterministic_sequences":
        values = [int(part.strip()) for part in prompt.removesuffix(", ?").split(",")]
        return values[-1] + (values[-1] - values[-2])
    if category == "language_classification":
        return "positive" if prompt.startswith("bright") else "negative"
    raise BenchmarkError(f"no symbolic adapter for category: {category}")


def evaluate_benchmark(
    config_path: Path, root: Path, expert: str, split: str = "validation"
) -> dict:
    config = load_benchmark_config(config_path)
    manifest = yaml.safe_load(_manifest_path(root, config).read_text(encoding="utf-8"))
    records = _read_records(root, manifest, split)
    if expert != "symbolic":
        return {
            "expert": expert,
            "split": split,
            "status": "not_applicable",
            "reason": "learned checkpoints have no structured-output adapter yet",
        }
    correct = 0
    by_category: Counter[str] = Counter()
    for record in records:
        if _symbolic_answer(record) == record["answer"]:
            correct += 1
            by_category[record["category"]] += 1
    return {
        "expert": expert,
        "split": split,
        "status": "scored",
        "samples": len(records),
        "correct": correct,
        "accuracy": correct / len(records),
        "correct_by_category": dict(by_category),
    }
