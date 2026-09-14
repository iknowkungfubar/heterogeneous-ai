from __future__ import annotations

import json
from pathlib import Path

import yaml

from hai.evaluation.benchmark import (
    _manifest_path,
    _read_records,
    load_benchmark_config,
)
from hai.experts.protocol import core_experts


class RouterError(ValueError):
    """Raised when router artifacts or inputs violate the routing contract."""


def load_router_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("id") != "router-v1":
        raise RouterError("router config must declare router-v1")
    config.setdefault("confidence_threshold", 0.75)
    return config


def extract_features(record: dict) -> dict[str, float]:
    prompt = str(record.get("prompt", ""))
    category = record.get("category", "unknown")
    return {
        "task_type": str(category),
        "input_length": float(len(prompt)),
        "contains_numbers": float(any(character.isdigit() for character in prompt)),
        "contains_equation": float("=" in prompt),
        "contains_graph_structure": float("path " in prompt),
        "symbolic_applicable": float(
            category
            in {
                "arithmetic",
                "linear_equations",
                "boolean_logic",
                "constraints",
                "graph_paths",
                "deterministic_sequences",
                "language_classification",
            }
        ),
    }


def build_router_dataset(config_path: Path, root: Path) -> dict:
    benchmark_config = load_benchmark_config(Path(config_path))
    manifest = yaml.safe_load(_manifest_path(root, benchmark_config).read_text(encoding="utf-8"))
    records = _read_records(root, manifest, "train")
    rows = []
    for record in records:
        rows.append({"id": record["id"], "features": extract_features(record), "label": "symbolic"})
    output = root / "data" / "benchmarks" / manifest["benchmark_id"] / "router-train.jsonl"
    if output.exists():
        raise RouterError(f"refusing to overwrite router dataset: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    return {
        "path": str(output.relative_to(root)),
        "records": len(rows),
        "labels": {"symbolic": len(rows)},
    }


def _feature_matrix(records: list[dict]):
    from sklearn.feature_extraction import DictVectorizer

    vectorizer = DictVectorizer(sparse=False)
    matrix = vectorizer.fit_transform([extract_features(record) for record in records])
    return vectorizer, matrix


def train_router(
    config_path: Path, benchmark_config_path: Path, root: Path, split: str = "train"
) -> dict:
    config = load_router_config(config_path)
    benchmark_config = load_benchmark_config(benchmark_config_path)
    manifest = yaml.safe_load(_manifest_path(root, benchmark_config).read_text(encoding="utf-8"))
    records = _read_records(root, manifest, split)
    if split != "train":
        raise RouterError("router training is restricted to the TRAIN benchmark split")
    try:
        import joblib
        from sklearn.tree import DecisionTreeClassifier, export_text
    except ImportError as exc:
        raise RouterError("scikit-learn and joblib are required for router training") from exc
    vectorizer, matrix = _feature_matrix(records)
    labels = ["symbolic"] * len(records)
    model = DecisionTreeClassifier(max_depth=3, random_state=1337)
    model.fit(matrix, labels)
    output = root / config.get("output_dir", "artifacts/routers/router-v1")
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "vectorizer": vectorizer}, output / "router.joblib")
    metadata = {
        "router_id": config["id"],
        "training_split": split,
        "records": len(records),
        "known_categories": sorted({str(record["category"]) for record in records}),
        "feature_names": list(vectorizer.get_feature_names_out()),
        "classes": list(model.classes_),
        "decision_trace": export_text(
            model, feature_names=list(vectorizer.get_feature_names_out())
        ),
        "confidence_threshold": config["confidence_threshold"],
        "label_source": "P10 core expert applicability; symbolic is the only structured adapter",
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "router": config["id"],
        "records": len(records),
        "classes": list(model.classes_),
        "decision_trace": metadata["decision_trace"],
    }


def evaluate_router(
    config_path: Path, benchmark_config_path: Path, root: Path, split: str = "validation"
) -> dict:
    config = load_router_config(config_path)
    benchmark_config = load_benchmark_config(benchmark_config_path)
    manifest = yaml.safe_load(_manifest_path(root, benchmark_config).read_text(encoding="utf-8"))
    records = _read_records(root, manifest, split)
    output = root / config.get("output_dir", "artifacts/routers/router-v1")
    try:
        import joblib
    except ImportError as exc:
        raise RouterError("joblib is required for router evaluation") from exc
    artifact = joblib.load(output / "router.joblib")
    model, vectorizer = artifact["model"], artifact["vectorizer"]
    predictions = model.predict(
        vectorizer.transform([extract_features(record) for record in records])
    )
    probabilities = model.predict_proba(
        vectorizer.transform([extract_features(record) for record in records])
    )
    threshold = config["confidence_threshold"]
    correct = 0
    abstained = 0
    selected_cost = 0
    traces = []
    symbolic = core_experts()["symbolic"]
    for record, prediction, probability in zip(
        records, predictions, probabilities, strict=True
    ):
        confidence = float(max(probability))
        selected = str(prediction) if confidence >= threshold else "abstain"
        if selected == "abstain":
            abstained += 1
        else:
            selected_cost += 1
            result = symbolic.answer(record["id"], record["prompt"])
            correct += int(selected == "symbolic" and result.answer == str(record["answer"]))
        traces.append(
            {
                "task_id": record["id"],
                "selected_expert": selected,
                "confidence": confidence,
                "features": extract_features(record),
            }
        )
    baseline_correct = sum(
        int(symbolic.answer(record["id"], record["prompt"]).answer == str(record["answer"]))
        for record in records
    )
    return {
        "router": config["id"],
        "split": split,
        "samples": len(records),
        "accuracy": correct / len(records) if records else 0.0,
        "always_symbolic_accuracy": baseline_correct / len(records) if records else 0.0,
        "abstained": abstained,
        "selected_expert_calls": selected_cost,
        "decision_trace": traces[:5],
    }


def route_prompt(config_path: Path, prompt: str, category: str, root: Path) -> dict:
    config = load_router_config(config_path)
    output = root / config.get("output_dir", "artifacts/routers/router-v1")
    try:
        import joblib
    except ImportError as exc:
        raise RouterError("joblib is required for routing") from exc
    artifact = joblib.load(output / "router.joblib")
    record = {"id": "route-request", "category": category, "prompt": prompt}
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    features = extract_features(record)
    if category not in metadata.get("known_categories", []):
        return {"selected_expert": "abstain", "confidence": 0.0, "features": features}
    matrix = artifact["vectorizer"].transform([features])
    prediction = artifact["model"].predict(matrix)[0]
    confidence = float(max(artifact["model"].predict_proba(matrix)[0]))
    return {
        "selected_expert": str(prediction)
        if confidence >= config["confidence_threshold"]
        else "abstain",
        "confidence": confidence,
        "features": features,
    }
