#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="Append a planned experiment to the registry")
    parser.add_argument("experiment_id")
    parser.add_argument("hypothesis")
    args = parser.parse_args()

    path = Path("experiments/registry.yaml")
    data = yaml.safe_load(path.read_text()) or {"schema_version": 1, "experiments": []}
    existing = {e["id"] for e in data.get("experiments", [])}
    if args.experiment_id in existing:
        raise SystemExit(f"experiment already exists: {args.experiment_id}")

    data.setdefault("experiments", []).append(
        {
            "id": args.experiment_id,
            "date": date.today().isoformat(),
            "hypothesis": args.hypothesis,
            "git_commit": "TO_CAPTURE",
            "parent_experiment": None,
            "dataset": None,
            "dataset_hash": None,
            "tokenizer_hash": None,
            "model_config": None,
            "training_config": None,
            "seed": 1337,
            "status": "planned",
            "metrics": {},
            "conclusion": None,
            "next_action": None,
        }
    )
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"registered {args.experiment_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
