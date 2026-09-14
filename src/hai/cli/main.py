from __future__ import annotations

import argparse
import json
from pathlib import Path

from hai.data.pipeline import (
    DataGovernanceError,
    fetch_dataset,
    inspect_config,
    prepare_dataset,
    verify_dataset,
)


def env_check() -> int:
    try:
        import torch
    except ImportError as exc:
        print(json.dumps({"ok": False, "error": f"torch import failed: {exc}"}, indent=2))
        return 2

    result = {
        "ok": bool(torch.cuda.is_available()),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "gpu_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "devices": [],
    }
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            result["devices"].append({"index": idx, "name": torch.cuda.get_device_name(idx)})
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(prog="hai")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("env-check", help="show PyTorch/ROCm GPU visibility")
    data = sub.add_parser("data", help="inspect and verify governed dataset inputs")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    inspect = data_sub.add_parser("inspect-config", help="validate a dataset config")
    inspect.add_argument("--config", type=Path, required=True)
    for command, help_text in (
        ("fetch", "acquire the allowlisted dataset into immutable raw JSONL"),
        ("prepare", "create deterministic processed splits"),
        ("verify", "verify manifest, checksums, and split disjointness"),
    ):
        action = data_sub.add_parser(command, help=help_text)
        action.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "env-check":
        return env_check()
    if args.command == "data":
        try:
            if args.data_command == "inspect-config":
                result = inspect_config(args.config)
            elif args.data_command == "fetch":
                result = fetch_dataset(args.config, Path.cwd())
            elif args.data_command == "prepare":
                result = prepare_dataset(args.config, Path.cwd())
            elif args.data_command == "verify":
                result = verify_dataset(args.config, Path.cwd())
            else:
                return 2
            print(json.dumps(result, indent=2, sort_keys=True))
        except (DataGovernanceError, OSError) as exc:
            parser.error(str(exc))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
