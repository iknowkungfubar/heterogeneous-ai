#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import yaml


def main() -> int:
    data = yaml.safe_load(Path("docs/status/phase-state.yaml").read_text())
    phases = data["phases"]
    ready = []
    for pid, info in phases.items():
        if info["status"] == "passed":
            continue
        deps_ok = all(phases[d]["status"] == "passed" for d in info.get("depends_on", []))
        if deps_ok:
            ready.append((pid, info["status"], info["name"]))
    if not ready:
        print("No dependency-ready incomplete phase found.")
        return 1
    pid, status, name = ready[0]
    print(f"First dependency-ready phase: {pid} — {name} (state={status})")
    if len(ready) > 1:
        print(
            "Note: multiple phases are technically dependency-ready; AGENTS.md "
            "requires working on the first unless authorized."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
