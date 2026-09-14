#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    'MASTER_RUNBOOK.md', 'AGENTS.md', 'ARCHITECTURE.md', 'RESEARCH_PLAN.md',
    'DATA.md', 'EVALUATION.md', 'SECURITY.md', 'REPRODUCIBILITY.md',
    'docs/task-graph.yaml', 'docs/status/phase-state.yaml', 'experiments/registry.yaml',
]


def fail(message: str) -> None:
    print(f'ERROR: {message}', file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    for rel in REQUIRED:
        if not (ROOT / rel).is_file():
            fail(f'missing required file: {rel}')

    graph = yaml.safe_load((ROOT / 'docs/task-graph.yaml').read_text())
    state = yaml.safe_load((ROOT / 'docs/status/phase-state.yaml').read_text())
    phases = graph['phases']
    state_phases = state['phases']
    if set(phases) != set(state_phases):
        fail('task graph and phase-state phase IDs differ')

    for pid, info in phases.items():
        plan = ROOT / info['plan']
        if not plan.is_file():
            fail(f'{pid} references missing phase plan: {info["plan"]}')
        for dep in info.get('depends_on', []):
            if dep not in phases:
                fail(f'{pid} has unknown dependency {dep}')

    ready = [pid for pid, info in state_phases.items() if info['status'] == 'ready']
    if ready != ['P00']:
        fail(f'fresh scaffold should have only P00 ready; found {ready}')

    print(f'OK: {len(phases)} phases, all plans/dependencies resolve, P00 is ready.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
