#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

for dev in /dev/kfd /dev/dri; do
  if [[ ! -e "$dev" ]]; then
    echo "ERROR: required AMD GPU device path missing: $dev" >&2
    echo "Stop at P01 and repair the host GPU compute path before debugging Python." >&2
    exit 2
  fi
done

# Uses the repository's built Compose service so project dependencies are
# present. The source tree is mounted and PYTHONPATH points at the mounted src.
exec docker compose run --rm hai bash
