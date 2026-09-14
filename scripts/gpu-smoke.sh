#!/usr/bin/env bash
set -euo pipefail
python -m hai.cli.main env-check
python -m hai.training.linear_smoke --steps 500
