#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/../heterogeneous-ai-metadata-backup-$(date +%Y%m%d-%H%M%S).tar.gz}"
cd "$ROOT"
# Metadata/source backup only. Large ignored data/checkpoints require separate storage policy.
tar --exclude='.git' --exclude='data/raw' --exclude='data/interim' --exclude='data/processed' \
    --exclude='data/multimodal' --exclude='artifacts/checkpoints' --exclude='mlruns' \
    -czf "$OUT" .
echo "$OUT"
