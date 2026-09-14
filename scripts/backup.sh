#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/../heterogeneous-ai-metadata-backup-$(date +%Y%m%d-%H%M%S).tar.gz}"
cd "$ROOT"
# Metadata/source backup only. Large ignored data/checkpoints and local secrets
# require separate storage policy and must never enter this archive implicitly.
tar --exclude='.git' --exclude='.env' --exclude='.env.*' \
    --exclude='*/.env' --exclude='*/.env.*' --exclude='.venv' \
    --exclude='*/.venv' --exclude='.serena' --exclude='*/.serena' \
    --exclude='.codegraph' --exclude='*/.codegraph' \
    --exclude='data/raw' --exclude='data/interim' --exclude='data/processed' \
    --exclude='data/multimodal' --exclude='artifacts/checkpoints' \
    --exclude='artifacts/environment' --exclude='mlruns' \
    -czf "$OUT" .
echo "$OUT"
