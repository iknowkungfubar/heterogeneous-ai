#!/usr/bin/env bash
set -euo pipefail

echo "== Host / device checks =="
uname -a
printf '\nGPU PCI devices:\n'
lspci | grep -Ei 'vga|display' || true
printf '\n/dev/kfd:\n'
ls -l /dev/kfd 2>&1 || true
printf '\nDRI render devices:\n'
ls -l /dev/dri/render* 2>&1 || true
printf '\nDocker:\n'
docker --version 2>&1 || true
