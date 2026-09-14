#!/usr/bin/env bash
set -euo pipefail

# Non-destructive prerequisite inspector. It deliberately does NOT install
# packages because host-level installation requires explicit human approval.
missing=0
for cmd in git docker lspci; do
  if command -v "$cmd" >/dev/null 2>&1; then
    printf 'OK: %s -> %s\n' "$cmd" "$(command -v "$cmd")"
  else
    printf 'MISSING: %s\n' "$cmd"
    missing=1
  fi
done

for path in /dev/kfd /dev/dri; do
  if [[ -e "$path" ]]; then
    printf 'OK: %s exists\n' "$path"
  else
    printf 'MISSING: %s\n' "$path"
    missing=1
  fi
done

if (( missing )); then
  echo
  echo 'One or more prerequisites are missing. Stop at P01 and diagnose them.'
  echo 'This script intentionally did not change the host.'
  exit 1
fi

echo 'Host prerequisite inspection passed.'
