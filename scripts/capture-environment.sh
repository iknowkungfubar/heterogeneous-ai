#!/usr/bin/env bash
set -euo pipefail
mkdir -p artifacts/environment

python -m torch.utils.collect_env > artifacts/environment/torch-collect-env.txt
python -m pip freeze > artifacts/environment/pip-freeze.txt
python - <<'PY' > artifacts/environment/gpu-summary.txt
import torch
print('torch:', torch.__version__)
print('hip:', torch.version.hip)
print('cuda_api_available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device_count:', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f'device_{i}:', torch.cuda.get_device_name(i))
        print(f'device_{i}:', torch.cuda.get_device_name(i))
PY
if command -v rocminfo >/dev/null 2>&1; then
  rocminfo > artifacts/environment/rocminfo.txt
fi
sha256sum artifacts/environment/* > artifacts/environment/SHA256SUMS
printf 'Environment evidence written to artifacts/environment/\n'
