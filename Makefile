SHELL := /usr/bin/env bash

.PHONY: help health build shell env-check gpu-smoke test lint format capture-env phase new-experiment release-check mlflow-up mlflow-down mlflow-url

help:
	@printf '%s\n' \
	  'make health       - host/device/Docker diagnostic' \
	  'make build        - build ROCm development image' \
	  'make shell        - enter pinned ROCm container' \
	  'make env-check    - inspect Python/PyTorch/GPU (inside container)' \
	  'make gpu-smoke    - GPU + optimizer smoke tests (inside container)' \
	  'make test         - default unit/integration tests' \
	  'make lint         - ruff checks' \
	  'make format       - ruff formatter' \
	  'make capture-env  - write environment evidence' \
	  'make phase        - show first dependency-ready phase' \
	  'make mlflow-up    - start the loopback-only experiment UI' \
	  'make mlflow-down  - stop the local experiment UI' \
	  'make mlflow-url   - print the local experiment UI URL' \
	  'make new-experiment ID=... HYPOTHESIS="..."' \
	  'make release-check - baseline repository release checks'

health:
	./scripts/health-check.sh

build:
	docker compose build

shell:
	./scripts/enter-rocm.sh

env-check:
	python -m hai.cli.main env-check

gpu-smoke:
	./scripts/gpu-smoke.sh

test:
	pytest -m 'not gpu and not slow'

lint:
	ruff check src tests scripts/new-experiment.py scripts/check-phase.py scripts/verify-scaffold.py

format:
	ruff format src tests scripts/new-experiment.py scripts/check-phase.py scripts/verify-scaffold.py

capture-env:
	./scripts/capture-environment.sh

phase:
	python scripts/check-phase.py

mlflow-up:
	docker compose up -d mlflow

mlflow-down:
	docker compose stop mlflow

mlflow-url:
	@echo 'http://127.0.0.1:5000'

new-experiment:
	@test -n "$(ID)" || (echo 'ID is required' >&2; exit 2)
	@test -n "$(HYPOTHESIS)" || (echo 'HYPOTHESIS is required' >&2; exit 2)
	python scripts/new-experiment.py "$(ID)" "$(HYPOTHESIS)"

release-check: lint test
	python scripts/check-phase.py || true
	git diff --check
