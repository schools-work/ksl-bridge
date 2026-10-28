# KSL Bridge developer commands. Run `make help`.
SHELL := /bin/bash
COMPOSE := docker compose -f infra/compose/compose.yaml --env-file .env
PSQL := psql "$${KSL_DATABASE_URL_SYNC:-postgresql://postgres:postgres@localhost:5432/ksl}"
TASK ?= islr

.PHONY: help install install-ml up down logs schema seed check-schema dev workers lint typecheck boundaries \
	test test-integration test-all contracts keys train eval clean

help:            ## show targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-18s %s\n",$$1,$$2}'

install:         ## install service + dev dependencies
	uv sync --extra dev

install-ml:      ## install ML extras (torch, mediapipe, …)
	uv sync --extra dev --extra ml

up:              ## start postgres, redis, minio, mlflow
	$(COMPOSE) up -d postgres redis minio mlflow

up-all:          ## start everything incl. gateway + workers
	$(COMPOSE) up -d --build

down:            ## stop the stack
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f --tail=200

schema:          ## load the canonical schema and run invariant checks
	$(PSQL) -v ON_ERROR_STOP=1 -f db/ksl_bridge_v1.sql
	$(MAKE) check-schema

check-schema:    ## executable schema invariants (rolls back)
	$(PSQL) -v ON_ERROR_STOP=1 -f db/checks/schema_invariants.sql

seed:            ## dev users + demo restaurant
	$(PSQL) -v ON_ERROR_STOP=1 -f db/seeds/dev_users.sql
	$(PSQL) -v ON_ERROR_STOP=1 -f db/seeds/demo_restaurant.sql

dev:             ## run the gateway with reload
	uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --ws websockets

workers:         ## run CPU workers locally
	uv run python -m workers.translate_worker & uv run python -m workers.production_worker & wait

lint:
	uv run ruff check . && uv run ruff format --check .

typecheck:
	uv run mypy ksl_core app workers

boundaries:      ## import-linter contracts (PROJECT_STRUCTURE §1.1)
	uv run lint-imports

contracts:       ## regenerate code + schemas from contracts/ (CI fails on drift)
	uv run python contracts/codegen/gen_python.py
	uv run python contracts/codegen/gen_kotlin.py --out build/kotlin
	uv run python -m tools.scripts.export_contracts

keys:            ## generate a dev Ed25519 key pair and print .env lines
	uv run python -m tools.scripts.gen_dev_keys

test:            ## unit + contract + golden tests
	uv run pytest tests/unit tests/contract tests/golden -q

test-integration: ## needs `make up` (or KSL_TEST_DATABASE_URL / KSL_TEST_REDIS_URL)
	uv run pytest tests/integration tests/security tests/chaos -m integration -q

test-all: lint typecheck boundaries test test-integration

train:           ## make train TASK=islr  (default config per task in ml/configs/)
	uv run python -m ml.train.train_$(TASK)

eval:            ## make eval BUNDLE=… MANIFEST=…  (CSLR)
	uv run python -m ml.eval.evaluate --task cslr --bundle $(BUNDLE) --manifest $(MANIFEST)

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist **/__pycache__
