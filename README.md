# KSL Bridge: Python AI Backend

AI-mediated, real-time, **two-way communication between Kenyan Sign Language (KSL) users and people or machines that do not know KSL** (hearing people, restaurant/hospital robots, kiosks).

```
SIGN (landmarks) ─┐                                    ┌─► TEXT
SPEECH (audio) ───┼──► UtteranceIR (glosses + English  ┼─► SPEECH
TEXT ─────────────┘     + intent + confidence)         └─► SIGN (pose sequence → avatar)
```

This repository holds the **Python backend**: the FastAPI real-time gateway and orchestrator, the inference workers, the ML research/training code, the canonical PostgreSQL schema, and the contracts shared with the Kotlin app.

## Read these first (in this order)

| File | What it is |
|---|---|
| [`MASTER_PROMPT.md`](MASTER_PROMPT.md) | Operating contract for every engineer (human or AI) working here |
| [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md) | Where code lives; Stages 0–10 with exit checks |
| [`db/ksl_bridge_v1.sql`](db/ksl_bridge_v1.sql) | Canonical database schema: the source of truth |
| [`KSL_Bridge_Backend_Implementation_Guide.md`](KSL_Bridge_Backend_Implementation_Guide.md) | The blueprint: algorithms, latency budgets, evaluation, ethics |
| [`docs/research/`](docs/research/) | Project brief, research questions, datasets, algorithm choices |
| [`contracts/`](contracts/) | Landmark spec, WebSocket protocol, error codes, IR, OpenAPI |

## Quick start (local development)

Prerequisites: Docker + Docker Compose, Python 3.12, [uv](https://docs.astral.sh/uv/), and `make`.

```bash
cp .env.example .env              # then edit secrets (or use the provided dev-only .env)
make keys                         # generate a dev Ed25519 JWT key pair into .env (optional)
make up                           # postgres(pgvector) · redis · minio · mlflow
make install                      # uv sync (service + dev tools)
make schema                       # load db/ksl_bridge_v1.sql + run db/checks/schema_invariants.sql
make seed                         # dev users + demo restaurant
make dev                          # gateway on http://localhost:8000 (docs at /docs)
make test                         # unit tests
make test-integration             # needs Postgres + Redis (make up)
```

ML work (needs the `ml` extra, ideally a CUDA GPU):

```bash
make install-ml
make train TASK=baselines         # k-NN+DTW and Random Forest (week-one numbers)
make train TASK=mpm               # masked pose pre-training
make train TASK=islr
make eval  TASK=islr
```

## Layout

```
contracts/  db/  ksl_core/  app/  workers/  ml/  tools/  infra/  tests/  docs/  data/  artifacts/
```

See `PROJECT_STRUCTURE.md` for the full tree, the dependency rules (enforced by `import-linter`) and the module ↔ schema map.

## Status

This repository is the **Stage 0–10 scaffold with working reference implementations** of the core algorithms from the guide: landmark codec, normalisation, augmentation, Kneser–Ney LM, CTC prefix beam search, Conformer pose encoder, ISLR/CSLR/MPM models, seq2seq, sign production, RAG chunking/BM25, the clarification policy, the dialogue manager, auth, rooms and the WebSocket relay.

Components that depend on data you don't have yet (for example the Maseno dataset, still restricted) or on group decisions (see `MASTER_PROMPT.md` §4.1) raise explicit errors instead of faking results.

## Group decisions still open

Recorded as PROVISIONAL in `docs/decisions/ADR-0003-no-hosted-ai-apis.md`:
- the scope of "no APIs";
- bootstrap speech models;
- v1 vocabulary and domains;
- Swahili scope;
- transcript storage;
- hosting;
- where the Kotlin app lives.

## Licence

Code: to be decided by the group (see `docs/decisions/`). Datasets keep their own licences; see `dataset.sources` in the schema and `docs/research/datasets.md`.
