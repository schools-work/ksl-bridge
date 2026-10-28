# KSL Bridge Backend: Project Structure

> **Companion files**
> - `KSL_Bridge_Backend_Implementation_Guide.md`: the blueprint (what each component does and why)
> - `db/ksl_bridge_v1.sql`: canonical database schema (source of truth)
> - `MASTER_PROMPT.md`: operating contract for the AI/human engineer
>
> This document defines **where every piece of code lives** and **in which order the tree grows**. The stages follow **technical dependency order**, not a calendar: a stage begins when the stage it depends on passes its exit check.

---

## 1. Top-level layout (the final shape)

```
ksl-bridge/
├── contracts/            # ① language-neutral contracts shared with the Kotlin app (SSOT)
├── db/                   # ② canonical SQL schema, functions, seeds (SSOT for the database)
├── ksl_core/             # ③ dependency-light Python package shared by service, workers and ML
├── app/                  # ④ the online service (FastAPI gateway + orchestrator), modular monolith
├── workers/              # ⑤ inference workers (Redis Streams consumers, GPU/CPU)
├── ml/                   # ⑥ research & training: data, models, decoding, training, evaluation, export
├── tools/                # ⑦ annotation tool backend, operational scripts, data-collection utilities
├── infra/                # ⑧ Docker, compose, proxy, observability, CI
├── tests/                # ⑨ cross-package test suites (unit, contract, integration, load, golden)
├── docs/                 # ⑩ ADRs, model cards, dataset cards, bias impact statements, runbooks
├── data/                 #     LOCAL ONLY (git-ignored): raw downloads, extracted landmarks, shards
├── artifacts/            #     LOCAL ONLY (git-ignored): checkpoints, exported models, eval reports
├── pyproject.toml        #     one workspace, several packages (uv workspaces)
├── uv.lock
├── Makefile              #     make dev | test | lint | schema | train-<task> | eval-<task>
├── .env.example
└── README.md
```

### 1.1 Dependency rules (enforced by `import-linter` in CI)

```
contracts  ◄── everyone may read (generated code only)
ksl_core   ◄── app, workers, ml, tools        (ksl_core imports NOTHING from them)
app        ──► ksl_core                       (app NEVER imports ml/ or workers/)
workers    ──► ksl_core, ml.models, ml.decoding (read-only use of model code; never app/)
ml         ──► ksl_core                       (ml NEVER imports app/ or workers/)
tools      ──► ksl_core, app.modules.*.application (through public use-case APIs only)
```

Inside `app/modules/<context>/` the layers are **hexagonal**:

```
api/  ──►  application/  ──►  domain/  ◄──  infrastructure/
(routers, WS handlers)  (use cases, DTOs)  (entities, value objects,   (SQLAlchemy repos, Redis,
                                            ports = abstract interfaces) S3, adapters implementing ports)
```

* `domain/` imports no framework (no FastAPI, SQLAlchemy, Redis, torch).
* `api/` contains no SQL and no business rules.
* Modules talk to each other **only** through another module's `application/` public functions or through **outbox events**. They never touch another module's tables directly.
* **Each `app/modules/<context>` owns exactly one schema in `db/ksl_bridge_v1.sql`** (identity ↔ `identity`, conversation ↔ `conversation`, …).

---

## 2. Progressive build: stage by stage

Each stage shows **only what it adds**. ✅ marks the exit check that unlocks the next stage.

### Stage 0: Foundation (repository, contracts, schema, shared core)

```
ksl-bridge/
├── pyproject.toml                         # uv workspace; ruff, mypy --strict, pytest, import-linter config
├── Makefile
├── .env.example                           # KSL_DATABASE_URL, KSL_REDIS_URL, KSL_S3_*, KSL_JWT_* …
├── .gitignore                             # data/, artifacts/, .env, *.pt, *.onnx
├── contracts/
│   ├── README.md                          # how contracts are versioned and code-generated
│   ├── landmarks/
│   │   └── landmark_spec_v1.json          # 107-point layout, FACE_IDX, slices, SPEC_VERSION
│   ├── realtime/
│   │   ├── ws_protocol_v1.json            # JSON Schema: envelope + every message type
│   │   ├── binary_frames_v1.md            # 0x01 landmarks · 0x02 audio · 0x03 poses · 0x04 tts
│   │   └── error_codes_v1.json            # 4001 … 4500
│   ├── ir/
│   │   └── utterance_ir_v1.json           # UtteranceIR JSON Schema (generated from ksl_core.ir)
│   ├── openapi/
│   │   └── openapi_v1.json                # generated from FastAPI; CI fails on drift
│   └── codegen/
│       ├── gen_python.py                  # contracts → ksl_core/_generated/*
│       └── gen_kotlin.py                  # contracts → Kotlin data classes (consumed by the app repo)
├── db/
│   ├── ksl_bridge_v1.sql                  # CANONICAL schema (15 schemas, RLS, functions, seeds)
│   ├── README.md                          # load order, roles, GUCs, partition maintenance
│   ├── seeds/
│   │   ├── dev_users.sql                  # local-only fixtures
│   │   └── demo_restaurant.sql            # demo venue, menu KB, venue_vocabulary
│   └── checks/
│       └── schema_invariants.sql          # executable tests: signer splits, gates, RLS, append-only audit
├── ksl_core/
│   ├── __init__.py
│   ├── py.typed
│   ├── _generated/                        # from contracts/ (never hand-edited)
│   │   ├── landmark_spec.py
│   │   └── ws_messages.py
│   ├── ir.py                              # UtteranceIR, Hypothesis, TokenSpan, Clarification, Intent
│   ├── landmarks.py                       # KSLB binary pack/unpack, index slices
│   ├── normalize.py                       # body-centred normalisation, hand-local, features (892-d)
│   ├── smoothing.py                       # constant-velocity Kalman / One-Euro
│   ├── vocab.py                           # Vocab with <blank>=0 and specials
│   ├── text.py                            # en/sw normalisation, numbers, KSh, Sheng map
│   └── errors.py                          # shared error types
├── infra/
│   ├── docker/
│   │   ├── base.Dockerfile                # python + uv, non-root user
│   │   ├── gateway.Dockerfile
│   │   └── worker-gpu.Dockerfile          # CUDA runtime base
│   ├── compose/
│   │   ├── compose.yaml                   # postgres(pgvector) · redis · minio · mlflow · gateway · workers
│   │   └── compose.observability.yaml     # prometheus · grafana · loki · otel-collector
│   └── ci/
│       └── github-actions.yaml            # lint · typecheck · import-linter · schema load · tests · contract drift
├── tests/
│   ├── conftest.py
│   └── unit/ksl_core/
│       ├── test_landmark_codec.py
│       ├── test_normalize_invariance.py   # hypothesis property tests
│       └── test_vocab.py
└── docs/
    ├── decisions/
    │   ├── ADR-0001-modular-monolith-plus-workers.md
    │   ├── ADR-0002-landmarks-not-video.md
    │   ├── ADR-0003-no-hosted-ai-apis.md   # MediaPipe-as-sensor, bootstrap ASR/TTS (group decision)
    │   └── ADR-0004-postgres-pgvector-single-db.md
    └── glossary.md
```

✅ `make schema` loads `db/ksl_bridge_v1.sql` clean, `db/checks/schema_invariants.sql` passes, the `ksl_core` tests pass, the contract codegen is reproducible, and CI is green.

---

### Stage 1: Identity & access

```
app/
├── __init__.py
├── main.py                                # FastAPI factory + lifespan (db, redis, bus)
├── settings.py                            # pydantic-settings, KSL_ prefix
├── platform/                              # cross-cutting technical plumbing (no business rules)
│   ├── db.py                              # async engine, session, SET LOCAL app.* GUCs per request
│   ├── redis.py
│   ├── s3.py
│   ├── outbox.py                          # write events in the same transaction; relay to Redis
│   ├── idempotency.py                     # ops.idempotency_keys helper
│   ├── ratelimit.py                       # Redis token bucket
│   ├── errors.py                          # problem+json responses, error codes
│   ├── security_headers.py
│   └── logging.py                         # structlog JSON; NO content logging by default
└── modules/
    └── identity/
        ├── domain/
        │   ├── user.py                    # User entity, Role value object, modality preferences
        │   ├── tokens.py                  # RefreshFamily, rotation policy (pure)
        │   └── ports.py                   # UserRepo, TokenRepo, OtpSender, Clock (abstract)
        ├── application/
        │   ├── register.py
        │   ├── login.py                   # password or OTP; lockout policy
        │   ├── refresh.py                 # calls identity.rotate_refresh_token()
        │   ├── guest_join.py              # room-scoped guest token
        │   ├── totp.py                    # 2FA for admin / annotator / ml_engineer
        │   └── dto.py
        ├── infrastructure/
        │   ├── passwords.py               # argon2id
        │   ├── jwt.py                     # EdDSA, kid rotation, JWKS
        │   ├── repositories.py            # SQLAlchemy, identity schema only
        │   └── otp_sms.py                 # SMS gateway adapter (non-AI third party)
        └── api/
            ├── router.py                  # /auth/*, /me, /.well-known/jwks.json
            ├── deps.py                    # current_claims, require_roles, require_mfa
            └── schemas.py
tests/
├── unit/app/identity/
└── integration/identity/                  # testcontainers Postgres + Redis
    ├── test_register_login.py
    ├── test_refresh_reuse_revokes_family.py
    └── test_rls_users_self.py
```

✅ Register → login → refresh → reuse detection → logout pass. RLS proves a user can only read themselves. The rate limits are tested.

---

### Stage 2: Venues, devices, rooms, real-time relay (text only)

```
app/modules/
├── venue/
│   ├── domain/        venue.py, device.py, ports.py
│   ├── application/   create_venue.py, register_device.py (returns API key once), rotate_key.py, staff.py
│   ├── infrastructure/ repositories.py, api_keys.py (HMAC-SHA256 + pepper)
│   └── api/           router.py (/venues, /venues/{id}/devices), device_auth.py
├── conversation/
│   ├── domain/        room.py (join code, lifecycle), member.py, utterance.py (wraps ksl_core.ir)
│   ├── application/   create_room.py, join_room.py, close_room.py, record_utterance.py (only if consented)
│   ├── infrastructure/ repositories.py, partitions.py (calls ops.create_monthly_partitions)
│   └── api/           router.py (/rooms, /rooms/join, /rooms/{id}/transcript)
└── realtime/                              # WebSocket edge, no business rules
    ├── protocol.py                        # envelope + typed messages (from ksl_core._generated)
    ├── endpoint.py                        # /ws/v1: hello-first auth, routing, watchdog
    ├── connection.py                      # ClientConn: bounded queue, droppable partials, replay buffer
    ├── local_registry.py                  # sockets on THIS replica, by room
    ├── room_bus.py                        # Redis pub/sub fan-out across replicas
    ├── resume.py                          # session replay from Redis after reconnect
    └── presence.py                        # typing / signing indicators
tests/
├── integration/realtime/
│   ├── test_two_clients_text_relay.py
│   ├── test_resume_after_disconnect.py
│   └── test_backpressure_drops_partials_only.py
└── load/
    └── locustfile_ws.py                   # 500 rooms × 2 clients
```

✅ Two phones (or two test clients) in a room exchange text across **two gateway replicas**. Graceful degradation works: text relay keeps going with no AI workers running.

---

### Stage 3: Data foundation (sources, consent, landmarks, datasets, annotation)

```
app/modules/
├── privacy/
│   ├── domain/        consent.py, retention.py
│   ├── application/   grant_consent.py, withdraw_consent.py, export_my_data.py, erase_me.py (privacy.erase_user)
│   └── api/           router.py (/me/consents, /me/export, DELETE /me)
├── dataset/
│   ├── application/   create_upload.py (presigned PUT), register_sample.py, build_dataset.py (freeze + manifest)
│   ├── infrastructure/ repositories.py, object_store.py
│   └── api/           router.py (/datasets/uploads, /datasets/samples)
└── annotation/
    ├── application/   next_task.py, submit_annotation.py, adjudicate.py, agreement.py (kappa, edit agreement)
    └── api/           router.py (/annotate/*)
ml/
├── __init__.py
└── data/
    ├── sources/                           # one loader per source → common Sample records
    │   ├── base.py                        # SourceLoader protocol; licence check against dataset.sources
    │   ├── ekitabu_kaggle.py
    │   ├── maseno_pose.py                 # enabled only once access is granted
    │   ├── google_asl_signs.py            # pretrain_only
    │   ├── own_recordings.py
    │   └── speech/{common_voice_sw.py, afrispeech.py, afrivoices_ke.py}
    ├── extract_landmarks.py               # video → (T,107,3) via local MediaPipe (sensor)
    ├── quality.py                         # hand-visibility ratio, blur, fps checks → quality_flags
    ├── shards.py                          # WebDataset .tar shards to MinIO
    ├── splits.py                          # signer-independent split (DB trigger is the 2nd line of defence)
    ├── augment.py                         # rotate, scale, time-warp, jitter, hand dropout, mirror (gloss-aware)
    ├── synth_continuous.py                # isolated → pseudo-sentences with cross-fades (labelled synthetic)
    └── datasets.py                        # torch Datasets + collate
tools/
├── annotate/                              # thin web UI backend for annotators (uses annotation module APIs)
├── collection/
│   ├── recording_protocol.md              # conditions matrix, scripts per domain
│   ├── scripts/{restaurant.csv, hospital.csv, bank.csv, transport.csv}
│   └── session_checklist.md
└── scripts/
    ├── download_sources.sh                # kaggle / huggingface-cli commands (run on a machine with access)
    └── import_source.py                   # load a source into dataset.samples
docs/
├── dataset_cards/{ekitabu.md, own_recordings.md}
└── gloss_conventions.md                   # IX-1, FS:, NUM:, CL:, variant suffixes
```

✅ A dataset build is created, frozen and reproducible from its manifest. A signer-leakage attempt is rejected. Consent withdrawal excludes samples from new builds. Inter-annotator agreement is computed on a pilot batch.

---

### Stage 4: Baselines, pre-training, isolated recognition, model registry

```
ml/
├── models/
│   ├── baselines.py                       # k-NN + DTW, Random Forest on handshape stats
│   ├── pose_encoder.py                    # Conformer encoder shared by every pose model
│   ├── mpm.py                             # masked pose modelling (self-supervised)
│   └── islr.py                            # attention pooling + ArcMargin head
├── train/
│   ├── loop.py                            # AMP, EMA, cosine LR, grad clip, early stopping
│   ├── carbon.py                          # CodeCarbon → ml.training_runs
│   ├── registry.py                        # writes ml.training_runs / model_versions / run_metrics
│   ├── train_baselines.py
│   ├── train_mpm.py
│   └── train_islr.py
├── eval/
│   ├── metrics.py                         # accuracy, WER, BLEU/chrF, ECE
│   ├── evaluate.py                        # writes ml.evaluations / evaluation_results
│   ├── robustness.py                      # perturbation matrix
│   └── fairness.py                        # per-group slices
└── configs/                               # (moved here from top-level configs/: training owns its configs)
    ├── baselines.yaml
    ├── mpm_pretrain.yaml
    └── islr_transformer.yaml
app/modules/mlops/                         # read/administer the registry (no training code here)
├── application/   list_models.py, record_gate.py, promote.py (ml.promote_model), rollback.py
└── api/           router.py (/admin/models/*)
docs/model_cards/islr_transformer.md
```

✅ The baseline and ISLR numbers sit in `ml.v_leaderboard` on the signer-independent test split. The MPM vs from-scratch ablation is recorded. `ml.promote_model` refuses a model with missing gates.

---

### Stage 5: Continuous recognition, decoding, streaming sign → text

```
ml/
├── models/cslr.py                         # streaming Conformer + CTC + intermediate CTC
├── models/fingerspell.py
├── decoding/
│   ├── ngram_lm.py                        # Kneser–Ney, domain interpolation
│   ├── ctc_beam.py                        # prefix beam search + LM fusion + context bias
│   ├── alignment.py                       # Viterbi forced alignment → per-token spans & confidence
│   └── calibration.py                     # temperature scaling, utterance-level logistic regression
├── train/{train_cslr.py, train_fingerspell.py, build_gloss_lm.py, fit_calibration.py}
└── configs/{cslr_conformer_ctc.yaml, fingerspell_ctc.yaml, gloss_lm.yaml}
workers/
├── __init__.py
├── runner.py                              # Redis Streams consumer group, micro-batching, deadlines, XAUTOCLAIM
├── adapters.py                            # ModelAdapter interface + registry
├── loader.py                              # resolves live model from ml.v_live_models, verifies sha256
├── sign_worker.py                         # CSLR + fingerspelling adapters
└── health.py                              # heartbeats for /readyz
app/modules/pipeline/                      # the orchestrator: per-session state machines
├── domain/
│   ├── session_state.py                   # IDLE → SIGNING → FINALISING
│   ├── endpointing.py                     # motion-energy + rest-position endpointer
│   └── decision.py                        # joint N-best × MT × NLU choice (pure)
├── application/
│   ├── on_landmarks.py                    # normalise (streaming EMA), enqueue CSLR jobs, partials
│   ├── on_text.py
│   └── finalise_utterance.py
└── infrastructure/
    ├── job_client.py                      # XADD + reply channel + deadlines
    └── session_store.py                   # Redis-backed state with TTL
tests/golden/cslr/                         # fixed inputs → expected N-best (regression)
```

✅ Live **sign → text** captions arrive in a room: partials with revisions and a final. The p95 latency budget is met on the reference GPU. A worker crash mid-utterance recovers.

---

### Stage 6: Two-way translation, lexicon, sign production, speech (bootstrap)

```
app/modules/lexicon/
├── application/   search_dictionary.py, approve_gloss.py (deaf_reviewer only), build_medoids.py
└── api/           router.py (/dictionary, /dictionary/{gloss}/pose)
ml/
├── models/
│   ├── seq2seq.py                         # gloss ⇄ text Transformer
│   ├── production.py                      # dictionary retrieval + Hermite blending + non-manual overlay
│   └── inbetween.py                       # learned transitions (later in this stage)
├── text2gloss/
│   ├── rules.py                           # rule baseline (validated by Deaf consultants)
│   └── validator.py                       # every gloss must exist; FS: fallback
├── tokenizers/train_bpe.py
├── train/{train_g2t.py, train_t2g.py, backtranslate.py, select_medoids.py}
└── configs/{gloss2text.yaml, text2gloss.yaml}
workers/
├── translate_worker.py                    # g2t, t2g, nlu (CPU)
├── production_worker.py                   # gloss → pose sequence (CPU)
├── asr_worker.py                          # BootstrapASRAdapter (local open weights) behind ModelAdapter
└── tts_worker.py                          # BootstrapTTSAdapter behind ModelAdapter
app/modules/pipeline/application/
├── on_audio.py                            # VAD, ASR partials, final → t2g → production
└── render_sign.py                         # 0x03 pose frames + captions broadcast
```

✅ Both directions of the single-phone scenario work: sign → text/speech and speech/text → avatar + captions. Critical glosses always show captions.

---

### Stage 7: Robustness and context (clarification, RAG biasing, feedback, personalisation)

```
app/modules/
├── pipeline/domain/clarify.py             # value-of-information policy (severity per venue, critical slots)
├── pipeline/domain/context.py             # dialogue context → expected categories → bias map
├── knowledge/
│   ├── domain/        chunking.py (recursive, structure-aware, table rows), grading.py
│   ├── application/   ingest_document.py (versioned, atomic re-index), answer_question.py (extractive),
│   │                  build_venue_vocabulary.py, translation_memory.py
│   ├── infrastructure/ bm25.py, embedder_client.py, repositories.py (calls knowledge.hybrid_search)
│   └── api/           router.py (/venues/{id}/kb/*)
└── feedback/
    ├── application/   record_feedback.py, enqueue_active_learning.py, problem_generator.py,
    │                  personal_prototypes.py, schedule_lora.py
    └── api/           (WS `feedback` messages handled via realtime → this module)
ml/
├── models/embedder.py                     # our sentence encoder (contrastive)
├── train/{train_embedder.py, train_lora_adapter.py}
└── eval/{ablation_context.py, calibration_report.py}
workers/embed_worker.py
```

✅ RQ2/RQ4 tables are produced: the robustness matrix plus the context ablations (LM, venue bias, one-turn clarification). ECE ≤ 0.05. The fairness gaps are reported in `ml.v_fairness_gaps`.

---

### Stage 8: Service agent, tools, devices (robot / kiosk)

```
app/modules/agent/
├── domain/
│   ├── frame.py                           # slot-filling frame (pure)
│   ├── policy.py                          # intent schemas, slot prompts, upsell, escalation rules
│   └── ports.py                           # VenueAdapter, ToolExecutor
├── application/
│   ├── dialogue_manager.py                # goal-based step(): ground → ask → execute
│   ├── execute_tool.py                    # idempotency key, confirmation, timeout, agent.tool_calls
│   └── escalate.py                        # staff.call, human in the loop
├── infrastructure/
│   ├── tool_registry.py                   # agent.tools ↔ Python callables
│   └── adapters/
│       ├── demo_restaurant.py
│       ├── generic_pos_rest.py
│       └── robot_bridge.py                # forwards agent.action to device sockets; motion needs confirm
└── api/router.py                          # /venues/{id}/agent, device events
ml/models/nlu.py                           # joint intent + slot model (baseline: TF-IDF + LogReg)
ml/train/train_nlu.py
```

✅ The mock-restaurant scenario completes end-to-end: sign an order, confirm it, the tool runs, staff can be escalated to. The `agent.tool_calls` guard rejects unbound or unconfirmed tools.

---

### Stage 9: Own speech models, mobile export, performance & CUDA

```
ml/
├── models/{asr.py, tts.py, vocoder.py, g2p_sw.py}
├── train/{train_asr.py, train_tts.py, train_vocoder.py}
├── export/
│   ├── to_onnx.py
│   ├── to_tflite.py                       # on-device ISLR core + fingerspelling, INT8
│   ├── quantize.py
│   └── mobile_manifest.py                 # ml.deployments target=mobile
└── kernels/
    ├── normalize_cuda.py                  # fused normalisation (load_inline)
    ├── ctc_beam_cuda/                     # C++/CUDA beam decoder (only if profiling demands)
    └── bench/                             # torch.profiler + CUDA-event benchmarks
app/modules/mlops/api/mobile.py            # GET /models/mobile/manifest
```

✅ Our ASR is compared with the bootstrap model on the Kenyan test set, and the result is recorded either way. Mobile models stay under the size/latency targets. Kernel speed-ups are measured against `torch.compile`.

---

### Stage 10: Hardening and release

```
tests/
├── contract/                              # OpenAPI + WS schema drift, Kotlin fixture round-trips
├── chaos/                                 # kill workers, drop Redis, partition network
├── security/                              # token reuse, IDOR, RLS bypass attempts, rate limits
└── load/                                  # sustained rooms, GPU saturation
infra/
├── proxy/Caddyfile                        # TLS, WS upgrade, timeouts
├── observability/{prometheus.yml, grafana/dashboards/*.json, alerts.yml}
└── backup/{pg_dump.sh, restore_test.sh}
docs/
├── model_cards/*.md                       # per production model: data, metrics, slices, limits
├── bias_impact_statement.md
├── dpia.md                                # data protection impact assessment (Kenya DPA 2019)
└── runbooks/{incident.md, rollback_model.md, rotate_keys.md, erase_user.md}
```

✅ All release gates pass, the documents above exist, and the restore test succeeds.

---

## 3. The complete tree (after Stage 10)

```
ksl-bridge/
├── contracts/
│   ├── landmarks/landmark_spec_v1.json
│   ├── realtime/{ws_protocol_v1.json, binary_frames_v1.md, error_codes_v1.json}
│   ├── ir/utterance_ir_v1.json
│   ├── openapi/openapi_v1.json
│   └── codegen/{gen_python.py, gen_kotlin.py}
├── db/
│   ├── ksl_bridge_v1.sql
│   ├── seeds/{dev_users.sql, demo_restaurant.sql}
│   └── checks/schema_invariants.sql
├── ksl_core/
│   ├── _generated/{landmark_spec.py, ws_messages.py}
│   ├── ir.py  landmarks.py  normalize.py  smoothing.py  vocab.py  text.py  errors.py
├── app/
│   ├── main.py  settings.py
│   ├── platform/{db.py, redis.py, s3.py, outbox.py, idempotency.py, ratelimit.py, errors.py,
│   │             security_headers.py, logging.py, metrics.py, tracing.py}
│   └── modules/
│       ├── identity/      {domain, application, infrastructure, api}
│       ├── venue/         {domain, application, infrastructure, api}
│       ├── conversation/  {domain, application, infrastructure, api}
│       ├── realtime/      protocol.py endpoint.py connection.py local_registry.py room_bus.py resume.py presence.py
│       ├── pipeline/      {domain: session_state, endpointing, decision, clarify, context;
│       │                   application: on_landmarks, on_audio, on_text, finalise_utterance, render_sign;
│       │                   infrastructure: job_client, session_store}
│       ├── lexicon/       {application, api}
│       ├── dataset/       {application, infrastructure, api}
│       ├── annotation/    {application, api}
│       ├── knowledge/     {domain, application, infrastructure, api}
│       ├── feedback/      {application, api}
│       ├── agent/         {domain, application, infrastructure/adapters, api}
│       ├── mlops/         {application, api}
│       ├── privacy/       {domain, application, api}
│       └── audit/         {application: record.py, query.py; api}
├── workers/
│   ├── runner.py  adapters.py  loader.py  health.py
│   ├── sign_worker.py  asr_worker.py  tts_worker.py  translate_worker.py  production_worker.py  embed_worker.py
├── ml/
│   ├── data/{sources/, extract_landmarks.py, quality.py, shards.py, splits.py, augment.py,
│   │         synth_continuous.py, datasets.py}
│   ├── models/{baselines.py, pose_encoder.py, mpm.py, islr.py, cslr.py, fingerspell.py, seq2seq.py,
│   │           production.py, inbetween.py, nlu.py, embedder.py, asr.py, tts.py, vocoder.py, g2p_sw.py}
│   ├── text2gloss/{rules.py, validator.py}
│   ├── tokenizers/train_bpe.py
│   ├── decoding/{ngram_lm.py, ctc_beam.py, alignment.py, calibration.py}
│   ├── train/{loop.py, carbon.py, registry.py, train_*.py, backtranslate.py, select_medoids.py}
│   ├── eval/{metrics.py, evaluate.py, robustness.py, fairness.py, ablation_context.py, calibration_report.py}
│   ├── export/{to_onnx.py, to_tflite.py, quantize.py, mobile_manifest.py}
│   ├── kernels/{normalize_cuda.py, ctc_beam_cuda/, bench/}
│   └── configs/*.yaml
├── tools/
│   ├── annotate/
│   ├── collection/{recording_protocol.md, scripts/*.csv, session_checklist.md}
│   └── scripts/{download_sources.sh, import_source.py, create_partitions.py}
├── infra/
│   ├── docker/{base, gateway, worker-gpu}.Dockerfile
│   ├── compose/{compose.yaml, compose.observability.yaml}
│   ├── proxy/Caddyfile
│   ├── observability/
│   ├── backup/
│   └── ci/github-actions.yaml
├── tests/{unit, contract, integration, load, chaos, security, golden}/
├── docs/{decisions/, model_cards/, dataset_cards/, runbooks/, gloss_conventions.md,
│         bias_impact_statement.md, dpia.md, glossary.md}
├── data/        (git-ignored)
└── artifacts/   (git-ignored)
```

### 3.1 Files added while implementing (as built)

| Path | Responsibility |
|---|---|
| `app/platform/principal.py` | Who is calling (user / guest / device / venue / roles / MFA), mapped onto the RLS GUCs |
| `app/platform/audit.py`, `app/platform/health.py` | Audit helper; `/healthz` + `/readyz` (DB, Redis, worker heartbeats) |
| `app/modules/identity/infrastructure/crypto.py` | AES-GCM for secrets at rest (TOTP, adapter credentials) |
| `app/modules/pipeline/domain/ports.py` | `Broadcaster` and `JobClient` ports (realtime and infrastructure implement them, so there are no cycles) |
| `app/modules/pipeline/application/session.py`, `messages.py` | Per-connection orchestration state; IR builders from worker outputs |
| `ml/db.py`, `ml/train/common.py`, `ml/train/pose_tasks.py`, `ml/train/seq2seq_data.py` | Training-side DB access, config/seed/manifest helpers, shared data loaders |
| `ml/models/asr_features.py` | Log-Mel + SpecAugment |
| `tools/scripts/export_contracts.py`, `tools/scripts/gen_dev_keys.py` | Export IR/OpenAPI into `contracts/`; dev key generation |
| `.github/workflows/ci.yml` | Symlink to `infra/ci/github-actions.yaml` (single source) |

---

## 4. Module ↔ schema ↔ responsibility map

| `app/modules/…` | DB schema | One responsibility |
|---|---|---|
| `identity` | `identity` | Who you are: accounts, roles, credentials, tokens, guests |
| `venue` | `venue` | Where you are: venues, staff, robot/kiosk devices |
| `conversation` | `conversation` | Rooms, membership, stored utterances |
| `realtime` | *(none: Redis only)* | Socket edge: protocol, fan-out, resume, backpressure |
| `pipeline` | *(reads ml, writes inference via workers)* | Per-session understanding: endpointing, decoding orchestration, clarification, decision |
| `lexicon` | `lexicon` | The KSL dictionary as SSOT for glosses |
| `dataset` | `dataset` | Sources & licences, samples, immutable builds |
| `annotation` | `annotation` | Human labelling, adjudication, agreement |
| `knowledge` | `knowledge` | RAG, translation memory, venue vocabulary for biasing |
| `feedback` | `feedback` | Corrections, active learning, personalisation |
| `agent` | `agent` | Goal-based service dialogue and tool execution |
| `mlops` | `ml` (read, gates, promote) | Model registry administration (training lives in `ml/`) |
| `privacy` | `privacy` | Consent, export, erasure, retention |
| `audit` | `audit` | Append-only accountability trail |
| *(platform)* | `ops` | Outbox, jobs, idempotency, partitions |

## 5. Naming and file conventions

* **Python:** `snake_case` modules. One cohesive responsibility per module (SRP is about cohesion, not tiny files).
* **Use cases** are verbs (`create_room.py`, `promote.py`) and expose one `async def execute(...)`.
* **Ports** are suffixed `Repo`, `Client` or `Adapter`. Implementations live in `infrastructure/`.
* **Configs:** `ml/configs/<task>_<arch>.yaml`. Every run copies its resolved config into `ml.training_runs.config`.
* **Model versions:** `YYYY.MM.patch` (`2026.11.0`) for ours, `local-N` for bootstrap open-weights models.
* **Glosses:** CAPITALS per `docs/gloss_conventions.md`, validated by the `lexicon.glosses` CHECK regex.
* **Tests mirror source paths:** `app/modules/identity/application/refresh.py` → `tests/unit/app/identity/test_refresh.py`.
