-- =====================================================================================================
--  KSL BRIDGE: canonical database schema  (ksl_bridge_v1.sql)
--  PostgreSQL 16+  ·  pgvector ≥ 0.5 (HNSW)  ·  btree_gist  ·  pgcrypto  ·  citext  ·  pg_trgm
--
--  This file is the SOURCE OF TRUTH for the schema. Alembic migrations must converge to it.
--  Workers and training jobs read and write it but NEVER migrate it.
--
--  Bounded-context schemas (mirrored 1:1 by app/modules/<context>/ in the Python backend):
--    shared        enums, GUC helpers, generic triggers
--    identity      users, roles, credentials, OTP, refresh-token families, guests
--    venue         venues (restaurants, hospitals…), staff, devices (robots/kiosks)
--    conversation  rooms, members, WebSocket sessions, utterances (partitioned)
--    lexicon       KSL glosses, translations, sign clips for production, prototypes
--    dataset       sources & licences, signers, recording sessions, samples, immutable builds
--    annotation    annotation tasks, annotations, adjudications, agreement reports
--    ml            vocabularies, experiments, training runs, model registry, deployments,
--                  evaluations, calibration, release gates
--    inference     per-request inference log (partitioned), clarifications
--    feedback      user corrections, active-learning queue, personal adaptations
--    agent         agent definitions, tools, venue tool bindings, sessions, turns, tool calls,
--                  escalations
--    knowledge     RAG documents & chunks (hybrid BM25 + vector), translation memory,
--                  venue vocabulary for decoder biasing, retrieval logs
--    privacy       consent texts, consents, erasure requests, retention policies
--    audit         append-only audit events (partitioned)
--    ops           transactional outbox, jobs, idempotency keys, partition maintenance
--
--  Conventions
--    * uuid PKs (gen_random_uuid()); bigserial only for very high-volume child rows.
--    * timestamptz everywhere (stored UTC; display in Africa/Nairobi).
--    * created_at / updated_at on mutable tables; updated_at maintained by trigger.
--    * `version` integer columns = optimistic locking on rows edited concurrently.
--    * Row-level security is FAIL-CLOSED: the app sets GUCs per request/transaction:
--          SET LOCAL app.current_user_id  = '<uuid>';
--          SET LOCAL app.current_roles    = 'signer,annotator';
--          SET LOCAL app.current_venue_id = '<uuid>';     -- optional (operators, devices)
--          SET LOCAL app.current_guest_id = '<uuid>';     -- QR guests (no account)
--          SET LOCAL app.current_device_id = '<uuid>';    -- robots / kiosks
--      With no GUCs set, protected tables return no rows.
-- =====================================================================================================

BEGIN;

-- -----------------------------------------------------------------------------------------------------
-- 0. Extensions, roles, schemas
-- -----------------------------------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS citext;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS btree_gist;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'ksl_migrator') THEN CREATE ROLE ksl_migrator NOLOGIN; END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'ksl_app')      THEN CREATE ROLE ksl_app      NOLOGIN; END IF;  -- gateway / orchestrator
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'ksl_worker')   THEN CREATE ROLE ksl_worker   NOLOGIN; END IF;  -- inference workers
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'ksl_trainer')  THEN CREATE ROLE ksl_trainer  NOLOGIN; END IF;  -- training / eval jobs
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'ksl_readonly') THEN CREATE ROLE ksl_readonly NOLOGIN; END IF;  -- analytics / dashboards
END $$;

CREATE SCHEMA IF NOT EXISTS shared;
CREATE SCHEMA IF NOT EXISTS identity;
CREATE SCHEMA IF NOT EXISTS venue;
CREATE SCHEMA IF NOT EXISTS conversation;
CREATE SCHEMA IF NOT EXISTS lexicon;
CREATE SCHEMA IF NOT EXISTS dataset;
CREATE SCHEMA IF NOT EXISTS annotation;
CREATE SCHEMA IF NOT EXISTS ml;
CREATE SCHEMA IF NOT EXISTS inference;
CREATE SCHEMA IF NOT EXISTS feedback;
CREATE SCHEMA IF NOT EXISTS agent;
CREATE SCHEMA IF NOT EXISTS knowledge;
CREATE SCHEMA IF NOT EXISTS privacy;
CREATE SCHEMA IF NOT EXISTS audit;
CREATE SCHEMA IF NOT EXISTS ops;

-- -----------------------------------------------------------------------------------------------------
-- 1. shared: enums, GUC helpers, generic triggers
-- -----------------------------------------------------------------------------------------------------
CREATE TYPE shared.modality        AS ENUM ('sign', 'speech', 'text', 'device');
CREATE TYPE shared.user_role       AS ENUM ('signer', 'speaker', 'operator', 'annotator', 'deaf_reviewer', 'ml_engineer', 'admin');
CREATE TYPE shared.venue_type      AS ENUM ('restaurant', 'hospital', 'bank', 'office', 'transport', 'education', 'general');
CREATE TYPE shared.lang            AS ENUM ('ksl', 'en', 'sw', 'other_sl');   -- other_sl = foreign sign language (pretraining only)
CREATE TYPE shared.hand            AS ENUM ('left', 'right', 'ambidextrous');
CREATE TYPE shared.hearing_status  AS ENUM ('deaf', 'hard_of_hearing', 'hearing_coda', 'hearing', 'undisclosed');
CREATE TYPE shared.consent_scope   AS ENUM ('store_transcripts', 'contribute_landmarks', 'contribute_video', 'contribute_audio',
                                            'personalisation', 'research_publication', 'demographics');
CREATE TYPE shared.review_status   AS ENUM ('pending', 'in_review', 'approved', 'rejected');
CREATE TYPE shared.sample_kind     AS ENUM ('isolated', 'continuous', 'fingerspelling', 'speech', 'synthetic_continuous');
CREATE TYPE shared.data_split      AS ENUM ('train', 'val', 'test');
CREATE TYPE shared.model_task      AS ENUM ('mpm', 'islr', 'cslr', 'fingerspell', 'gloss_lm', 'gloss2text', 'text2gloss',
                                            'nlu', 'asr', 'tts', 'sign2pose', 'nonmanual', 'embedder', 'reranker', 'calibrator');
CREATE TYPE shared.deploy_stage    AS ENUM ('shadow', 'canary', 'production', 'retired');
CREATE TYPE shared.deploy_target   AS ENUM ('server', 'mobile');
CREATE TYPE shared.job_status      AS ENUM ('queued', 'running', 'succeeded', 'failed', 'cancelled');
CREATE TYPE shared.feedback_kind   AS ENUM ('confirm', 'correct', 'reject', 'clarify_choice');
CREATE TYPE shared.clarify_kind    AS ENUM ('choose_one', 'confirm', 'repeat', 'fingerspell');
CREATE TYPE shared.turn_actor      AS ENUM ('user', 'agent', 'device', 'staff', 'system');
CREATE TYPE shared.tool_effect     AS ENUM ('none', 'write', 'physical');
CREATE TYPE shared.call_status     AS ENUM ('pending', 'succeeded', 'failed', 'rejected', 'timeout');

-- GUC helpers (return NULL when unset → policies fail closed)
CREATE FUNCTION shared.current_user_id() RETURNS uuid
LANGUAGE sql STABLE AS $$ SELECT NULLIF(current_setting('app.current_user_id', true), '')::uuid $$;

CREATE FUNCTION shared.current_venue_id() RETURNS uuid
LANGUAGE sql STABLE AS $$ SELECT NULLIF(current_setting('app.current_venue_id', true), '')::uuid $$;

CREATE FUNCTION shared.current_guest_id() RETURNS uuid
LANGUAGE sql STABLE AS $$ SELECT NULLIF(current_setting('app.current_guest_id', true), '')::uuid $$;

CREATE FUNCTION shared.current_device_id() RETURNS uuid
LANGUAGE sql STABLE AS $$ SELECT NULLIF(current_setting('app.current_device_id', true), '')::uuid $$;

CREATE FUNCTION shared.has_role(r text) RETURNS boolean
LANGUAGE sql STABLE AS $$
  SELECT r = ANY (string_to_array(COALESCE(current_setting('app.current_roles', true), ''), ','))
$$;

CREATE FUNCTION shared.set_updated_at() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
  NEW.updated_at := now();
  RETURN NEW;
END $$;

CREATE FUNCTION shared.bump_version() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
  IF NEW.version <> OLD.version THEN
    RAISE EXCEPTION 'optimistic lock: caller must not set version directly' USING ERRCODE = '40001';
  END IF;
  NEW.version := OLD.version + 1;
  RETURN NEW;
END $$;

-- -----------------------------------------------------------------------------------------------------
-- 2. ops: partition maintenance, outbox, jobs, idempotency
-- -----------------------------------------------------------------------------------------------------
CREATE FUNCTION ops.create_monthly_partitions(parent regclass, start_month date, months int)
RETURNS void LANGUAGE plpgsql AS $$
DECLARE
  m date := date_trunc('month', start_month)::date;
  part text;
  sch text;
  tbl text;
BEGIN
  SELECT n.nspname, c.relname INTO sch, tbl
  FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE c.oid = parent;
  FOR i IN 0 .. months - 1 LOOP
    part := format('%I.%I', sch, tbl || '_' || to_char(m + make_interval(months => i), 'YYYY_MM'));
    EXECUTE format('CREATE TABLE IF NOT EXISTS %s PARTITION OF %s FOR VALUES FROM (%L) TO (%L)',
                   part, parent, m + make_interval(months => i), m + make_interval(months => i + 1));
  END LOOP;
END $$;

COMMENT ON FUNCTION ops.create_monthly_partitions IS
  'Run monthly (scheduler) for every partitioned table: SELECT ops.create_monthly_partitions(''<table>'', now()::date, 3);';

CREATE TABLE ops.outbox (
  id            bigserial PRIMARY KEY,
  aggregate     text        NOT NULL,                  -- 'model_version', 'utterance', 'knowledge.document'
  aggregate_id  text        NOT NULL,
  event_type    text        NOT NULL,                  -- 'ModelPromoted', 'FeedbackRecorded', 'DocumentReindexed'
  event_version int         NOT NULL DEFAULT 1,
  payload       jsonb       NOT NULL,
  created_at    timestamptz NOT NULL DEFAULT now(),
  published_at  timestamptz,
  attempts      int         NOT NULL DEFAULT 0
);
CREATE INDEX outbox_unpublished_idx ON ops.outbox (created_at) WHERE published_at IS NULL;

CREATE TABLE ops.jobs (
  id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  kind        text            NOT NULL,                -- 'train', 'evaluate', 'extract_landmarks', 'reindex_kb', 'erase_user'
  payload     jsonb           NOT NULL DEFAULT '{}',
  status      shared.job_status NOT NULL DEFAULT 'queued',
  priority    int             NOT NULL DEFAULT 100,
  attempts    int             NOT NULL DEFAULT 0,
  max_attempts int            NOT NULL DEFAULT 3,
  run_after   timestamptz     NOT NULL DEFAULT now(),
  locked_by   text,
  locked_at   timestamptz,
  last_error  text,
  created_by  uuid,
  created_at  timestamptz     NOT NULL DEFAULT now(),
  finished_at timestamptz
);
CREATE INDEX jobs_ready_idx ON ops.jobs (priority, run_after) WHERE status = 'queued';

-- claim the next job safely across many workers
CREATE FUNCTION ops.claim_job(p_kind text, p_worker text) RETURNS ops.jobs
LANGUAGE plpgsql AS $$
DECLARE j ops.jobs;
BEGIN
  SELECT * INTO j FROM ops.jobs
   WHERE status = 'queued' AND kind = p_kind AND run_after <= now()
   ORDER BY priority, run_after
   FOR UPDATE SKIP LOCKED LIMIT 1;
  IF NOT FOUND THEN RETURN NULL; END IF;
  UPDATE ops.jobs SET status = 'running', locked_by = p_worker, locked_at = now(), attempts = attempts + 1
   WHERE id = j.id RETURNING * INTO j;
  RETURN j;
END $$;

CREATE TABLE ops.idempotency_keys (
  key          text        NOT NULL,
  scope        text        NOT NULL,                   -- 'agent.tool_call', 'dataset.upload', 'admin.promote'
  request_hash bytea       NOT NULL,
  response     jsonb,
  created_at   timestamptz NOT NULL DEFAULT now(),
  expires_at   timestamptz NOT NULL DEFAULT now() + interval '24 hours',
  PRIMARY KEY (scope, key)
);

-- -----------------------------------------------------------------------------------------------------
-- 3. identity
-- -----------------------------------------------------------------------------------------------------
CREATE TABLE identity.users (
  id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  email             citext UNIQUE,
  phone_e164        text   UNIQUE CHECK (phone_e164 ~ '^\+[1-9][0-9]{7,14}$'),
  password_hash     text,                                         -- argon2id; NULL = OTP-only
  display_name      text   NOT NULL CHECK (length(display_name) BETWEEN 1 AND 80),
  locale            text   NOT NULL DEFAULT 'en-KE' CHECK (locale IN ('en-KE', 'sw-KE')),
  preferred_in      shared.modality NOT NULL DEFAULT 'text',
  preferred_out     shared.modality NOT NULL DEFAULT 'text',
  dominant_hand     shared.hand,
  sign_speed        real   NOT NULL DEFAULT 1.0 CHECK (sign_speed BETWEEN 0.5 AND 1.5),
  caption_scale     real   NOT NULL DEFAULT 1.0 CHECK (caption_scale BETWEEN 0.75 AND 2.5),
  status            text   NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'suspended', 'erased')),
  email_verified_at timestamptz,
  phone_verified_at timestamptz,
  last_login_at     timestamptz,
  version           int    NOT NULL DEFAULT 1,
  created_at        timestamptz NOT NULL DEFAULT now(),
  updated_at        timestamptz NOT NULL DEFAULT now(),
  erased_at         timestamptz,
  CHECK (email IS NOT NULL OR phone_e164 IS NOT NULL OR status = 'erased')
);
CREATE TRIGGER users_updated BEFORE UPDATE ON identity.users FOR EACH ROW EXECUTE FUNCTION shared.set_updated_at();
CREATE TRIGGER users_version BEFORE UPDATE ON identity.users FOR EACH ROW EXECUTE FUNCTION shared.bump_version();

CREATE TABLE identity.user_roles (
  user_id    uuid NOT NULL REFERENCES identity.users(id) ON DELETE CASCADE,
  role       shared.user_role NOT NULL,
  granted_by uuid REFERENCES identity.users(id),
  granted_at timestamptz NOT NULL DEFAULT now(),
  revoked_at timestamptz,
  PRIMARY KEY (user_id, role)
);

CREATE TABLE identity.totp_credentials (                            -- required for admin / annotator / ml_engineer
  user_id     uuid PRIMARY KEY REFERENCES identity.users(id) ON DELETE CASCADE,
  secret_enc  bytea NOT NULL,                                       -- AES-GCM, key from secret manager
  enabled_at  timestamptz,
  last_used_step bigint                                             -- replay protection
);

CREATE TABLE identity.otp_codes (
  id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  target      text NOT NULL,                                        -- phone or email
  purpose     text NOT NULL CHECK (purpose IN ('login', 'verify_phone', 'verify_email', 'reset_password')),
  code_hash   bytea NOT NULL,                                       -- HMAC(pepper, code)
  attempts    int  NOT NULL DEFAULT 0 CHECK (attempts <= 5),
  expires_at  timestamptz NOT NULL,
  consumed_at timestamptz,
  created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX otp_active_idx ON identity.otp_codes (target, purpose) WHERE consumed_at IS NULL;

CREATE TABLE identity.refresh_tokens (
  id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id      uuid  NOT NULL REFERENCES identity.users(id) ON DELETE CASCADE,
  family_id    uuid  NOT NULL,                                      -- rotation chain
  token_hash   bytea NOT NULL UNIQUE,                               -- SHA-256 of opaque token
  device_label text,
  ip_inet      inet,
  issued_at    timestamptz NOT NULL DEFAULT now(),
  expires_at   timestamptz NOT NULL,
  used_at      timestamptz,
  revoked_at   timestamptz,
  revoke_reason text
);
CREATE INDEX refresh_user_idx   ON identity.refresh_tokens (user_id) WHERE revoked_at IS NULL;
CREATE INDEX refresh_family_idx ON identity.refresh_tokens (family_id);

-- Rotation with reuse detection. Returns: 'rotated' | 'reuse_detected' | 'invalid'
CREATE FUNCTION identity.rotate_refresh_token(p_old_hash bytea, p_new_hash bytea, p_ttl interval DEFAULT '30 days',
                                              OUT result text, OUT user_id uuid)
LANGUAGE plpgsql AS $$
DECLARE t identity.refresh_tokens;
BEGIN
  SELECT * INTO t FROM identity.refresh_tokens WHERE token_hash = p_old_hash FOR UPDATE;
  IF NOT FOUND OR t.revoked_at IS NOT NULL OR t.expires_at < now() THEN
    result := 'invalid'; RETURN;
  END IF;
  user_id := t.user_id;
  IF t.used_at IS NOT NULL THEN                                     -- stolen-token signal
    UPDATE identity.refresh_tokens SET revoked_at = now(), revoke_reason = 'reuse_detected'
     WHERE family_id = t.family_id AND revoked_at IS NULL;
    INSERT INTO audit.events (actor_user_id, action, target, meta)
    VALUES (t.user_id, 'auth.refresh_reuse_detected', t.family_id::text, '{}'::jsonb);
    result := 'reuse_detected'; RETURN;
  END IF;
  UPDATE identity.refresh_tokens SET used_at = now() WHERE id = t.id;
  INSERT INTO identity.refresh_tokens (user_id, family_id, token_hash, device_label, expires_at)
  VALUES (t.user_id, t.family_id, p_new_hash, t.device_label, now() + p_ttl);
  result := 'rotated';
END $$;

-- Credential lookup for login/OTP. SECURITY DEFINER because RLS hides users from anonymous callers.
CREATE FUNCTION identity.find_login(p_email citext DEFAULT NULL, p_phone text DEFAULT NULL)
RETURNS TABLE (user_id uuid, password_hash text, status text, roles text[])
LANGUAGE sql STABLE SECURITY DEFINER SET search_path = identity, shared, pg_temp AS $$
  SELECT u.id, u.password_hash, u.status,
         COALESCE(array_agg(r.role::text) FILTER (WHERE r.role IS NOT NULL AND r.revoked_at IS NULL), '{}')
  FROM identity.users u LEFT JOIN identity.user_roles r ON r.user_id = u.id
  WHERE (p_email IS NOT NULL AND u.email = p_email) OR (p_phone IS NOT NULL AND u.phone_e164 = p_phone)
  GROUP BY u.id
$$;
REVOKE EXECUTE ON FUNCTION identity.find_login(citext, text) FROM PUBLIC;

CREATE TABLE identity.guests (                                      -- hearing party who joined by QR, no account
  id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  display_name text NOT NULL DEFAULT 'Guest' CHECK (length(display_name) <= 80),
  created_at   timestamptz NOT NULL DEFAULT now(),
  expires_at   timestamptz NOT NULL DEFAULT now() + interval '2 hours'
);

-- -----------------------------------------------------------------------------------------------------
-- 4. venue
-- -----------------------------------------------------------------------------------------------------
CREATE TABLE venue.venues (
  id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name         text NOT NULL,
  venue_type   shared.venue_type NOT NULL,
  owner_id     uuid NOT NULL REFERENCES identity.users(id),
  county_code  text CHECK (county_code ~ '^[0-9]{3}$'),             -- Kenya county code 001–047
  address      text,
  timezone     text NOT NULL DEFAULT 'Africa/Nairobi',
  settings     jsonb NOT NULL DEFAULT '{}',                         -- {"upsell_drink":true,"clarify_cost":0.15}
  version      int  NOT NULL DEFAULT 1,
  created_at   timestamptz NOT NULL DEFAULT now(),
  updated_at   timestamptz NOT NULL DEFAULT now(),
  archived_at  timestamptz
);
CREATE TRIGGER venues_updated BEFORE UPDATE ON venue.venues FOR EACH ROW EXECUTE FUNCTION shared.set_updated_at();
CREATE TRIGGER venues_version BEFORE UPDATE ON venue.venues FOR EACH ROW EXECUTE FUNCTION shared.bump_version();

CREATE TABLE venue.staff (
  venue_id   uuid NOT NULL REFERENCES venue.venues(id) ON DELETE CASCADE,
  user_id    uuid NOT NULL REFERENCES identity.users(id) ON DELETE CASCADE,
  role       text NOT NULL CHECK (role IN ('manager', 'staff')),
  added_at   timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (venue_id, user_id)
);

CREATE TABLE venue.devices (                                        -- robots, kiosks, POS tablets
  id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  venue_id       uuid NOT NULL REFERENCES venue.venues(id) ON DELETE CASCADE,
  name           text NOT NULL,
  kind           text NOT NULL CHECK (kind IN ('robot', 'kiosk', 'tablet')),
  api_key_prefix text NOT NULL UNIQUE CHECK (length(api_key_prefix) = 8),
  api_key_hash   bytea NOT NULL,                                    -- HMAC-SHA256(pepper, key)
  capabilities   jsonb NOT NULL DEFAULT '{}',                       -- {"screen":true,"speaker":true,"camera":true,"motion":false}
  app_version    text,
  last_seen_at   timestamptz,
  created_at     timestamptz NOT NULL DEFAULT now(),
  revoked_at     timestamptz
);
CREATE INDEX devices_venue_idx ON venue.devices (venue_id) WHERE revoked_at IS NULL;

-- -----------------------------------------------------------------------------------------------------
-- 5. privacy (before tables that reference consents)
-- -----------------------------------------------------------------------------------------------------
CREATE TABLE privacy.consent_texts (
  scope          shared.consent_scope NOT NULL,
  version        text NOT NULL,                                     -- '2026-11-01'
  language       shared.lang NOT NULL,                              -- KSL version = consent video
  body           text,
  video_uri      text,                                              -- signed KSL explanation
  effective_from timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (scope, version, language),
  CHECK (body IS NOT NULL OR video_uri IS NOT NULL)
);

CREATE TABLE privacy.consents (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id       uuid REFERENCES identity.users(id) ON DELETE CASCADE,
  signer_code   text,                                               -- contributors without an app account
  scope         shared.consent_scope NOT NULL,
  text_version  text NOT NULL,
  text_language shared.lang NOT NULL,
  granted_at    timestamptz NOT NULL DEFAULT now(),
  withdrawn_at  timestamptz,
  evidence_uri  text,                                               -- signed consent video / form scan
  witness_id    uuid REFERENCES identity.users(id),
  FOREIGN KEY (scope, text_version, text_language) REFERENCES privacy.consent_texts (scope, version, language),
  CHECK (user_id IS NOT NULL OR signer_code IS NOT NULL)
);
CREATE UNIQUE INDEX consents_active_user_uq ON privacy.consents (user_id, scope) WHERE withdrawn_at IS NULL AND user_id IS NOT NULL;
CREATE INDEX consents_signer_idx ON privacy.consents (signer_code) WHERE signer_code IS NOT NULL;

CREATE FUNCTION privacy.has_consent(p_user uuid, p_scope shared.consent_scope) RETURNS boolean
LANGUAGE sql STABLE AS $$
  SELECT EXISTS (SELECT 1 FROM privacy.consents
                  WHERE user_id = p_user AND scope = p_scope AND withdrawn_at IS NULL)
$$;

CREATE TABLE privacy.retention_policies (
  data_class   text PRIMARY KEY,                                    -- 'realtime_landmarks', 'transcripts', ...
  retain_days  int  NOT NULL CHECK (retain_days >= 0),              -- 0 = never stored
  legal_basis  text NOT NULL,
  description  text NOT NULL
);

CREATE TABLE privacy.erasure_requests (
  id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id      uuid NOT NULL REFERENCES identity.users(id),
  requested_at timestamptz NOT NULL DEFAULT now(),
  status       shared.job_status NOT NULL DEFAULT 'queued',
  completed_at timestamptz,
  report       jsonb                                                -- what was deleted / anonymised / excluded
);

-- -----------------------------------------------------------------------------------------------------
-- 6. conversation
-- -----------------------------------------------------------------------------------------------------
CREATE FUNCTION conversation.generate_join_code() RETURNS text
LANGUAGE plpgsql VOLATILE AS $$
DECLARE
  alphabet constant text := 'ABCDEFGHJKMNPQRSTUVWXYZ23456789';   -- no 0/O, 1/I/L
  code text := '';
BEGIN
  FOR i IN 1..6 LOOP
    code := code || substr(alphabet, 1 + (get_byte(gen_random_bytes(1), 0) % length(alphabet)), 1);
    IF i = 3 THEN code := code || '-'; END IF;
  END LOOP;
  RETURN code;
END $$;

CREATE TABLE conversation.rooms (
  id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  venue_id         uuid REFERENCES venue.venues(id),
  created_by_user  uuid REFERENCES identity.users(id),
  created_by_device uuid REFERENCES venue.devices(id),
  join_code        text NOT NULL UNIQUE DEFAULT conversation.generate_join_code(),
  context_profile  shared.venue_type NOT NULL DEFAULT 'general',     -- selects LM interpolation & biasing
  status           text NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'closed')),
  store_transcript boolean NOT NULL DEFAULT false,
  created_at       timestamptz NOT NULL DEFAULT now(),
  expires_at       timestamptz NOT NULL DEFAULT now() + interval '12 hours',
  closed_at        timestamptz,
  CHECK (num_nonnulls(created_by_user, created_by_device) = 1)
);
CREATE INDEX rooms_open_idx ON conversation.rooms (venue_id) WHERE status = 'open';

CREATE TABLE conversation.room_members (
  id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  room_id     uuid NOT NULL REFERENCES conversation.rooms(id) ON DELETE CASCADE,
  user_id     uuid REFERENCES identity.users(id),
  guest_id    uuid REFERENCES identity.guests(id),
  device_id   uuid REFERENCES venue.devices(id),
  role        text NOT NULL CHECK (role IN ('signer', 'speaker', 'device', 'observer')),
  modality_in  shared.modality NOT NULL,
  modality_out shared.modality NOT NULL,
  joined_at   timestamptz NOT NULL DEFAULT now(),
  left_at     timestamptz,
  CHECK (num_nonnulls(user_id, guest_id, device_id) = 1)
);
CREATE UNIQUE INDEX room_members_user_uq   ON conversation.room_members (room_id, user_id)   WHERE user_id   IS NOT NULL;
CREATE UNIQUE INDEX room_members_guest_uq  ON conversation.room_members (room_id, guest_id)  WHERE guest_id  IS NOT NULL;
CREATE UNIQUE INDEX room_members_device_uq ON conversation.room_members (room_id, device_id) WHERE device_id IS NOT NULL;

CREATE TABLE conversation.ws_sessions (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  member_id       uuid REFERENCES conversation.room_members(id) ON DELETE SET NULL,
  user_id         uuid,
  gateway_node    text NOT NULL,
  client_app      text NOT NULL,                                     -- 'ksl-android', 'ksl-kiosk'
  client_version  text NOT NULL,
  protocol_version int NOT NULL DEFAULT 1,
  landmark_spec_version int NOT NULL DEFAULT 1,
  connected_at    timestamptz NOT NULL DEFAULT now(),
  disconnected_at timestamptz,
  close_code      int,
  last_seq_in     bigint NOT NULL DEFAULT 0,
  last_seq_out    bigint NOT NULL DEFAULT 0,
  rtt_ms_p50      int
);
CREATE INDEX ws_sessions_open_idx ON conversation.ws_sessions (gateway_node) WHERE disconnected_at IS NULL;

-- Utterances: only written when the room stores transcripts (consent) or for clarification/feedback linkage.
CREATE TABLE conversation.utterances (
  id                  uuid        NOT NULL DEFAULT gen_random_uuid(),
  created_at          timestamptz NOT NULL DEFAULT now(),
  room_id             uuid        NOT NULL REFERENCES conversation.rooms(id) ON DELETE CASCADE,
  member_id           uuid        REFERENCES conversation.room_members(id) ON DELETE SET NULL,
  source              shared.modality NOT NULL,
  language            shared.lang NOT NULL,
  glosses             text[],
  text_en             text,
  text_sw             text,
  confidence          real CHECK (confidence BETWEEN 0 AND 1),
  intent              jsonb,                                         -- {"name":"order_food","confidence":0.91,"slots":{...}}
  non_manual          jsonb,                                         -- {"question":0.8,"negation":0.0}
  needs_clarification boolean NOT NULL DEFAULT false,
  revision            int NOT NULL DEFAULT 0,
  model_versions      jsonb NOT NULL DEFAULT '{}',                   -- {"cslr":"2026.11.0","g2t":"2026.11.0"}
  e2e_latency_ms      int,
  PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);
CREATE INDEX utterances_room_time_idx ON conversation.utterances (room_id, created_at);
CREATE TABLE conversation.utterances_default PARTITION OF conversation.utterances DEFAULT;

-- -----------------------------------------------------------------------------------------------------
-- 7. dataset
-- -----------------------------------------------------------------------------------------------------
CREATE TABLE dataset.sources (
  id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  slug                text NOT NULL UNIQUE,                          -- 'ekitabu-ksl-video'
  name                text NOT NULL,
  modality            shared.modality NOT NULL,
  language            shared.lang NOT NULL,
  url                 text,
  licence             text NOT NULL,                                 -- SPDX id or 'restricted' / 'unknown'
  allows_commercial   boolean,                                       -- NULL = unknown → treat as research only
  allows_derivatives  boolean,
  usage               text NOT NULL CHECK (usage IN ('train', 'pretrain_only', 'eval_only', 'blocked')),
  access_notes        text,
  citation            text,
  added_at            timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE dataset.signers (                                       -- pseudonymous; demographics only with consent
  signer_code     text PRIMARY KEY CHECK (signer_code ~ '^S[0-9]{4,6}$'),
  user_id         uuid UNIQUE REFERENCES identity.users(id) ON DELETE SET NULL,
  source_id       uuid NOT NULL REFERENCES dataset.sources(id),
  gender          text CHECK (gender IN ('female', 'male', 'non_binary', 'undisclosed')),
  age_band        text CHECK (age_band IN ('18-24', '25-34', '35-44', '45-54', '55+', 'undisclosed')),
  region          text,                                              -- county / region of signing background
  dominant_hand   shared.hand,
  hearing_status  shared.hearing_status NOT NULL DEFAULT 'undisclosed',
  ksl_fluency     text CHECK (ksl_fluency IN ('native', 'fluent', 'intermediate', 'learner')),
  school_for_deaf text,                                              -- regional-variant signal
  skin_tone_band  smallint CHECK (skin_tone_band BETWEEN 1 AND 10),  -- Monk scale, ONLY with 'demographics' consent
  created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE dataset.recording_sessions (
  id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  recorded_at  timestamptz NOT NULL,
  location     text,
  device_model text,
  device_tier  text CHECK (device_tier IN ('low', 'mid', 'high')),
  lighting     text CHECK (lighting IN ('bright', 'dim', 'outdoor_shade', 'mixed')),
  background   text CHECK (background IN ('plain', 'cluttered', 'moving')),
  posture      text CHECK (posture IN ('seated', 'standing')),
  distance_m   real CHECK (distance_m BETWEEN 0.3 AND 5),
  operator_id  uuid REFERENCES identity.users(id),
  notes        text
);

CREATE TABLE dataset.samples (
  id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  source_id         uuid NOT NULL REFERENCES dataset.sources(id),
  external_id       text,                                            -- id inside the source dataset
  signer_code       text REFERENCES dataset.signers(signer_code),
  speaker_code      text,                                            -- speech samples
  session_id        uuid REFERENCES dataset.recording_sessions(id),
  consent_id        uuid REFERENCES privacy.consents(id),            -- required for own recordings
  kind              shared.sample_kind NOT NULL,
  domain            shared.venue_type,
  video_uri         text,
  landmarks_uri     text,
  audio_uri         text,
  landmark_spec_version int,
  landmarks_sha256  bytea,
  fps               real CHECK (fps > 0 AND fps <= 120),
  n_frames          int  CHECK (n_frames > 0),
  duration_ms       int  CHECK (duration_ms > 0),
  glosses           text[],                                          -- current gold (after adjudication)
  text_en           text,
  text_sw           text,
  is_synthetic      boolean NOT NULL DEFAULT false,
  quality_flags     text[] NOT NULL DEFAULT '{}',                    -- 'hand_out_of_frame','blurry','partial'
  review_status     shared.review_status NOT NULL DEFAULT 'pending',
  excluded_at       timestamptz,                                     -- consent withdrawn / erasure → never in new builds
  excluded_reason   text,
  created_at        timestamptz NOT NULL DEFAULT now(),
  updated_at        timestamptz NOT NULL DEFAULT now(),
  UNIQUE (source_id, external_id),
  CHECK (video_uri IS NOT NULL OR landmarks_uri IS NOT NULL OR audio_uri IS NOT NULL),
  CHECK (kind = 'speech' OR signer_code IS NOT NULL OR is_synthetic)
);
CREATE TRIGGER samples_updated BEFORE UPDATE ON dataset.samples FOR EACH ROW EXECUTE FUNCTION shared.set_updated_at();
CREATE INDEX samples_signer_idx   ON dataset.samples (signer_code);
CREATE INDEX samples_review_idx   ON dataset.samples (review_status) WHERE excluded_at IS NULL;
CREATE INDEX samples_glosses_gin  ON dataset.samples USING gin (glosses);

-- own recordings must carry a valid contribution consent
CREATE FUNCTION dataset.require_contribution_consent() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE src_slug text; ok boolean;
BEGIN
  SELECT slug INTO src_slug FROM dataset.sources WHERE id = NEW.source_id;
  IF src_slug = 'ksl-bridge-own' THEN
    SELECT EXISTS (SELECT 1 FROM privacy.consents c
                    WHERE c.id = NEW.consent_id AND c.withdrawn_at IS NULL
                      AND c.scope IN ('contribute_landmarks', 'contribute_video', 'contribute_audio'))
      INTO ok;
    IF NOT ok THEN
      RAISE EXCEPTION 'own-recorded sample % requires an active contribution consent', NEW.id USING ERRCODE = '23514';
    END IF;
  END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER samples_consent BEFORE INSERT OR UPDATE OF consent_id ON dataset.samples
  FOR EACH ROW EXECUTE FUNCTION dataset.require_contribution_consent();

CREATE TABLE dataset.builds (                                         -- immutable dataset versions
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name            text NOT NULL,                                     -- 'ksl-cslr'
  version         text NOT NULL,                                     -- 'v7'
  purpose         shared.model_task NOT NULL,
  split_seed      text NOT NULL,
  code_commit     text NOT NULL,
  params          jsonb NOT NULL DEFAULT '{}',                       -- filters, augment flags, min quality
  manifest_uri    text,
  manifest_sha256 bytea,
  frozen_at       timestamptz,                                       -- once set, membership is read-only
  created_by      uuid REFERENCES identity.users(id),
  created_at      timestamptz NOT NULL DEFAULT now(),
  UNIQUE (name, version)
);

CREATE TABLE dataset.build_members (
  build_id    uuid NOT NULL REFERENCES dataset.builds(id) ON DELETE CASCADE,
  sample_id   uuid NOT NULL REFERENCES dataset.samples(id),
  split       shared.data_split NOT NULL,
  signer_code text,                                                  -- denormalised for the split invariant
  PRIMARY KEY (build_id, sample_id)
);
CREATE INDEX build_members_signer_idx ON dataset.build_members (build_id, signer_code, split);

-- INVARIANTS: (1) signer-independent splits; (2) frozen builds are immutable; (3) excluded samples never enter
CREATE FUNCTION dataset.enforce_build_member() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE frozen timestamptz; s dataset.samples;
BEGIN
  SELECT frozen_at INTO frozen FROM dataset.builds WHERE id = COALESCE(NEW.build_id, OLD.build_id);
  IF frozen IS NOT NULL THEN
    RAISE EXCEPTION 'dataset build is frozen' USING ERRCODE = '42501';
  END IF;
  IF TG_OP = 'DELETE' THEN RETURN OLD; END IF;
  SELECT * INTO s FROM dataset.samples WHERE id = NEW.sample_id;
  IF s.excluded_at IS NOT NULL THEN
    RAISE EXCEPTION 'sample % is excluded (%)', s.id, s.excluded_reason USING ERRCODE = '23514';
  END IF;
  NEW.signer_code := s.signer_code;
  IF NEW.signer_code IS NOT NULL AND EXISTS (
       SELECT 1 FROM dataset.build_members m
        WHERE m.build_id = NEW.build_id AND m.signer_code = NEW.signer_code AND m.split <> NEW.split) THEN
    RAISE EXCEPTION 'signer % already in another split of this build (signer-independent splits required)',
      NEW.signer_code USING ERRCODE = '23514';
  END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER build_members_guard BEFORE INSERT OR UPDATE OR DELETE ON dataset.build_members
  FOR EACH ROW EXECUTE FUNCTION dataset.enforce_build_member();

-- -----------------------------------------------------------------------------------------------------
-- 8. lexicon (KSL dictionary): SSOT for glosses used by recognition, translation and production
-- -----------------------------------------------------------------------------------------------------
CREATE TABLE lexicon.glosses (
  gloss          text PRIMARY KEY CHECK (gloss ~ '^[A-Z0-9][A-Z0-9:_.\-]*$'),   -- WATER, IX-1, NUM:2, EAT.v2
  base_gloss     text REFERENCES lexicon.glosses(gloss),             -- variants point to their base
  categories     text[] NOT NULL DEFAULT '{}',                       -- FOOD, DRINK, NUMBER, HEALTH, PLACE, TIME, YES_NO
  regions        text[] NOT NULL DEFAULT '{}',                       -- where the variant is used
  hamnosys       text,                                               -- articulatory transcription
  handshape      text,
  location       text,
  movement       text,
  is_directional boolean NOT NULL DEFAULT false,                     -- true ⇒ never mirror in augmentation
  is_critical    boolean NOT NULL DEFAULT false,                     -- allergy, pain, emergency → forced captions
  status         shared.review_status NOT NULL DEFAULT 'pending',
  approved_by    uuid REFERENCES identity.users(id),                 -- must hold role deaf_reviewer
  notes          text,
  created_at     timestamptz NOT NULL DEFAULT now(),
  updated_at     timestamptz NOT NULL DEFAULT now()
);
CREATE TRIGGER glosses_updated BEFORE UPDATE ON lexicon.glosses FOR EACH ROW EXECUTE FUNCTION shared.set_updated_at();
CREATE INDEX glosses_categories_gin ON lexicon.glosses USING gin (categories);

CREATE TABLE lexicon.translations (
  gloss      text NOT NULL REFERENCES lexicon.glosses(gloss) ON DELETE CASCADE,
  language   shared.lang NOT NULL CHECK (language IN ('en', 'sw')),
  phrase     text NOT NULL,
  is_primary boolean NOT NULL DEFAULT false,
  PRIMARY KEY (gloss, language, phrase)
);
CREATE INDEX translations_phrase_trgm ON lexicon.translations USING gin (phrase gin_trgm_ops);
CREATE UNIQUE INDEX translations_primary_uq ON lexicon.translations (gloss, language) WHERE is_primary;

CREATE TABLE lexicon.sign_clips (                                     -- canonical clips for sign production
  id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  gloss       text NOT NULL REFERENCES lexicon.glosses(gloss) ON DELETE CASCADE,
  sample_id   uuid NOT NULL REFERENCES dataset.samples(id),
  start_ms    int NOT NULL DEFAULT 0,
  end_ms      int,
  pose_uri    text NOT NULL,                                         -- trimmed, normalised (T,107,3) float16
  dtw_cost    real,                                                  -- total DTW distance to other clips (medoid = min)
  is_medoid   boolean NOT NULL DEFAULT false,
  approved_by uuid REFERENCES identity.users(id),
  created_at  timestamptz NOT NULL DEFAULT now(),
  CHECK (end_ms IS NULL OR end_ms > start_ms)
);
CREATE UNIQUE INDEX sign_clips_medoid_uq ON lexicon.sign_clips (gloss) WHERE is_medoid;

CREATE TABLE lexicon.fingerspell_letters (
  letter     char(1) PRIMARY KEY CHECK (letter ~ '^[A-Z0-9]$'),
  pose_uri   text NOT NULL,
  approved_by uuid REFERENCES identity.users(id)
);

CREATE TABLE lexicon.nonmanual_templates (
  name       text PRIMARY KEY CHECK (name IN ('yes_no_question', 'wh_question', 'negation', 'intensity')),
  face_offsets_uri text NOT NULL,                                    -- (40,3) offsets over face landmarks
  approved_by uuid REFERENCES identity.users(id)
);

-- -----------------------------------------------------------------------------------------------------
-- 9. ml: registry, experiments, runs, evaluations, deployments
-- -----------------------------------------------------------------------------------------------------
CREATE TABLE ml.vocabularies (                                        -- token-id mapping = model output contract
  id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name       text NOT NULL,                                          -- 'ksl-gloss', 'en-sw-bpe8k', 'fs-letters'
  version    text NOT NULL,
  kind       text NOT NULL CHECK (kind IN ('gloss', 'bpe', 'letters', 'phonemes')),
  size       int  NOT NULL CHECK (size > 0),
  sha256     bytea NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (name, version)
);

CREATE TABLE ml.vocabulary_tokens (
  vocabulary_id uuid NOT NULL REFERENCES ml.vocabularies(id) ON DELETE CASCADE,
  idx           int  NOT NULL CHECK (idx >= 0),
  token         text NOT NULL,
  gloss         text REFERENCES lexicon.glosses(gloss),             -- set for gloss vocabularies
  PRIMARY KEY (vocabulary_id, idx),
  UNIQUE (vocabulary_id, token)
);

CREATE TABLE ml.experiments (
  id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  task        shared.model_task NOT NULL,
  name        text NOT NULL UNIQUE,
  hypothesis  text NOT NULL,                                         -- what this experiment tests (RQ link)
  research_question text CHECK (research_question IN ('RQ1', 'RQ2', 'RQ3', 'RQ4', 'RQ5')),
  created_by  uuid REFERENCES identity.users(id),
  created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE ml.training_runs (
  id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  experiment_id    uuid NOT NULL REFERENCES ml.experiments(id),
  task             shared.model_task NOT NULL,
  parent_run_id    uuid REFERENCES ml.training_runs(id),            -- fine-tuned from (e.g. MPM → CSLR)
  dataset_build_id uuid REFERENCES dataset.builds(id),
  config           jsonb NOT NULL,
  config_sha256    bytea NOT NULL,
  code_commit      text  NOT NULL,
  image_digest     text,                                             -- docker image sha256
  seed             bigint NOT NULL,
  status           shared.job_status NOT NULL DEFAULT 'queued',
  hardware         jsonb,                                            -- {"gpu":"RTX 4090","count":1}
  gpu_hours        numeric(10,3),
  energy_kwh       numeric(12,4),
  emissions_kgco2e numeric(12,4),                                    -- CodeCarbon
  mlflow_run_id    text UNIQUE,
  artifact_uri     text,
  started_at       timestamptz,
  finished_at      timestamptz,
  error            text,
  created_by       uuid REFERENCES identity.users(id),
  created_at       timestamptz NOT NULL DEFAULT now(),
  CHECK (finished_at IS NULL OR started_at IS NOT NULL AND finished_at >= started_at)
);
CREATE INDEX training_runs_exp_idx ON ml.training_runs (experiment_id, created_at DESC);

CREATE TABLE ml.run_metrics (
  run_id      uuid NOT NULL REFERENCES ml.training_runs(id) ON DELETE CASCADE,
  key         text NOT NULL,                                         -- 'train_loss', 'val_wer'
  step        bigint NOT NULL,
  value       double precision NOT NULL,
  recorded_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (run_id, key, step)
);

CREATE TABLE ml.model_versions (
  id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  task           shared.model_task NOT NULL,
  name           text NOT NULL,                                      -- 'cslr_conformer_ctc'
  version        text NOT NULL CHECK (version ~ '^[0-9]{4}\.[0-9]{1,2}\.[0-9]+$|^local-[0-9]+$'),
  run_id         uuid REFERENCES ml.training_runs(id),               -- NULL only for bootstrap models
  is_bootstrap   boolean NOT NULL DEFAULT false,                     -- third-party open weights run locally
  bootstrap_source text,                                             -- e.g. upstream model id + licence
  vocabulary_id  uuid REFERENCES ml.vocabularies(id),
  format         text NOT NULL CHECK (format IN ('pytorch', 'onnx', 'tensorrt', 'tflite', 'executorch', 'pickle', 'arpa')),
  artifact_uri   text NOT NULL,
  sha256         bytea NOT NULL,
  size_bytes     bigint CHECK (size_bytes > 0),
  params_count   bigint,
  quantization   text CHECK (quantization IN ('fp32', 'bf16', 'fp16', 'int8', 'int4')),
  licence        text NOT NULL DEFAULT 'proprietary-ksl-bridge',
  model_card_uri text,
  created_at     timestamptz NOT NULL DEFAULT now(),
  UNIQUE (task, name, version, format),
  CHECK (is_bootstrap OR run_id IS NOT NULL),
  CHECK (NOT is_bootstrap OR bootstrap_source IS NOT NULL)
);

CREATE TABLE lexicon.gloss_prototypes (                               -- ISLR embedding prototypes (open-set rejection)
  gloss            text NOT NULL REFERENCES lexicon.glosses(gloss) ON DELETE CASCADE,
  model_version_id uuid NOT NULL REFERENCES ml.model_versions(id) ON DELETE CASCADE,
  embedding        vector(256) NOT NULL,
  n_clips          int NOT NULL CHECK (n_clips > 0),
  reject_threshold real,
  PRIMARY KEY (gloss, model_version_id)
);
CREATE INDEX gloss_prototypes_hnsw ON lexicon.gloss_prototypes USING hnsw (embedding vector_cosine_ops);

CREATE TABLE ml.calibrations (
  model_version_id uuid NOT NULL REFERENCES ml.model_versions(id) ON DELETE CASCADE,
  method           text NOT NULL CHECK (method IN ('temperature', 'utterance_logreg', 'isotonic')),
  params           jsonb NOT NULL,                                   -- {"T":1.42} or LR coefficients
  ece_before       real,
  ece_after        real,
  fitted_on_build  uuid REFERENCES dataset.builds(id),
  created_at       timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (model_version_id, method)
);

CREATE TABLE ml.decoder_configs (                                     -- beam/LM fusion settings are versioned too
  id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  task             shared.model_task NOT NULL CHECK (task IN ('cslr', 'asr', 'fingerspell', 'gloss2text', 'text2gloss')),
  lm_version_id    uuid REFERENCES ml.model_versions(id),
  beam             int  NOT NULL CHECK (beam BETWEEN 1 AND 128),
  alpha            real NOT NULL DEFAULT 0.6,
  beta             real NOT NULL DEFAULT 1.0,
  prune_logp       real NOT NULL DEFAULT -8.0,
  bias_max_nats    real NOT NULL DEFAULT 2.0 CHECK (bias_max_nats <= 2.0),   -- guide: biasing stays mild
  extra            jsonb NOT NULL DEFAULT '{}',
  created_at       timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE ml.evaluations (
  id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  model_version_id  uuid NOT NULL REFERENCES ml.model_versions(id) ON DELETE CASCADE,
  decoder_config_id uuid REFERENCES ml.decoder_configs(id),
  dataset_build_id  uuid NOT NULL REFERENCES dataset.builds(id),
  split             shared.data_split NOT NULL,
  suite             text NOT NULL CHECK (suite IN ('clean', 'robustness', 'fairness', 'safety', 'latency',
                                                   'calibration', 'human', 'ablation')),
  perturbation      text,                                            -- 'rot15', 'drop30' for robustness rows
  hardware          text,
  code_commit       text NOT NULL,
  created_at        timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX evaluations_model_idx ON ml.evaluations (model_version_id, suite);

CREATE TABLE ml.evaluation_results (
  evaluation_id uuid NOT NULL REFERENCES ml.evaluations(id) ON DELETE CASCADE,
  metric        text NOT NULL,                                       -- 'wer', 'bleu', 'chrf', 'ece', 'top1', 'p95_ms', 'mos'
  slice_key     text NOT NULL DEFAULT 'all',                         -- 'gender', 'region', 'dominant_hand', 'lighting'
  slice_value   text NOT NULL DEFAULT 'all',
  value         double precision NOT NULL,
  ci_low        double precision,
  ci_high       double precision,
  n             int NOT NULL CHECK (n > 0),
  PRIMARY KEY (evaluation_id, metric, slice_key, slice_value)
);

CREATE TABLE ml.release_gates (
  model_version_id uuid NOT NULL REFERENCES ml.model_versions(id) ON DELETE CASCADE,
  gate             text NOT NULL CHECK (gate IN ('quality', 'no_regression', 'fairness', 'calibration',
                                                 'latency', 'safety_set', 'licence', 'human_review')),
  passed           boolean NOT NULL,
  details          jsonb NOT NULL DEFAULT '{}',
  evaluation_id    uuid REFERENCES ml.evaluations(id),
  checked_by       uuid REFERENCES identity.users(id),
  checked_at       timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (model_version_id, gate)
);

CREATE TABLE ml.deployments (
  id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  model_version_id uuid NOT NULL REFERENCES ml.model_versions(id),
  task             shared.model_task NOT NULL,
  target           shared.deploy_target NOT NULL DEFAULT 'server',
  stage            shared.deploy_stage NOT NULL,
  traffic_fraction real NOT NULL DEFAULT 0 CHECK (traffic_fraction BETWEEN 0 AND 1),
  decoder_config_id uuid REFERENCES ml.decoder_configs(id),
  min_app_version  text,                                             -- mobile manifests
  started_at       timestamptz NOT NULL DEFAULT now(),
  ended_at         timestamptz,
  promoted_by      uuid REFERENCES identity.users(id),
  notes            text,
  CHECK (ended_at IS NULL OR ended_at > started_at),
  CHECK (stage <> 'production' OR traffic_fraction = 1),
  -- at most ONE production model per (task, target) at any instant
  EXCLUDE USING gist (task WITH =, target WITH =, tstzrange(started_at, COALESCE(ended_at, 'infinity')) WITH &&)
    WHERE (stage = 'production')
);
CREATE INDEX deployments_live_idx ON ml.deployments (task, target) WHERE ended_at IS NULL;

-- Promotion: all mandatory gates must have passed; previous production deployment is closed atomically.
CREATE FUNCTION ml.promote_model(p_model uuid, p_target shared.deploy_target, p_actor uuid,
                                 p_decoder uuid DEFAULT NULL, p_min_app text DEFAULT NULL) RETURNS uuid
LANGUAGE plpgsql AS $$
DECLARE
  v_task shared.model_task;
  v_missing text[];
  v_id uuid;
  v_now timestamptz := clock_timestamp();
BEGIN
  SELECT task INTO v_task FROM ml.model_versions WHERE id = p_model;
  IF v_task IS NULL THEN RAISE EXCEPTION 'unknown model version %', p_model; END IF;

  SELECT array_agg(g) INTO v_missing
  FROM unnest(ARRAY['quality', 'no_regression', 'fairness', 'latency', 'safety_set', 'licence']) AS g
  WHERE NOT EXISTS (SELECT 1 FROM ml.release_gates r WHERE r.model_version_id = p_model AND r.gate = g AND r.passed);
  IF v_missing IS NOT NULL THEN
    RAISE EXCEPTION 'release gates not passed: %', v_missing USING ERRCODE = '23514';
  END IF;

  UPDATE ml.deployments SET ended_at = v_now
   WHERE task = v_task AND target = p_target AND stage = 'production' AND ended_at IS NULL;

  INSERT INTO ml.deployments (model_version_id, task, target, stage, traffic_fraction, decoder_config_id,
                              min_app_version, started_at, promoted_by)
  VALUES (p_model, v_task, p_target, 'production', 1, p_decoder, p_min_app, v_now, p_actor)
  RETURNING id INTO v_id;

  INSERT INTO ops.outbox (aggregate, aggregate_id, event_type, payload)
  VALUES ('model_version', p_model::text, 'ModelPromoted',
          jsonb_build_object('task', v_task, 'target', p_target, 'deployment_id', v_id));
  INSERT INTO audit.events (actor_user_id, action, target, meta)
  VALUES (p_actor, 'ml.promote', p_model::text, jsonb_build_object('target', p_target));
  RETURN v_id;
END $$;

-- -----------------------------------------------------------------------------------------------------
-- 10. inference
-- -----------------------------------------------------------------------------------------------------
CREATE TABLE inference.requests (
  id                uuid        NOT NULL DEFAULT gen_random_uuid(),
  created_at        timestamptz NOT NULL DEFAULT now(),
  room_id           uuid,
  utterance_id      uuid,
  utterance_created_at timestamptz,
  task              shared.model_task NOT NULL,
  model_version_id  uuid NOT NULL REFERENCES ml.model_versions(id),
  deployment_stage  shared.deploy_stage NOT NULL,                    -- shadow results logged, never served
  decoder_config_id uuid REFERENCES ml.decoder_configs(id),
  input_frames      int,
  input_ms          int,
  batch_size        int,
  queue_ms          int,
  compute_ms        int,
  status            text NOT NULL CHECK (status IN ('ok', 'timeout', 'error', 'dropped_stale')),
  error_code        text,
  confidence        real,
  nbest             jsonb,                                           -- ONLY when room stores transcripts
  energy_mj         real,
  PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);
CREATE INDEX inference_task_time_idx  ON inference.requests (task, created_at);
CREATE INDEX inference_model_time_idx ON inference.requests (model_version_id, created_at);
CREATE TABLE inference.requests_default PARTITION OF inference.requests DEFAULT;

CREATE TABLE inference.clarifications (
  id                   uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  utterance_id         uuid NOT NULL,
  utterance_created_at timestamptz NOT NULL,
  kind                 shared.clarify_kind NOT NULL,
  options              text[] NOT NULL DEFAULT '{}',
  span_index           int,
  p_best               real NOT NULL CHECK (p_best BETWEEN 0 AND 1),
  expected_loss        real NOT NULL,                                -- (1-p)·loss_if_wrong, the VOI decision input
  venue_type           shared.venue_type NOT NULL,
  answered_choice      text,
  answered_at          timestamptz,
  outcome              text CHECK (outcome IN ('resolved', 'retry', 'abandoned', 'escalated')),
  created_at           timestamptz NOT NULL DEFAULT now(),
  FOREIGN KEY (utterance_id, utterance_created_at) REFERENCES conversation.utterances (id, created_at) ON DELETE CASCADE
);

-- -----------------------------------------------------------------------------------------------------
-- 11. feedback & active learning (the learning-agent "critic" and "problem generator")
-- -----------------------------------------------------------------------------------------------------
CREATE TABLE feedback.events (
  id                   uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  utterance_id         uuid NOT NULL,
  utterance_created_at timestamptz NOT NULL,
  user_id              uuid REFERENCES identity.users(id) ON DELETE SET NULL,
  kind                 shared.feedback_kind NOT NULL,
  corrected_glosses    text[],
  corrected_text       text,
  consent_id           uuid REFERENCES privacy.consents(id),         -- needed to keep the clip for training
  clip_uri             text,
  created_at           timestamptz NOT NULL DEFAULT now(),
  FOREIGN KEY (utterance_id, utterance_created_at) REFERENCES conversation.utterances (id, created_at) ON DELETE CASCADE,
  CHECK (clip_uri IS NULL OR consent_id IS NOT NULL),
  CHECK (kind <> 'correct' OR corrected_glosses IS NOT NULL OR corrected_text IS NOT NULL)
);
CREATE INDEX feedback_user_idx ON feedback.events (user_id, created_at DESC);

CREATE TABLE feedback.active_learning_queue (
  id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  sample_id    uuid REFERENCES dataset.samples(id) ON DELETE CASCADE,
  feedback_id  uuid REFERENCES feedback.events(id) ON DELETE CASCADE,
  reason       text NOT NULL CHECK (reason IN ('low_confidence', 'user_correction', 'model_disagreement',
                                               'rare_gloss', 'open_set_reject', 'fairness_gap', 'drift')),
  score        real NOT NULL,                                        -- higher = more informative
  model_version_id uuid REFERENCES ml.model_versions(id),
  status       text NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'tasked', 'done', 'discarded')),
  created_at   timestamptz NOT NULL DEFAULT now(),
  CHECK (num_nonnulls(sample_id, feedback_id) >= 1)
);
CREATE INDEX al_queue_open_idx ON feedback.active_learning_queue (score DESC) WHERE status = 'open';

CREATE TABLE feedback.personal_adaptations (                          -- per-user prototypes & LoRA adapters
  id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id          uuid NOT NULL REFERENCES identity.users(id) ON DELETE CASCADE,
  kind             text NOT NULL CHECK (kind IN ('prototype', 'personal_sign', 'lora')),
  gloss            text REFERENCES lexicon.glosses(gloss),
  base_model_id    uuid NOT NULL REFERENCES ml.model_versions(id),
  embedding        vector(256),                                      -- prototype / personal sign
  artifact_uri     text,                                             -- LoRA weights
  n_examples       int NOT NULL CHECK (n_examples > 0),
  eval_delta_wer   real,                                             -- negative = improvement
  enabled          boolean NOT NULL DEFAULT false,
  created_at       timestamptz NOT NULL DEFAULT now(),
  CHECK (kind = 'lora' AND artifact_uri IS NOT NULL OR kind <> 'lora' AND embedding IS NOT NULL)
);
CREATE INDEX personal_adapt_user_idx ON feedback.personal_adaptations (user_id) WHERE enabled;

-- personalisation requires consent
CREATE FUNCTION feedback.require_personalisation_consent() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
  IF NOT privacy.has_consent(NEW.user_id, 'personalisation') THEN
    RAISE EXCEPTION 'user % has not consented to personalisation', NEW.user_id USING ERRCODE = '23514';
  END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER personal_adapt_consent BEFORE INSERT ON feedback.personal_adaptations
  FOR EACH ROW EXECUTE FUNCTION feedback.require_personalisation_consent();

-- -----------------------------------------------------------------------------------------------------
-- 12. annotation
-- -----------------------------------------------------------------------------------------------------
CREATE TABLE annotation.tasks (
  id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  sample_id    uuid NOT NULL REFERENCES dataset.samples(id) ON DELETE CASCADE,
  kind         text NOT NULL CHECK (kind IN ('gloss', 'translation', 'non_manual', 'time_align', 'review', 'adjudicate')),
  priority     real NOT NULL DEFAULT 0,
  source_queue_id uuid REFERENCES feedback.active_learning_queue(id),
  status       text NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'assigned', 'submitted', 'closed')),
  assigned_to  uuid REFERENCES identity.users(id),
  required_role shared.user_role NOT NULL DEFAULT 'annotator',
  assigned_at  timestamptz,
  due_at       timestamptz,
  created_at   timestamptz NOT NULL DEFAULT now(),
  version      int NOT NULL DEFAULT 1
);
CREATE TRIGGER tasks_version BEFORE UPDATE ON annotation.tasks FOR EACH ROW EXECUTE FUNCTION shared.bump_version();
CREATE INDEX annotation_tasks_open_idx ON annotation.tasks (priority DESC) WHERE status = 'open';

CREATE TABLE annotation.annotations (
  id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  task_id      uuid NOT NULL REFERENCES annotation.tasks(id) ON DELETE CASCADE,
  sample_id    uuid NOT NULL REFERENCES dataset.samples(id) ON DELETE CASCADE,
  annotator_id uuid NOT NULL REFERENCES identity.users(id),
  glosses      text[],
  gloss_times  int4range[],                                          -- per-gloss [start_ms, end_ms)
  text_en      text,
  text_sw      text,
  non_manual   jsonb,
  self_confidence smallint CHECK (self_confidence BETWEEN 1 AND 5),
  time_spent_s int CHECK (time_spent_s >= 0),
  created_at   timestamptz NOT NULL DEFAULT now(),
  CHECK (gloss_times IS NULL OR glosses IS NOT NULL AND cardinality(gloss_times) = cardinality(glosses))
);
CREATE INDEX annotations_sample_idx ON annotation.annotations (sample_id);
CREATE UNIQUE INDEX annotations_one_per_task_uq ON annotation.annotations (task_id, annotator_id);

CREATE TABLE annotation.adjudications (
  sample_id       uuid PRIMARY KEY REFERENCES dataset.samples(id) ON DELETE CASCADE,
  chosen_id       uuid REFERENCES annotation.annotations(id),
  final_glosses   text[] NOT NULL,
  final_text_en   text,
  adjudicator_id  uuid NOT NULL REFERENCES identity.users(id),       -- senior Deaf annotator
  notes           text,
  decided_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE annotation.agreement_reports (
  id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  scope       text NOT NULL,                                         -- 'batch-2026-11-03', 'non_manual'
  metric      text NOT NULL CHECK (metric IN ('cohen_kappa', 'gloss_edit_agreement', 'krippendorff_alpha')),
  value       real NOT NULL,
  n_items     int  NOT NULL,
  computed_at timestamptz NOT NULL DEFAULT now()
);

-- -----------------------------------------------------------------------------------------------------
-- 13. agent: goal-based service agents with tools
-- -----------------------------------------------------------------------------------------------------
CREATE TABLE agent.definitions (
  id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name        text NOT NULL,                                         -- 'restaurant-ordering'
  version     int  NOT NULL,
  venue_type  shared.venue_type NOT NULL,
  intents     jsonb NOT NULL,                                        -- {"order_food":["item","qty"], ...}
  slot_prompts jsonb NOT NULL,                                       -- {"qty":{"text":"How many?","expect":["NUMBER"]}}
  policy      jsonb NOT NULL DEFAULT '{}',                           -- upsell rules, min intent confidence, escalation
  enabled     boolean NOT NULL DEFAULT true,
  created_at  timestamptz NOT NULL DEFAULT now(),
  UNIQUE (name, version)
);

CREATE TABLE agent.tools (
  id                    uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name                  text NOT NULL UNIQUE CHECK (name ~ '^[a-z_]+\.[a-z_]+$'),   -- 'menu.lookup'
  description           text NOT NULL,
  input_schema          jsonb NOT NULL,                              -- JSON Schema
  output_schema         jsonb NOT NULL,
  side_effect           shared.tool_effect NOT NULL,
  requires_confirmation boolean NOT NULL DEFAULT false,              -- signer confirms before execution
  timeout_ms            int NOT NULL DEFAULT 2000 CHECK (timeout_ms BETWEEN 50 AND 30000),
  enabled               boolean NOT NULL DEFAULT true,
  CHECK (side_effect <> 'physical' OR requires_confirmation)        -- motion always needs confirmation
);

CREATE TABLE agent.venue_tools (                                     -- binds a tool to a venue's adapter
  venue_id   uuid NOT NULL REFERENCES venue.venues(id) ON DELETE CASCADE,
  tool_id    uuid NOT NULL REFERENCES agent.tools(id),
  adapter    text NOT NULL,                                          -- 'demo_restaurant', 'pos_generic_rest'
  config_enc bytea,                                                  -- encrypted adapter credentials
  enabled    boolean NOT NULL DEFAULT true,
  PRIMARY KEY (venue_id, tool_id)
);

CREATE TABLE agent.sessions (
  id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  room_id        uuid NOT NULL REFERENCES conversation.rooms(id) ON DELETE CASCADE,
  definition_id  uuid NOT NULL REFERENCES agent.definitions(id),
  venue_id       uuid REFERENCES venue.venues(id),
  frame          jsonb NOT NULL DEFAULT '{}',                        -- {"intent":"order_food","slots":{...}}
  goal_status    text NOT NULL DEFAULT 'active' CHECK (goal_status IN ('active', 'completed', 'escalated', 'abandoned')),
  started_at     timestamptz NOT NULL DEFAULT now(),
  ended_at       timestamptz,
  version        int NOT NULL DEFAULT 1
);
CREATE TRIGGER agent_sessions_version BEFORE UPDATE ON agent.sessions FOR EACH ROW EXECUTE FUNCTION shared.bump_version();
CREATE UNIQUE INDEX agent_sessions_active_uq ON agent.sessions (room_id) WHERE goal_status = 'active';

CREATE TABLE agent.turns (
  id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id          uuid NOT NULL REFERENCES agent.sessions(id) ON DELETE CASCADE,
  turn_index          int  NOT NULL CHECK (turn_index >= 0),
  actor               shared.turn_actor NOT NULL,
  utterance_id        uuid,
  utterance_created_at timestamptz,
  intent              text,
  intent_confidence   real,
  slots               jsonb NOT NULL DEFAULT '{}',
  reply_text          text,
  reply_glosses       text[],
  expected_categories text[] NOT NULL DEFAULT '{}',                  -- fed back into decoder biasing
  created_at          timestamptz NOT NULL DEFAULT now(),
  UNIQUE (session_id, turn_index)
);

CREATE TABLE agent.tool_calls (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id      uuid NOT NULL REFERENCES agent.sessions(id) ON DELETE CASCADE,
  turn_id         uuid REFERENCES agent.turns(id) ON DELETE SET NULL,
  tool_id         uuid NOT NULL REFERENCES agent.tools(id),
  idempotency_key text NOT NULL UNIQUE,
  arguments       jsonb NOT NULL,
  result          jsonb,
  status          shared.call_status NOT NULL DEFAULT 'pending',
  confirmed_by_user boolean,
  error           text,
  latency_ms      int,
  created_at      timestamptz NOT NULL DEFAULT now(),
  completed_at    timestamptz
);
CREATE INDEX tool_calls_session_idx ON agent.tool_calls (session_id, created_at);

-- tools with side effects must be enabled for the session's venue, and confirmed when required
CREATE FUNCTION agent.guard_tool_call() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE t agent.tools; v uuid;
BEGIN
  SELECT * INTO t FROM agent.tools WHERE id = NEW.tool_id;
  IF NOT t.enabled THEN RAISE EXCEPTION 'tool % disabled', t.name; END IF;
  SELECT venue_id INTO v FROM agent.sessions WHERE id = NEW.session_id;
  IF t.side_effect <> 'none' AND NOT EXISTS (
       SELECT 1 FROM agent.venue_tools vt WHERE vt.venue_id = v AND vt.tool_id = t.id AND vt.enabled) THEN
    RAISE EXCEPTION 'tool % not enabled for venue %', t.name, v USING ERRCODE = '42501';
  END IF;
  IF t.requires_confirmation AND NEW.status = 'succeeded' AND COALESCE(NEW.confirmed_by_user, false) = false THEN
    RAISE EXCEPTION 'tool % requires user confirmation before execution', t.name USING ERRCODE = '23514';
  END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER tool_calls_guard BEFORE INSERT OR UPDATE ON agent.tool_calls
  FOR EACH ROW EXECUTE FUNCTION agent.guard_tool_call();

CREATE TABLE agent.escalations (                                     -- "call a staff member": human in the loop
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id      uuid NOT NULL REFERENCES agent.sessions(id) ON DELETE CASCADE,
  reason          text NOT NULL CHECK (reason IN ('low_confidence', 'unknown_intent', 'kb_no_answer',
                                                  'critical_domain', 'user_request', 'tool_failure')),
  requested_at    timestamptz NOT NULL DEFAULT now(),
  acknowledged_by uuid REFERENCES identity.users(id),
  acknowledged_at timestamptz,
  resolved_at     timestamptz
);
CREATE INDEX escalations_open_idx ON agent.escalations (requested_at) WHERE resolved_at IS NULL;

-- -----------------------------------------------------------------------------------------------------
-- 14. knowledge: RAG (hybrid BM25 + vectors), translation memory, decoder biasing vocabulary
-- -----------------------------------------------------------------------------------------------------
CREATE TABLE knowledge.documents (
  id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  venue_id       uuid REFERENCES venue.venues(id) ON DELETE CASCADE,   -- NULL = global
  kind           text NOT NULL CHECK (kind IN ('menu', 'faq', 'directions', 'policy', 'hours', 'other')),
  title          text NOT NULL,
  source_uri     text,
  content_sha256 bytea NOT NULL,
  version        int  NOT NULL DEFAULT 1,
  status         text NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'superseded', 'deleted')),
  created_by     uuid REFERENCES identity.users(id),
  created_at     timestamptz NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX documents_active_uq ON knowledge.documents (COALESCE(venue_id, '00000000-0000-0000-0000-000000000000'::uuid), title)
  WHERE status = 'active';

CREATE TABLE knowledge.chunks (
  id                 bigserial PRIMARY KEY,
  document_id        uuid NOT NULL REFERENCES knowledge.documents(id) ON DELETE CASCADE,
  venue_id           uuid,                                           -- denormalised for tenant filtering (RLS)
  section_path       text,                                           -- 'Menu > Drinks'
  ordinal            int  NOT NULL,
  content            text NOT NULL CHECK (length(content) BETWEEN 1 AND 4000),
  tsv                tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(section_path, '') || ' ' || content)) STORED,
  embedding          vector(384),
  embedder_version_id uuid REFERENCES ml.model_versions(id),
  token_count        int,
  meta               jsonb NOT NULL DEFAULT '{}',                    -- {"price":650,"item_id":17}
  UNIQUE (document_id, ordinal)
);
CREATE INDEX chunks_tsv_gin    ON knowledge.chunks USING gin (tsv);
CREATE INDEX chunks_venue_idx  ON knowledge.chunks (venue_id);
CREATE INDEX chunks_embed_hnsw ON knowledge.chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

CREATE FUNCTION knowledge.chunk_venue_from_document() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
  SELECT venue_id INTO NEW.venue_id FROM knowledge.documents WHERE id = NEW.document_id;
  RETURN NEW;
END $$;
CREATE TRIGGER chunks_venue BEFORE INSERT OR UPDATE OF document_id ON knowledge.chunks
  FOR EACH ROW EXECUTE FUNCTION knowledge.chunk_venue_from_document();

-- Hybrid search (vector + keyword), tenant-scoped. Weights per the guide (0.6 / 0.4).
CREATE FUNCTION knowledge.hybrid_search(p_venue uuid, p_query text, p_qvec vector(384), p_k int DEFAULT 5,
                                        p_w_vec real DEFAULT 0.6, p_w_kw real DEFAULT 0.4)
RETURNS TABLE (chunk_id bigint, document_id uuid, section_path text, content text, score real)
LANGUAGE sql STABLE AS $$
  WITH vec AS (
    SELECT c.id, (1 - (c.embedding <=> p_qvec))::real AS vs
    FROM knowledge.chunks c JOIN knowledge.documents d ON d.id = c.document_id AND d.status = 'active'
    WHERE (c.venue_id = p_venue OR c.venue_id IS NULL) AND c.embedding IS NOT NULL
    ORDER BY c.embedding <=> p_qvec LIMIT 20
  ), kw AS (
    SELECT c.id, ts_rank_cd(c.tsv, plainto_tsquery('english', p_query))::real AS ks
    FROM knowledge.chunks c JOIN knowledge.documents d ON d.id = c.document_id AND d.status = 'active'
    WHERE (c.venue_id = p_venue OR c.venue_id IS NULL) AND c.tsv @@ plainto_tsquery('english', p_query)
    ORDER BY ks DESC LIMIT 20
  ), merged AS (
    SELECT COALESCE(vec.id, kw.id) AS id, COALESCE(vec.vs, 0) AS vs, COALESCE(kw.ks, 0) AS ks
    FROM vec FULL OUTER JOIN kw ON vec.id = kw.id
  )
  SELECT c.id, c.document_id, c.section_path, c.content,
         (p_w_vec * m.vs + p_w_kw * m.ks / NULLIF(max(m.ks) OVER (), 0)::real)::real AS score
  FROM merged m JOIN knowledge.chunks c ON c.id = m.id
  ORDER BY score DESC NULLS LAST
  LIMIT p_k
$$;

-- array_to_string is only STABLE in core; for text[] it is deterministic, so wrap it as IMMUTABLE for the generated key
CREATE FUNCTION knowledge.gloss_key(text[]) RETURNS text
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$ SELECT array_to_string($1, ' ') $$;

CREATE TABLE knowledge.translation_memory (                           -- verified gloss ⇄ English/Swahili pairs
  id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  glosses     text[] NOT NULL CHECK (cardinality(glosses) > 0),
  gloss_key   text GENERATED ALWAYS AS (knowledge.gloss_key(glosses)) STORED,
  text_en     text NOT NULL,
  text_sw     text,
  domain      shared.venue_type NOT NULL DEFAULT 'general',
  source      text NOT NULL CHECK (source IN ('annotation', 'confirmed_feedback', 'deaf_reviewer', 'template')),
  verified_by uuid REFERENCES identity.users(id),
  uses        int  NOT NULL DEFAULT 0,
  created_at  timestamptz NOT NULL DEFAULT now(),
  UNIQUE (gloss_key, domain, text_en)
);
CREATE INDEX tm_gloss_key_trgm ON knowledge.translation_memory USING gin (gloss_key gin_trgm_ops);
CREATE INDEX tm_text_en_trgm   ON knowledge.translation_memory USING gin (text_en gin_trgm_ops);

CREATE TABLE knowledge.venue_vocabulary (                             -- RAG → recognition biasing
  venue_id   uuid NOT NULL REFERENCES venue.venues(id) ON DELETE CASCADE,
  term       text NOT NULL,                                          -- 'chicken burger', 'Pharmacy', 'Wanjiru'
  gloss      text REFERENCES lexicon.glosses(gloss),                 -- NULL ⇒ fingerspelling lexicon entry
  category   text NOT NULL,                                          -- FOOD, DRINK, PLACE, PERSON
  bias_nats  real NOT NULL DEFAULT 0.5 CHECK (bias_nats BETWEEN 0 AND 2),
  source_chunk_id bigint REFERENCES knowledge.chunks(id) ON DELETE SET NULL,
  PRIMARY KEY (venue_id, term)
);

CREATE TABLE knowledge.retrieval_logs (                               -- grounding audit (naive-RAG failure fixes)
  id              bigserial PRIMARY KEY,
  room_id         uuid,
  venue_id        uuid,
  query           text NOT NULL,
  rewritten_query text,                                              -- second attempt (rewrite & retry)
  chunk_ids       bigint[] NOT NULL DEFAULT '{}',
  scores          real[]   NOT NULL DEFAULT '{}',
  graded_relevant boolean  NOT NULL,
  answer_chunk_id bigint,                                            -- extractive answer provenance
  answered        boolean  NOT NULL,                                 -- false ⇒ "I don't know" + escalation
  latency_ms      int,
  created_at      timestamptz NOT NULL DEFAULT now(),
  CHECK (cardinality(chunk_ids) = cardinality(scores))
);

-- -----------------------------------------------------------------------------------------------------
-- 15. audit (append-only, partitioned)
-- -----------------------------------------------------------------------------------------------------
CREATE TABLE audit.events (
  id            bigserial,
  at            timestamptz NOT NULL DEFAULT now(),
  actor_user_id uuid,
  actor_device_id uuid,
  action        text NOT NULL,                                       -- 'auth.login', 'ml.promote', 'privacy.erase'
  target        text,
  ip_inet       inet,
  meta          jsonb NOT NULL DEFAULT '{}',
  PRIMARY KEY (id, at)
) PARTITION BY RANGE (at);
CREATE INDEX audit_actor_idx  ON audit.events (actor_user_id, at);
CREATE INDEX audit_action_idx ON audit.events (action, at);
CREATE TABLE audit.events_default PARTITION OF audit.events DEFAULT;

CREATE FUNCTION audit.prevent_mutation() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
  RAISE EXCEPTION 'audit.events is append-only' USING ERRCODE = '42501';
END $$;
CREATE TRIGGER audit_no_update BEFORE UPDATE OR DELETE ON audit.events
  FOR EACH ROW EXECUTE FUNCTION audit.prevent_mutation();

-- -----------------------------------------------------------------------------------------------------
-- 16. privacy.erase_user: right to erasure as anonymisation
-- -----------------------------------------------------------------------------------------------------
CREATE FUNCTION privacy.erase_user(p_user uuid, p_actor uuid) RETURNS jsonb
LANGUAGE plpgsql AS $$
DECLARE
  v_signer text;
  v_samples int := 0;
  v_report jsonb;
BEGIN
  SELECT signer_code INTO v_signer FROM dataset.signers WHERE user_id = p_user;

  UPDATE privacy.consents SET withdrawn_at = now() WHERE user_id = p_user AND withdrawn_at IS NULL;
  UPDATE identity.refresh_tokens SET revoked_at = now(), revoke_reason = 'erasure'
   WHERE user_id = p_user AND revoked_at IS NULL;
  DELETE FROM identity.totp_credentials WHERE user_id = p_user;
  DELETE FROM feedback.personal_adaptations WHERE user_id = p_user;

  IF v_signer IS NOT NULL THEN
    UPDATE dataset.samples SET excluded_at = now(), excluded_reason = 'erasure'
     WHERE signer_code = v_signer AND excluded_at IS NULL;
    GET DIAGNOSTICS v_samples = ROW_COUNT;
    UPDATE dataset.signers SET user_id = NULL, gender = NULL, age_band = NULL, region = NULL,
                               skin_tone_band = NULL, school_for_deaf = NULL
     WHERE signer_code = v_signer;
  END IF;

  UPDATE identity.users
     SET email = NULL, phone_e164 = NULL, password_hash = NULL, display_name = 'Erased user',
         status = 'erased', erased_at = now()
   WHERE id = p_user;
  UPDATE identity.user_roles SET revoked_at = now() WHERE user_id = p_user AND revoked_at IS NULL;

  v_report := jsonb_build_object('user_id', p_user, 'signer_code', v_signer, 'samples_excluded', v_samples,
                                 'note', 'already-trained models are not retroactively retrained; excluded from all future builds');
  INSERT INTO ops.jobs (kind, payload, created_by)                    -- object-storage deletion happens out of band
  VALUES ('erase_user_objects', v_report, p_actor);
  INSERT INTO audit.events (actor_user_id, action, target, meta) VALUES (p_actor, 'privacy.erase', p_user::text, v_report);
  RETURN v_report;
END $$;

-- -----------------------------------------------------------------------------------------------------
-- 17. Views (read models)
-- -----------------------------------------------------------------------------------------------------
CREATE VIEW ml.v_live_models AS
SELECT d.task, d.target, d.stage, d.traffic_fraction, mv.name, mv.version, mv.format, mv.is_bootstrap,
       mv.artifact_uri, d.decoder_config_id, d.min_app_version, d.started_at
FROM ml.deployments d JOIN ml.model_versions mv ON mv.id = d.model_version_id
WHERE d.ended_at IS NULL;

CREATE VIEW ml.v_leaderboard AS
SELECT mv.task, mv.name, mv.version, e.suite, e.perturbation, r.metric, r.value, r.ci_low, r.ci_high, r.n,
       b.name || ':' || b.version AS dataset, e.split, e.created_at
FROM ml.evaluation_results r
JOIN ml.evaluations e    ON e.id = r.evaluation_id
JOIN ml.model_versions mv ON mv.id = e.model_version_id
JOIN dataset.builds b     ON b.id = e.dataset_build_id
WHERE r.slice_key = 'all';

CREATE VIEW ml.v_fairness_gaps AS
SELECT e.model_version_id, e.id AS evaluation_id, r.metric, r.slice_key,
       max(r.value) - min(r.value) AS gap,
       (array_agg(r.slice_value ORDER BY r.value DESC))[1] AS worst_slice_if_error_metric,
       count(*) AS n_groups
FROM ml.evaluation_results r JOIN ml.evaluations e ON e.id = r.evaluation_id
WHERE r.slice_key <> 'all'
GROUP BY e.model_version_id, e.id, r.metric, r.slice_key;

CREATE VIEW ml.v_training_footprint AS
SELECT ex.task, ex.name AS experiment, count(*) AS runs,
       sum(tr.gpu_hours) AS gpu_hours, sum(tr.energy_kwh) AS energy_kwh, sum(tr.emissions_kgco2e) AS kgco2e
FROM ml.training_runs tr JOIN ml.experiments ex ON ex.id = tr.experiment_id
GROUP BY ex.task, ex.name;

CREATE VIEW dataset.v_build_composition AS
SELECT b.name, b.version, m.split, count(*) AS samples, count(DISTINCT m.signer_code) AS signers,
       count(*) FILTER (WHERE s.kind = 'continuous') AS continuous,
       count(*) FILTER (WHERE s.is_synthetic) AS synthetic,
       count(DISTINCT sg.region) AS regions,
       count(*) FILTER (WHERE sg.dominant_hand = 'left') AS left_handed_samples
FROM dataset.build_members m
JOIN dataset.builds b  ON b.id = m.build_id
JOIN dataset.samples s ON s.id = m.sample_id
LEFT JOIN dataset.signers sg ON sg.signer_code = s.signer_code
GROUP BY b.name, b.version, m.split;

CREATE VIEW dataset.v_licence_risk AS                                 -- anything that must not reach production models
SELECT src.slug, src.licence, src.allows_commercial, src.usage, count(s.id) AS samples
FROM dataset.sources src LEFT JOIN dataset.samples s ON s.source_id = src.id
WHERE src.allows_commercial IS DISTINCT FROM true OR src.usage <> 'train'
GROUP BY src.slug, src.licence, src.allows_commercial, src.usage;

CREATE VIEW feedback.v_quality_by_model AS
SELECT ir.model_version_id, date_trunc('day', fe.created_at) AS day,
       count(*) FILTER (WHERE fe.kind = 'confirm') AS confirms,
       count(*) FILTER (WHERE fe.kind = 'correct') AS corrections,
       count(*) FILTER (WHERE fe.kind = 'reject')  AS rejects
FROM feedback.events fe
JOIN inference.requests ir ON ir.utterance_id = fe.utterance_id AND ir.task = 'cslr' AND ir.deployment_stage = 'production'
GROUP BY ir.model_version_id, date_trunc('day', fe.created_at);

-- -----------------------------------------------------------------------------------------------------
-- 18. Row-level security (fail-closed)
-- -----------------------------------------------------------------------------------------------------
ALTER TABLE identity.users          ENABLE ROW LEVEL SECURITY;
ALTER TABLE conversation.rooms      ENABLE ROW LEVEL SECURITY;
ALTER TABLE conversation.room_members ENABLE ROW LEVEL SECURITY;
ALTER TABLE conversation.utterances ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledge.documents     ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledge.chunks        ENABLE ROW LEVEL SECURITY;
ALTER TABLE dataset.signers         ENABLE ROW LEVEL SECURITY;
ALTER TABLE privacy.consents        ENABLE ROW LEVEL SECURITY;
ALTER TABLE feedback.events         ENABLE ROW LEVEL SECURITY;
ALTER TABLE feedback.personal_adaptations ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent.sessions          ENABLE ROW LEVEL SECURITY;

CREATE POLICY users_self ON identity.users
  USING (id = shared.current_user_id() OR shared.has_role('admin'))
  WITH CHECK (id = shared.current_user_id() OR shared.has_role('admin'));

CREATE FUNCTION conversation.is_room_member(p_room uuid) RETURNS boolean
LANGUAGE sql STABLE SECURITY DEFINER SET search_path = conversation, pg_temp AS $$
  SELECT EXISTS (SELECT 1 FROM conversation.room_members m
                  WHERE m.room_id = p_room AND m.left_at IS NULL
                    AND (m.user_id = shared.current_user_id()
                         OR m.guest_id = shared.current_guest_id()
                         OR m.device_id = shared.current_device_id()))
$$;

-- Join-by-code lookup (QR). SECURITY DEFINER: the joiner is not a member yet, so RLS would hide the room.
CREATE FUNCTION conversation.find_room_by_code(p_code text)
RETURNS TABLE (room_id uuid, status text, expires_at timestamptz, venue_id uuid, context_profile shared.venue_type)
LANGUAGE sql STABLE SECURITY DEFINER SET search_path = conversation, pg_temp AS $$
  SELECT id, status, expires_at, venue_id, context_profile FROM conversation.rooms WHERE join_code = upper(p_code)
$$;

CREATE FUNCTION conversation.room_is_joinable(p_room uuid) RETURNS boolean
LANGUAGE sql STABLE SECURITY DEFINER SET search_path = conversation, pg_temp AS $$
  SELECT EXISTS (SELECT 1 FROM conversation.rooms r WHERE r.id = p_room AND r.status = 'open' AND r.expires_at > now())
$$;

CREATE POLICY rooms_members ON conversation.rooms FOR SELECT
  USING (conversation.is_room_member(id) OR created_by_user = shared.current_user_id()
         OR (venue_id IS NOT NULL AND venue_id = shared.current_venue_id())
         OR shared.has_role('admin'));
CREATE POLICY rooms_insert ON conversation.rooms FOR INSERT
  WITH CHECK (created_by_user = shared.current_user_id()
              OR (created_by_device = shared.current_device_id() AND venue_id = shared.current_venue_id()));

CREATE POLICY members_visible ON conversation.room_members FOR SELECT
  USING (user_id = shared.current_user_id() OR guest_id = shared.current_guest_id()      -- own rows (also makes
         OR device_id = shared.current_device_id()                                       -- INSERT … RETURNING work)
         OR conversation.is_room_member(room_id) OR shared.has_role('admin')
         OR EXISTS (SELECT 1 FROM conversation.rooms r WHERE r.id = room_id AND r.venue_id = shared.current_venue_id()));
CREATE POLICY members_join ON conversation.room_members FOR INSERT
  WITH CHECK (conversation.room_is_joinable(room_id)
              AND (user_id = shared.current_user_id() OR guest_id = shared.current_guest_id()
                   OR device_id = shared.current_device_id()));
CREATE POLICY members_self_update ON conversation.room_members FOR UPDATE
  USING (user_id = shared.current_user_id() OR guest_id = shared.current_guest_id()
         OR device_id = shared.current_device_id() OR shared.has_role('admin'));
CREATE POLICY members_admin_delete ON conversation.room_members FOR DELETE USING (shared.has_role('admin'));

CREATE POLICY rooms_update ON conversation.rooms FOR UPDATE
  USING (created_by_user = shared.current_user_id() OR shared.has_role('admin')
         OR (venue_id IS NOT NULL AND venue_id = shared.current_venue_id()));

CREATE POLICY utterances_members ON conversation.utterances
  USING (conversation.is_room_member(room_id) OR shared.has_role('admin'));

CREATE POLICY kb_docs_tenant ON knowledge.documents
  USING (venue_id IS NULL OR venue_id = shared.current_venue_id() OR shared.has_role('admin'));
CREATE POLICY kb_chunks_tenant ON knowledge.chunks
  USING (venue_id IS NULL OR venue_id = shared.current_venue_id() OR shared.has_role('admin'));

CREATE POLICY signers_restricted ON dataset.signers
  USING (shared.has_role('admin') OR shared.has_role('ml_engineer') OR user_id = shared.current_user_id());

CREATE POLICY consents_self ON privacy.consents
  USING (user_id = shared.current_user_id() OR shared.has_role('admin'))
  WITH CHECK (user_id = shared.current_user_id() OR shared.has_role('admin'));

CREATE POLICY feedback_self ON feedback.events
  USING (user_id = shared.current_user_id() OR shared.has_role('admin') OR shared.has_role('ml_engineer'))
  WITH CHECK (user_id = shared.current_user_id());

CREATE POLICY adapt_self ON feedback.personal_adaptations
  USING (user_id = shared.current_user_id() OR shared.has_role('admin'));

CREATE POLICY agent_sessions_room ON agent.sessions
  USING (conversation.is_room_member(room_id) OR venue_id = shared.current_venue_id() OR shared.has_role('admin'));

-- -----------------------------------------------------------------------------------------------------
-- 19. Grants (least privilege)
-- -----------------------------------------------------------------------------------------------------
GRANT USAGE ON SCHEMA shared, identity, venue, conversation, lexicon, dataset, annotation, ml, inference,
                      feedback, agent, knowledge, privacy, audit, ops
  TO ksl_app, ksl_worker, ksl_trainer, ksl_readonly;

-- gateway / orchestrator
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA identity, venue, conversation, agent, knowledge, feedback, privacy TO ksl_app;
GRANT SELECT ON ALL TABLES IN SCHEMA lexicon, ml TO ksl_app;
GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA inference TO ksl_app;
GRANT UPDATE ON inference.clarifications TO ksl_app;
GRANT SELECT, INSERT, UPDATE ON dataset.samples, annotation.tasks, annotation.annotations TO ksl_app;
GRANT SELECT ON dataset.sources, dataset.signers TO ksl_app;
GRANT INSERT ON audit.events TO ksl_app, ksl_worker, ksl_trainer;
GRANT SELECT, INSERT, UPDATE ON ops.outbox, ops.jobs, ops.idempotency_keys TO ksl_app, ksl_worker, ksl_trainer;

-- inference workers: read the registry, write inference logs only
GRANT SELECT ON ALL TABLES IN SCHEMA ml, lexicon TO ksl_worker;
GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA inference TO ksl_worker;
GRANT SELECT ON knowledge.chunks, knowledge.documents, knowledge.venue_vocabulary, knowledge.translation_memory TO ksl_worker;

-- training / evaluation jobs: datasets + registry, never identity
GRANT SELECT ON ALL TABLES IN SCHEMA dataset, annotation, lexicon, knowledge TO ksl_trainer;
GRANT INSERT, UPDATE ON dataset.builds, dataset.build_members TO ksl_trainer;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA ml TO ksl_trainer;
GRANT INSERT, UPDATE ON lexicon.gloss_prototypes, lexicon.sign_clips TO ksl_trainer;
GRANT SELECT ON feedback.events, feedback.active_learning_queue TO ksl_trainer;
GRANT INSERT, UPDATE ON feedback.active_learning_queue TO ksl_trainer;

-- read-only analytics: views + non-personal tables
GRANT SELECT ON ml.v_live_models, ml.v_leaderboard, ml.v_fairness_gaps, ml.v_training_footprint,
                dataset.v_build_composition, dataset.v_licence_risk TO ksl_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA ml TO ksl_readonly;

GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA ops, knowledge, audit TO ksl_app, ksl_worker, ksl_trainer;
GRANT EXECUTE ON FUNCTION ml.promote_model(uuid, shared.deploy_target, uuid, uuid, text) TO ksl_trainer;
REVOKE EXECUTE ON FUNCTION privacy.erase_user(uuid, uuid) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION privacy.erase_user(uuid, uuid) TO ksl_app;
GRANT EXECUTE ON FUNCTION identity.find_login(citext, text) TO ksl_app;

-- -----------------------------------------------------------------------------------------------------
-- 20. Seed / reference data
-- -----------------------------------------------------------------------------------------------------
INSERT INTO privacy.retention_policies (data_class, retain_days, legal_basis, description) VALUES
  ('realtime_landmarks', 0,    'data minimisation',          'Live landmark streams are processed in memory and never stored'),
  ('transcripts',        30,   'consent: store_transcripts', 'Utterances of rooms with store_transcript = true'),
  ('inference_logs',     90,   'legitimate interest',        'Latency/status logs without content (nbest only with consent)'),
  ('feedback_clips',     1825, 'consent: contribute_*',      'Clips attached to corrections for retraining'),
  ('dataset_samples',    1825, 'consent: contribute_*',      'Own recordings; removed from future builds on withdrawal'),
  ('audit_events',       730,  'security & accountability',  'Append-only audit trail'),
  ('ws_sessions',        30,   'operations',                 'Connection metadata for debugging');

INSERT INTO dataset.sources (slug, name, modality, language, url, licence, allows_commercial, allows_derivatives, usage, access_notes) VALUES
  ('ksl-bridge-own',       'KSL Bridge own recordings',                 'sign',   'ksl', NULL, 'proprietary-consented', true, true, 'train', 'Collected under ethics approval with per-signer consent'),
  ('ekitabu-ksl-video',    'eKitabu KSL Video Dataset (Kaggle)',        'sign',   'ksl', 'https://www.kaggle.com/datasets/ekitabu/ksl-video-dataset', 'CC0-1.0', true, true, 'train', '≈2,237 videos, 30 classes'),
  ('maseno-ksl-pose',      'Maseno KSL Word-Based Pose Dataset',        'sign',   'ksl', 'https://zenodo.org/records/14974973', 'restricted', NULL, NULL, 'blocked', 'Access by request; update licence + usage when granted'),
  ('ai4ksl-corpus',        'AI4KSL corpus (arXiv 2410.18295)',          'sign',   'ksl', 'https://arxiv.org/abs/2410.18295', 'unknown', NULL, NULL, 'blocked', 'Contact authors'),
  ('zindi-ksl-images',     'Zindi KSL Classification Challenge',        'sign',   'ksl', 'https://zindi.africa/competitions/kenyan-sign-language-classification-challenge', 'CC-BY-SA-4.0', true, true, 'pretrain_only', 'Share-alike obligations on derivatives'),
  ('google-asl-signs',     'Google Isolated SLR (ASL Signs, Kaggle)',   'sign',   'other_sl', 'https://www.kaggle.com/competitions/asl-signs', 'competition-rules', NULL, NULL, 'pretrain_only', 'ASL, not KSL: motion priors only; check rules'),
  ('afrispeech-200',       'AfriSpeech-200',                            'speech', 'en',  'https://huggingface.co/datasets/intronhealth/afrispeech-200', 'CC-BY-NC-SA-4.0', false, true, 'eval_only', 'Non-commercial: research/eval only'),
  ('common-voice-sw',      'Mozilla Common Voice Swahili',              'speech', 'sw',  'https://datacollective.mozillafoundation.org', 'CC0-1.0', true, true, 'train', 'Mozilla Data Collective account'),
  ('afrivoices-ke',        'AfriVoices-KE',                             'speech', 'sw',  'https://huggingface.co/Anv-ke', 'CC-BY-4.0', true, true, 'train', 'Dholuo/Kikuyu/Kalenjin/Maasai/Somali; request form');

INSERT INTO agent.tools (name, description, input_schema, output_schema, side_effect, requires_confirmation, timeout_ms) VALUES
  ('menu.lookup', 'Find a menu item by name', '{"type":"object","required":["name"],"properties":{"name":{"type":"string"}}}',
   '{"type":["object","null"]}', 'none', false, 500),
  ('order.add', 'Add an item and quantity to the room''s order', '{"type":"object","required":["item_id","qty"],"properties":{"item_id":{"type":"integer"},"qty":{"type":"integer","minimum":1,"maximum":50}}}',
   '{"type":"object"}', 'write', true, 2000),
  ('kb.answer', 'Answer a venue question from the knowledge base (extractive, grounded)', '{"type":"object","required":["question"],"properties":{"question":{"type":"string"}}}',
   '{"type":["string","null"]}', 'none', false, 800),
  ('staff.call', 'Call a human staff member to the room', '{"type":"object","required":["reason"],"properties":{"reason":{"type":"string"}}}',
   '{"type":"object"}', 'write', false, 1000),
  ('ui.show_menu', 'Show the menu on the signer''s screen', '{"type":"object"}', '{"type":"object"}', 'none', false, 300),
  ('robot.navigate', 'Ask the robot to move to a location (vendor safety layer has final authority)',
   '{"type":"object","required":["destination"],"properties":{"destination":{"type":"string"}}}', '{"type":"object"}', 'physical', true, 5000);

INSERT INTO lexicon.glosses (gloss, categories, is_critical, status) VALUES
  ('IX-1', '{PRONOUN}', false, 'approved'), ('IX-2', '{PRONOUN}', false, 'approved'), ('IX-3', '{PRONOUN}', false, 'approved'),
  ('YES', '{YES_NO}', false, 'pending'), ('NO', '{YES_NO}', false, 'pending'),
  ('WANT', '{VERB}', false, 'pending'), ('WATER', '{DRINK}', false, 'pending'), ('FOOD', '{FOOD}', false, 'pending'),
  ('HELP', '{HELP}', false, 'pending'), ('PAIN', '{HEALTH}', true, 'pending'), ('ALLERGY', '{HEALTH}', true, 'pending'),
  ('EMERGENCY', '{HEALTH}', true, 'pending'), ('THANK-YOU', '{SOCIAL}', false, 'pending');
-- Only Deaf reviewers move glosses to 'approved'. Seed glosses other than pointing are 'pending' on purpose.

-- -----------------------------------------------------------------------------------------------------
-- 21. Initial partitions (scheduler keeps 3 months ahead)
-- -----------------------------------------------------------------------------------------------------
SELECT ops.create_monthly_partitions('conversation.utterances', now()::date, 3);
SELECT ops.create_monthly_partitions('inference.requests',      now()::date, 3);
SELECT ops.create_monthly_partitions('audit.events',            now()::date, 3);

COMMIT;
