# ADR-0004: One PostgreSQL 16 database with pgvector; schema file is the source of truth

* **Status:** accepted
* **Date:** 2026-10-27

## Context
We need relational integrity (consent, splits, registry, auth), vectors (RAG chunks, sign prototypes) and full-text search. The guest RAG lecture contrasted pgvector, Pinecone and Chroma, and noted that with pgvector, tenant isolation is the application's job.

## Decision
* PostgreSQL 16 + pgvector (HNSW) + `tsvector` for hybrid search, in one database.
* `db/ksl_bridge_v1.sql` is canonical: 15 schemas, one per bounded context; fail-closed RLS driven by `app.current_user_id`, `app.current_roles` and `app.current_venue_id`.
* Invariants live **in the database** where they protect research validity or rights: signer-independent splits, frozen builds, consent for contributions and personalisation, release gates before promotion, one production model per task, append-only audit, tool-confirmation rules.
* Workers and training jobs never migrate. Alembic migrations converge to the SQL file.

## Consequences
+ One backup, one consistency model; invariants can't be bypassed by a buggy script.
− RLS requires every request transaction to set GUCs (`app/platform/db.py`).
