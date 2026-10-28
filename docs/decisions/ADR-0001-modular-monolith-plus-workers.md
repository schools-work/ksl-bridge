# ADR-0001: Modular monolith + inference workers (not microservices)

* **Status:** accepted
* **Date:** 2026-10-27

## Context
A student team must ship a real-time system with heterogeneous workloads: I/O-bound WebSockets and compute-bound GPU inference. Microservices would multiply deployment, networking and consistency problems with no benefit at this scale.

## Decision
* `app/`: one FastAPI process type, split into bounded-context modules (`app/modules/<context>`) with hexagonal layers and import-linter-enforced boundaries.
* `workers/`: separate processes consuming **Redis Streams**, with micro-batching on GPU/CPU.
* `ml/`: offline training, never imported by `app/`.
* One PostgreSQL database with one schema per bounded context.

## Consequences
+ A slow model can't block sockets. Gateways and workers scale independently.
+ One deployable service is easy to test and reason about.
− Module discipline relies on CI (import-linter) and review.

## Alternatives considered
* **Microservices:** rejected; operational overhead, distributed transactions.
* **Single process with models in the gateway:** rejected; GPU work blocks the event loop and there's no batching.
