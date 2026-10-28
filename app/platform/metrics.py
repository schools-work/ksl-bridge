"""Prometheus metrics. Per-stage latency is a research metric (RQ3), not just ops."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.5, 5.0)

WS_CONNECTIONS = Gauge("ksl_ws_connections", "Open WebSocket connections")
WS_MESSAGES = Counter("ksl_ws_messages_total", "WebSocket messages", ["direction", "type"])
STAGE_LATENCY = Histogram("ksl_stage_latency_seconds", "Pipeline stage latency", ["stage"], buckets=LATENCY_BUCKETS)
E2E_LATENCY = Histogram("ksl_e2e_latency_seconds", "End-to-end latency", ["path"], buckets=LATENCY_BUCKETS)
CLARIFICATIONS = Counter("ksl_clarifications_total", "Clarification requests", ["venue_type", "kind"])
FEEDBACK = Counter("ksl_feedback_total", "User feedback", ["kind"])
DROPPED_PARTIALS = Counter("ksl_dropped_partials_total", "Partial captions dropped by backpressure")
AUTH_EVENTS = Counter("ksl_auth_events_total", "Auth events", ["event"])
