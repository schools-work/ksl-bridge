# Runbook: incident response

1. **Detect:** a Grafana alert (`infra/observability/alerts.yml`) or a user report.
2. **Stabilise:** if translation quality or latency is degraded, keep rooms alive. Text relay works without workers. Scale workers, or roll back the model (`rollback_model.md`).
3. **Communicate:** post status to the team channel. If personal data may be affected, start the breach assessment (the ODPC must be notified within 72 hours when required).
4. **Investigate:** traces by `utterance_id` (OpenTelemetry), `inference.requests` by `model_version_id`, `audit.events`.
5. **Recover and review:** write a post-mortem in `docs/runbooks/postmortems/YYYY-MM-DD-<slug>.md`.
