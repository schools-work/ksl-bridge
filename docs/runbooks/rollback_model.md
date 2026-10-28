# Runbook: roll back a production model

```sql
-- 1. find the previous production version for the task/target
SELECT d.id, mv.version, d.started_at, d.ended_at
FROM ml.deployments d JOIN ml.model_versions mv ON mv.id = d.model_version_id
WHERE d.task = 'cslr' AND d.target = 'server' AND d.stage = 'production'
ORDER BY d.started_at DESC LIMIT 3;

-- 2. re-promote it (its gates are already recorded)
SELECT ml.promote_model('<previous model_version_id>', 'server', '<your user id>');
```

Workers pick up the change through the `ModelPromoted` outbox event (`workers/loader.py`). Verify it in `ml.v_live_models` and the Grafana panel "live model per task". Record the reason in `ml.deployments.notes`.
