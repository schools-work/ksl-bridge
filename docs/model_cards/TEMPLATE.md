# Model card: <name> <version>

| Field | Value |
|---|---|
| Task | `shared.model_task` |
| Registry id | `ml.model_versions.id` |
| Training run | `ml.training_runs.id` (MLflow run id) |
| Dataset build | `<name>:<version>` (frozen) |
| Bootstrap? | yes/no (+ upstream source and licence) |
| Params / size / quantisation | |
| Energy / emissions (training) | kWh / kg CO₂e |

## Intended use
Who uses it, for what, and where it must **not** be used.

## Metrics (signer-independent test split)
| Metric | Value | 95% CI | n |
|---|---|---|---|

## Robustness matrix
| Condition | WER | Δ vs clean |
|---|---|---|

## Per-group results (fairness)
| Slice | Group | Metric | Value | n |
|---|---|---|---|---|

## Calibration
ECE before / after; method.

## Limitations and known failure modes

## Release gates
quality · no_regression · fairness · calibration · latency · safety_set · licence · human_review
