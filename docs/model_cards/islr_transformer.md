# Model card: islr_transformer (first release pending)

| Field | Value |
|---|---|
| Task | `islr` |
| Architecture | Conformer pose encoder (d = 256, 8 layers, ×2 subsampling) + attention pooling + ArcMargin cosine head (s = 30, m = 0.2) |
| Initialisation | Masked Pose Modelling pre-training (`ml/configs/mpm_pretrain.yaml`) |
| Input | 892-d features (`ksl_core/normalize.py`) + presence mask |
| Training config | `ml/configs/islr_transformer.yaml` |
| Status | **Not trained yet.** Fill this card from `ml.v_leaderboard` and `ml.v_fairness_gaps` when the first run passes the gates |

## Planned evaluation
* Top-1/top-5 and macro-F1 on the signer-independent test split; open-set AUROC.
* Robustness: rot15, rot30, slow, fast, jitter05, drop30 (`ml/eval/robustness.py`).
* Groups: gender, age band, region, dominant hand, hearing status, device tier, lighting.
* Baselines to beat: k-NN + DTW, Random Forest (`ml/models/baselines.py`).
