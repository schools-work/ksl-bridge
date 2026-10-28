# Dataset card: eKitabu KSL Video Dataset

| Field | Value |
|---|---|
| Source slug | `ekitabu-ksl-video` |
| Location | Kaggle `ekitabu/ksl-video-dataset` (duplicates exist under other accounts; use this one) |
| Licence | CC0-1.0 (verify at download time) → `usage = train` |
| Content | ≈2,237 `.mov` videos, 30 sign classes, ≈14.7 GB (per a third-party catalogue; verify after download) |
| Modality | Isolated signs, RGB video |
| Signers | To be determined after download (needed for signer-independent splits) |

## Processing
1. `tools/scripts/download_sources.sh ekitabu`
2. `ml/data/extract_landmarks.py` → `(T,107,3)` float16 per clip
3. `ml/data/quality.py` → `quality_flags`
4. `tools/scripts/import_source.py --source ekitabu-ksl-video` → `dataset.samples`

## Known gaps
* Isolated words only: no co-articulation or sentences.
* Signer metadata may be missing. Without signer IDs, signer-independent evaluation is impossible, so infer them conservatively (cluster by appearance) or keep this set for training only.
* 30 classes is a small vocabulary.
