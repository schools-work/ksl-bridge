# Datasets: where to get them, licences, how we use them

> The licence decides what a dataset may be used for, because the app is meant for use across Kenya, possibly commercially. Every source is registered in `dataset.sources` with `licence`, `allows_commercial` and `usage` (`train` / `pretrain_only` / `eval_only` / `blocked`). `dataset.v_licence_risk` lists everything that must not reach a shipped model. **Re-verify licences at the source before ingest.**

## Kenyan Sign Language

| Dataset | Content | Access | Licence | `usage` |
|---|---|---|---|---|
| **eKitabu KSL Video Dataset** (Kaggle `ekitabu/ksl-video-dataset`) | ≈2,237 videos, 30 classes, 14.7 GB | Kaggle account | CC0-1.0 | `train` |
| **Maseno KSL Word-Based Pose Dataset** ([Zenodo 14974973](https://zenodo.org/records/14974973); [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11999445/)) | Isolated signs as MediaPipe landmarks; 685 participants | **Restricted**. Request through your supervisor | Listed CC BY 4.0 | `blocked` until access is granted |
| **AI4KSL corpus** ([arXiv 2410.18295](https://arxiv.org/abs/2410.18295)) | ≈14k English sentences + KSL gloss, ≈20k sign videos, HamNoSys | No public download; contact the authors | Not stated | `blocked` |
| **Zindi KSL Classification Challenge** | ≈1.1 GB images, ~9–10 classes | Zindi account | CC BY-SA 4.0 | `pretrain_only` |
| Small Kaggle sets (`joanwachuka/ksl-hand-landmarks`, `michaelmwai/ksl-numpy-array-files-dataset`) | 5–10 words, landmarks | Kaggle | Apache 2.0 / MIT | `train` (tiny) |
| Peace Corps KSL Glossary ([GitHub](https://github.com/peterkhess/KSL-Glossary)) | 487 short videos | GitHub | Unclear | ask first |

Catalogue with more sources and licence notes: [marionet issue #133](https://github.com/juleshenry/marionet/issues/133).

## Other sign languages (motion pre-training only, never KSL labels)

* **Google Isolated SLR "ASL Signs"** (Kaggle competition `asl-signs`): 250 signs, ≈94k sequences, already in 543-point MediaPipe Holistic parquet ([format](https://hackster.io/AlbertaBeef/insightful-datasets-for-asl-recognition-f786b9)). Check the competition rules.
* **How2Sign Holistic** (Hugging Face `PSewmuthu/How2Sign_Holistic`).

## Speech

| Dataset | Use | Licence | `usage` |
|---|---|---|---|
| **Common Voice Swahili** ([Mozilla Data Collective](https://datacollective.mozillafoundation.org/datasets/cmn3ailbd008nmb07mjyu3xro)) | Swahili ASR | CC0 | `train` |
| **AfriSpeech-200** (HF `intronhealth/afrispeech-200`; [paper](https://arxiv.org/pdf/2310.00274)) | Kenyan-accented English (≈21 h from Kenya) | **CC BY-NC-SA 4.0** | `eval_only` (non-commercial) |
| **Google FLEURS** (HF `google/fleurs`, config `sw_ke`) | Swahili test set | verify (believed CC BY 4.0) | `eval_only` until verified |
| **AfriVoices-KE** ([paper](https://arxiv.org/pdf/2604.08448), HF org `Anv-ke`) | ≈3,000 h Dholuo, Kikuyu, Kalenjin, Maasai, Somali | CC BY 4.0, request form | `train` |

## The gap no dataset fills

No dataset has **continuous KSL in service situations** (ordering, reception). We record it ourselves (`tools/collection/recording_protocol.md`). That's also the research contribution.

## Download commands

See `tools/scripts/download_sources.sh`. Run it on a machine that can reach Kaggle and Hugging Face. Use campus Wi-Fi: eKitabu alone is 14.7 GB.
