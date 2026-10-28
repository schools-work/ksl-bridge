# Algorithm choices (and why they suit a nationwide app)

There isn't one algorithm. Each sub-problem gets the right one, and classical baselines from the course's ML cheat sheet are mandatory.

| Sub-problem | Baseline (course) | Main model | Why |
|---|---|---|---|
| Isolated sign recognition (ISLR) | **k-NN + DTW**; **Random Forest / XGBoost** on handshape statistics | Conformer pose encoder + attention pooling + ArcMargin cosine head | DTW is speed-invariant and explainable; cosine heads give tight clusters and open-set rejection |
| Self-supervised pre-training | — | **Masked Pose Modelling** (BERT-style, on landmarks) | Labelled KSL is scarce, unlabelled motion is cheaper (Han et al. 2021) |
| Continuous signing (CSLR) | — | Streaming Conformer + **CTC** (+ intermediate CTC) | No frame-level labels needed; CTC ≈ HMM with a blank state (AIMA Ch. 15) |
| Context / "small mistakes" | **Kneser–Ney n-gram LM** (AIMA Ch. 22) | **Prefix beam search** with LM shallow fusion + mild venue biasing (≤ 2 nats) | Informed search with a pruned frontier (AIMA Ch. 3–4); context resolves ambiguous signs |
| Confidence | **Logistic regression** on decoder features | + temperature scaling | Calibrated probabilities drive the clarification policy |
| When to ask "Did you mean…?" | Threshold | **Value of information** (AIMA Ch. 16): ask iff (1 − p)·loss > cost | Asks more in hospitals than restaurants; never guesses on critical slots |
| Gloss ⇄ English | Rule-based templates | Small Transformer seq2seq + back-translation + decoder pre-training | Low-resource pair; rules are the safety net |
| Intents / slots | **TF-IDF + logistic regression** | Joint Transformer intent + slot | Baseline works on English *and* raw gloss strings |
| Speech recognition | Bootstrap local open-weights model | Conformer-CTC + BPE + domain LM | Kenyan accents, Swahili, code-switching |
| Speech synthesis | Bootstrap local TTS | FastSpeech2 + HiFi-GAN; rule-based Swahili G2P | Swahili is phonemic, so G2P is cheap |
| Sign production | — | Dictionary medoid retrieval (DTW) + Hermite blending + non-manual overlay → learned in-betweening | Deterministic on day one; validated by Deaf reviewers |
| Regional variants / label errors | **K-means / DBSCAN** on sign embeddings | — | Unsupervised discovery (*Theories of ML*) |
| Visualisation / speed | **PCA** | — | 892-d features → 64-d for DTW |

## Why landmarks (not raw video) for the whole of Kenya

* About 20 KB/s instead of video, so it works over Kenyan mobile data.
* Small models (10–20 M parameters) run on low-end Android phones, and offline via TFLite/ExecuTorch.
* Privacy: faces don't leave the phone in real-time mode.
* Signer-invariant normalisation removes body size, distance and camera roll.
* I believe top solutions to the Google ASL Signs competition used landmark-based 1D-CNN/Transformer models. Verify this against the published write-ups before citing it.

## Scaling to the whole country

Record signers from several regions and schools for the Deaf, report per-region error rates (`ml.v_fairness_gaps`), keep variant glosses (`EAT.v2`) rather than deleting them, and personalise per user (prototypes, LoRA).
