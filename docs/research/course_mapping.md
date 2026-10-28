# How the course material maps onto the build

| Material | Concept | Where it shows up |
|---|---|---|
| *Introduction to AI* (BBT 3201) | Rational agents; AI should complement humans (Weizenbaum) | Human escalation (`staff.call`); interpreter recommendation in critical domains |
| *Intelligent Agents* (ICS 3202), AIMA Ch. 2 | PEAS, environment types, agent structures | `docs/research/project_brief.md`; the architecture rationale |
| *Problem Solving Agents*, AIMA Ch. 3–4 | State space, path cost, informed search, beam search | `ml/decoding/ctc_beam.py`: states = gloss prefixes, cost = −log p |
| *Learning Agents & Data Mining* | Learning-agent components, CRISP-DM | Critic = `feedback.events`; problem generator = `feedback.active_learning_queue`; CRISP-DM = `ml/data` → `ml/train` → `ml/eval` → `mlops` |
| *Theories of Machine Learning* | Supervised/unsupervised, clustering, PCA, distances | Baselines, DBSCAN variant discovery, PCA for DTW |
| *ML Cheat Sheet* | Logistic regression, RF, XGBoost, k-NN | `ml/models/baselines.py`, `ml/models/nlu.py`, `ml/decoding/calibration.py` |
| *Pre-trained Models* (Han et al., 2021) | Transfer learning, masked modelling | `ml/models/mpm.py`, decoder pre-training for MT |
| Guest lecture: *Naive RAG* | Ingestion vs retrieval, chunking, pgvector, failure modes | `app/modules/knowledge/*`, `knowledge.hybrid_search`, `knowledge.retrieval_logs` |
| AIMA Ch. 13–17 | Uncertainty, HMMs, Kalman, utility, value of information | `ksl_core/smoothing.py`, CTC, `pipeline/domain/clarify.py` |
| AIMA Ch. 22–23 | N-gram LMs, smoothing, MT, speech | `ml/decoding/ngram_lm.py`, `ml/models/seq2seq.py`, `ml/models/asr.py` |
| AIMA Ch. 24–27, *Sustainable AI* topic, van Wynsberghe (2021), Wu et al. (2022), Ren & Wierman (2024) | Perception, robotics, ethics, AI footprint | Landmarks as percepts; robot safety (`side_effect='physical'`); `ml.v_training_footprint` |
| Brookings: *Algorithmic bias detection and mitigation* | Bias impact statements, audits | `docs/bias_impact_statement.md`, `ml/eval/fairness.py` |
| Sustainable development literature review | Three pillars | Social pillar: SDG 10, SDG 8 (employment RQ), SDG 3/4 |
| *AI and Art* | Generative AI, authorship, provenance | Sign production validated by Deaf reviewers; contributor consent and credit |
