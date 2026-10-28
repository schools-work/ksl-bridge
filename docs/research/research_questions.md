# Research questions

## Social / background (literature review and survey)

1. What communication challenges do Deaf and hard-of-hearing individuals face when interacting with people who do not understand sign language?
2. How do communication barriers affect employment opportunities for Deaf and hard-of-hearing individuals?
3. What methods are currently used to overcome these barriers?

## Technical (evidence produced by the backend)

| RQ | Question | Evidence | Stored in |
|---|---|---|---|
| RQ1 | What barriers do KSL users experience with non-signers and technology-mediated services? | Interviews/surveys + session analytics (clarification rate, stalls) | `inference.clarifications`, `agent.escalations` |
| RQ2 | How accurately can an AI system recognise **continuous** KSL under signer, environment and performance variation? | CSLR WER on signer-independent splits + robustness matrix | `ml.evaluations` (suite `clean`, `robustness`) |
| RQ3 | How effectively can it translate between KSL and text/speech in real time? | BLEU/chrF + human ratings + p50/p95 latency | `ml.evaluation_results` (`bleu`, `chrf`, `p95_ms`) |
| RQ4 | How can context improve recognition when signs are ambiguous or imperfect? | Ablations: no LM / LM / + venue bias / + clarification | `ml.evaluations` (suite `ablation`) |
| RQ5 | How usable is AI-mediated KSL communication in real service interactions? | Task success, clarifications per task, SUS, with Deaf participants | `agent.sessions`, study records |

Every experiment row in `ml.experiments` names the RQ it answers.
