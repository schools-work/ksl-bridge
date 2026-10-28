# ADR-0003: No hosted AI APIs; we build and train our own models

* **Status:** **PROVISIONAL**. The group must confirm the items marked ⚠
* **Date:** 2026-10-27

## Context
The group decided "we won't be using APIs at any point. We will build our own AI models and agents." Some components (hand-landmark detection, high-quality ASR/TTS) are extremely data-hungry, and the demo has a deadline.

## Decision
| Allowed | Not allowed |
|---|---|
| Open-source libraries running on our machines (PyTorch, NumPy, FastAPI, MediaPipe, librosa, ONNX Runtime) | Hosted AI services (OpenAI, Google Cloud Speech, Azure Cognitive Services, hosted translation) |
| Public datasets under their licences | Sending user video, audio or text to third-party AI services |
| Architectures from papers, implemented and trained by us | Presenting a downloaded end-to-end translator as "our model" |

⚠ **Provisional items:**
1. **MediaPipe as the sensor.** The landmark detector is treated like a camera (AIMA: sensors are given); our AI starts at the percept.
2. **Bootstrap ASR/TTS:** local open-weights models behind `ModelAdapter` (`ml.model_versions.is_bootstrap = true`) until our own models beat them on the Kenyan test set. The comparison is reported as an experiment.
3. **v1 vocabulary:** 300–500 glosses; domains restaurant + hospital reception.
4. **Swahili scope:** speech/text in and out for captions; KSL ⇄ English is the core translation pair.
5. **Transcript storage:** off by default, on only with `store_transcripts` consent.
6. **Pilot hosting:** single GPU server; data residency in or near Kenya.
7. **Kotlin app** lives in its own repository and consumes `contracts/`.

## Consequences
Every provisional item is implemented behind an interface, so changing the decision is a configuration change. Code that depends on an unconfirmed item raises a **Decision Required** (MASTER_PROMPT §12).
