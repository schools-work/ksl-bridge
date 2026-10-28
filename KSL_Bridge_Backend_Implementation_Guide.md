# KSL Bridge — Python AI Backend Implementation Guide

> **Project:** AI-mediated multimodal communication bridge between Kenyan Sign Language (KSL) users and people or machines that don't understand KSL. The machines include service robots, kiosks and reception terminals.
> **Scope of this guide:** the **Python backend**: AI models, training, inference, real-time streaming, and the normal platform services (authentication, sessions, WebSocket broadcast, storage, observability). The Kotlin app is covered only where it touches the backend contract.
> **Working name:** "KSL Bridge" is a placeholder. Rename it freely.
> **Hackathon framing:** *Develop a technology-based solution to bridge communication barriers for sign language users.*

---

## Table of contents

- [Part I — Understanding the project](#part-i--understanding-the-project)
  - [1. The problem, stated precisely](#1-the-problem-stated-precisely)
  - [2. Users, scenarios and system boundary](#2-users-scenarios-and-system-boundary)
  - [3. The system as an intelligent agent (PEAS + environment)](#3-the-system-as-an-intelligent-agent-peas--environment)
  - [4. The five hidden problems and the interlingua](#4-the-five-hidden-problems-and-the-interlingua)
  - [5. What "small mistakes" means technically](#5-what-small-mistakes-means-technically)
  - [6. Existing systems and our differentiation](#6-existing-systems-and-our-differentiation)
  - [7. The "no APIs, our own models" constraint, interpreted](#7-the-no-apis-our-own-models-constraint-interpreted)
  - [8. How the course material maps onto the build](#8-how-the-course-material-maps-onto-the-build)
- [Part II — Architecture](#part-ii--architecture)
- [Part III — Data (CRISP-DM)](#part-iii--data-crisp-dm)
- [Part IV — Models](#part-iv--models)
- [Part V — Real-time inference and error tolerance](#part-v--real-time-inference-and-error-tolerance)
- [Part VI — Platform services: auth, sessions, WebSockets, storage](#part-vi--platform-services-auth-sessions-websockets-storage)
- [Part VII — Agent layer, retrieval (RAG) and robot/kiosk integration](#part-vii--agent-layer-retrieval-rag-and-robotkiosk-integration)
- [Part VIII — Evaluation, testing, MLOps, performance, CUDA](#part-viii--evaluation-testing-mlops-performance-cuda)
- [Part IX — Ethics, bias, privacy and sustainability](#part-ix--ethics-bias-privacy-and-sustainability)
- [Part X — Roadmap, team split and Kotlin contract](#part-x--roadmap-team-split-and-kotlin-contract)
- [Appendices](#appendices)

---

# Part I — Understanding the project

## 1. The problem, stated precisely

Don't describe the project as "a sign-language translator app". The problem is:

> **How can a Deaf or hard-of-hearing KSL user communicate naturally and in real time with a hearing person, or an intelligent service machine, that does not know KSL, without a human interpreter, and with enough robustness that ordinary variation in signing doesn't break the conversation?**

Three qualifiers make it a research project and not just an app:

| Qualifier | What it forces into the backend |
|---|---|
| **Real time** | Streaming inference, partial hypotheses, a latency budget per stage, WebSockets. |
| **Two-way, multimodal** | Every modality goes in and out: Sign ⇄ Text ⇄ Speech. A shared internal meaning representation (§4). |
| **Robust to "small mistakes"** | Probabilistic decoding, calibrated confidence, context rescoring, clarification dialogue, learning from corrections (§5). |
| **Human ⇄ machine as well as human ⇄ human** | An agent layer that can turn understood meaning into actions such as placing an order or answering a question (Part VII). |

**Terminology rule for the written report:** *Deaf* (capital D: culturally Deaf, often sign-first) and *hard of hearing* are not interchangeable. Many hard-of-hearing people use spoken language with hearing aids or cochlear implants. Write the target population as:

> Deaf and hard-of-hearing people, particularly **KSL users**, who face communication barriers with people and services that do not understand Kenyan Sign Language.

The system should also support users who prefer **text** both ways. Many hard-of-hearing users will pick that.

### 1.1 Research questions, technical version

The social RQs from the brainstorm stay. These are the technical ones the backend must produce evidence for:

| RQ | Question | Backend evidence |
|---|---|---|
| RQ1 | What communication barriers do KSL users experience with non-signers and technology-mediated services? | Survey and interview data (not backend), plus session analytics: where conversations stall and how often clarification is needed. |
| RQ2 | How accurately can an AI system recognise **continuous** KSL under signer, environment and performance variation? | CSLR WER on signer-independent splits, plus the robustness matrix (§VIII-2). |
| RQ3 | How effectively can the system translate between KSL and text/speech in real time? | BLEU/chrF for gloss↔English, end-to-end latency percentiles. |
| RQ4 | How can context improve recognition when individual signs are ambiguous or imperfect? | Ablation: decoder with vs without LM fusion, domain context and RAG (§V-3). |
| RQ5 | How usable is AI-mediated KSL communication in real service interactions? | Task-completion rate, clarification rate, SUS scores from pilot sessions in a mock restaurant or reception setting. |

## 2. Users, scenarios and system boundary

### 2.1 Actors

| Actor | Description | Primary modality | Backend role |
|---|---|---|---|
| **Signer** | Deaf/HoH KSL user | Sign in; sign/text out | `signer` |
| **Hearing party** | Waiter, nurse, receptionist, friend | Speech/text in; speech/text out | `speaker` |
| **Service machine** | Robot, kiosk, POS terminal | Structured intents in/out | `device` (API-key auth) |
| **Venue operator** | Restaurant or hospital that registers devices and knowledge (menus, FAQs) | Web/admin | `operator` |
| **Annotator / Deaf consultant** | Validates recordings, glosses, translations | Annotation tools | `annotator` |
| **Admin / ML engineer** | Runs training, deploys models | CLI/admin | `admin` |

### 2.2 Canonical scenarios (use these as the test suite for the whole system)

1. **Face-to-face, one phone.** The phone sits on the table. The signer signs to the front camera, and the hearing person sees text and hears speech. The hearing person speaks, and the signer sees an avatar signing plus captions.
2. **Face-to-face, two phones.** Each person uses their own device, joined to the same **conversation room** by QR code. *This is where WebSocket broadcast matters.*
3. **Restaurant robot or kiosk.** The signer signs an order to the robot's camera, which runs our Kotlin/Android client or calls the backend. The agent layer maps meaning to `place_order(items=[chicken_burger×2, water×1])`, the robot replies in sign via an avatar on its screen, and the waiter's tablet joins the same room.
4. **Hospital reception.** Domain vocabulary such as PAIN, HEAD and APPOINTMENT. RAG over the hospital FAQ answers "Where is the pharmacy?" in KSL.
5. **Offline / low connectivity.** The phone runs a quantised on-device ISLR model for a core vocabulary and syncs later. The backend must accept deferred uploads and resolve versions.

### 2.3 System boundary for the backend

```
                 IN SCOPE (Python backend)                        OUT OF SCOPE (Kotlin / hardware)
 ┌─────────────────────────────────────────────────────────┐   ┌──────────────────────────────┐
 │ Auth, users, devices, venues, rooms                      │   │ Camera capture, UI, avatar    │
 │ WebSocket gateway + room broadcast                       │   │ 3D rendering, mic capture,    │
 │ Landmark ingestion, normalisation                        │   │ audio playback, on-device     │
 │ Sign recognition (isolated + continuous)                 │   │ landmark extraction*          │
 │ Gloss⇄English translation, ASR, TTS                      │   │ robot motors / POS hardware   │
 │ Sign production (gloss → pose sequence for avatar)       │   └──────────────────────────────┘
 │ Context engine, clarification, agent/tool layer, RAG     │   * the backend ALSO supports
 │ Data collection, annotation, training, evaluation, MLOps │     server-side extraction from
 │ Observability, privacy, consent, retention               │     uploaded video for datasets
 └─────────────────────────────────────────────────────────┘
```

**Key design decision: the phone sends landmarks, not video, in real-time mode.** The Kotlin app runs a landmark extractor on-device and streams about 1–2 KB per frame of keypoints instead of video. That gives privacy (no faces leave the phone), bandwidth suited to Kenyan mobile data, and lower latency. Raw video is uploaded only in *dataset-contribution* mode, with explicit consent.

## 3. The system as an intelligent agent (PEAS + environment)

This applies the *Intelligent Agents* topic (ICS 3202) and AIMA Ch. 2.

### 3.1 PEAS

| | KSL Bridge agent |
|---|---|
| **Performance measure** | Meaning correctly conveyed (task success), low gloss/word error, low end-to-end latency (< 1.5 s sign→text after end of utterance), few unnecessary clarifications, no harmful mistranslation in critical domains (allergies, medication), equal performance across signer groups, low energy per inference. |
| **Environment** | Signers with varying style, speed, handedness and regional variation; hearing people with Kenyan English/Swahili accents and code-switching (Sheng); noisy restaurants; variable lighting; phones of varying quality; intermittent networks; robots and kiosks with domain-specific tasks. |
| **Actuators** | Text on screen, synthesised speech, avatar pose sequences, clarification prompts, structured intents/tool calls to devices, broadcast messages to room participants. |
| **Sensors** | Landmark streams (hands, body, face) from the camera, audio from the mic, typed text, user corrections and confirmations, device state (menu, availability). |

### 3.2 Environment classification

| Dimension | Classification | Consequence for design |
|---|---|---|
| Observable? | **Partially observable.** Hands leave the frame or occlude each other, and the landmark detector is noisy. | Probabilistic models (CTC/HMM-style decoding) and belief over hypotheses (N-best), not a single hard label. |
| Deterministic? | **Stochastic.** The same sign is performed differently every time. | Calibrated probabilities; decisions under uncertainty (AIMA Ch. 13, 16). |
| Episodic? | **Sequential.** The meaning of sign *t* depends on signs *t−1…t−k* and on the dialogue state. | Sequence models, a language model, a dialogue state tracker. |
| Static? | **Dynamic.** The person keeps signing while we compute. | Streaming inference with partial results; time budgets. |
| Agents | **Multi-agent.** Signer, hearing party, robot. | Room model, turn-taking, broadcast. |
| Discrete? | **Continuous** input (coordinates over time), discrete output (glosses/words). | Continuous-to-discrete decoding (CTC + beam search). |

This classification is the most important conceptual input to the architecture. A partially observable, stochastic, sequential, dynamic, multi-agent environment rules out a lookup-table translator, and that justifies everything that follows.

### 3.3 The system as a learning agent

From *Learning Agents & Data Mining* and AIMA §2.4.6:

| Learning-agent component | KSL Bridge implementation |
|---|---|
| **Performance element** | The inference pipeline (recogniser → decoder → translator → generator). |
| **Critic** | User feedback: "👍 correct", "✏️ I meant …", hearing-party confirmations, task success from the agent layer, annotator review. |
| **Learning element** | Periodic fine-tuning jobs on corrected samples, LM updates from confirmed sentences, per-user adaptation (signer embeddings). |
| **Problem generator** | Active learning: surfaces low-confidence or high-disagreement clips to annotators and suggests vocabulary to record next (signs with the highest confusion). |

### 3.4 Agent type

The core pipeline is a **model-based reflex agent**. It keeps internal state (signing buffer, dialogue context) and maps percepts to outputs. The **service layer** (Part VII) is a **goal-based / utility-based agent**: it has goals (complete an order), uses tools (menu lookup) and chooses clarifying questions by **expected utility** (AIMA Ch. 16, value of information).

## 4. The five hidden problems and the interlingua

| # | Problem | Question it answers | Backend module |
|---|---|---|---|
| 1 | **Perception** | What are the hands, body and face physically doing? | `perception/` (landmarks, normalisation) |
| 2 | **Sign recognition** | Which signs, in which order? | `recognition/` (ISLR, CSLR, fingerspelling) |
| 3 | **Language understanding** | What does the gloss sequence mean in context? | `language/` (LM, NLU, context) |
| 4 | **Translation** | How is that meaning expressed in the target language? | `translation/` (gloss⇄English, English⇄Swahili optional) |
| 5 | **Generation** | How do we present it naturally? | `generation/` (TTS, sign production, text) |

### 4.1 Don't build 9 translators. Build a hub.

```
 SIGN (landmarks) ──► CSLR ──► gloss lattice ─┐                          ┌─► text renderer ──► TEXT
                                               │                          │
 SPEECH (audio) ────► ASR ───► word lattice ──┼──►  UTTERANCE IR (hub) ──┼─► TTS ─────────────► SPEECH
                                               │   (meaning + glosses +   │
 TEXT (keyboard) ───► normaliser ─────────────┘    English + intent +     └─► text→gloss ─► sign
                                                    confidence + context)                  production ─► SIGN (pose seq)
```

Every input becomes an **Utterance IR** (intermediate representation). Every output is generated *from* the IR. N inputs and M outputs then need N + M components, not N × M. It also lets the agent layer read meaning directly through the intent/slots, with no need to parse English.

### 4.2 Utterance IR schema (this is the central data contract)

```python
# ksl_core/ir.py
from __future__ import annotations
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class Modality(str, Enum):
    SIGN = "sign"
    SPEECH = "speech"
    TEXT = "text"
    DEVICE = "device"          # structured message from a robot/kiosk


class Hypothesis(BaseModel):
    """One candidate reading of the input (an entry of the N-best list)."""
    glosses: list[str] = Field(default_factory=list)   # e.g. ["IX-1", "WANT", "BURGER", "TWO"]
    text: str | None = None                            # English surface form
    log_prob: float                                    # total decoder score
    confidence: float = Field(ge=0, le=1)              # calibrated probability (§V-2)


class TokenSpan(BaseModel):
    token: str
    start_ms: int
    end_ms: int
    confidence: float
    alternatives: list[tuple[str, float]] = Field(default_factory=list)  # for "did you mean?"


class NonManual(BaseModel):
    """Non-manual markers: these change meaning (question, negation, intensity)."""
    question: float = 0.0       # raised/furrowed brows → yes/no or WH question
    negation: float = 0.0       # head shake
    intensity: float = 0.0      # puffed cheeks, tense mouth


class Intent(BaseModel):
    name: str                   # "order_food", "ask_location", "greet", "unknown"
    confidence: float
    slots: dict[str, str | int | list[str]] = Field(default_factory=dict)


class UtteranceIR(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    room_id: UUID
    speaker_id: UUID
    source: Modality
    language: str = "ksl"                    # "ksl" | "en-KE" | "sw-KE"
    created_at: datetime
    is_final: bool = False                   # partial (streaming) vs final
    revision: int = 0                        # partial results increase this
    best: Hypothesis
    nbest: list[Hypothesis] = Field(default_factory=list)
    spans: list[TokenSpan] = Field(default_factory=list)
    non_manual: NonManual | None = None
    intent: Intent | None = None
    needs_clarification: bool = False
    clarification: "Clarification | None" = None
    model_versions: dict[str, str] = Field(default_factory=dict)  # provenance for every stage


class Clarification(BaseModel):
    kind: str                                # "choose_one" | "confirm" | "repeat" | "fingerspell"
    prompt_text: str
    prompt_glosses: list[str]
    options: list[str] = Field(default_factory=list)
    span_index: int | None = None


UtteranceIR.model_rebuild()
```

**Glosses** are the written labels of signs (by convention in CAPITALS, e.g. `WANT`, `BURGER`, `IX-1` for pointing to oneself). A gloss is *not* an English translation. KSL glosses follow KSL grammar, so `IX-1 WANT BURGER TWO` translates to "I want two burgers."

## 5. What "small mistakes" means technically

The brainstorm's "the system must recognise even the small, small, small mistakes" breaks down into six engineering requirements:

| Source of "mistake" | Technical name | Mitigation in this guide |
|---|---|---|
| Slightly wrong handshape or finger angle | Intra-class variation | Pose normalisation (§III-3), augmentation (§III-5), metric-learning embeddings (§IV-2.4) |
| Faster or slower movement | Temporal variation | Time-warp augmentation, CTC alignment-free training (§IV-3) |
| Different signer body, style, dominant hand | Signer variation / domain shift | Signer-independent splits, mirroring, per-user adaptation (§V-6) |
| Bad angle, lighting, partial frame | Sensor noise / occlusion | Landmark dropout augmentation, missing-data masks, interpolation |
| Model unsure between two signs | Ambiguity | N-best lattices + LM + domain context rescoring (§V-3) |
| Sign not in vocabulary | Open-set / OOV | Open-set rejection, fingerspelling module, "please fingerspell" clarification (§IV-4, §V-4) |
| Hearing person's accent, noise, code-switching | ASR robustness | Noise augmentation, Kenyan-accented data, custom LM with Sheng/Swahili tokens (§IV-7) |

The system should **degrade gracefully**. It may say *"I think you signed WATER (72%) or WAIT (21%). Which one?"*. It must never confidently output the wrong thing in a critical domain.

## 6. Existing systems and our differentiation

| System | What it does (publicly described) | Gap we target |
|---|---|---|
| **Ishara AI** (Kenya, ishara.co.ke) | KSL ⇄ text/speech; recognised in IRCAI Global Top 100 2025 | Our angle: robot/kiosk **service interactions**, agentic task completion, explicit error-tolerance research, offline core |
| **Signvrse** (Kenya) | AI sign-language translation platform with avatars | Same: our research contribution is robustness + context + service-agent integration |
| **SignAll** (hackathon reference) | Speech → text, generates sign-hand images | Image generation isn't sign production. We generate *pose sequences* that respect sign structure |
| **Signa AI, VocalSign, Hand Talk, SignComm, Kozha** | Various text/speech → avatar, or sign → text/speech | Mostly other sign languages; we are KSL-specific with Kenyan English/Swahili speech |
| **Maseno University AI4KSL** and related academic work | KSL datasets and recognition research | We build on their data and contribute continuous-signing data in service contexts |

**Differentiation statement for the proposal:**

> A context-aware, error-tolerant, multimodal KSL communication engine designed for real-time interactions between KSL users, non-signers and intelligent public-service interfaces. It uses its own trained models and works with limited connectivity.

Don't claim "first KSL translator". Claim the four-way combination: **Kenyan context × multimodal two-way × robustness research × service-agent integration**.

## 7. The "no APIs, our own models" constraint, interpreted

The group decided: *"we won't be using APIs at any point. We will build our own AI models and agents."* Interpret this as follows, and **agree it as a group and write it into the proposal**:

| Allowed | Not allowed |
|---|---|
| Open-source **libraries** that run on our machines: PyTorch, NumPy, FastAPI, MediaPipe (local landmark extraction), librosa, ONNX Runtime | Calling hosted AI services: OpenAI, Google Cloud Speech, Azure Cognitive Services, hosted translation APIs |
| Public **datasets** (KSL datasets, Common Voice, etc.) under their licences | Sending user video, audio or text to third-party AI services |
| Architectures from papers, implemented and **trained by us** | Treating a downloaded end-to-end translator as "our model" |

**Decisions to record explicitly**, because each is a trade-off between "fully ours" and "works by the demo date":

| Component | Fully-ours route | Pragmatic bootstrap (still local, no API) | Recommendation |
|---|---|---|---|
| Hand/body landmarks | Train our own keypoint detector (very hard, needs huge labelled data) | MediaPipe Holistic/Hands **as a fixed sensor**, running locally | Use MediaPipe as the *sensor*. AIMA treats sensors as given; our AI starts at the percept. State this clearly. |
| Sign recognition | **Ours** (this is the core contribution) | — | Ours, trained from scratch plus self-supervised pretraining. |
| Gloss ⇄ English | **Ours** (small seq2seq transformer) | Rule-based fallback (also ours) | Ours. |
| ASR (speech → text) | Train a Conformer-CTC on Kenyan English/Swahili speech | Local open-weights model (e.g. Whisper-small) for the early prototype only | Prototype with the bootstrap, then replace it with our trained model and report the comparison as an experiment. |
| TTS (text → speech) | Train a small FastSpeech2/VITS-style model | Local open-weights TTS (e.g. Piper voices) | Same as ASR. TTS quality matters least to the research question. |
| Sign production | **Ours** (dictionary retrieval + neural blending) | — | Ours. |
| Embeddings for RAG | Train a small sentence encoder, or use BM25 (no neural net) | Local sentence-transformer | Start with BM25 (ours, simple), add a trained encoder later. |

All bootstrap components sit behind the **same interface** as the final ones (see `ModelAdapter`, §V-1). Swapping one is then a configuration change.

**CUDA clarification:** CUDA is not a language. It is NVIDIA's parallel computing platform and C/C++ extension for GPUs. We'll mostly use CUDA *through* PyTorch. §VIII-6 shows one hand-written CUDA kernel and where custom kernels actually pay off.

## 8. How the course material maps onto the build

| Material | Concept | Where it shows up |
|---|---|---|
| *Introduction to AI* (BBT 3201) | Thinking vs acting rationally; AI should complement humans | Design principle: the system *assists* communication and never pretends to be an interpreter in critical settings; human escalation path |
| *Intelligent Agents* (ICS 3202) / AIMA Ch. 2 | PEAS, environment types, agent structures | §3, the architecture rationale |
| *Problem Solving Agents* (BBT 3201) / AIMA Ch. 3–4 | State space, path cost, uninformed vs informed search, A*, heuristics | **Beam search** decoding over gloss hypotheses (§V-3) is informed search with a pruned frontier. Path cost = −log probability. Sign production path planning between poses |
| *Learning Agents & Data Mining* | Learning-agent components, **CRISP-DM** | §3.3 and the whole of Part III follows CRISP-DM phases |
| *Theories of Machine Learning* | Supervised vs unsupervised, clustering, PCA, distance measures | Supervised ISLR/CSLR; **k-means/DBSCAN on sign embeddings** to discover regional variants and annotation errors; PCA on landmarks for visualisation and dimensionality reduction; Euclidean vs cosine distance for kNN sign retrieval |
| *ML Cheat Sheet* | Logistic regression, random forest, XGBoost, k-NN, … | **Baselines** before deep learning: random forest on handcrafted handshape features, k-NN + DTW for isolated signs, logistic regression for intent classification. Report them. Examiners like baselines. |
| *Pre-trained Models: Past, Present, Future* (Han et al., 2021) | Transfer learning, self-supervised pre-training, masked modelling, fine-tuning | **Masked pose modelling** pre-training on unlabelled signing (§IV-2.3), then fine-tuning for ISLR/CSLR; pretraining the translator decoder on English text |
| *Guest lecture: Naive RAG* (notebook + slides) | Ingestion vs retrieval pipelines, chunking, embeddings, vector stores, "say so if not in context", naive-RAG failure modes | Venue knowledge bases (menus, hospital FAQs), **translation memory**, KSL dictionary lookup (§VII-3). We swap OpenAI/Pinecone for **local embeddings + pgvector** to respect the no-API rule, and we add relevance grading to fix naive RAG's failure modes |
| AIMA Ch. 13–15 | Uncertainty, Bayes, **HMMs**, temporal models, Viterbi | CTC ≈ HMM with a blank state; decoding = most-likely path; smoothing landmark noise with a **Kalman filter** (§III-3.4) |
| AIMA Ch. 16–17 | Utility, **value of information**, POMDPs | Clarification policy: ask only when the expected gain beats the cost of interrupting (§V-4) |
| AIMA Ch. 18, 20 | Learning from examples, EM, probabilistic models | Training methodology, calibration, n-gram LM |
| AIMA Ch. 22–23 | **N-gram LMs, smoothing**, MT, speech recognition | Gloss LM with Kneser–Ney smoothing (§IV-5), gloss⇄English MT (§IV-6), ASR (§IV-7) |
| AIMA Ch. 24–25 | Perception, robotics | Landmark perception; robot integration (§VII-4) |
| AIMA Ch. 26–27, *Sustainable AI* topic, van Wynsberghe (2021), Wu et al. (2022), Ren & Wierman (HBR 2024) | Ethics, AI for sustainability vs sustainability of AI, carbon and water costs | Part IX: efficient models, quantisation, carbon tracking, small models instead of giant ones |
| Brookings: *Algorithmic bias detection and mitigation* | Bias impact statements, audits, diverse stakeholders | Part IX: per-group evaluation, bias impact statement, Deaf community in the loop |
| *Sustainable development* literature review | Three pillars: economic, social, environmental | Frame the project under the social pillar (inclusion, SDG 10 *Reduced inequalities*, SDG 8 *Decent work*, linked to RQ on employment) |
| *AI and Art* | Generative AI, authorship | Sign *production* is generative. Signing avatars must be validated by Deaf signers, because generated signs can be "fluent-looking nonsense" |


---

# Part II — Architecture

## 1. High-level topology

```
                ┌──────────────────────────── Kotlin clients ────────────────────────────┐
                │  Signer phone      Hearing phone      Robot/Kiosk (Android)   Admin web │
                └──────┬──────────────────┬───────────────────┬──────────────────────┬───┘
                       │ WSS (landmarks,  │ WSS (audio PCM,   │ WSS + REST           │ HTTPS
                       │  text, control)  │  text, control)   │ (API key)            │
                ┌──────▼──────────────────▼───────────────────▼──────────────────────▼───┐
                │                    API / WebSocket GATEWAY  (FastAPI, uvicorn)          │
                │  auth · rate limit · room membership · protocol validation · fan-out    │
                └──────┬───────────────────────────┬───────────────────────────┬─────────┘
                       │ Redis Streams (jobs)      │ Redis Pub/Sub (room bus)  │ SQL
             ┌─────────▼─────────┐       ┌─────────▼─────────┐        ┌────────▼────────┐
             │ INFERENCE WORKERS │       │  ORCHESTRATOR     │        │ PostgreSQL 16   │
             │ (GPU / CPU pools) │◄─────►│ (session pipeline,│        │ + pgvector      │
             │  sign · asr · tts │       │  context, agent,  │        └─────────────────┘
             │  translate · prod │       │  clarification)   │        ┌─────────────────┐
             └─────────┬─────────┘       └─────────┬─────────┘        │ MinIO / S3      │
                       │ model files               │ tools            │ (clips, models, │
             ┌─────────▼─────────┐       ┌─────────▼─────────┐        │  datasets)      │
             │ MODEL REGISTRY    │       │ VENUE ADAPTERS    │        └─────────────────┘
             │ (MLflow + MinIO)  │       │ (menu, POS, robot)│        ┌─────────────────┐
             └───────────────────┘       └───────────────────┘        │ Prometheus,     │
                                                                       │ Grafana, Loki,  │
             ┌──────────────────────────────────────────────────┐     │ OpenTelemetry   │
             │ OFFLINE: training jobs (Celery/Arq + GPU node),  │     └─────────────────┘
             │ data pipeline, annotation service, evaluation    │
             └──────────────────────────────────────────────────┘
```

### 1.1 Why this split

* **Gateway ≠ inference.** The gateway is I/O-bound (thousands of sockets, async). Inference is compute-bound (GPU, batching). Mixing them in one process means a slow model blocks every socket. They communicate over **Redis Streams** (durable, consumer groups, acks).
* **Pub/Sub for rooms.** With several gateway replicas, participants of one room can sit on different replicas. Each replica subscribes to `room:{id}` channels for the rooms its sockets belong to (§VI-5).
* **Orchestrator** holds per-session logic: buffering, the context window, clarification state, agent calls. It's stateless across restarts because state lives in Redis with a TTL and in Postgres.
* **The model registry** decouples "which model version is live" from code deploys.

### 1.2 Latency budget (sign → text on the other phone)

| Stage | Budget (p95) |
|---|---|
| Client landmark extraction (on-device) | 30 ms/frame (not backend) |
| Uplink (frame batches of 4 at 30 fps) | 80 ms |
| Gateway validate + enqueue | 5 ms |
| Worker batching wait | ≤ 20 ms |
| CSLR encoder (chunk of 16 frames, GPU) | 15 ms |
| Streaming CTC beam step | 10 ms |
| Endpoint detection (end of utterance) | 300–500 ms of stillness (inherent) |
| Final rescoring + gloss→English | 40 ms |
| Broadcast to room | 10 ms |
| **Partial caption latency** | **≈ 250 ms** |
| **Final sentence after signer stops** | **≈ 600–900 ms** |

Speech → sign adds ASR (streaming, ~300 ms partials), text→gloss (~20 ms) and sign production (~50 ms for retrieval-based). The avatar playback itself takes as long as the signing does.

## 2. Technology choices

| Concern | Choice | Reason |
|---|---|---|
| Language | Python 3.12 | ML ecosystem; asyncio |
| Web framework | **FastAPI** + Uvicorn | Async, native WebSockets, Pydantic v2 validation, OpenAPI for the Kotlin team |
| Deep learning | **PyTorch 2.x** (+ `torch.compile`, AMP) | Research flexibility; CUDA access; ONNX export |
| Audio | torchaudio, librosa, soundfile, webrtcvad | Features, resampling, voice-activity detection |
| Landmarks (server-side, dataset mode) | MediaPipe Tasks (local) | Sensor; same as the Kotlin side for consistency |
| DB | PostgreSQL 16 + **pgvector** | Relational data plus vectors in one store (the lecture's pgvector option) |
| Migrations / ORM | SQLAlchemy 2.0 (async) + Alembic | Typed, async |
| Cache / bus / queue | **Redis 7** (Streams + Pub/Sub) | One dependency covers three needs |
| Background jobs | **Arq** (async Redis jobs) or Celery | Training/eval/ingestion jobs |
| Object storage | MinIO (S3-compatible) | Self-hosted; datasets, clips, model artefacts |
| Experiment tracking | MLflow (self-hosted) | Runs, metrics, registry |
| Serialization (real-time) | **MessagePack** frames over WS + JSON control | Compact binary for landmarks and audio |
| Auth | Argon2id passwords, JWT access + rotating refresh tokens, device API keys | §VI-2 |
| Observability | OpenTelemetry, Prometheus, Grafana, structlog | Per-stage latency is a research metric |
| Packaging | `uv` or Poetry, Docker, docker compose | Reproducible environments |
| Testing | pytest, pytest-asyncio, hypothesis, locust | Unit, property-based and load tests |

## 3. Repository layout

A monorepo with a clear split between the **service** (`app/`) and **ML research/training** (`ml/`). They share `ksl_core/`, which holds the pure-Python pieces both need: IR, landmark spec, normalisation and the vocabulary.

```
ksl-bridge-backend/
├── pyproject.toml
├── docker/
│   ├── gateway.Dockerfile
│   ├── worker-gpu.Dockerfile          # CUDA base image
│   └── compose.yaml
├── configs/
│   ├── app.yaml                        # service config (12-factor overrides via env)
│   ├── models.yaml                     # which model version serves which task
│   └── train/
│       ├── islr_transformer.yaml
│       ├── cslr_conformer_ctc.yaml
│       ├── mpm_pretrain.yaml
│       ├── gloss2text.yaml
│       ├── text2gloss.yaml
│       ├── asr_conformer.yaml
│       └── tts_fastspeech2.yaml
├── ksl_core/                           # shared, dependency-light
│   ├── ir.py                           # UtteranceIR & friends (Part I §4.2)
│   ├── landmarks.py                    # landmark spec, index maps, packing
│   ├── normalize.py                    # pose normalisation (Part III §3)
│   ├── vocab.py                        # gloss vocabulary, special tokens
│   └── text.py                         # English/Swahili normalisation, numbers
├── app/                                # the online service
│   ├── main.py                         # FastAPI factory
│   ├── settings.py
│   ├── deps.py                         # DI: db session, current user, redis
│   ├── db/
│   │   ├── base.py, session.py
│   │   ├── models/                     # SQLAlchemy ORM
│   │   └── migrations/                 # Alembic
│   ├── auth/
│   │   ├── passwords.py, tokens.py, api_keys.py, router.py, rbac.py
│   ├── rooms/
│   │   ├── router.py                   # REST: create/join/QR
│   │   ├── manager.py                  # local connection registry
│   │   └── bus.py                      # Redis pub/sub fan-out
│   ├── ws/
│   │   ├── protocol.py                 # message schemas & codec
│   │   ├── endpoint.py                 # /ws handler
│   │   └── backpressure.py
│   ├── pipeline/
│   │   ├── orchestrator.py             # per-utterance state machine
│   │   ├── context.py                  # dialogue context window
│   │   ├── clarify.py                  # value-of-information policy
│   │   └── endpointing.py              # end-of-utterance detection
│   ├── agent/
│   │   ├── dialogue.py                 # goal-based dialogue manager
│   │   ├── tools.py                    # tool registry (menu, order, faq)
│   │   └── adapters/                   # robot / POS adapters
│   ├── rag/
│   │   ├── ingest.py, chunking.py, bm25.py, embed.py, retrieve.py, grade.py
│   ├── feedback/                       # corrections → training queue
│   ├── datasets/                       # contribution uploads, consent
│   ├── observability/
│   └── routers/                        # users, venues, devices, admin, health
├── workers/
│   ├── runner.py                       # Redis Streams consumer w/ micro-batching
│   ├── adapters.py                     # ModelAdapter interface
│   ├── sign_worker.py
│   ├── asr_worker.py
│   ├── tts_worker.py
│   ├── translate_worker.py
│   └── production_worker.py
├── ml/                                 # research & training
│   ├── data/
│   │   ├── sources/                    # dataset-specific loaders (KSL datasets, own recordings)
│   │   ├── extract_landmarks.py        # video → landmarks (MediaPipe, offline)
│   │   ├── shards.py                   # WebDataset/Zarr writer
│   │   ├── augment.py
│   │   ├── splits.py                   # signer-independent splits
│   │   └── datasets.py                 # torch Datasets
│   ├── models/
│   │   ├── pose_encoder.py             # shared spatio-temporal encoder
│   │   ├── islr.py, cslr.py, mpm.py
│   │   ├── fingerspell.py
│   │   ├── seq2seq.py                  # gloss⇄text transformer
│   │   ├── asr.py, tts.py
│   │   └── production.py
│   ├── decoding/
│   │   ├── ctc_beam.py                 # prefix beam search + LM fusion
│   │   ├── ngram_lm.py                 # Kneser–Ney gloss LM
│   │   └── calibration.py              # temperature scaling
│   ├── train/
│   │   ├── loop.py                     # generic trainer (AMP, grad accumulation, EMA)
│   │   └── *.py                        # per task entrypoints
│   ├── eval/
│   │   ├── metrics.py                  # WER, BLEU, chrF, ECE
│   │   ├── robustness.py               # perturbation matrix
│   │   └── fairness.py                 # per-group slices
│   ├── export/                         # ONNX, TorchScript, TFLite for Kotlin
│   └── kernels/                        # custom CUDA (Part VIII §6)
├── tools/
│   ├── annotate/                       # annotation web tool backend
│   └── scripts/
└── tests/
    ├── unit/, integration/, load/, golden/
```

## 4. Configuration

```yaml
# configs/app.yaml
env: dev
http:
  cors_origins: ["*"]          # tighten in prod
auth:
  access_ttl_seconds: 900       # 15 min
  refresh_ttl_days: 30
  jwt_issuer: "ksl-bridge"
  jwt_audience: "ksl-bridge-clients"
ws:
  max_frame_bytes: 65536
  heartbeat_seconds: 15
  idle_timeout_seconds: 60
  max_inflight_landmark_batches: 8      # backpressure threshold per socket
pipeline:
  endpoint_still_ms: 450
  max_utterance_seconds: 20
  partial_every_frames: 8
  clarify_cost: 0.15                    # see Part V §4
retention:
  realtime_landmarks_days: 0            # do NOT store unless consented
  consented_clips_days: 1825
```

```yaml
# configs/models.yaml — the "serving map"; change versions here, not in code
sign:
  cslr:      { name: cslr_conformer_ctc, version: "2026.11.0", device: cuda, batch_max: 32, batch_wait_ms: 15 }
  islr:      { name: islr_transformer,    version: "2026.10.2", device: cuda }
  fingerspell: { name: fs_ctc,            version: "2026.10.0", device: cuda }
  lm:        { name: gloss_kn4,           version: "2026.11.0" }
  calibration: { temperature: 1.42 }
translate:
  gloss2text: { name: g2t_transformer, version: "2026.11.0", beam: 4 }
  text2gloss: { name: t2g_transformer, version: "2026.11.0", beam: 4, rule_fallback: true }
speech:
  asr: { name: asr_bootstrap_whisper_small, version: "local-1", device: cuda }   # → replace with asr_conformer_ctc
  tts: { name: tts_bootstrap_piper_en,      version: "local-1", device: cpu }
production:
  sign2pose: { name: dict_blend, version: "2026.11.0", dictionary: "s3://models/sign_dict/v3" }
```

```python
# app/settings.py
from functools import lru_cache
from pydantic import Field, PostgresDsn, RedisDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KSL_", env_file=".env", extra="ignore")

    env: str = "dev"
    database_url: PostgresDsn
    redis_url: RedisDsn = "redis://localhost:6379/0"
    s3_endpoint: str = "http://localhost:9000"
    s3_access_key: str
    s3_secret_key: str = Field(repr=False)
    jwt_private_key_pem: str = Field(repr=False)   # EdDSA (Ed25519) or RS256 private key
    jwt_public_key_pem: str
    access_ttl_seconds: int = 900
    refresh_ttl_days: int = 30


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
```


---

# Part III — Data (CRISP-DM)

The data work follows the six CRISP-DM phases from the *Learning Agents & Data Mining* topic. **Most of the project's success is decided here, not in the model code.**

```
 Business understanding → Data understanding → Data preparation → Modelling → Evaluation → Deployment
        (Part I)            (§1, §2)              (§3–§6)         (Part IV)    (Part VIII)   (Part VIII)
           ▲                                                                                     │
           └───────────────────── feedback loop: corrections, new vocabulary ◄──────────────────┘
```

## 1. Data understanding: sources

| Source | Type | Use | Notes / risks |
|---|---|---|---|
| **KSL word-based pose dataset** (2025, *Data in Brief*; Maseno/partners; reported ≈30k word/phrase videos with MediaPipe pose/3D coordinates) | Isolated signs, landmarks | ISLR pretraining and fine-tuning; sign dictionary for production | The Zenodo record indicates **restricted access**. Request it formally through your supervisor and read the licence. |
| **KSL videos on Kaggle** (multiple signers; listed CC0) | Isolated sign videos | ISLR; landmark extraction offline | Check signer IDs and consistency of the label taxonomy. |
| **Our own recordings** (§6) | **Continuous** sentences in service scenarios | CSLR, gloss⇄English pairs, evaluation | This is the novel contribution, and only we can produce it. |
| Other sign-language datasets (WLASL, PHOENIX-2014T, How2Sign, etc.) | Other languages | **Self-supervised pretraining only** (motion priors) | Never mix their labels into KSL. Different languages. Check each licence. |
| Mozilla Common Voice (English, Swahili), other Kenyan speech corpora | Speech | ASR training/eval | Kenyan-accented English is under-represented, so we record our own (§6.4). |
| English/Swahili text corpora | Text | LM pretraining for the translator decoder, Text→Gloss input distribution | Domain text such as menus and hospital phrases is especially useful. |

**What the data doesn't give us** (write this in the report): isolated-sign datasets don't contain co-articulation, the way signs blend into each other in continuous signing, and they don't contain natural service dialogues. Hence §6.

## 2. Landmark specification (shared with Kotlin)

MediaPipe Holistic outputs 33 pose + 468 face + 21 × 2 hand landmarks (543 total). We use a **107-point subset** that keeps what matters for sign language:

| Group | Count | Indices (MediaPipe) | Why |
|---|---|---|---|
| Left hand | 21 | all | Handshape |
| Right hand | 21 | all | Handshape |
| Pose (upper body) | 25 | 0–24 | Arm movement, location relative to the body |
| Face | 40 | lips outer/inner, eyebrows, eye corners, nose tip | Non-manual markers (questions, negation, mouthing) |

```python
# ksl_core/landmarks.py
from __future__ import annotations
import struct
from dataclasses import dataclass
import numpy as np

# Face subset (MediaPipe FaceMesh indices). Lips, brows, eye corners, nose.
FACE_IDX: tuple[int, ...] = (
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,     # outer lips (11)
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324,          # inner lips (10)
    70, 63, 105, 66, 107,                                  # left brow (5)
    336, 296, 334, 293, 300,                               # right brow (5)
    33, 133, 362, 263,                                     # eye corners (4)
    1, 4, 168, 152, 10,                                    # nose tip, nose, between eyes, chin, forehead (5)
)
assert len(FACE_IDX) == 40

N_LH, N_RH, N_POSE, N_FACE = 21, 21, 25, 40
N_POINTS = N_LH + N_RH + N_POSE + N_FACE           # 107
SL_LH = slice(0, 21)
SL_RH = slice(21, 42)
SL_POSE = slice(42, 67)
SL_FACE = slice(67, 107)

# Pose indices *within* our pose slice (MediaPipe pose numbering preserved 0..24)
POSE_NOSE, POSE_L_SHOULDER, POSE_R_SHOULDER = 0, 11, 12
POSE_L_ELBOW, POSE_R_ELBOW, POSE_L_WRIST, POSE_R_WRIST = 13, 14, 15, 16

SPEC_VERSION = 1


@dataclass(slots=True)
class LandmarkBatch:
    """T frames of landmarks. xyz: (T, 107, 3) float32; mask: (T, 107) bool (True = present)."""
    seq_start: int          # client frame counter of first frame
    t0_ms: int              # client timestamp of first frame (monotonic clock)
    fps: float
    xyz: np.ndarray
    mask: np.ndarray


# ---- Binary wire format (little-endian) ----------------------------------------------
# header: magic 'KSLB' | u8 spec_version | u8 flags | u16 n_frames | u32 seq_start | u64 t0_ms | f32 fps
# body:   n_frames * 107 * 3 float16 coordinates, then n_frames * 107 bits mask (packed, padded to bytes)
_HDR = struct.Struct("<4sBBHIQf")


def pack(batch: LandmarkBatch) -> bytes:
    t = batch.xyz.shape[0]
    hdr = _HDR.pack(b"KSLB", SPEC_VERSION, 0, t, batch.seq_start, batch.t0_ms, batch.fps)
    coords = np.nan_to_num(batch.xyz, nan=0.0).astype("<f2").tobytes()
    bits = np.packbits(batch.mask.reshape(-1).astype(np.uint8)).tobytes()
    return hdr + coords + bits


def unpack(buf: bytes) -> LandmarkBatch:
    magic, ver, _flags, t, seq, t0, fps = _HDR.unpack_from(buf, 0)
    if magic != b"KSLB" or ver != SPEC_VERSION:
        raise ValueError("bad landmark frame header")
    off = _HDR.size
    n = t * N_POINTS * 3
    coords = np.frombuffer(buf, dtype="<f2", count=n, offset=off).astype(np.float32)
    off += n * 2
    nbits = t * N_POINTS
    bits = np.unpackbits(np.frombuffer(buf, dtype=np.uint8, offset=off, count=(nbits + 7) // 8))[:nbits]
    mask = bits.astype(bool).reshape(t, N_POINTS)
    xyz = coords.reshape(t, N_POINTS, 3)
    xyz[~mask] = np.nan
    return LandmarkBatch(seq, t0, fps, xyz, mask)
```

Size: 107 × 3 × 2 B ≈ 642 B per frame plus 14 B of mask. At 30 fps that's about 20 KB/s, far less than video. Clients send **batches of 4 frames**, about 7.5 messages per second.

> **Contract rule:** Kotlin and Python must use identical index maps. Put `FACE_IDX` and the slice definitions in a shared JSON (`landmark_spec_v1.json`) that both sides generate code from. Bump `SPEC_VERSION` on any change.

## 3. Data preparation I: normalisation

Raw coordinates depend on where the person stands, how far they are from the camera, body size and camera roll. Normalisation removes these **nuisance variables** so the model learns the sign, not the signer.

### 3.1 Steps

1. **Origin:** midpoint of the shoulders → (0, 0, 0).
2. **Scale:** divide by shoulder width. Bodies of all sizes and distances become comparable.
3. **Roll:** rotate in the x–y plane so the shoulder line is horizontal.
4. **Handedness canonicalisation (optional, per model):** if the signer is left-dominant, mirror x and swap hands, so the "dominant hand" is always channel 0. Detect dominance from motion energy over the first N frames or take it from the user profile.
5. **Hand-local features:** each hand also centred on its wrist and scaled by palm size (wrist → middle-finger MCP). This captures **handshape** independent of location.
6. **Gap filling:** linear interpolation for gaps ≤ 6 frames. Longer gaps stay missing (mask = 0).
7. **Smoothing:** One-Euro filter or a constant-velocity Kalman filter (AIMA §15.4) to remove jitter without lag.

### 3.2 Code

```python
# ksl_core/normalize.py
from __future__ import annotations
import numpy as np
from .landmarks import (SL_LH, SL_RH, SL_POSE, SL_FACE, POSE_L_SHOULDER, POSE_R_SHOULDER)

EPS = 1e-6


def _interp_gaps(x: np.ndarray, max_gap: int = 6) -> np.ndarray:
    """x: (T, P, 3) with NaNs. Linearly interpolate along time per point/coord for short gaps."""
    T = x.shape[0]
    out = x.copy()
    t = np.arange(T)
    flat = out.reshape(T, -1)
    for j in range(flat.shape[1]):
        col = flat[:, j]
        nan = np.isnan(col)
        if nan.all() or not nan.any():
            continue
        good = ~nan
        filled = np.interp(t, t[good], col[good])
        # only fill runs whose length ≤ max_gap and which are bounded on both sides
        run_start = None
        for i in range(T + 1):
            is_nan = i < T and nan[i]
            if is_nan and run_start is None:
                run_start = i
            elif not is_nan and run_start is not None:
                if run_start > 0 and i < T and (i - run_start) <= max_gap:
                    col[run_start:i] = filled[run_start:i]
                run_start = None
        flat[:, j] = col
    return out


def normalize_sequence(xyz: np.ndarray, mirror: bool = False) -> np.ndarray:
    """
    xyz: (T, 107, 3) raw image-normalised coords (x,y in [0,1], z relative), NaN where missing.
    returns (T, 107, 3) body-centred, scale- and roll-normalised coordinates.
    """
    x = _interp_gaps(xyz.astype(np.float32))
    pose = x[:, SL_POSE]
    ls, rs = pose[:, POSE_L_SHOULDER], pose[:, POSE_R_SHOULDER]           # (T, 3)
    # robust per-sequence reference: median over frames (signers mostly don't move their torso)
    center = np.nanmedian((ls + rs) / 2.0, axis=0)                          # (3,)
    width = np.nanmedian(np.linalg.norm((ls - rs)[:, :2], axis=-1))
    if not np.isfinite(width) or width < EPS:
        width = 0.25  # fallback: typical shoulder width in image units
    x = (x - center) / width

    # roll correction from the median shoulder vector
    v = np.nanmedian((ls - rs)[:, :2], axis=0)
    angle = np.arctan2(v[1], v[0])
    c, s = np.cos(-angle), np.sin(-angle)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    x[..., :2] = x[..., :2] @ R.T

    if mirror:  # make the dominant hand canonical "right"
        x[..., 0] *= -1.0
        lh = x[:, SL_LH].copy()
        x[:, SL_LH] = x[:, SL_RH]
        x[:, SL_RH] = lh
    return x


def hand_local(hand: np.ndarray) -> np.ndarray:
    """hand: (T, 21, 3) → wrist-centred, palm-scaled coords (handshape only)."""
    wrist = hand[:, 0:1]
    palm = np.linalg.norm(hand[:, 9] - hand[:, 0], axis=-1)[:, None, None]   # wrist → middle MCP
    return (hand - wrist) / np.maximum(palm, EPS)


# Finger bones (parent, child) for direction features
HAND_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
              (0, 17), (17, 18), (18, 19), (19, 20)]


def build_features(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    x: normalised (T, 107, 3). Returns (features (T, F), mask (T, F)).
    Features = positions + velocities + hand-local shape + bone directions + key distances.
    """
    T = x.shape[0]
    lh, rh, face = x[:, SL_LH], x[:, SL_RH], x[:, SL_FACE]
    pos = x.reshape(T, -1)                                                  # 321
    vel = np.diff(x, axis=0, prepend=x[:1]).reshape(T, -1)                  # 321
    shape = np.concatenate([hand_local(lh).reshape(T, -1), hand_local(rh).reshape(T, -1)], -1)  # 126
    bones = []
    for hand in (lh, rh):
        for a, b in HAND_BONES:
            d = hand[:, b] - hand[:, a]
            bones.append(d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), EPS))
    bones = np.concatenate(bones, axis=-1)                                  # 2*20*3 = 120
    nose = x[:, 42 + 0]
    dists = np.stack([
        np.linalg.norm(lh[:, 0] - rh[:, 0], axis=-1),                       # hand-to-hand
        np.linalg.norm(lh[:, 8] - nose, axis=-1),                           # index tip → nose
        np.linalg.norm(rh[:, 8] - nose, axis=-1),
        np.linalg.norm(face[:, 21] - face[:, 26], axis=-1),                 # brow spread proxy
    ], axis=-1)                                                             # 4
    feats = np.concatenate([pos, vel, shape, bones, dists], axis=-1).astype(np.float32)
    mask = np.isfinite(feats)
    return np.nan_to_num(feats, nan=0.0), mask
```

Feature dimension: 321 + 321 + 126 + 120 + 4 = **892**. The model also receives the **mask**, so "hand not visible" is information rather than zeros pretending to be data.

> **Ablation to run (report material):** raw coords vs normalised vs normalised + hand-local + bones. This shows that the preprocessing contributes to robustness (RQ2).

### 3.3 Streaming variant

Real-time streams can't compute a per-sequence median. Keep an **exponential moving estimate** of shoulder centre and width per session (`alpha = 0.05`), initialised from the first 15 frames where both shoulders are visible. Put the same function in `ksl_core` so training and serving share the same code. Training/serving skew is a classic silent bug.

### 3.4 Smoothing with a constant-velocity Kalman filter (AIMA §15.4)

```python
# ksl_core/smoothing.py
import numpy as np


class ConstantVelocityKalman:
    """Independent 1-D CV Kalman filters for every coordinate. State = [pos, vel]."""
    def __init__(self, n: int, q: float = 1e-3, r: float = 4e-3, dt: float = 1 / 30):
        self.F = np.array([[1, dt], [0, 1]], dtype=np.float32)
        self.Q = q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]], dtype=np.float32)
        self.R = r
        self.x = np.zeros((n, 2), np.float32)
        self.P = np.tile(np.eye(2, dtype=np.float32), (n, 1, 1))
        self.init = np.zeros(n, bool)

    def step(self, z: np.ndarray) -> np.ndarray:
        """z: (n,) observation, NaN = missing. Returns filtered positions (n,)."""
        # predict
        self.x = self.x @ self.F.T
        self.P = self.F @ self.P @ self.F.T + self.Q
        obs = np.isfinite(z)
        first = obs & ~self.init
        self.x[first, 0], self.x[first, 1] = z[first], 0.0
        self.init |= obs
        upd = obs & ~first
        if upd.any():
            P = self.P[upd]
            S = P[:, 0, 0] + self.R
            K = P[:, :, 0] / S[:, None]                     # (k, 2)
            y = z[upd] - self.x[upd, 0]
            self.x[upd] += K * y[:, None]
            self.P[upd] = P - K[:, :, None] * P[:, 0:1, :]
        out = self.x[:, 0].copy()
        out[~self.init] = np.nan
        return out
```

## 4. Data preparation II: storage format

* **Per sample:** `features.npy` (float16, T × 892), `mask.npy` (packed bits), `meta.json` (`sample_id`, `signer_id`, `glosses`, `english`, `source`, `consent_id`, `fps`, `dominant_hand`, `split`).
* **Sharding:** WebDataset `.tar` shards of about 1 GB in MinIO (`s3://datasets/ksl/v{N}/train-{000000..}.tar`). This gives streaming reads with no millions-of-small-files problem.
* **Versioning:** every dataset build is immutable and gets a manifest with a hash of its inputs and the code commit. Models record the dataset version they were trained on.

## 5. Data preparation III: augmentation (the "small mistakes" simulator)

Each augmentation corresponds to a real-world mistake from Part I §5. **Apply them on landmarks before feature building**, with random probabilities.

```python
# ml/data/augment.py
from __future__ import annotations
import numpy as np
from ksl_core.landmarks import SL_LH, SL_RH

rng = np.random.default_rng()


def rotate3d(x, max_deg=15):                 # camera viewpoint change
    a, b, c = np.deg2rad(rng.uniform(-max_deg, max_deg, 3))
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return x @ (Rz @ Ry @ Rx).T.astype(np.float32)


def scale_shift(x, s=0.1, t=0.1):            # distance from camera, framing
    return x * rng.uniform(1 - s, 1 + s) + rng.uniform(-t, t, size=(1, 1, 3)).astype(np.float32)


def time_warp(x, lo=0.7, hi=1.3):            # faster / slower signing
    T = x.shape[0]
    newT = max(4, int(round(T * rng.uniform(lo, hi))))
    src = np.linspace(0, T - 1, newT)
    i0 = np.floor(src).astype(int); i1 = np.minimum(i0 + 1, T - 1); w = (src - i0)[:, None, None]
    return (1 - w) * x[i0] + w * x[i1]


def finger_jitter(x, sigma=0.02):            # slightly imperfect handshape
    y = x.copy()
    for sl in (SL_LH, SL_RH):
        y[:, sl] += rng.normal(0, sigma, size=y[:, sl].shape).astype(np.float32)
    return y


def hand_dropout(x, p_frame=0.1, max_span=8):  # hand leaves frame / detector misses
    y = x.copy()
    T = x.shape[0]
    for sl in (SL_LH, SL_RH):
        if rng.random() < 0.5:
            n_spans = rng.integers(1, 3)
            for _ in range(n_spans):
                L = int(rng.integers(1, max_span + 1)); s = int(rng.integers(0, max(1, T - L)))
                y[s:s + L, sl] = np.nan
        drop = rng.random(T) < p_frame
        y[drop, sl] = np.nan
    return y


def mirror(x):                               # left-handed signers
    y = x.copy()
    y[..., 0] *= -1
    lh = y[:, SL_LH].copy(); y[:, SL_LH] = y[:, SL_RH]; y[:, SL_RH] = lh
    return y


def temporal_crop(x, min_keep=0.85):         # late start / early stop of capture
    T = x.shape[0]
    keep = int(T * rng.uniform(min_keep, 1.0))
    s = int(rng.integers(0, T - keep + 1))
    return x[s:s + keep]


def augment(x: np.ndarray, allow_mirror: bool) -> np.ndarray:
    if rng.random() < 0.8: x = rotate3d(x)
    if rng.random() < 0.8: x = scale_shift(x)
    if rng.random() < 0.6: x = time_warp(x)
    if rng.random() < 0.5: x = finger_jitter(x)
    if rng.random() < 0.5: x = hand_dropout(x)
    if rng.random() < 0.3: x = temporal_crop(x)
    if allow_mirror and rng.random() < 0.5: x = mirror(x)
    return x
```

> ⚠️ **Mirroring caveat:** some signs may be lexically or spatially directional (pointing to a referent's location, for example). Ask a Deaf KSL consultant whether mirroring changes meaning for any vocabulary items, and disable mirroring for those glosses (`allow_mirror` per gloss).

**Continuous-signing synthesis** (to bootstrap CSLR before enough real sentences exist): concatenate isolated-sign clips into pseudo-sentences sampled from the gloss LM, with **cross-fade interpolation over 4–8 frames** between clips to mimic co-articulation. Label the result as synthetic and always **evaluate on real continuous data only**.

## 6. Our own data collection (the novel dataset)

### 6.1 Ethics first

* Ethics approval through the university's research ethics committee before any recording.
* Informed consent **in KSL**: a signed consent video, plus a written form. Plain-language explanation of what is stored, for how long, who can see it, and the right to withdraw (§IX-3).
* **Pay Deaf contributors** for their time. Involve the Kenya National Association of the Deaf (KNAD) or a local Deaf association and KSL interpreters from the start: *nothing about us without us*.

### 6.2 Recording protocol

| Item | Specification |
|---|---|
| Signers | ≥ 20 (aim 30+), balanced by gender, age band, region and dominant hand; include hard-of-hearing KSL users and fluent hearing CODAs separately labelled |
| Devices | 3+ phone models (low/mid/high end); front camera at 30 fps; 720p |
| Conditions | Indoor bright, indoor dim, outdoor shade, cluttered background, seated/standing, distance 0.8–2 m |
| Content | (a) **Service scripts:** restaurant (60 sentences), hospital reception (60), bank/office (40), transport (40); (b) **free dialogue** role-play with a hearing participant; (c) isolated core vocabulary (500 glosses) × 3 reps; (d) fingerspelling of names/numbers |
| Annotation | Gloss sequence (time-aligned where possible), English translation, Swahili translation (optional), non-manual tags, quality flag |

### 6.3 Annotation workflow and quality

1. Recording uploaded via the contribution endpoint (Part VI §8), landmarks extracted server-side.
2. **Annotator A** writes glosses + English. **Annotator B** (Deaf KSL user) reviews them independently.
3. Measure **inter-annotator agreement** on a 10% overlap: gloss-level edit-distance agreement and **Cohen's κ** for non-manual tags. Target κ ≥ 0.7.
4. Maintain a **gloss style guide** (`docs/gloss_conventions.md`): naming rules (`IX-1`, `IX-2`, `IX-3` for pointing; `FS:JOHN` for fingerspelling; `NUM:2`; `CL:VEHICLE-MOVE` for classifiers), variant suffixes (`EAT.v2` for regional variants).
5. Disagreements go to adjudication by a senior Deaf annotator. Never resolve them by majority hearing vote.

Unsupervised learning helps here (*Theories of ML*, clustering): embed all clips of a gloss with the ISLR encoder, run **DBSCAN**, and outlier clips are either annotation errors or **regional variants**. Variants should get their own variant labels, not be deleted.

### 6.4 Speech data

Record the hearing participants in the same sessions (Kenyan English, Swahili, code-switched) to get **in-domain ASR test data** with restaurant/hospital noise. Add noise augmentation from recorded ambient noise of real venues.

## 7. Signer-independent splits (non-negotiable)

If the same signer appears in train and test, accuracy is inflated and you're measuring memorisation of people. Split **by signer**.

```python
# ml/data/splits.py
from __future__ import annotations
import hashlib
from collections import defaultdict


def signer_split(samples: list[dict], test_frac=0.15, val_frac=0.10, seed="ksl-v1") -> dict[str, list[dict]]:
    """Deterministic split by signer_id, stratified so every split has both dominant hands."""
    by_signer: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        by_signer[s["signer_id"]].append(s)

    def h(sid: str) -> float:
        return int(hashlib.sha256(f"{seed}:{sid}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF

    out = {"train": [], "val": [], "test": []}
    for sid, items in by_signer.items():
        r = h(sid)
        split = "test" if r < test_frac else "val" if r < test_frac + val_frac else "train"
        out[split].extend(items)
    return out
```

Then check the split for vocabulary coverage (every test gloss must appear in train) and for group balance (gender, region, handedness). If the hash split is unbalanced, change the seed or do an explicit stratified assignment.

## 8. PyTorch datasets

```python
# ml/data/datasets.py
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset
from ksl_core.normalize import normalize_sequence, build_features
from ksl_core.vocab import Vocab
from .augment import augment


class PoseSeqDataset(Dataset):
    """Generic: returns features, mask, and targets (gloss ids)."""
    def __init__(self, items: list[dict], vocab: Vocab, train: bool, max_len: int = 384,
                 no_mirror_glosses: set[str] | None = None):
        self.items, self.vocab, self.train, self.max_len = items, vocab, train, max_len
        self.no_mirror = no_mirror_glosses or set()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        xyz = np.load(it["landmarks_path"]).astype(np.float32)        # (T,107,3) raw with NaN
        if self.train:
            allow_mirror = not (set(it["glosses"]) & self.no_mirror)
            xyz = augment(xyz, allow_mirror)
        x = normalize_sequence(xyz, mirror=(it.get("dominant_hand") == "left"))
        feats, mask = build_features(x)
        if feats.shape[0] > self.max_len:                               # uniform subsample
            idx = np.linspace(0, feats.shape[0] - 1, self.max_len).astype(int)
            feats, mask = feats[idx], mask[idx]
        target = [self.vocab.stoi[g] for g in it["glosses"]]
        return torch.from_numpy(feats), torch.from_numpy(mask), torch.tensor(target, dtype=torch.long)


def collate(batch):
    feats, masks, targets = zip(*batch)
    lens = torch.tensor([f.shape[0] for f in feats])
    T, F = int(lens.max()), feats[0].shape[1]
    X = torch.zeros(len(feats), T, F)
    M = torch.zeros(len(feats), T, F, dtype=torch.bool)
    pad = torch.ones(len(feats), T, dtype=torch.bool)                   # True = padding
    for i, (f, m) in enumerate(zip(feats, masks)):
        X[i, : f.shape[0]] = f; M[i, : f.shape[0]] = m; pad[i, : f.shape[0]] = False
    tgt_lens = torch.tensor([len(t) for t in targets])
    Y = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=-100)
    return X, M, pad, lens, Y, tgt_lens
```

```python
# ksl_core/vocab.py
from __future__ import annotations
import json
from dataclasses import dataclass, field

SPECIALS = ["<blank>", "<pad>", "<unk>", "<bos>", "<eos>"]   # <blank> must be index 0 for CTC


@dataclass
class Vocab:
    itos: list[str]
    stoi: dict[str, int] = field(init=False)

    def __post_init__(self):
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    @classmethod
    def build(cls, glosses: list[str]) -> "Vocab":
        uniq = sorted(set(glosses) - set(SPECIALS))
        return cls(SPECIALS + uniq)

    def encode(self, seq: list[str]) -> list[int]:
        unk = self.stoi["<unk>"]
        return [self.stoi.get(g, unk) for g in seq]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.itos[i] for i in ids if i >= len(SPECIALS)]

    def save(self, path: str):
        json.dump(self.itos, open(path, "w"))

    @classmethod
    def load(cls, path: str) -> "Vocab":
        return cls(json.load(open(path)))
```


---

# Part IV — Models

Model map (each row is its own training job, model card and registry entry):

| # | Model | Input → Output | Architecture | Trained on |
|---|---|---|---|---|
| M0 | Baselines | landmarks → gloss | k-NN + DTW; Random Forest on handshape features | Isolated KSL |
| M1 | **Pose encoder** (shared) | features → embeddings | Conv subsampling + Conformer/Transformer | Self-supervised MPM (§2.3) |
| M2 | **ISLR** | isolated clip → gloss | M1 + attention pooling + cosine classifier | Isolated KSL |
| M3 | **CSLR** | continuous stream → gloss sequence | M1 (streaming) + CTC head | Own continuous + synthetic |
| M4 | Fingerspelling | stream → letters | Small CTC model on hand features | Own fingerspelling data |
| M5 | Gloss LM | gloss history → next gloss | Kneser–Ney 4-gram (+ optional tiny Transformer LM) | Gloss corpora + scripts |
| M6 | Gloss→Text | gloss seq → English | Transformer seq2seq | Gloss–English pairs (+ synthetic) |
| M7 | Text→Gloss | English → gloss seq | Rules + Transformer seq2seq | Same pairs, reversed |
| M8 | NLU | English/gloss → intent + slots | TF-IDF + LogReg baseline → joint Transformer | Service-script dialogues |
| M9 | ASR | audio → text | Conformer-CTC + BPE + LM fusion | Common Voice + own Kenyan recordings |
| M10 | TTS | text → audio | FastSpeech2-style + HiFi-GAN vocoder | Recorded voice talent |
| M11 | Sign production | gloss seq → pose seq | Dictionary retrieval + transition blending (+ learned in-betweening) | Isolated + continuous KSL |
| M12 | Non-manual classifier | face features → question/negation/intensity | Small temporal CNN | Own annotated data |

Start with **M0 baselines** in week 1. They give you a number to beat and they test the data pipeline end-to-end.

## 1. Baselines (M0): classical ML from the course

### 1.1 k-NN with Dynamic Time Warping

A classic, explainable baseline and a direct application of the distance measures from *Theories of ML*. DTW aligns two sequences of different speeds, which makes it naturally robust to "signed faster or slower".

```python
# ml/models/baselines.py
from __future__ import annotations
import numpy as np


def dtw_distance(a: np.ndarray, b: np.ndarray, window: int | None = None) -> float:
    """a: (Ta, F), b: (Tb, F). Euclidean frame cost, Sakoe–Chiba band."""
    Ta, Tb = len(a), len(b)
    w = max(window or max(Ta, Tb), abs(Ta - Tb))
    D = np.full((Ta + 1, Tb + 1), np.inf, dtype=np.float32)
    D[0, 0] = 0.0
    for i in range(1, Ta + 1):
        lo, hi = max(1, i - w), min(Tb, i + w)
        cost = np.linalg.norm(b[lo - 1:hi] - a[i - 1], axis=1)
        for k, j in enumerate(range(lo, hi + 1)):
            D[i, j] = cost[k] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[Ta, Tb] / (Ta + Tb))


class KnnDTW:
    def __init__(self, k: int = 3, window: int = 20):
        self.k, self.window = k, window

    def fit(self, X: list[np.ndarray], y: list[str]):
        self.X, self.y = X, y
        return self

    def predict_proba(self, q: np.ndarray) -> dict[str, float]:
        d = np.array([dtw_distance(q, x, self.window) for x in self.X])
        idx = np.argsort(d)[: self.k]
        w = np.exp(-d[idx] / (d[idx].mean() + 1e-6))
        out: dict[str, float] = {}
        for i, wi in zip(idx, w):
            out[self.y[i]] = out.get(self.y[i], 0) + wi
        s = sum(out.values())
        return {k: v / s for k, v in sorted(out.items(), key=lambda kv: -kv[1])}
```

Use a **PCA-reduced** feature space (e.g. 892 → 64 dimensions, fitted on training data) to make DTW roughly 10× faster. That's the dimensionality-reduction use case from the course. DTW is O(Ta·Tb) and doesn't scale to large vocabularies, which motivates the neural models.

### 1.2 Random Forest on summary features

Per clip, compute mean, std, min and max of hand-local shape and bone features over time, plus motion energy. Then `sklearn.ensemble.RandomForestClassifier(n_estimators=500)`. It's interpretable: feature importance shows which fingers matter. Report it beside DTW and the deep models.

## 2. Shared pose encoder (M1) and ISLR (M2)

### 2.1 Encoder

```python
# ml/models/pose_encoder.py
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Subsample(nn.Module):
    """Temporal conv subsampling ×factor (CSLR uses 4: 30 fps → 7.5 Hz tokens)."""
    def __init__(self, d_in: int, d: int, factor: int):
        super().__init__()
        layers, c = [], d_in
        for _ in range(int(math.log2(factor))):
            layers += [nn.Conv1d(c, d, kernel_size=5, stride=2, padding=2), nn.GELU()]
            c = d
        self.net = nn.Sequential(*layers) if layers else nn.Conv1d(d_in, d, 1)
        self.factor = factor

    def forward(self, x, pad):              # x: (B,T,D) pad: (B,T) True=pad
        y = self.net(x.transpose(1, 2)).transpose(1, 2)
        if self.factor > 1:
            pad = pad[:, :: self.factor][:, : y.shape[1]]
        return y, pad


class ConvModule(nn.Module):
    """Conformer convolution module: captures local motion patterns."""
    def __init__(self, d: int, k: int = 15, p: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.pw1 = nn.Linear(d, 2 * d)
        self.dw = nn.Conv1d(d, d, k, padding=k // 2, groups=d)
        self.bn = nn.BatchNorm1d(d)
        self.pw2 = nn.Linear(d, d)
        self.do = nn.Dropout(p)

    def forward(self, x, pad):
        y = F.glu(self.pw1(self.ln(x)), dim=-1)
        y = y.masked_fill(pad[..., None], 0.0).transpose(1, 2)
        y = F.silu(self.bn(self.dw(y))).transpose(1, 2)
        return x + self.do(self.pw2(y))


class ConformerBlock(nn.Module):
    def __init__(self, d: int, heads: int, ff: int = 4, p: float = 0.1):
        super().__init__()
        self.ff1 = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, ff * d), nn.SiLU(), nn.Dropout(p), nn.Linear(ff * d, d))
        self.ln_att = nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, dropout=p, batch_first=True)
        self.conv = ConvModule(d, p=p)
        self.ff2 = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, ff * d), nn.SiLU(), nn.Dropout(p), nn.Linear(ff * d, d))
        self.ln_out = nn.LayerNorm(d)
        self.do = nn.Dropout(p)

    def forward(self, x, pad, attn_mask=None):
        x = x + 0.5 * self.ff1(x)
        h = self.ln_att(x)
        a, _ = self.att(h, h, h, key_padding_mask=pad, attn_mask=attn_mask, need_weights=False)
        x = x + self.do(a)
        x = self.conv(x, pad)
        x = x + 0.5 * self.ff2(x)
        return self.ln_out(x)


class SinusoidalPE(nn.Module):
    def __init__(self, d: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len)[:, None]
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x, offset: int = 0):
        return x + self.pe[offset: offset + x.shape[1]]


class PoseEncoder(nn.Module):
    def __init__(self, d_feat: int = 892, d: int = 256, layers: int = 8, heads: int = 4,
                 subsample: int = 4, p: float = 0.1):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(2 * d_feat, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(p))
        self.sub = Subsample(d, d, subsample)
        self.pe = SinusoidalPE(d)
        self.blocks = nn.ModuleList([ConformerBlock(d, heads, p=p) for _ in range(layers)])
        self.d = d

    def forward(self, x, m, pad, attn_mask=None):
        """x,m: (B,T,F) features and presence mask; pad: (B,T) True=pad."""
        h = self.inp(torch.cat([x * m, m.float()], dim=-1))      # mask is an input, not just a multiplier
        h, pad = self.sub(h, pad)
        h = self.pe(h)
        for blk in self.blocks:
            h = blk(h, pad, attn_mask)
        return h, pad
```

### 2.2 ISLR head with a cosine classifier

A **cosine (normalised) classifier with additive angular margin** (ArcFace-style) produces compact class clusters. That improves robustness to small handshape deviations, and the embedding doubles as a **prototype** for open-set rejection and k-NN retrieval.

```python
# ml/models/islr.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .pose_encoder import PoseEncoder


class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q = nn.Linear(d, 1)

    def forward(self, h, pad):
        w = self.q(h).squeeze(-1).masked_fill(pad, float("-inf")).softmax(-1)
        return torch.einsum("bt,btd->bd", w, h)


class ArcMarginHead(nn.Module):
    def __init__(self, d, n_classes, s=30.0, m=0.2):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_classes, d) * 0.01)
        self.s, self.m = s, m

    def forward(self, z, y=None):
        cos = F.linear(F.normalize(z), F.normalize(self.W))           # (B,C)
        if y is None:
            return self.s * cos
        theta = torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))
        target = torch.cos(theta + self.m)
        onehot = F.one_hot(y, cos.shape[1]).bool()
        return self.s * torch.where(onehot, target, cos)


class ISLR(nn.Module):
    def __init__(self, n_classes, **enc_kw):
        super().__init__()
        self.enc = PoseEncoder(subsample=2, **enc_kw)
        self.pool = AttnPool(self.enc.d)
        self.proj = nn.Sequential(nn.Linear(self.enc.d, 256), nn.BatchNorm1d(256))
        self.head = ArcMarginHead(256, n_classes)

    def embed(self, x, m, pad):
        h, p = self.enc(x, m, pad)
        return self.proj(self.pool(h, p))

    def forward(self, x, m, pad, y=None):
        return self.head(self.embed(x, m, pad), y)
```

Training: AdamW (lr 1e-3, wd 0.05), cosine schedule with 5% warm-up, label smoothing 0.1, batch 128, mixed precision, 100 epochs, EMA of weights (decay 0.999). Metric: top-1/top-5 on signer-independent test.

### 2.3 Self-supervised pre-training: Masked Pose Modelling (MPM)

This applies *Pre-trained Models: Past, Present, Future*. The masked language modelling idea of BERT is transferred to motion. Labelled KSL is scarce, but **unlabelled signing is much cheaper**: all collected video, including unlabelled free dialogues, plus other sign-language datasets used *only for motion priors*.

```python
# ml/models/mpm.py
import torch
import torch.nn as nn
from .pose_encoder import PoseEncoder


class MaskedPoseModel(nn.Module):
    """Mask random time spans; reconstruct normalised hand/body features at masked positions."""
    def __init__(self, d_feat=892, **enc_kw):
        super().__init__()
        self.enc = PoseEncoder(d_feat=d_feat, subsample=1, **enc_kw)
        self.mask_token = nn.Parameter(torch.zeros(d_feat))
        self.dec = nn.Linear(self.enc.d, d_feat)

    @staticmethod
    def span_mask(B, T, ratio=0.4, span=6, device="cpu"):
        m = torch.zeros(B, T, dtype=torch.bool, device=device)
        n_spans = max(1, int(T * ratio / span))
        starts = torch.randint(0, max(1, T - span), (B, n_spans), device=device)
        for k in range(span):
            m.scatter_(1, (starts + k).clamp(max=T - 1), True)
        return m

    def forward(self, x, m, pad):
        B, T, _ = x.shape
        tm = self.span_mask(B, T, device=x.device) & ~pad
        x_in = torch.where(tm[..., None], self.mask_token.expand_as(x), x)
        m_in = m & ~tm[..., None]
        h, _ = self.enc(x_in, m_in, pad)
        rec = self.dec(h)
        valid = tm[..., None] & m                       # only score observed values in masked frames
        loss = (torch.abs(rec - x) * valid).sum() / valid.sum().clamp(min=1)
        return loss
```

Then load `mpm.enc` weights into ISLR/CSLR encoders and fine-tune. **Report the ablation** of from-scratch vs MPM-pretrained against the amount of labelled data (10%, 25%, 50%, 100%). This curve is one of the strongest results you can show.

### 2.4 Open-set rejection (sign not in vocabulary)

Store a **prototype** per gloss: the mean normalised embedding of its training clips. At inference:

* `max_cos < τ_reject` (tuned on validation with held-out "unknown" signs) → return `<unk>` and trigger the *"Please fingerspell"* clarification.
* Margin `cos_1 − cos_2 < τ_ambig` → ambiguous, so return top-k for context rescoring.

## 3. Continuous recognition: CSLR (M3)

### 3.1 Why CTC

Continuous signing has no per-frame labels: we know the sentence's gloss sequence, not where each sign starts. **Connectionist Temporal Classification** adds a `<blank>` symbol and sums over every alignment of the label sequence to the frames. It's the neural counterpart of an HMM forward algorithm (AIMA §15.3), and it needs no segmentation.

```python
# ml/models/cslr.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .pose_encoder import PoseEncoder


class CSLR(nn.Module):
    def __init__(self, vocab_size: int, inter_ctc_layer: int = 4, **enc_kw):
        super().__init__()
        self.enc = PoseEncoder(subsample=4, **enc_kw)
        self.ctc = nn.Linear(self.enc.d, vocab_size)
        self.inter_layer = inter_ctc_layer           # intermediate CTC regularisation
        self.inter = nn.Linear(self.enc.d, vocab_size)

    def forward(self, x, m, pad, attn_mask=None):
        enc = self.enc
        h = enc.inp(torch.cat([x * m, m.float()], -1))
        h, pad = enc.sub(h, pad)
        h = enc.pe(h)
        inter_logits = None
        for i, blk in enumerate(enc.blocks):
            h = blk(h, pad, attn_mask)
            if i + 1 == self.inter_layer:
                inter_logits = self.inter(h)
        return self.ctc(h), inter_logits, pad

    @staticmethod
    def loss(logits, inter, pad, targets, tgt_lens, inter_w=0.3):
        out_lens = (~pad).sum(1)
        def ctc(lg):
            lp = lg.float().log_softmax(-1).transpose(0, 1)          # (T,B,C)
            tg = torch.cat([t[:l] for t, l in zip(targets, tgt_lens)])
            return F.ctc_loss(lp, tg, out_lens, tgt_lens, blank=0, zero_infinity=True)
        main = ctc(logits)
        return main + inter_w * ctc(inter) if inter is not None else main
```

**Subsampling ×4** turns 30 fps into 7.5 tokens per second. Signing produces roughly 1.5–3 signs per second, so the CTC constraint T_out ≥ label length (+ repeats) holds comfortably.

### 3.2 Streaming: chunked attention

For live captions the encoder must not look into the future indefinitely. Use **chunk-wise causal attention**: each frame sees its own chunk plus `L` previous chunks. Train with random chunk sizes ("dynamic chunk training") so a single model serves both streaming and full-context modes.

```python
def chunk_attn_mask(T: int, chunk: int, left_chunks: int, device) -> torch.Tensor:
    """Bool mask (T,T); True = NOT allowed to attend (PyTorch MHA convention)."""
    idx = torch.arange(T, device=device)
    c = idx // chunk
    allowed = (c[None, :] <= c[:, None]) & (c[None, :] >= c[:, None] - left_chunks)
    return ~allowed

# training: chunk = random.choice([4, 8, 16, full]) per batch (in subsampled frames)
```

At serving time the worker keeps a **cache** of the last `left_chunks × chunk` encoder inputs per session and re-runs the encoder on `[cache ‖ new chunk]`. That's a simpler alternative to full KV-caching of Conformer state, and fast enough at d = 256.

### 3.3 Training recipe

1. Initialise the encoder from MPM.
2. Curriculum: (a) synthetic concatenated sentences + isolated clips as 1-gloss sentences, (b) mix in real continuous data with upsampling, (c) fine-tune on real continuous data only.
3. Loss: CTC + 0.3 × intermediate CTC. SpecAugment-style **time masking** on features (in addition to §III-5).
4. Evaluation: **WER** (word error rate on glosses) = (S + D + I) / N, greedy and beam + LM.

## 4. Fingerspelling (M4)

Fingerspelling covers names, places, menu items not in the vocabulary, and numbers. It uses the same CTC recipe with a letter vocabulary (`A–Z`, digits, `<blank>`) and dominant-hand features only (hand-local shape + bones, 63 + 60 dims), in a smaller encoder (d = 128, 4 layers). A **lexicon-constrained decoder** with the venue's menu names, staff names and local places (via RAG, §VII-3) turns noisy letter sequences into likely words.

## 5. Gloss language model (M5): Kneser–Ney n-gram (AIMA Ch. 22)

The LM supplies **context**: P(next gloss | previous glosses). That is exactly what resolves the brainstorm example (`I WANT …` → `FOOD` beats `FOOT`). Use n-grams first. They're tiny, fast, explainable, and work with small corpora, where neural LMs overfit.

```python
# ml/decoding/ngram_lm.py
from __future__ import annotations
import math
from collections import Counter, defaultdict

BOS, EOS = "<s>", "</s>"


class KneserNeyLM:
    """Interpolated Kneser–Ney with a fixed discount (AIMA §22.1; Chen & Goodman 1999)."""
    def __init__(self, order: int = 4, discount: float = 0.75):
        self.n, self.D = order, discount

    def fit(self, sentences: list[list[str]]) -> "KneserNeyLM":
        n = self.n
        raw = [defaultdict(Counter) for _ in range(n + 1)]   # raw[k][ctx][w], len(ctx) = k-1
        vocab = set()
        for s in sentences:
            toks = [BOS] * (n - 1) + s + [EOS]
            vocab.update(s); vocab.add(EOS)
            for i in range(n - 1, len(toks)):
                for k in range(1, n + 1):
                    raw[k][tuple(toks[i - k + 1:i])][toks[i]] += 1
        # highest order uses raw counts; lower orders use continuation counts N1+(• ctx w)
        self.tab = [None] * (n + 1)
        self.tab[n] = raw[n]
        for k in range(n - 1, 0, -1):
            cont = defaultdict(Counter)
            for ctx, words in raw[k + 1].items():
                for w in words:
                    cont[ctx[1:]][w] += 1
            self.tab[k] = cont
        self.tot = [None] + [{c: sum(ws.values()) for c, ws in self.tab[k].items()} for k in range(1, n + 1)]
        self.V = len(vocab) + 1                               # + <unk>
        self._cache: dict = {}
        return self

    def _p(self, k: int, ctx: tuple, w: str) -> float:
        if k == 0:
            return 1.0 / self.V
        key = (k, ctx, w)
        if key in self._cache:
            return self._cache[key]
        lower = self._p(k - 1, ctx[1:], w)
        table = self.tab[k].get(ctx)
        if not table:
            p = lower
        else:
            total, types = self.tot[k][ctx], len(table)
            p = max(table.get(w, 0) - self.D, 0) / total + self.D * types / total * lower
        self._cache[key] = p
        return p

    def logp(self, history: list[str] | tuple, w: str) -> float:
        ctx = tuple(([BOS] * (self.n - 1) + list(history))[-(self.n - 1):])
        return math.log(self._p(self.n, ctx, w))

    def perplexity(self, sentences: list[list[str]]) -> float:
        lp, N = 0.0, 0
        for s in sentences:
            for i, w in enumerate(s + [EOS]):
                lp += self.logp(s[:i], w); N += 1
        return math.exp(-lp / N)
```

**Training text for the gloss LM**: glosses from all annotated sentences; Text→Gloss outputs on large English domain text (menus, hospital dialogue), which gives a synthetic but domain-relevant corpus; and the service scripts. Build **domain-specific LMs** (restaurant, hospital, general) and interpolate them at runtime by the room's venue type: `P = λ_domain·P_domain + (1−λ)·P_general`.

## 6. Gloss → English translation (M6)

### 6.1 Why it's needed

The CSLR output `IX-1 WANT BURGER TWO WATER ONE` is not English. KSL, like other sign languages, has its own word order and grammar: time references often come first, topic–comment structures are used, and question words often appear at the end. *Verify each grammatical claim with your Deaf KSL consultants and cite KSL linguistics sources. Don't assume English or ASL rules apply.* M6 learns this mapping.

### 6.2 Model

A standard encoder–decoder Transformer (Vaswani et al., 2017), small: 4 + 4 layers, d = 256, 4 heads, which is enough for a low-resource pair.

* Source vocab: glosses (the CSLR vocab + `FS:*` passthrough handled by copying).
* Target vocab: **BPE** (8k merges) trained by us on English + Swahili text with the local `tokenizers` or `sentencepiece` library.
* **Copy mechanism / placeholder trick**: replace `FS:JOHN`, `NUM:2` by placeholders `<FS0>`, `<NUM0>` before translation and restore them after. The model never has to learn to spell names.

```python
# ml/models/seq2seq.py
import math
import torch
import torch.nn as nn


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d=256, heads=4, enc_layers=4, dec_layers=4, p=0.1, pad_id=1):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab, d, padding_idx=pad_id)
        self.tgt_emb = nn.Embedding(tgt_vocab, d, padding_idx=pad_id)
        self.pos = nn.Embedding(512, d)
        self.tf = nn.Transformer(d, heads, enc_layers, dec_layers, 4 * d, p, batch_first=True, norm_first=True)
        self.out = nn.Linear(d, tgt_vocab)
        self.out.weight = self.tgt_emb.weight                  # weight tying
        self.d, self.pad = d, pad_id

    def _embed(self, emb, ids):
        pos = torch.arange(ids.shape[1], device=ids.device)
        return emb(ids) * math.sqrt(self.d) + self.pos(pos)

    def forward(self, src, tgt_in):
        causal = nn.Transformer.generate_square_subsequent_mask(tgt_in.shape[1], device=src.device, dtype=torch.bool)
        h = self.tf(self._embed(self.src_emb, src), self._embed(self.tgt_emb, tgt_in),
                    tgt_mask=causal,
                    src_key_padding_mask=src.eq(self.pad), tgt_key_padding_mask=tgt_in.eq(self.pad),
                    memory_key_padding_mask=src.eq(self.pad))
        return self.out(h)

    @torch.no_grad()
    def beam_search(self, src, bos, eos, beam=4, max_len=64, len_alpha=0.7):
        """src: (1,S). Returns list of (tokens, score) sorted best-first."""
        mem = self.tf.encoder(self._embed(self.src_emb, src), src_key_padding_mask=src.eq(self.pad))
        beams = [([bos], 0.0)]
        done = []
        for _ in range(max_len):
            cand = []
            for toks, sc in beams:
                t = torch.tensor([toks], device=src.device)
                causal = nn.Transformer.generate_square_subsequent_mask(t.shape[1], device=src.device, dtype=torch.bool)
                h = self.tf.decoder(self._embed(self.tgt_emb, t), mem, tgt_mask=causal,
                                    memory_key_padding_mask=src.eq(self.pad))
                lp = self.out(h[:, -1]).log_softmax(-1).squeeze(0)
                top = lp.topk(beam)
                for v, i in zip(top.values.tolist(), top.indices.tolist()):
                    cand.append((toks + [i], sc + v))
            cand.sort(key=lambda x: -x[1])
            beams = []
            for toks, sc in cand[: beam * 2]:
                (done if toks[-1] == eos else beams).append((toks, sc))
                if len(beams) == beam:
                    break
            if len(done) >= beam or not beams:
                break
        done = done or beams
        norm = lambda t, s: s / (((5 + len(t)) / 6) ** len_alpha)   # GNMT length penalty
        return sorted(((t, norm(t, s)) for t, s in done), key=lambda x: -x[1])
```

### 6.3 The data problem and how to solve it

Gloss–English pairs are the scarcest resource. Use four strategies together:

1. **Annotated real pairs** from §III-6. Gold quality, small (a few thousand).
2. **Rule-based synthetic pairs.** Write a KSL→English grammar template system with Deaf consultants: for each service intent, generate many gloss sequences and their English realisations by slot filling (`IX-1 WANT {FOOD} {NUM}` ↔ "I want {num} {food}s").
3. **Back-translation.** Train M7 (Text→Gloss) on the pairs, run it on large English domain text to create (synthetic gloss, real English) pairs, then train M6 on them. The target side is real English, so fluency improves.
4. **Decoder pretraining.** Pretrain the decoder as an English/Swahili LM on domain text (the PTM paper's "pre-train then fine-tune"), then fine-tune the full seq2seq.

Evaluate with **BLEU-4** and **chrF++** (sacreBLEU, run locally) and with human adequacy/fluency ratings (1–5) from bilingual KSL users.

### 6.4 Using N-best inputs

Don't translate only the top CSLR hypothesis. Translate the **top-3** and pick the final output by `score = λ1·CSLR_logp + λ2·MT_logp + λ3·NLU_intent_confidence`. This *joint* decision is one of the cheapest robustness wins (§V-3).

## 7. Text → Gloss (M7)

Two-stage design for reliability:

```
English text ─► normalise (numbers, contractions, Sheng map) ─► NEURAL t2g (M7, beam 4)
                                                           └─► RULE t2g (fallback + validator)
                                       ─► gloss validator: every gloss must exist in the sign dictionary;
                                          unknown → synonym lookup (RAG over dictionary) → FS: fallback
```

```python
# app/pipeline/text2gloss_rules.py
import re

DROP = {"a", "an", "the", "is", "are", "am", "was", "were", "be", "to", "of", "do", "does", "did"}
WH = {"what", "where", "when", "who", "why", "how", "which"}
TIME = {"today", "tomorrow", "yesterday", "now", "later", "morning", "evening", "tonight"}
PRON = {"i": "IX-1", "me": "IX-1", "my": "POSS-1", "you": "IX-2", "your": "POSS-2",
        "he": "IX-3", "she": "IX-3", "they": "IX-3-PL", "we": "IX-1-PL"}
NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def simple_lemma(w: str) -> str:
    for suf, rep in (("ies", "y"), ("es", ""), ("s", ""), ("ing", ""), ("ed", "")):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)] + rep
    return w


def rule_text2gloss(text: str, dictionary: set[str], synonyms: dict[str, str]) -> list[str]:
    """
    Baseline ordering heuristic: TIME first, then the rest, WH-word last.
    THIS IS A PLACEHOLDER GRAMMAR. Validate the ordering rules with Deaf KSL consultants.
    """
    words = re.findall(r"[a-zA-Z']+|\d+", text.lower())
    time_part, body, wh = [], [], []
    for w in words:
        if w in DROP:
            continue
        if w in PRON:
            body.append(PRON[w]); continue
        if w.isdigit() or w in NUM_WORDS:
            body.append(f"NUM:{w if w.isdigit() else NUM_WORDS[w]}"); continue
        g = simple_lemma(w).upper()
        g = synonyms.get(g, g)
        target = time_part if w in TIME else wh if w in WH else body
        if g in dictionary:
            target.append(g)
        else:
            target.append(f"FS:{w.upper()}")          # fingerspell unknown words
    return time_part + body + wh
```

**Safety rule for critical domains** (hospital, allergies, medication): when the neural and rule outputs disagree on content glosses, or a content word falls back to `FS:`, show **captions alongside the avatar**, and add a "show the text" button for the hearing party's original words.

## 8. NLU: intents and slots (M8)

The agent layer (Part VII) needs structured meaning. Work in two stages.

**Baseline (week 2):** a classic pipeline straight from the ML cheat sheet.

```python
# ml/models/nlu_baseline.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union

def build_intent_clf():
    feats = make_union(
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2),   # robust to ASR/CSLR typos
    )
    return make_pipeline(feats, LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"))
```

Train it on **both** English text and gloss strings, so the agent can work even when M6 translation is weak. Slots come from gazetteers (menu items, numbers) via RAG lookup.

**Upgrade:** a joint intent + slot model (small Transformer encoder with a CRF or token-classification head) trained on the annotated service dialogues. Report the comparison.

## 9. ASR (M9)

### 9.1 Pipeline

```
PCM16 16 kHz ─► WebRTC VAD ─► log-Mel (80 bins, 25 ms / 10 ms) ─► SpecAugment (train)
           ─► Conformer-CTC (12 layers, d=256, BPE-1k vocab, ×4 subsampling)
           ─► CTC prefix beam search + domain n-gram LM (same decoder as CSLR!) ─► text
```

That the decoder is reused between ASR and CSLR is a nice architectural point: both are "continuous signal → token sequence under uncertainty". The CTC beam decoder in §V-3 is **modality-agnostic**.

```python
# ml/models/asr_features.py
import torch
import torchaudio

mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=400, win_length=400,
                                           hop_length=160, n_mels=80)

def logmel(wave: torch.Tensor) -> torch.Tensor:          # wave: (N,) float32 in [-1,1]
    x = mel(wave).clamp(min=1e-10).log()                 # (80, T)
    return ((x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-5)).T   # (T, 80)

spec_aug = torch.nn.Sequential(
    torchaudio.transforms.FrequencyMasking(27),
    torchaudio.transforms.TimeMasking(40, p=0.05),
)
```

The model is the **same `ConformerBlock`** stack as `PoseEncoder`, with an 80-dim input. Reuse the code.

### 9.2 Kenyan specifics

* Train on **Kenyan-accented English + Swahili** jointly with a shared BPE. Code-switching is common, so don't force a single-language model.
* Add a Sheng/colloquial lexicon to the LM text.
* Noise augmentation with real restaurant/hospital ambience at SNR 0–20 dB.
* Until the trained model beats it on our Kenyan test set, the **bootstrap adapter** (a local open-weights ASR) serves. Report both. The comparison is itself a finding.

## 10. TTS (M10)

| Stage | Plan |
|---|---|
| Text normalisation | Numbers, currency (KSh), abbreviations, Sheng. Shared with `ksl_core/text.py` |
| G2P | English: CMUdict-style lexicon + a small seq2seq G2P for OOV. **Swahili is highly phonemic**, so a rule-based G2P works well. That's a nice, cheap, explainable component |
| Acoustic model | FastSpeech2-style: phoneme encoder, duration/pitch/energy predictors (durations from forced alignment), mel decoder |
| Vocoder | HiFi-GAN (V2 size, ~1M params) trained on the same voice |
| Data | 4–8 hours from one consenting Kenyan voice talent per language (paid, with a written licence), recorded in a quiet room |
| Evaluation | MOS from 15+ listeners, intelligibility via ASR-WER on the TTS output |

TTS is the component least central to the research questions. Keep the bootstrap TTS until M10 is trained, and prioritise it last.

## 11. Sign production (M11): gloss → pose sequence for the avatar

**Never generate photorealistic video** (deepfake risk, compute cost, uncanny valley). Generate **pose sequences** that the Kotlin app retargets onto a 3D avatar skeleton, or draws as a stylised 2D figure.

### 11.1 Stage 1: dictionary retrieval + transition blending (deterministic, works on day one)

1. **Build the sign dictionary.** For each gloss, choose the **medoid clip**: the clip with minimum total DTW distance to all others of that gloss from different signers. That's clustering in practice. Trim rest frames at start and end with a motion-energy threshold.
2. **Concatenate** the clips for the gloss sequence.
3. **Blend transitions** with cubic Hermite interpolation over `k = 6` frames between the end of one sign and the start of the next. This removes "teleporting hands".
4. **Non-manual overlay:** for yes/no questions, blend raised-brow face landmarks over the last sign; for WH-questions, furrowed brows. Templates come from annotated data.
5. **Timing:** speed factor per user preference (0.75×–1.25×).
6. **Fingerspelling** for `FS:` tokens: concatenate letter handshapes with short holds.

```python
# ml/models/production.py
from __future__ import annotations
import numpy as np


def hermite_blend(a_end: np.ndarray, a_vel: np.ndarray, b_start: np.ndarray, b_vel: np.ndarray, k: int) -> np.ndarray:
    """Cubic Hermite between two poses (107,3) with end/start velocities; returns (k,107,3)."""
    t = np.linspace(0, 1, k + 2)[1:-1][:, None, None]
    h00, h10 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t
    h01, h11 = -2 * t**3 + 3 * t**2, t**3 - t**2
    return h00 * a_end + h10 * a_vel * k + h01 * b_start + h11 * b_vel * k


def trim_rest(clip: np.ndarray, thr: float = 0.02) -> np.ndarray:
    v = np.linalg.norm(np.diff(clip[:, :42], axis=0), axis=-1).mean(-1)   # hand motion energy
    active = np.where(v > thr)[0]
    if len(active) == 0:
        return clip
    return clip[max(0, active[0] - 2): active[-1] + 3]


def produce(glosses: list[str], dictionary: dict[str, np.ndarray], fs_letters: dict[str, np.ndarray],
            k: int = 6, speed: float = 1.0, question: str | None = None,
            face_templates: dict[str, np.ndarray] | None = None) -> np.ndarray:
    clips: list[np.ndarray] = []
    for g in glosses:
        if g.startswith("FS:"):
            for ch in g[3:]:
                if ch in fs_letters:
                    clips.append(np.repeat(fs_letters[ch][-1:], 6, axis=0))  # brief hold per letter
        elif g in dictionary:
            clips.append(trim_rest(dictionary[g]))
    if not clips:
        return np.zeros((0, 107, 3), np.float32)
    out = [clips[0]]
    for nxt in clips[1:]:
        prev = out[-1]
        a_vel = prev[-1] - prev[-2] if len(prev) > 1 else np.zeros_like(prev[-1])
        b_vel = nxt[1] - nxt[0] if len(nxt) > 1 else np.zeros_like(nxt[0])
        out.append(hermite_blend(prev[-1], a_vel, nxt[0], b_vel, k))
        out.append(nxt)
    seq = np.concatenate(out, axis=0)
    if question and face_templates and question in face_templates:      # brows over the final sign
        tpl = face_templates[question]                                     # (40,3) target face offsets
        n = min(len(seq), len(clips[-1]))
        w = np.linspace(0, 1, n)[:, None, None]
        seq[-n:, 67:107] = (1 - w) * seq[-n:, 67:107] + w * (seq[-n:, 67:107] + tpl)
    if speed != 1.0:                                                       # resample in time
        T = len(seq); newT = max(2, int(T / speed))
        src = np.linspace(0, T - 1, newT); i0 = np.floor(src).astype(int); i1 = np.minimum(i0 + 1, T - 1)
        w = (src - i0)[:, None, None]
        seq = (1 - w) * seq[i0] + w * seq[i1]
    return seq.astype(np.float32)
```

### 11.2 Stage 2: learned in-betweening

Train a small Transformer that takes `(end of sign A, start of sign B, gloss ids)` and predicts the `k` transition frames. Ground truth comes from the real continuous recordings (§III-6), where real co-articulation is visible. That's more natural than Hermite curves.

### 11.3 Evaluation of production

* **Back-translation score:** feed the produced pose sequence into **our own CSLR** and measure WER against the input glosses. It's automatic and cheap, but optimistic.
* **Human evaluation (essential):** Deaf KSL users rate comprehension (they write what they understood) and naturalness (1–5). This is the real metric. Generated signing can look fluent and still mean nothing.

## 12. Generic training loop

```python
# ml/train/loop.py
from __future__ import annotations
import copy
import math
import time
import mlflow
import torch


class EMA:
    def __init__(self, model, decay=0.999):
        self.m = copy.deepcopy(model).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.d = decay

    @torch.no_grad()
    def update(self, model):
        for e, p in zip(self.m.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(self.d).add_(p.detach(), alpha=1 - self.d)
            else:
                e.copy_(p)


def cosine_lr(step, total, warmup, base):
    if step < warmup:
        return base * step / max(1, warmup)
    q = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * q))


def train(model, loss_fn, train_dl, val_fn, epochs, lr=1e-3, wd=0.05, accum=1, clip=1.0,
          device="cuda", run_name="run", patience=10):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.98))
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
    ema = EMA(model)
    total = epochs * len(train_dl) // accum
    step, best, bad = 0, float("inf"), 0
    with mlflow.start_run(run_name=run_name):
        for ep in range(epochs):
            model.train(); t0 = time.time()
            for i, batch in enumerate(train_dl):
                batch = [b.to(device, non_blocking=True) if torch.is_tensor(b) else b for b in batch]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                    loss = loss_fn(model, batch) / accum
                scaler.scale(loss).backward()
                if (i + 1) % accum == 0:
                    for g in opt.param_groups:
                        g["lr"] = cosine_lr(step, total, int(0.05 * total), lr)
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                    ema.update(model); step += 1
                    if step % 50 == 0:
                        mlflow.log_metric("train_loss", loss.item() * accum, step=step)
            val = val_fn(ema.m)                      # e.g. {"wer": 0.23}
            mlflow.log_metrics({f"val_{k}": v for k, v in val.items()}, step=step)
            mlflow.log_metric("epoch_seconds", time.time() - t0, step=step)
            key = next(iter(val.values()))
            if key < best:
                best, bad = key, 0
                torch.save(ema.m.state_dict(), f"checkpoints/{run_name}_best.pt")
            else:
                bad += 1
                if bad >= patience:
                    break
    return best
```

Log to MLflow per run: config, git commit, dataset version, and **energy/carbon** (CodeCarbon, Part IX-4).


---

# Part V — Real-time inference and error tolerance

## 1. Model adapters: one interface, swappable implementations

Every model is served through a narrow interface. Bootstrap models (local open-weights ASR, say) and our trained models are interchangeable through `configs/models.yaml`.

```python
# workers/adapters.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ModelInfo:
    task: str            # "cslr" | "asr" | "tts" | "g2t" | "t2g" | "sign2pose" | "nlu"
    name: str
    version: str
    device: str


class ModelAdapter(ABC):
    info: ModelInfo

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def infer_batch(self, inputs: list[Any]) -> list[Any]:
        """Pure function of inputs → outputs. Stateful streaming adapters receive session state in inputs."""

    def warmup(self) -> None:
        """Run dummy inputs so CUDA kernels/compilation happen before real traffic."""


REGISTRY: dict[str, type[ModelAdapter]] = {}

def register(name: str):
    def deco(cls):
        REGISTRY[name] = cls
        return cls
    return deco
```

```python
# workers/sign_worker.py (excerpt)
import numpy as np
import torch
from ksl_core.normalize import build_features
from ml.models.cslr import CSLR
from .adapters import ModelAdapter, ModelInfo, register


@register("cslr_conformer_ctc")
class CSLRAdapter(ModelAdapter):
    def __init__(self, cfg):
        self.info = ModelInfo("cslr", cfg["name"], cfg["version"], cfg["device"])
        self.cfg = cfg

    def load(self):
        ckpt = torch.load(f"/models/cslr/{self.info.version}/model.pt", map_location="cpu")
        self.model = CSLR(vocab_size=ckpt["vocab_size"], **ckpt["enc_kw"]).eval().to(self.info.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.T = torch.tensor(self.cfg.get("temperature", 1.0))
        if self.info.device.startswith("cuda"):
            self.model = torch.compile(self.model, mode="reduce-overhead", dynamic=True)

    @torch.inference_mode()
    def infer_batch(self, inputs):
        """inputs: list of dicts {xyz_window: (T,107,3) normalised, new_from: int}. Returns log-probs for new tokens."""
        feats = [build_features(i["xyz_window"]) for i in inputs]
        T = max(f.shape[0] for f, _ in feats)
        B, F = len(feats), feats[0][0].shape[1]
        X = torch.zeros(B, T, F); M = torch.zeros(B, T, F, dtype=torch.bool); P = torch.ones(B, T, dtype=torch.bool)
        for b, (f, m) in enumerate(feats):
            X[b, : len(f)] = torch.from_numpy(f); M[b, : len(f)] = torch.from_numpy(m); P[b, : len(f)] = False
        dev = self.info.device
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev.startswith("cuda")):
            logits, _, pad = self.model(X.to(dev), M.to(dev), P.to(dev))
        lp = (logits.float() / self.T.to(dev)).log_softmax(-1).cpu().numpy()
        out = []
        for b, inp in enumerate(inputs):
            valid = int((~pad[b]).sum())
            out.append(lp[b, inp["new_from"] // 4: valid])          # only the newly covered tokens
        return out
```

## 2. Workers: Redis Streams consumers with micro-batching

GPUs are efficient only with batches. Many sessions each sending small chunks are **micro-batched**: wait up to `batch_wait_ms` or until `batch_max` jobs, run one forward pass, and reply to each session.

```python
# workers/runner.py
from __future__ import annotations
import asyncio
import time
import msgpack
import redis.asyncio as aioredis
from .adapters import ModelAdapter


class StreamWorker:
    def __init__(self, redis: aioredis.Redis, stream: str, group: str, consumer: str,
                 adapter: ModelAdapter, batch_max: int = 32, batch_wait_ms: int = 15):
        self.r, self.stream, self.group, self.consumer = redis, stream, group, consumer
        self.adapter, self.batch_max, self.wait = adapter, batch_max, batch_wait_ms / 1000

    async def ensure_group(self):
        try:
            await self.r.xgroup_create(self.stream, self.group, id="$", mkstream=True)
        except aioredis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    async def run(self):
        await self.ensure_group()
        self.adapter.load(); self.adapter.warmup()
        while True:
            batch: list[tuple[str, dict]] = []
            deadline = time.monotonic() + self.wait
            while len(batch) < self.batch_max:
                timeout_ms = max(1, int((deadline - time.monotonic()) * 1000))
                resp = await self.r.xreadgroup(self.group, self.consumer, {self.stream: ">"},
                                               count=self.batch_max - len(batch), block=timeout_ms)
                for _, entries in resp or []:
                    for msg_id, fields in entries:
                        batch.append((msg_id, msgpack.unpackb(fields[b"p"], raw=False)))
                if time.monotonic() >= deadline:
                    break
            if not batch:
                continue
            t0 = time.perf_counter()
            # drop stale jobs (client already moved on) — deadline set by the orchestrator
            now_ms = int(time.time() * 1000)
            live = [(i, j) for i, j in batch if j.get("deadline_ms", now_ms + 1) > now_ms]
            try:
                outs = await asyncio.to_thread(self.adapter.infer_batch, [j["input"] for _, j in live])
                pipe = self.r.pipeline()
                for (msg_id, job), out in zip(live, outs):
                    pipe.publish(job["reply_to"], msgpack.packb({"job_id": job["job_id"], "output": out},
                                                                default=_np_default, use_bin_type=True))
                await pipe.execute()
            finally:
                await self.r.xack(self.stream, self.group, *[i for i, _ in batch])
            INFER_LATENCY.labels(self.adapter.info.task).observe(time.perf_counter() - t0)
            BATCH_SIZE.labels(self.adapter.info.task).observe(len(live))


def _np_default(o):
    import numpy as np
    if isinstance(o, np.ndarray):
        return {"__nd__": True, "dtype": str(o.dtype), "shape": o.shape, "data": o.tobytes()}
    raise TypeError(type(o))

# Prometheus metrics defined in app/observability/metrics.py and imported here
from app.observability.metrics import INFER_LATENCY, BATCH_SIZE  # noqa: E402
```

**Recovery:** a worker that crashes leaves un-acked messages in the PEL (pending entries list). A janitor task runs `XAUTOCLAIM` on messages idle > 5 s and either re-processes them or, if `deadline_ms` has passed, acks and drops them. Real-time results that arrive late are worse than useless.

## 3. The streaming sign pipeline (orchestrator)

### 3.1 Per-session state machine

```
            ┌──────────┐ hands enter / motion ┌───────────┐  stillness ≥ 450 ms   ┌────────────┐
   ───────► │  IDLE    │ ───────────────────► │ SIGNING   │ ────────────────────► │ FINALISING │
            └──────────┘                      └─────┬─────┘                       └─────┬──────┘
                 ▲                                  │ every 8 frames:                   │ rescoring, MT, NLU,
                 │                                  │ encoder chunk → beam step         │ clarification check
                 │                                  ▼ → partial caption (revision++)    ▼
                 │                            broadcast "partial"                 broadcast "final"
                 └──────────────────────────────────────────────────────────────────────┘
```

**Endpointing** (end of utterance): hand motion energy below threshold for 450 ms, or hands dropped to rest position (below the hips), or an explicit "done" tap. Implement it as a small classifier trained on annotated utterance boundaries if the heuristic misfires. An HMM over {signing, pause, rest} states is a good course-aligned choice.

```python
# app/pipeline/endpointing.py
import numpy as np
from ksl_core.landmarks import SL_LH, SL_RH


class Endpointer:
    def __init__(self, fps=30, still_ms=450, energy_thr=0.015, rest_y=1.6):
        self.need = int(still_ms / 1000 * fps)
        self.thr, self.rest_y = energy_thr, rest_y
        self.still, self.active, self.prev = 0, False, None

    def update(self, frame_norm: np.ndarray) -> str | None:
        """frame_norm: (107,3) normalised. Returns 'start', 'end' or None."""
        hands = np.concatenate([frame_norm[SL_LH], frame_norm[SL_RH]])
        visible = np.isfinite(hands).all(-1)
        at_rest = (not visible.any()) or np.nanmean(hands[visible][:, 1]) > self.rest_y  # y grows downward
        energy = 0.0
        if self.prev is not None and visible.any():
            d = hands - self.prev
            energy = float(np.nanmean(np.linalg.norm(d, axis=-1)))
        self.prev = hands
        moving = energy > self.thr and not at_rest
        if not self.active and moving:
            self.active, self.still = True, 0
            return "start"
        if self.active:
            self.still = 0 if moving else self.still + 1
            if self.still >= self.need:
                self.active = False
                return "end"
        return None
```

### 3.2 CTC prefix beam search with LM shallow fusion (streaming)

This is **informed search** from *Problem Solving Agents* (AIMA Ch. 3–4). States are gloss prefixes, path cost is −log P(acoustic) − α·log P_LM − β·length, and the frontier is pruned to the best *B* states (**beam search**). It's incomplete, but fast and good in practice.

```python
# ml/decoding/ctc_beam.py
from __future__ import annotations
import math
from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np

NEG_INF = -float("inf")


def lse(*xs: float) -> float:
    m = max(xs)
    if m == NEG_INF:
        return NEG_INF
    return m + math.log(sum(math.exp(x - m) for x in xs))


@dataclass
class Beam:
    pb: float = NEG_INF          # log P(prefix, ending in blank)
    pnb: float = NEG_INF         # log P(prefix, ending in non-blank)

    def acoustic(self) -> float:
        return lse(self.pb, self.pnb)


@dataclass
class CTCPrefixBeamSearch:
    itos: list[str]
    lm: object | None = None                 # needs .logp(history: list[str], w: str)
    beam: int = 8
    alpha: float = 0.6                       # LM weight
    beta: float = 1.0                        # insertion bonus (counters LM's short-sentence preference)
    prune_logp: float = -8.0                 # skip tokens with very low frame prob
    blank: int = 0
    skip_ids: frozenset = frozenset({1, 3, 4})  # <pad>,<bos>,<eos> are never emitted; <unk> is kept on purpose
    context_bias: dict[str, float] = field(default_factory=dict)   # gloss → log-boost (domain/RAG)
    beams: dict[tuple, Beam] = field(default_factory=lambda: {(): Beam(pb=0.0)})
    lm_cache: dict[tuple, float] = field(default_factory=lambda: {(): 0.0})  # prefix → alpha*LM + beta*len + bias

    def _extend_lm(self, prefix: tuple, new: tuple) -> None:
        if new in self.lm_cache:
            return
        g = self.itos[new[-1]]
        s = self.beta + self.context_bias.get(g, 0.0)
        if self.lm is not None:
            s += self.alpha * self.lm.logp([self.itos[i] for i in prefix], g)
        self.lm_cache[new] = self.lm_cache[prefix] + s

    def _total(self, prefix: tuple, b: Beam) -> float:
        return b.acoustic() + self.lm_cache[prefix]

    def step(self, log_probs: np.ndarray) -> None:
        """log_probs: (T_new, C) frame log-posteriors (already temperature-scaled)."""
        for t in range(log_probs.shape[0]):
            lp = log_probs[t]
            cand = np.where(lp > self.prune_logp)[0]
            if len(cand) == 0:
                cand = np.argsort(lp)[-self.beam:]
            cand = [c for c in cand if int(c) not in self.skip_ids]
            nxt: dict[tuple, Beam] = defaultdict(Beam)
            for prefix, b in self.beams.items():
                for c in cand:
                    c = int(c); p = float(lp[c])
                    if c == self.blank:                               # stay on prefix, now ending in blank
                        nb = nxt[prefix]
                        nb.pb = lse(nb.pb, b.pb + p, b.pnb + p)
                        continue
                    last = prefix[-1] if prefix else None
                    new = prefix + (c,)
                    self._extend_lm(prefix, new)
                    nn_ = nxt[new]
                    if c == last:
                        nn_.pnb = lse(nn_.pnb, b.pb + p)              # true repeat needs a blank between
                        same = nxt[prefix]
                        same.pnb = lse(same.pnb, b.pnb + p)           # otherwise collapses into same token
                    else:
                        nn_.pnb = lse(nn_.pnb, b.pb + p, b.pnb + p)
            self.beams = dict(sorted(nxt.items(), key=lambda kv: -self._total(*kv))[: self.beam])

    def nbest(self, n: int = 5, final: bool = False) -> list[tuple[list[str], float]]:
        scored = []
        for prefix, b in self.beams.items():
            s = self._total(prefix, b)
            if final and self.lm is not None:
                s += self.alpha * self.lm.logp([self.itos[i] for i in prefix], "</s>")
            scored.append(([self.itos[i] for i in prefix], s))
        scored.sort(key=lambda x: -x[1])
        return scored[:n]

    def reset(self):
        self.beams = {(): Beam(pb=0.0)}
        self.lm_cache = {(): 0.0}
```

Tune `alpha` (LM weight), `beta` (insertion bonus) and `beam` by **grid search on validation WER**. Record the WER-vs-beam curve and the latency-vs-beam curve. That trade-off is a nice result for the report.

### 3.3 Contextual biasing: the "I WANT … FOOD" mechanism

Context enters decoding at three levels, each a separate ablation for RQ4:

| Level | Mechanism | Example |
|---|---|---|
| **Language** | Gloss LM (M5), domain-interpolated | `WANT` → `FOOD`/`WATER` more likely than `FOOT` |
| **Situation** | `context_bias` boosts for glosses relevant to the venue and dialogue state, from RAG (§VII-3) | At a restaurant with the waiter having just asked "Drink?", boost `WATER`, `SODA`, `JUICE` by +1.5 nats |
| **Dialogue** | Expected-answer priors from the agent | After "How many?", boost `NUM:*` |

```python
# app/pipeline/context.py
from dataclasses import dataclass, field


@dataclass
class DialogueContext:
    venue_type: str = "general"                     # restaurant | hospital | bank | general
    last_turns: list[str] = field(default_factory=list)   # English of the last k turns (both parties)
    expected: set[str] = field(default_factory=set)       # gloss categories expected next ("DRINK","NUMBER")
    entities: dict[str, list[str]] = field(default_factory=dict)  # venue gazetteer: menu, rooms, staff names

    def bias(self, category_index: dict[str, set[str]], strength: float = 1.5) -> dict[str, float]:
        boost: dict[str, float] = {}
        for cat in self.expected:
            for g in category_index.get(cat, ()):
                boost[g] = max(boost.get(g, 0.0), strength)
        for items in self.entities.values():
            for item in items:
                boost.setdefault(item.upper(), 0.5)
        return boost
```

> **Biasing must stay mild** (≤ 2 nats). Strong bias makes the system "hear what it expects", which is a measurement-bias failure (Part IX). Always evaluate biasing on held-out **adversarial** sentences where the signer says something unexpected.

### 3.4 Joint final decision (N-best × translation × NLU)

```python
# app/pipeline/orchestrator.py (excerpt)
def choose_final(nbest, translate, nlu, w=(1.0, 0.5, 0.8)):
    """nbest: [(glosses, cslr_score)]. translate(glosses) -> (text, mt_logp). nlu(text) -> (intent, conf)."""
    best, best_s = None, -1e9
    for glosses, s_cslr in nbest[:3]:
        text, s_mt = translate(glosses)
        intent, conf = nlu(text)
        s = w[0] * s_cslr + w[1] * s_mt + w[2] * math.log(max(conf, 1e-6))
        if s > best_s:
            best, best_s = (glosses, text, intent, conf), s
    return best
```

## 4. Calibration and the clarification policy

### 4.1 Calibrated confidence

Neural networks are typically **over-confident**. Raw softmax 0.9 doesn't mean "right 90% of the time". Calibrate it:

* **Token level (ISLR/CSLR frames):** temperature scaling. Fit a single scalar `T` on validation logits by minimising NLL.
* **Utterance level:** a **logistic regression** (from the cheat sheet) predicting *P(utterance fully correct)* from features: length-normalised CSLR score, top-1 − top-2 margin, LM log-prob, mean min-frame-posterior, number of `<unk>`/`FS:` tokens, MT log-prob, hand-visibility ratio. Train it on validation utterances labelled correct or incorrect.

```python
# ml/decoding/calibration.py
import numpy as np
import torch


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, iters: int = 200) -> float:
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=iters)
    nll = torch.nn.CrossEntropyLoss()
    def closure():
        opt.zero_grad()
        loss = nll(logits / logT.exp(), labels)
        loss.backward()
        return loss
    opt.step(closure)
    return float(logT.detach().exp())


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(ece)
```

Report **ECE before and after** calibration, with a reliability diagram.

### 4.2 When to ask: value of information (AIMA Ch. 16)

Asking a clarifying question has a **cost**: it interrupts and slows the conversation. Not asking has an **expected cost of error**, which depends on the domain: a wrong drink is mild, a wrong allergy is severe. Ask if and only if the expected gain beats the cost:

```
EU(proceed) = Σ_h P(h)·U(act on best | h)         = P(best)·U_correct + (1 − P(best))·U_wrong(domain)
EU(ask)     = −c_ask + Σ_h P(h)·U_correct          (assume the answer resolves the ambiguity)
Ask  ⇔  EU(ask) > EU(proceed)  ⇔  (1 − P(best))·(U_correct − U_wrong) > c_ask
```

```python
# app/pipeline/clarify.py
from ksl_core.ir import Clarification

SEVERITY = {"general": 1.0, "restaurant": 1.0, "bank": 2.0, "hospital": 4.0}
CRITICAL_SLOTS = {"allergy", "medication", "amount", "dosage"}


def should_clarify(p_best: float, venue: str, slots: set[str], c_ask: float = 0.15) -> bool:
    loss_if_wrong = SEVERITY.get(venue, 1.0) * (3.0 if slots & CRITICAL_SLOTS else 1.0)
    return (1.0 - p_best) * loss_if_wrong > c_ask


def build_clarification(spans, nbest, threshold=0.6) -> Clarification:
    """Pick the least confident span; offer its alternatives."""
    weakest = min(range(len(spans)), key=lambda i: spans[i].confidence) if spans else None
    if weakest is not None and spans[weakest].alternatives:
        opts = [spans[weakest].token] + [a for a, _ in spans[weakest].alternatives[:2]]
        return Clarification(kind="choose_one", span_index=weakest,
                             prompt_text=f"Did you mean: {' / '.join(opts)}?",
                             prompt_glosses=["IX-2", "MEAN", "WHICH"], options=opts)
    if any(g == "<unk>" for g in nbest[0][0]):
        return Clarification(kind="fingerspell", prompt_text="Please fingerspell that word.",
                             prompt_glosses=["PLEASE", "FINGERSPELL"])
    return Clarification(kind="repeat", prompt_text="Please sign that again.",
                         prompt_glosses=["PLEASE", "AGAIN"])
```

The clarification is **sent to the signer in their modality**: an avatar signing the question plus tappable options. The hearing party sees "⏳ Clarifying…", so they understand the pause.

### 4.3 Per-token alternatives for "Did you mean?"

Get them from the beam: for each position in the best hypothesis, collect the tokens that the other N-best hypotheses place at the aligned position (Levenshtein alignment). Their relative scores, softmaxed, become `alternatives`. Per-token confidence is the mean posterior of the frames the token occupies in the best CTC alignment (Viterbi forced alignment on the final log-probs).

## 5. Learning from mistakes: feedback loop and active learning

```
 user ✏️ correction / 👍 confirm ─► feedback table ─► consent check ─► training queue
                                                                │
     nightly: low-confidence + corrected samples ─► annotator UI (Deaf reviewers) ─► gold set
                                                                │
     weekly: fine-tune CSLR / LM / MT on new gold ─► eval gates (§VIII-3) ─► registry "staging"
                                                                │
                                         canary 10% of sessions ─► promote / rollback
```

* **Problem generator** (learning-agent role): rank vocabulary by confusion-matrix mass and recording count; produce a "signs we need more of" list for the next data-collection session.
* Store **only consented** feedback clips. Corrections without consent update the LM text only (glosses/English), never the landmark data.

## 6. Personalisation (signer adaptation)

Different signers sign differently, the deepest form of "small mistakes". Three levels:

1. **Prototype adaptation (no training):** when a user confirms a sign, add its embedding to a per-user prototype set. ISLR then uses `score = max(cos(z, global_proto), cos(z, user_proto) + δ)`.
2. **Personal signs:** users can register name signs or idiosyncratic signs (3 recordings each, few-shot via prototypes).
3. **Lightweight fine-tuning:** per-user **LoRA adapters** (rank 8) on the attention projections, trained in a background job after ≥ 200 confirmed utterances and stored under the user's ID. Evaluate on that user's held-out data, and only enable the adapter if it helps.

All personalisation data is the user's own. Provide export and delete (§IX-3).

## 7. The speech → sign path

```
audio chunks (PCM16, 20 ms) ─► VAD ─► streaming ASR (partials every 300 ms)
   final text ─► normalise ─► Text→Gloss (neural + rule validator) ─► sign production (pose seq)
   ─► broadcast: {"type":"sign.render", "poses": <binary>, "glosses": [...], "captions": "..."}
```

* Send **partial captions** to the signer immediately. Captions are often faster to read than waiting for the avatar.
* Start sign production only on **final** ASR segments (sentence-level), so the avatar doesn't restart.
* If T2G confidence is low, send the avatar with a ⚠️ marker plus the caption.

## 8. Text in/out paths

* **Text → speech** (the signer types, the hearing person listens): normalise → TTS → stream audio frames (Opus or PCM) to the hearing device.
* **Text → sign:** the same as speech → sign without ASR.
* **Sign → text:** the default display for the hearing party. **Sign → speech** additionally runs TTS on final English.


---

# Part VI — Platform services: auth, sessions, WebSockets, storage

## 1. Database schema (PostgreSQL 16 + pgvector)

```sql
-- app/db/migrations/0001_init.sql  (generate via Alembic; shown as SQL for clarity)
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS citext;

CREATE TYPE user_role AS ENUM ('signer','speaker','operator','annotator','admin');
CREATE TYPE modality  AS ENUM ('sign','speech','text','device');

CREATE TABLE users (
  id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  email            citext UNIQUE,
  phone_e164       text UNIQUE,
  password_hash    text,                                -- argon2id; NULL for OTP-only accounts
  display_name     text NOT NULL,
  roles            user_role[] NOT NULL DEFAULT '{speaker}',
  preferred_in     modality NOT NULL DEFAULT 'text',
  preferred_out    modality NOT NULL DEFAULT 'text',
  dominant_hand    text CHECK (dominant_hand IN ('left','right')),
  sign_speed       real NOT NULL DEFAULT 1.0,
  locale           text NOT NULL DEFAULT 'en-KE',
  email_verified   boolean NOT NULL DEFAULT false,
  created_at       timestamptz NOT NULL DEFAULT now(),
  deleted_at       timestamptz
);

CREATE TABLE refresh_tokens (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id       uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  family_id     uuid NOT NULL,                   -- rotation chain; reuse ⇒ revoke whole family
  token_hash    bytea NOT NULL UNIQUE,           -- SHA-256 of the opaque token
  device_label  text,
  issued_at     timestamptz NOT NULL DEFAULT now(),
  expires_at    timestamptz NOT NULL,
  used_at       timestamptz,
  revoked_at    timestamptz
);
CREATE INDEX ON refresh_tokens(user_id);
CREATE INDEX ON refresh_tokens(family_id);

CREATE TABLE venues (
  id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name        text NOT NULL,
  venue_type  text NOT NULL CHECK (venue_type IN ('restaurant','hospital','bank','office','transport','general')),
  owner_id    uuid NOT NULL REFERENCES users(id),
  created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE devices (                             -- robots / kiosks
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  venue_id      uuid NOT NULL REFERENCES venues(id) ON DELETE CASCADE,
  name          text NOT NULL,
  api_key_prefix text NOT NULL UNIQUE,           -- first 8 chars, for lookup
  api_key_hash  bytea NOT NULL,                  -- HMAC-SHA256(server_pepper, key)
  capabilities  jsonb NOT NULL DEFAULT '{}',     -- {"screen":true,"speaker":true,"tools":["place_order"]}
  last_seen_at  timestamptz,
  revoked_at    timestamptz
);

CREATE TABLE rooms (
  id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  venue_id    uuid REFERENCES venues(id),
  created_by  uuid REFERENCES users(id),
  join_code   text UNIQUE NOT NULL,              -- short code for QR, e.g. 'K7Q-2MX'
  status      text NOT NULL DEFAULT 'open' CHECK (status IN ('open','closed')),
  store_transcript boolean NOT NULL DEFAULT false,
  created_at  timestamptz NOT NULL DEFAULT now(),
  closed_at   timestamptz
);

CREATE TABLE room_members (
  room_id     uuid REFERENCES rooms(id) ON DELETE CASCADE,
  user_id     uuid REFERENCES users(id),
  device_id   uuid REFERENCES devices(id),
  role        text NOT NULL CHECK (role IN ('signer','speaker','device','observer')),
  joined_at   timestamptz NOT NULL DEFAULT now(),
  left_at     timestamptz,
  CHECK ((user_id IS NULL) <> (device_id IS NULL))
);
CREATE UNIQUE INDEX ON room_members(room_id, user_id) WHERE user_id IS NOT NULL;
CREATE UNIQUE INDEX ON room_members(room_id, device_id) WHERE device_id IS NOT NULL;

CREATE TABLE utterances (                          -- only if room.store_transcript
  id            uuid PRIMARY KEY,
  room_id       uuid NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
  speaker_user  uuid REFERENCES users(id),
  speaker_device uuid REFERENCES devices(id),
  source        modality NOT NULL,
  glosses       text[],
  text_en       text,
  confidence    real,
  intent        jsonb,
  model_versions jsonb NOT NULL,
  created_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ON utterances(room_id, created_at);

CREATE TABLE consents (
  id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id      uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  scope        text NOT NULL CHECK (scope IN ('store_transcripts','contribute_landmarks','contribute_video','personalisation','research_publication')),
  version      text NOT NULL,                    -- consent text version
  granted_at   timestamptz NOT NULL DEFAULT now(),
  withdrawn_at timestamptz,
  evidence_uri text                              -- e.g. signed consent video in S3
);

CREATE TABLE feedback (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  utterance_id  uuid NOT NULL,
  user_id       uuid NOT NULL REFERENCES users(id),
  kind          text NOT NULL CHECK (kind IN ('confirm','correct','reject','clarify_choice')),
  corrected_glosses text[],
  corrected_text text,
  clip_uri      text,                            -- only with contribute_* consent
  created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE dataset_samples (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  contributor   uuid REFERENCES users(id),
  consent_id    uuid NOT NULL REFERENCES consents(id),
  signer_code   text NOT NULL,                   -- pseudonymous signer id for splits
  kind          text NOT NULL CHECK (kind IN ('isolated','continuous','fingerspelling','speech')),
  landmarks_uri text,
  video_uri     text,
  audio_uri     text,
  glosses       text[],
  text_en       text,
  text_sw       text,
  meta          jsonb NOT NULL DEFAULT '{}',
  review_status text NOT NULL DEFAULT 'pending' CHECK (review_status IN ('pending','in_review','approved','rejected')),
  created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE annotations (
  id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  sample_id    uuid NOT NULL REFERENCES dataset_samples(id) ON DELETE CASCADE,
  annotator_id uuid NOT NULL REFERENCES users(id),
  glosses      text[] NOT NULL,
  gloss_times  int4range[],                      -- per-gloss ms ranges when aligned
  text_en      text,
  non_manual   jsonb,
  created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE sign_dictionary (
  gloss        text PRIMARY KEY,
  variants     text[] NOT NULL DEFAULT '{}',
  english      text[] NOT NULL,                  -- translations / synonyms
  category     text[],                           -- 'FOOD','DRINK','NUMBER','HEALTH' (for biasing)
  pose_uri     text NOT NULL,                    -- medoid clip for production
  no_mirror    boolean NOT NULL DEFAULT false,
  embedding    vector(256)                       -- ISLR prototype
);

-- RAG knowledge (Part VII)
CREATE TABLE kb_documents (
  id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  venue_id   uuid REFERENCES venues(id) ON DELETE CASCADE,   -- NULL = global
  title      text NOT NULL,
  source_uri text,
  version    int NOT NULL DEFAULT 1,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE kb_chunks (
  id          bigserial PRIMARY KEY,
  document_id uuid NOT NULL REFERENCES kb_documents(id) ON DELETE CASCADE,
  venue_id    uuid,                               -- denormalised for tenant filtering
  section     text,
  content     text NOT NULL,
  tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
  embedding   vector(384),
  meta        jsonb NOT NULL DEFAULT '{}'
);
CREATE INDEX ON kb_chunks USING gin(tsv);
CREATE INDEX ON kb_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON kb_chunks(venue_id);

CREATE TABLE audit_log (
  id bigserial PRIMARY KEY, actor uuid, action text NOT NULL, target text, at timestamptz DEFAULT now(), meta jsonb
);
```

> **Multi-tenancy (lecture point on pgvector):** pgvector doesn't isolate tenants. *Isolation is your application's job.* Every KB query **must** filter `venue_id = :venue OR venue_id IS NULL`. Enforce it in one repository function, and add PostgreSQL **row-level security** as a second line of defence.

## 2. Authentication and authorisation

### 2.1 Design

| Client | Mechanism |
|---|---|
| Mobile users | Email/phone + password (Argon2id) **or** phone OTP (SMS gateway is a non-AI API and acceptable). Returns a **15-min access JWT** + **30-day opaque refresh token** |
| Refresh | **Rotation:** each refresh issues a new refresh token and marks the old one used. **Reuse detection:** if a used token is presented again, revoke the whole `family_id` (stolen-token signal) |
| WebSocket | Access JWT sent in the first `hello` message (not in the URL, which ends up in logs). Re-auth with `auth.refresh` messages on long sessions |
| Robots/kiosks | **Device API key** `kslb_dev_<prefix>_<secret>`. Server stores HMAC-SHA256(pepper, key). Scoped to one venue |
| Guest hearing party | Joins a room by scanning the QR: gets a **room-scoped guest token** (JWT with `scope=room:{id}`, 2 h TTL), no account needed. *Removes friction for the hearing person, which is critical for adoption.* |
| Admin/annotator | Password + **TOTP 2FA** required |

### 2.2 Code

```python
# app/auth/passwords.py
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

_ph = PasswordHasher(time_cost=3, memory_cost=64 * 1024, parallelism=2)   # Argon2id

def hash_password(pw: str) -> str:
    return _ph.hash(pw)

def verify_password(hash_: str, pw: str) -> tuple[bool, str | None]:
    """Returns (ok, new_hash_if_rehash_needed)."""
    try:
        _ph.verify(hash_, pw)
    except VerifyMismatchError:
        return False, None
    return True, (_ph.hash(pw) if _ph.check_needs_rehash(hash_) else None)
```

```python
# app/auth/tokens.py
from __future__ import annotations
import hashlib
import secrets
import uuid
from datetime import datetime, timedelta, timezone
import jwt  # PyJWT
from app.settings import get_settings

ALG = "EdDSA"   # Ed25519 keys; small, fast, no RSA padding pitfalls


def issue_access(user_id: uuid.UUID, roles: list[str], scope: str | None = None, ttl: int | None = None) -> str:
    s = get_settings()
    now = datetime.now(timezone.utc)
    claims = {
        "sub": str(user_id), "roles": roles, "iat": now, "nbf": now,
        "exp": now + timedelta(seconds=ttl or s.access_ttl_seconds),
        "iss": "ksl-bridge", "aud": "ksl-bridge-clients", "jti": uuid.uuid4().hex,
    }
    if scope:
        claims["scope"] = scope
    return jwt.encode(claims, s.jwt_private_key_pem, algorithm=ALG)


def decode_access(token: str) -> dict:
    s = get_settings()
    return jwt.decode(token, s.jwt_public_key_pem, algorithms=[ALG],
                      audience="ksl-bridge-clients", issuer="ksl-bridge",
                      options={"require": ["exp", "sub", "iat"]}, leeway=10)


def new_refresh_token() -> tuple[str, bytes]:
    raw = secrets.token_urlsafe(48)
    return raw, hashlib.sha256(raw.encode()).digest()
```

```python
# app/auth/router.py (refresh with rotation + reuse detection)
from datetime import datetime, timedelta, timezone
import hashlib
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, update
from app.deps import db_session
from app.db.models import RefreshToken, User
from .tokens import issue_access, new_refresh_token

router = APIRouter(prefix="/auth", tags=["auth"])


class RefreshIn(BaseModel):
    refresh_token: str


@router.post("/refresh")
async def refresh(body: RefreshIn, db=Depends(db_session)):
    h = hashlib.sha256(body.refresh_token.encode()).digest()
    rt = (await db.execute(select(RefreshToken).where(RefreshToken.token_hash == h).with_for_update())).scalar_one_or_none()
    now = datetime.now(timezone.utc)
    if rt is None or rt.revoked_at or rt.expires_at < now:
        raise HTTPException(401, "invalid refresh token")
    if rt.used_at is not None:
        # REUSE ⇒ token theft suspected ⇒ revoke entire family
        await db.execute(update(RefreshToken).where(RefreshToken.family_id == rt.family_id)
                         .values(revoked_at=now))
        await db.commit()
        raise HTTPException(401, "refresh token reuse detected")
    rt.used_at = now
    user = await db.get(User, rt.user_id)
    raw, new_hash = new_refresh_token()
    db.add(RefreshToken(user_id=user.id, family_id=rt.family_id, token_hash=new_hash,
                        device_label=rt.device_label, expires_at=now + timedelta(days=30)))
    await db.commit()
    return {"access_token": issue_access(user.id, user.roles), "refresh_token": raw, "token_type": "bearer"}
```

```python
# app/auth/rbac.py
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from .tokens import decode_access

bearer = HTTPBearer(auto_error=True)


async def current_claims(cred: HTTPAuthorizationCredentials = Depends(bearer)) -> dict:
    try:
        return decode_access(cred.credentials)
    except Exception:
        raise HTTPException(401, "invalid token")


def require_roles(*roles: str):
    async def dep(claims: dict = Depends(current_claims)):
        if not set(roles) & set(claims.get("roles", [])):
            raise HTTPException(403, "forbidden")
        return claims
    return dep
```

```python
# app/auth/api_keys.py
import hashlib, hmac, secrets

def new_device_key(pepper: bytes) -> tuple[str, str, bytes]:
    prefix = secrets.token_hex(4)
    secret = secrets.token_urlsafe(32)
    key = f"kslb_dev_{prefix}_{secret}"
    return key, prefix, hmac.new(pepper, key.encode(), hashlib.sha256).digest()

def verify_device_key(pepper: bytes, presented: str, stored_hash: bytes) -> bool:
    return hmac.compare_digest(hmac.new(pepper, presented.encode(), hashlib.sha256).digest(), stored_hash)
```

**Security checklist:** rate-limit `/auth/*` (5/min/IP + per account, Redis token bucket); lock out after 10 failures, with exponential back-off; uniform error messages (no user enumeration); constant-time comparisons; rotate the JWT signing key with a `kid` header and JWKS endpoint; TLS everywhere; secrets via environment/secret manager, never in git; audit-log admin actions.

## 3. Rooms (conversation sessions)

REST:

| Method | Path | Who | Purpose |
|---|---|---|---|
| POST | `/rooms` | signer/speaker/device | Create room `{venue_id?, store_transcript}` → `{room_id, join_code, qr_payload}` |
| POST | `/rooms/join` | anyone with code | `{join_code, display_name?}` → room-scoped guest token if unauthenticated |
| GET | `/rooms/{id}` | member | Room info, members, their modality preferences |
| POST | `/rooms/{id}/close` | creator/operator | Close; triggers retention policy |
| GET | `/rooms/{id}/transcript` | member, if stored | Paginated utterances |

`qr_payload = "kslbridge://join?c=K7Q-2MX"`. The Kotlin app deep-links it, and the web fallback serves a minimal join page.

## 4. WebSocket protocol (v1)

One socket per client: `wss://api.example/ws/v1`. Two frame kinds:

* **Text frames:** JSON control and data messages with a common envelope.
* **Binary frames:** high-rate media with a 1-byte type prefix: `0x01` landmarks (`KSLB` format, §III-2), `0x02` audio PCM16 16 kHz mono (20 ms per frame), `0x03` server→client pose sequence, `0x04` TTS audio.

### 4.1 Envelope

```json
{ "v": 1, "type": "utterance.partial", "id": "c0a1…", "seq": 42, "ts": 1761550000123,
  "room": "8f2e…", "ack": 17, "data": { } }
```

Client messages use the same envelope. The message-specific fields go in `data`, for example `{"v":1,"type":"hello","seq":1,"ts":…,"data":{"token":"…","client":{"app":"ksl-android","version":"0.3.0"}}}`. The server validates the envelope first, then `{"type": …, **data}` against the typed schema.

`seq` increases monotonically per direction per socket. `ack` acknowledges the last received peer `seq`. Both enable **resume** after reconnects on flaky mobile networks.

### 4.2 Message catalogue

| Direction | `type` | `data` | Notes |
|---|---|---|---|
| C→S | `hello` | `{token, client: {app, version, platform}, resume?: {session_id, last_seq}}` | Must be first, within 5 s |
| S→C | `welcome` | `{session_id, server_time, heartbeat_s, limits}` | |
| C→S | `room.join` | `{room_id, role}` | Must be a member (REST join first) |
| S→C | `room.state` | `{members:[{id,name,role,modality_in,modality_out,online}]}` | Broadcast on change |
| C→S | `sign.start` / `sign.stop` | `{fps, dominant_hand?}` | Optional manual endpointing |
| C→S (bin 0x01) | landmarks | `KSLB` batch | 4 frames per message |
| C→S | `speech.start` / `speech.stop` | `{sample_rate:16000, lang_hint?}` | |
| C→S (bin 0x02) | audio | PCM16 | |
| C→S | `text.send` | `{text, lang}` | |
| S→room | `utterance.partial` | `UtteranceIR` (`is_final=false`, `revision`) | Live captions; client replaces by `id`+`revision` |
| S→room | `utterance.final` | `UtteranceIR` (`is_final=true`) | |
| S→signer | `clarify.request` | `Clarification` + `utterance_id` | Avatar signs the question |
| C→S | `clarify.answer` | `{utterance_id, choice \| "none" \| retry: true}` | |
| S→room | `sign.render` | `{utterance_id, glosses, captions, n_frames, fps}` then bin `0x03` | Avatar data |
| S→room | `speech.audio` | `{utterance_id, format:"pcm16", sr:22050}` then bin `0x04` chunks | TTS for the hearing party |
| C→S | `feedback` | `{utterance_id, kind, corrected_glosses?, corrected_text?}` | Learning-agent critic |
| S→C | `typing` / `signing` | `{member_id, active}` | Presence: "Wanjiku is signing…" |
| both | `ping` / `pong` | `{t}` | Heartbeat every 15 s; RTT metric |
| S→C | `error` | `{code, message, retry_after?}` | Codes in §4.4 |
| C→S | `auth.refresh` | `{token}` | Swap the access token mid-session |

### 4.3 Message schemas and codec

```python
# app/ws/protocol.py
from __future__ import annotations
from typing import Annotated, Any, Literal, Union
from pydantic import BaseModel, Field, TypeAdapter


class Env(BaseModel):
    v: Literal[1] = 1
    type: str
    id: str | None = None
    seq: int
    ts: int
    room: str | None = None
    ack: int | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class Hello(BaseModel):
    type: Literal["hello"]
    token: str
    client: dict[str, str]
    resume: dict[str, Any] | None = None

class RoomJoin(BaseModel):
    type: Literal["room.join"]
    room_id: str
    role: Literal["signer", "speaker", "device", "observer"]

class TextSend(BaseModel):
    type: Literal["text.send"]
    text: Annotated[str, Field(min_length=1, max_length=2000)]
    lang: Literal["en", "sw"] = "en"

class Feedback(BaseModel):
    type: Literal["feedback"]
    utterance_id: str
    kind: Literal["confirm", "correct", "reject", "clarify_choice"]
    corrected_glosses: list[str] | None = None
    corrected_text: str | None = None

ClientMsg = TypeAdapter(Annotated[Union[Hello, RoomJoin, TextSend, Feedback], Field(discriminator="type")])

BIN_LANDMARKS, BIN_AUDIO, BIN_POSES, BIN_TTS = 0x01, 0x02, 0x03, 0x04
```

### 4.4 Error codes

| Code | Meaning | Client action |
|---|---|---|
| 4001 | auth required / invalid | Refresh the token, reconnect |
| 4003 | not a room member | Re-join via REST |
| 4008 | protocol violation | Bug: log and reconnect |
| 4009 | backpressure (too many in-flight frames) | Reduce send rate (drop to 15 fps) |
| 4029 | rate limited | Wait `retry_after` |
| 4500 | pipeline unavailable | Show "Translation temporarily unavailable"; keep text chat working |

**Graceful degradation is a requirement:** if every AI worker is down, the room must still relay **plain text** between participants. Communication never fully breaks.

## 5. Connection manager and cross-replica broadcast

```python
# app/rooms/bus.py
from __future__ import annotations
import asyncio
import json
from collections import defaultdict
import redis.asyncio as aioredis
from fastapi import WebSocket


class LocalConnections:
    """Sockets connected to THIS gateway replica, indexed by room."""
    def __init__(self):
        self.by_room: dict[str, set["ClientConn"]] = defaultdict(set)

    def add(self, room: str, c: "ClientConn"):
        self.by_room[room].add(c)

    def remove(self, room: str, c: "ClientConn"):
        self.by_room[room].discard(c)
        if not self.by_room[room]:
            self.by_room.pop(room, None)


class RoomBus:
    """Redis pub/sub fan-out: publish once, every replica delivers to its local sockets."""
    def __init__(self, redis: aioredis.Redis, local: LocalConnections):
        self.r, self.local = redis, local
        self.pubsub = redis.pubsub()
        self.subscribed: set[str] = set()
        self._task: asyncio.Task | None = None

    async def start(self):
        await self.pubsub.subscribe("__keepalive__")
        self._task = asyncio.create_task(self._reader())

    async def join(self, room: str):
        if room not in self.subscribed:
            await self.pubsub.subscribe(f"room:{room}")
            self.subscribed.add(room)

    async def leave_if_empty(self, room: str):
        if room in self.subscribed and room not in self.local.by_room:
            await self.pubsub.unsubscribe(f"room:{room}")
            self.subscribed.discard(room)

    async def publish(self, room: str, msg: dict, binary: bytes | None = None, exclude: str | None = None,
                      only_roles: list[str] | None = None):
        payload = {"msg": msg, "exclude": exclude, "only_roles": only_roles}
        if binary is not None:
            payload["bin"] = binary.hex()          # for large payloads, store in Redis key and send a ref
        await self.r.publish(f"room:{room}", json.dumps(payload))

    async def _reader(self):
        async for m in self.pubsub.listen():
            if m["type"] != "message":
                continue
            room = m["channel"].decode().split(":", 1)[1]
            p = json.loads(m["data"])
            for c in list(self.local.by_room.get(room, ())):
                if p["exclude"] == c.member_id:
                    continue
                if p["only_roles"] and c.role not in p["only_roles"]:
                    continue
                c.enqueue(p["msg"], bytes.fromhex(p["bin"]) if p.get("bin") else None)
```

For pose sequences (tens of KB) don't push hex through pub/sub. Write the blob to Redis with `SET pose:{utt_id} <bytes> EX 60` and publish a reference. Each replica fetches it once and fans it out.

### 5.1 Per-connection writer with backpressure

A slow phone on 3G mustn't block the room. Each connection has a **bounded outbound queue**. Partial captions are droppable and superseded by newer revisions. Finals are not.

```python
# app/ws/backpressure.py
from __future__ import annotations
import asyncio
import json
import time
from fastapi import WebSocket

DROPPABLE = {"utterance.partial", "typing", "signing"}


class ClientConn:
    def __init__(self, ws: WebSocket, member_id: str, role: str, maxsize: int = 256):
        self.ws, self.member_id, self.role = ws, member_id, role
        self.q: asyncio.Queue[tuple[dict, bytes | None]] = asyncio.Queue(maxsize)
        self.seq = 0
        self.last_peer_seq = 0
        self.replay: list[tuple[int, dict]] = []     # last N for resume
        self.inflight_landmarks = 0

    def enqueue(self, msg: dict, binary: bytes | None = None):
        if self.q.full():
            if msg.get("type") in DROPPABLE:
                return                               # drop stale partials
            try:                                     # make room by dropping the oldest droppable
                tmp = [self.q.get_nowait() for _ in range(self.q.qsize())]
                kept = [t for t in tmp if t[0].get("type") not in DROPPABLE]
                for t in kept[-(self.q.maxsize - 1):]:
                    self.q.put_nowait(t)
            except asyncio.QueueEmpty:
                pass
        self.q.put_nowait((msg, binary))

    async def writer(self):
        while True:
            msg, binary = await self.q.get()
            self.seq += 1
            env = {"v": 1, "seq": self.seq, "ts": int(time.time() * 1000), "ack": self.last_peer_seq, **msg}
            await self.ws.send_text(json.dumps(env, default=str))
            if binary is not None:
                await self.ws.send_bytes(binary)
            self.replay.append((self.seq, env))
            del self.replay[:-200]
```

## 6. The WebSocket endpoint

```python
# app/ws/endpoint.py
from __future__ import annotations
import asyncio
import json
import time
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from app.auth.tokens import decode_access
from ksl_core.landmarks import unpack
from .protocol import ClientMsg, Env, BIN_LANDMARKS, BIN_AUDIO
from .backpressure import ClientConn

router = APIRouter()


@router.websocket("/ws/v1")
async def ws_endpoint(ws: WebSocket):
    app = ws.app
    await ws.accept(subprotocol=None)
    # 1) hello + auth within 5 s
    def parse(raw: str):
        env = Env.model_validate_json(raw)                       # envelope first…
        return env, ClientMsg.validate_python({"type": env.type, **env.data})   # …then typed payload

    try:
        env, hello = parse(await asyncio.wait_for(ws.receive_text(), timeout=5))
        if hello.type != "hello":
            await ws.close(4008); return
        claims = decode_access(hello.token)
    except Exception:                                           # timeout, bad JSON, bad token
        await ws.close(4001); return

    member_id = claims["sub"]
    conn = ClientConn(ws, member_id=member_id, role="observer")
    writer = asyncio.create_task(conn.writer())
    session = app.state.sessions.create(member_id, claims)          # orchestrator session (§V-3)
    conn.enqueue({"type": "welcome", "data": {"session_id": session.id, "heartbeat_s": 15}})
    room: str | None = None
    last_rx = time.monotonic()

    async def watchdog():
        while True:
            await asyncio.sleep(5)
            if time.monotonic() - last_rx > 60:
                await ws.close(1001); return
    wd = asyncio.create_task(watchdog())

    try:
        while True:
            m = await ws.receive()
            last_rx = time.monotonic()
            if m.get("bytes") is not None:
                b = m["bytes"]
                if room is None or len(b) > 65536:
                    await ws.close(4008); return
                kind, payload = b[0], b[1:]
                if kind == BIN_LANDMARKS:
                    if conn.inflight_landmarks > app.state.cfg.ws.max_inflight_landmark_batches:
                        conn.enqueue({"type": "error", "data": {"code": 4009, "message": "slow down"}})
                        continue
                    batch = unpack(payload)
                    await session.on_landmarks(batch, conn)         # normalise, endpoint, enqueue CSLR job
                elif kind == BIN_AUDIO:
                    await session.on_audio(payload, conn)
                continue

            try:
                env, msg = parse(m["text"])
            except ValidationError:
                conn.enqueue({"type": "error", "data": {"code": 4008, "message": "invalid message"}})
                continue
            conn.last_peer_seq = env.seq
            match msg.type:
                case "room.join":
                    if not await app.state.rooms.is_member(msg.room_id, member_id, claims):
                        conn.enqueue({"type": "error", "data": {"code": 4003}}); continue
                    room, conn.role = msg.room_id, msg.role
                    app.state.local.add(room, conn)
                    await app.state.bus.join(room)
                    await session.attach_room(room, msg.role)
                    await app.state.rooms.broadcast_state(room)
                case "text.send":
                    await session.on_text(msg.text, msg.lang, conn)
                case "feedback":
                    await app.state.feedback.record(member_id, msg)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        wd.cancel(); writer.cancel()
        if room:
            app.state.local.remove(room, conn)
            await app.state.bus.leave_if_empty(room)
            await app.state.rooms.broadcast_state(room)
        await session.close()
```

**Note on the `hello`-first pattern.** Browsers can't set headers on WebSocket requests, and tokens in query strings leak into logs. Sending the token as the first message avoids both. Validate the `Origin` header for browser clients.

### 6.1 Resume after reconnect

The client keeps `session_id` and `last_seq` received. On reconnect it sends `hello.resume`. The server looks up the session's replay buffer in Redis (`session:{id}:replay`, last 200 messages, TTL 2 min) and re-sends messages with `seq > last_seq`. A mid-utterance CSLR state survives for 10 s. After that the utterance is finalised with what was received.

## 7. Application factory and lifespan

```python
# app/main.py
from contextlib import asynccontextmanager
import redis.asyncio as aioredis
from fastapi import FastAPI
from prometheus_client import make_asgi_app
from app.settings import get_settings
from app.rooms.bus import LocalConnections, RoomBus
from app.ws.endpoint import router as ws_router
from app.auth.router import router as auth_router
from app.routers import rooms, users, venues, devices, datasets, admin, health


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    app.state.redis = aioredis.from_url(str(s.redis_url))
    app.state.local = LocalConnections()
    app.state.bus = RoomBus(app.state.redis, app.state.local)
    await app.state.bus.start()
    # sessions/orchestrator, rooms service, feedback service wired here (DI container)
    yield
    await app.state.redis.aclose()


def create_app() -> FastAPI:
    app = FastAPI(title="KSL Bridge API", version="1.0.0", lifespan=lifespan)
    for r in (auth_router, rooms.router, users.router, venues.router, devices.router,
              datasets.router, admin.router, health.router, ws_router):
        app.include_router(r)
    app.mount("/metrics", make_asgi_app())
    return app

app = create_app()
```

## 8. REST API summary

| Area | Endpoints |
|---|---|
| Auth | `POST /auth/register`, `/auth/login`, `/auth/refresh`, `/auth/logout`, `/auth/otp/request`, `/auth/otp/verify`, `/auth/2fa/enable`, `GET /.well-known/jwks.json` |
| Users | `GET/PATCH /me` (modality prefs, dominant hand, sign speed), `GET /me/export` (all personal data, JSON + files), `DELETE /me` (erasure) |
| Consents | `GET /me/consents`, `POST /me/consents`, `DELETE /me/consents/{scope}` |
| Rooms | §3 |
| Venues & devices | `POST /venues`, `POST /venues/{id}/devices` (returns the API key **once**), `POST /devices/{id}/rotate-key`, `DELETE /devices/{id}` |
| Knowledge (RAG) | `POST /venues/{id}/kb/documents` (upload menu/FAQ PDF/Markdown), `GET …/kb/documents`, `DELETE …/{doc}`, `POST …/kb/reindex` |
| One-shot translation | `POST /translate/text-to-sign` `{text}` → glosses + pose URL; `POST /translate/sign-to-text` (upload `KSLB` file) → IR. Useful for testing and for kiosk apps without WebSockets |
| Dictionary | `GET /dictionary?q=water` (search by English or gloss), `GET /dictionary/{gloss}/pose` (for the app's "learn KSL" mode) |
| Dataset contribution | `POST /datasets/uploads` → presigned MinIO PUT URL; `POST /datasets/samples` (register after upload, requires a `contribute_*` consent) |
| Annotation | `GET /annotate/queue`, `POST /annotate/{sample_id}` (annotator role) |
| Admin/ML | `GET /admin/models`, `POST /admin/models/{task}/promote`, `GET /admin/metrics/summary` |
| Health | `GET /healthz` (liveness), `GET /readyz` (DB, Redis, workers heartbeats) |

## 9. File storage

* MinIO buckets: `datasets` (private), `models` (private), `kb` (private), `sign-dict` (read via signed URLs).
* Uploads use **presigned PUT URLs** with 15-min expiry, so large videos never pass through the API process.
* A server-side job runs after upload: virus/format check (`ffprobe`), landmark extraction, face-blur derivative for annotators who don't need faces, and registration in `dataset_samples` as `pending`.

## 10. Observability

| Signal | Examples |
|---|---|
| Metrics (Prometheus) | `ws_connections`, `ws_msgs_total{type}`, `stage_latency_seconds{stage}` histogram (normalise, cslr, beam, mt, nlu, tts, production, broadcast), `e2e_latency_seconds{path="sign->text"}`, `clarifications_total{venue_type}`, `feedback_total{kind}`, `infer_batch_size{task}`, `gpu_util`, `energy_joules_total{task}` |
| Traces (OpenTelemetry) | One trace per utterance: span per stage, attributes `model_version`, `n_frames`, `beam` |
| Logs (structlog JSON) | **No content by default**: log IDs, lengths and timings, never glosses or text unless the room has `store_transcript` and debug sampling is consented |
| Dashboards | Live latency SLOs, error rates, clarification rate per model version (a quality proxy), drift: mean confidence over time |


---

# Part VII — Agent layer, retrieval (RAG) and robot/kiosk integration

## 1. Where an agent is actually justified

Not every component is an agent (Part I §3.4). The pipeline is a model-based reflex agent. An actual **goal-based agent** is justified only where the system must *decide and act toward a goal using tools*. That's the service interaction with a robot or kiosk:

```
Signer: 🤟 IX-1 WANT BURGER CHICKEN TWO
  └─► IR.intent = order_food {item: "chicken burger", qty: 2}
        └─► Agent: goal = complete order
              ├─ tool: menu.lookup("chicken burger")  → found, KSh 650, in stock
              ├─ tool: order.add(item_id=17, qty=2)
              ├─ policy: missing slot "drink"? venue rule says offer drink
              └─ respond: "Two chicken burgers. Would you like a drink?"
                    └─► Text→Gloss → avatar to signer; text to waiter tablet; robot screen
```

## 2. Dialogue manager: goal-based, frame-filling

Use a **frame-based dialogue manager** with explicit state. It's deterministic, testable and explainable, which is the right choice under the no-hosted-LLM constraint. It's also a direct application of goal formulation → problem formulation → search/plan → execute from *Problem Solving Agents*.

```python
# app/agent/dialogue.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

ToolFn = Callable[..., Awaitable[Any]]          # async tools (adapters, KB)


@dataclass
class Frame:
    intent: str
    required: list[str]
    slots: dict[str, Any] = field(default_factory=dict)

    def missing(self) -> list[str]:
        return [s for s in self.required if s not in self.slots]


@dataclass
class AgentReply:
    text: str
    expected: set[str]                       # gloss categories to bias next decoding (Part V §3.3)
    actions: list[dict] = field(default_factory=list)
    done: bool = False


INTENT_SCHEMAS = {
    "order_food":   ["item", "qty"],
    "ask_location": ["place"],
    "book_appointment": ["department", "time"],
    "pay":          ["method"],
    "greet":        [],
    "help":         [],
}

SLOT_QUESTIONS = {   # English; Text→Gloss converts for the signer
    "item": ("What would you like?", {"FOOD", "DRINK"}),
    "qty": ("How many?", {"NUMBER"}),
    "place": ("Where do you want to go?", {"PLACE"}),
    "department": ("Which department?", {"HEALTH"}),
    "time": ("What time?", {"TIME", "NUMBER"}),
    "method": ("Cash or M-Pesa?", {"PAYMENT"}),
}


class DialogueManager:
    def __init__(self, tools: dict[str, ToolFn], venue_rules: dict[str, Any]):
        self.tools, self.rules = tools, venue_rules
        self.frame: Frame | None = None
        self.history: list[dict] = []

    async def step(self, intent: str, conf: float, slots: dict[str, Any]) -> AgentReply:
        if conf < 0.4 or intent == "unknown":
            return AgentReply("Sorry, I didn't understand. Could you sign it again?", set())
        if self.frame is None or (intent != self.frame.intent and intent in INTENT_SCHEMAS):
            self.frame = Frame(intent, INTENT_SCHEMAS.get(intent, []))
        self.frame.slots.update({k: v for k, v in slots.items() if v not in (None, "")})

        # validate slots with tools (grounding): e.g. item must exist on the menu
        if "item" in self.frame.slots:
            hit = await self.tools["menu.lookup"](self.frame.slots["item"])
            if not hit:
                self.frame.slots.pop("item")
                return AgentReply("We don't have that. Would you like to see the menu?", {"FOOD", "DRINK"},
                                  actions=[{"type": "show_menu"}])
            self.frame.slots["item_id"], self.frame.slots["item"] = hit["id"], hit["name"]

        missing = self.frame.missing()
        if missing:
            q, expect = SLOT_QUESTIONS[missing[0]]
            return AgentReply(q, expect)

        return await self._execute()

    async def _execute(self) -> AgentReply:
        f = self.frame
        if f.intent == "order_food":
            res = await self.tools["order.add"](item_id=f.slots["item_id"], qty=int(f.slots["qty"]))
            follow = self.rules.get("upsell_drink") and "drink" not in f.slots
            self.frame = None
            if follow:
                return AgentReply(f"{f.slots['qty']} {f.slots['item']}, added. Would you like a drink?",
                                  {"DRINK", "YES_NO"}, actions=[{"type": "order_added", **res}])
            return AgentReply(f"{f.slots['qty']} {f.slots['item']}, added. Anything else?", {"YES_NO"},
                              actions=[{"type": "order_added", **res}], done=True)
        if f.intent == "ask_location":
            ans = await self.tools["kb.answer"](f"Where is the {f.slots['place']}?")
            self.frame = None
            return AgentReply(ans or "I don't know. Let me call a staff member.",
                              set(), actions=[] if ans else [{"type": "call_staff"}], done=True)
        if f.intent == "greet":
            self.frame = None
            return AgentReply("Hello! How can I help you?", {"FOOD", "DRINK", "HELP"})
        self.frame = None
        return AgentReply("Let me call a staff member to help.", set(), actions=[{"type": "call_staff"}], done=True)
```

**Always keep a human escalation path** (`call_staff`). The system must complement humans, not replace them. That's the Weizenbaum point from *Introduction to AI*.

### 2.1 Optional: a learned policy

Once you have logged dialogues, you can learn **which clarification or upsell to ask** with a contextual bandit, or frame the problem as a small POMDP (AIMA Ch. 17, belief over the user's intent). Treat this as stretch work. The rule-based manager is the baseline.

### 2.2 Optional: a small local language model

If the group later wants freer responses, train or fine-tune a **small** local model (≤ 1B parameters, quantised) for response phrasing only, never for decisions. It sits behind the same `ModelAdapter` interface. This respects the no-API rule, but weigh the sustainability cost (Part IX-4).

## 3. Retrieval-Augmented Generation (from the guest lecture), adapted

The guest lecture's naive RAG has two pipelines: **ingestion** (load → chunk → embed → store) and **retrieval** (embed query → top-k → prompt → generate). We use the same structure with three changes:

1. **No OpenAI / Pinecone** (no-API rule): local embeddings (§3.3) + **pgvector**, plus **BM25** for hybrid retrieval.
2. **The "generator" isn't an LLM.** It's extractive answer selection plus our Text→Gloss. Retrieval also **feeds context into recognition** (decoder biasing), which is unusual and a novelty point.
3. **The four naive-RAG failure modes are fixed explicitly** (§3.6).

### 3.1 Three knowledge bases

| KB | Content | Used for |
|---|---|---|
| **Venue KB** | Menus, prices, opening hours, directions, FAQs (per venue, tenant-isolated) | Answering questions; slot grounding; **biasing CSLR/ASR toward venue vocabulary** |
| **Sign dictionary KB** | Gloss ↔ English synonyms ↔ categories ↔ prototype embeddings | Text→Gloss synonym fallback ("soda" → `SODA` or `SOFT-DRINK`), category index for biasing, "did you mean" alternatives |
| **Translation memory** | Confirmed (gloss sequence ↔ English) pairs | Exact or near-exact match → reuse a verified translation instead of a model guess (higher precision for frequent phrases) |

### 3.2 Ingestion and chunking

The lecture's guidance applies directly: **start with recursive character splitting**, and use **structure-aware** splitting for menus and FAQs, because they have real headings and tables.

```python
# app/rag/chunking.py
from __future__ import annotations
import re


def recursive_split(text: str, size: int = 500, overlap: int = 50,
                    seps: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")) -> list[str]:
    """Same behaviour as RecursiveCharacterTextSplitter, without the dependency."""
    if len(text) <= size:
        return [text.strip()] if text.strip() else []
    sep = next((s for s in seps if s and s in text), "")
    parts = text.split(sep) if sep else list(text)
    chunks, cur = [], ""
    for p in parts:
        piece = (cur + sep + p) if cur else p
        if len(piece) <= size:
            cur = piece
        else:
            if cur:
                chunks.append(cur)
            if len(p) > size:
                chunks.extend(recursive_split(p, size, overlap, seps[seps.index(sep) + 1:] if sep else ("",)))
                cur = ""
            else:
                cur = (cur[-overlap:] + sep + p) if (cur and overlap) else p
    if cur.strip():
        chunks.append(cur)
    return [c.strip() for c in chunks if c.strip()]


def markdown_sections(md: str) -> list[tuple[str, str]]:
    """Structure-aware: split on headings, keep the heading path as metadata (lecture technique 4)."""
    out, path, buf = [], [], []
    for line in md.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            if buf:
                out.append((" > ".join(path), "\n".join(buf).strip())); buf = []
            level = len(m.group(1))
            path = path[: level - 1] + [m.group(2).strip()]
        else:
            buf.append(line)
    if buf:
        out.append((" > ".join(path), "\n".join(buf).strip()))
    return [(p, t) for p, t in out if t]


def menu_rows(table_rows: list[dict]) -> list[str]:
    """Tables: never split mid-row; one chunk per item, header repeated (lecture 'real documents' slide)."""
    return [f"Menu item: {r['name']}. Category: {r.get('category','')}. Price: KSh {r.get('price','')}. "
            f"Description: {r.get('description','')}" for r in table_rows]
```

### 3.3 Embeddings without APIs

Two options, both local:

* **BM25** (no neural net, fully ours, strong on keyword-heavy menus). Implement it or use PostgreSQL full-text search (`tsvector`, already in the schema).
* **Our own small sentence encoder:** a 6-layer, 384-d Transformer trained with a contrastive (InfoNCE) objective on (question, answer-chunk) pairs from the venue FAQs plus paraphrase pairs. Bootstrap it with a local open-weights sentence-embedding model if needed, behind the `ModelAdapter` interface.

```python
# app/rag/bm25.py
from __future__ import annotations
import math
import re
from collections import Counter

TOK = re.compile(r"[a-z0-9]+")


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [TOK.findall(d.lower()) for d in docs]
        self.N = len(self.docs)
        self.avgdl = sum(map(len, self.docs)) / max(1, self.N)
        self.df = Counter(t for d in self.docs for t in set(d))
        self.tf = [Counter(d) for d in self.docs]
        self.k1, self.b = k1, b

    def idf(self, t: str) -> float:
        n = self.df.get(t, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def scores(self, q: str) -> list[float]:
        qt = TOK.findall(q.lower())
        out = []
        for d, tf in zip(self.docs, self.tf):
            s = 0.0
            for t in qt:
                f = tf.get(t, 0)
                if f:
                    s += self.idf(t) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl))
            out.append(s)
        return out
```

### 3.4 Hybrid retrieval with tenant isolation

```python
# app/rag/retrieve.py
from sqlalchemy import text

HYBRID_SQL = text("""
WITH vec AS (
  SELECT id, content, section, 1 - (embedding <=> :qvec) AS vscore
  FROM kb_chunks
  WHERE (venue_id = :venue OR venue_id IS NULL)
  ORDER BY embedding <=> :qvec
  LIMIT 20
), kw AS (
  SELECT id, content, section, ts_rank_cd(tsv, plainto_tsquery('english', :q)) AS kscore
  FROM kb_chunks
  WHERE (venue_id = :venue OR venue_id IS NULL) AND tsv @@ plainto_tsquery('english', :q)
  ORDER BY kscore DESC
  LIMIT 20
)
SELECT COALESCE(v.id, k.id) AS id, COALESCE(v.content, k.content) AS content,
       COALESCE(v.section, k.section) AS section,
       COALESCE(v.vscore, 0) AS vscore, COALESCE(k.kscore, 0) AS kscore
FROM vec v FULL OUTER JOIN kw k ON v.id = k.id
""")


async def hybrid_search(db, venue_id, q: str, qvec: list[float], k: int = 5, w_vec=0.6, w_kw=0.4):
    rows = (await db.execute(HYBRID_SQL, {"venue": venue_id, "q": q, "qvec": str(qvec)})).mappings().all()
    if not rows:
        return []
    kmax = max(r["kscore"] for r in rows) or 1.0
    scored = [(w_vec * r["vscore"] + w_kw * (r["kscore"] / kmax), r) for r in rows]
    scored.sort(key=lambda x: -x[0])
    return [dict(r, score=s) for s, r in scored[:k]]
```

### 3.5 Answer generation without an LLM

For FAQ-style answering, the "generation" step is **extractive**: pick the best sentence(s) from the top chunk, then compress them for signing (short, concrete sentences translate to KSL far better):

1. Split the top-3 chunks into sentences.
2. Score each sentence by BM25 overlap with the question plus entity match (e.g. the place asked about).
3. Return the top sentence and run it through Text→Gloss.

### 3.6 Fixing naive RAG's four failure modes (from the lecture's limitations slide)

| Lecture limitation | Our fix |
|---|---|
| 1. *Similar is not the same as useful* | **Relevance grading:** a small cross-encoder (trained by us on Q–chunk relevance labels), or simply the requirement that the question's key entity appears in the chunk |
| 2. *One shot, no second chances* | **Rewrite and retry:** if the grade is below threshold, expand the query with dictionary synonyms (`toilet` → `bathroom`, `washroom`, `choo`) and search again, at most once (latency budget) |
| 3. *Nothing verifies the answer* | **Grounding check:** the answer is extractive, so it's literally a span from a source chunk. Store the chunk ID with the utterance for audit |
| 4. *What if the answer isn't there?* | **Say so**, the lecture's key instruction. Reply "I don't know. Let me call a staff member" and trigger `call_staff`. Never improvise venue facts (prices, directions, medical information) |

```python
# app/rag/grade.py
def grade(question_entities: set[str], chunk: str, score: float, min_score: float = 0.35) -> bool:
    text = chunk.lower()
    entity_hit = (not question_entities) or any(e.lower() in text for e in question_entities)
    return score >= min_score and entity_hit
```

**Freshness** (lecture question 5): menus change daily. `POST …/kb/documents` bumps the `version`, deletes the old chunks and re-embeds **in one transaction**, so a half-updated index is never visible. Re-index nightly from the POS integration if one is available.

### 3.7 RAG → recognition biasing (the novel part)

When a room is attached to a venue, the orchestrator pre-computes:

* the **venue vocabulary**: menu items, room names, staff names. Map each to glosses via the dictionary KB, or to `FS:` lexicon entries for the fingerspelling decoder;
* a `context_bias` map for the CTC decoder (Part V §3.3) and a hot-word list for the ASR decoder.

The KSL recogniser then "knows" the restaurant's menu without retraining, the same open-book idea as RAG applied to perception. Evaluate it as an ablation for RQ4.

## 4. Robot / kiosk integration

### 4.1 Integration modes

| Mode | Description | When |
|---|---|---|
| **A. Android client on the robot** | The robot's screen runs our Kotlin app in *kiosk mode*. The camera → landmarks pipeline is identical to phones | Most robots and kiosks with Android tablets |
| **B. Device API** | The robot's own software connects with a device API key, sends `KSLB` landmarks (or raw video to a server-side extractor), and receives **structured intents** + pose sequences | Custom robots (ROS-based) |
| **C. Tool callbacks** | Our agent calls the venue's systems (POS, queue system) through **adapters** | Ordering, queue tickets |

### 4.2 Device message extensions

| Direction | `type` | `data` |
|---|---|---|
| S→device | `agent.action` | `{action: "order_added" \| "show_menu" \| "call_staff" \| "navigate", params}` |
| device→S | `device.event` | `{event: "item_unavailable" \| "order_ready" \| "staff_arrived", params}` (the agent turns these into messages for the signer) |
| device→S | `device.capabilities` | `{screen: true, speaker: true, arm: false}` |

### 4.3 Adapter interface

```python
# app/agent/adapters/base.py
from abc import ABC, abstractmethod


class VenueAdapter(ABC):
    @abstractmethod
    async def menu_lookup(self, name: str) -> dict | None: ...
    @abstractmethod
    async def order_add(self, item_id: int, qty: int, room_id: str) -> dict: ...
    @abstractmethod
    async def call_staff(self, room_id: str, reason: str) -> None: ...


class DemoRestaurantAdapter(VenueAdapter):
    """In-memory adapter for demos and tests; ships with a sample Kenyan menu."""
    MENU = [{"id": 17, "name": "chicken burger", "price": 650}, {"id": 3, "name": "chips", "price": 250},
            {"id": 21, "name": "water", "price": 100}, {"id": 22, "name": "soda", "price": 120},
            {"id": 30, "name": "ugali beef", "price": 450}, {"id": 31, "name": "chapati", "price": 50}]

    def __init__(self):
        self.orders: dict[str, list] = {}

    async def menu_lookup(self, name: str):
        name = name.lower().strip()
        return next((m for m in self.MENU if name == m["name"] or name in m["name"]), None)

    async def order_add(self, item_id, qty, room_id):
        self.orders.setdefault(room_id, []).append((item_id, qty))
        return {"order_lines": len(self.orders[room_id])}

    async def call_staff(self, room_id, reason):
        pass
```

**Safety for physical robots** (*Sustainable AI* topic, physical safety): the agent may only call **whitelisted, non-motion** tools by default. Any motion command such as `navigate` needs the robot's own safety layer to accept it, and the robot vendor's controller always has final authority.


---

# Part VIII — Evaluation, testing, MLOps, performance, CUDA

## 1. Metrics

```python
# ml/eval/metrics.py
from __future__ import annotations


def edit_ops(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """Returns (substitutions, deletions, insertions) via Levenshtein DP with backtrace."""
    R, H = len(ref), len(hyp)
    D = [[0] * (H + 1) for _ in range(R + 1)]
    for i in range(R + 1): D[i][0] = i
    for j in range(H + 1): D[0][j] = j
    for i in range(1, R + 1):
        for j in range(1, H + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, S, Dl, I = R, H, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            S += ref[i - 1] != hyp[j - 1]; i -= 1; j -= 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            Dl += 1; i -= 1
        else:
            I += 1; j -= 1
    return S, Dl, I


def wer(refs: list[list[str]], hyps: list[list[str]]) -> dict[str, float]:
    S = Dl = I = N = 0
    for r, h in zip(refs, hyps):
        s, d, i = edit_ops(r, h); S += s; Dl += d; I += i; N += len(r)
    N = max(N, 1)
    return {"wer": (S + Dl + I) / N, "sub": S / N, "del": Dl / N, "ins": I / N}
```

| Component | Primary metric | Secondary |
|---|---|---|
| ISLR | Top-1 / top-5 accuracy (signer-independent) | Macro-F1 (rare signs), open-set AUROC |
| CSLR | **Gloss WER** (with sub/del/ins breakdown) | Latency per chunk, partial-result stability (how often partials get revised) |
| Fingerspelling | Character error rate | Name accuracy with the lexicon |
| Gloss LM | Perplexity on held-out sentences | WER improvement when fused |
| Gloss→Text | BLEU-4, chrF++ | Human adequacy/fluency (1–5) |
| Text→Gloss | BLEU on glosses, gloss WER | Dictionary coverage (% of glosses producible) |
| NLU | Intent accuracy, slot F1 | Task-completion rate |
| ASR | WER on the **Kenyan in-domain** test set | WER by accent/noise level |
| TTS | MOS | Intelligibility (ASR-WER of the TTS output) |
| Production | Back-translation WER | **Deaf comprehension rate**, naturalness MOS |
| Calibration | ECE | Reliability diagram |
| End-to-end | Task success in scripted scenarios, p50/p95 latency | Clarification rate, clarifications per successful task |
| Fairness | Max gap in WER between groups | §IX-2 |
| Efficiency | ms/utterance, J/utterance, model size MB | gCO₂e per 1,000 utterances |

## 2. Robustness matrix (RQ2): turning "small mistakes" into science

Evaluate the CSLR model on the clean test set and on **controlled perturbations**, plus real recordings under each condition from §III-6:

| Condition | How produced | WER | Δ vs clean |
|---|---|---|---|
| Clean, seen-style signer | test split | | — |
| **Unseen signer** | signer-independent split | | |
| Left-handed signers | real subset | | |
| Viewpoint ±15° / ±30° | `rotate3d` on test | | |
| Speed 0.7× / 1.3× | `time_warp` on test | | |
| Finger jitter σ = 0.02 / 0.05 | synthetic | | |
| Hand dropout 10% / 30% frames | synthetic | | |
| Dim lighting | real recordings (landmark detector degrades naturally) | | |
| Cluttered background | real recordings | | |
| Low-end phone | real recordings | | |
| **+ LM fusion** | decoder ablation | | |
| **+ venue context biasing** | decoder ablation | | |
| **+ clarification (1 turn)** | interactive eval: utterance accuracy after ≤ 1 clarification | | |

```python
# ml/eval/robustness.py
from ml.data import augment as A

PERTURBATIONS = {
    "clean": lambda x: x,
    "rot15": lambda x: A.rotate3d(x, 15),
    "rot30": lambda x: A.rotate3d(x, 30),
    "slow": lambda x: A.time_warp(x, 1.3, 1.3),
    "fast": lambda x: A.time_warp(x, 0.7, 0.7),
    "jitter05": lambda x: A.finger_jitter(x, 0.05),
    "drop30": lambda x: A.hand_dropout(x, p_frame=0.3),
}

def robustness_report(model_fn, test_items, seed=0):
    import numpy as np
    from .metrics import wer
    A.rng = np.random.default_rng(seed)       # reproducible perturbations
    rows = {}
    for name, f in PERTURBATIONS.items():
        refs, hyps = [], []
        for it in test_items:
            hyps.append(model_fn(f(it["xyz"]))); refs.append(it["glosses"])
        rows[name] = wer(refs, hyps)
    return rows
```

Plot it as a bar chart with error bars (bootstrap 95% CI over test utterances). This becomes a figure in the report.

## 3. Release gates (automated)

A model is promoted from `staging` to `production` only if:

| Gate | Rule |
|---|---|
| Quality | WER ≤ production WER − 0.5 pt (absolute) on the frozen gold test set, with a significant bootstrap test |
| No regression | No robustness-matrix row worse by > 1 pt |
| Fairness | Max group gap not larger than production's (§IX-2) |
| Calibration | ECE ≤ 0.05 |
| Latency | p95 chunk latency ≤ budget on the reference GPU and the reference CPU |
| Safety set | 100% correct on the "critical phrases" golden set (allergy, emergency, pain, medication: about 50 sentences) **or** a clarification triggered |

## 4. Testing strategy

| Layer | Tools | Examples |
|---|---|---|
| Unit | pytest | `pack/unpack` round-trip; `normalize_sequence` invariances; KN LM probabilities sum to 1 over the vocab; with no LM, prefix probabilities over an unpruned beam sum to 1, and for peaked (near one-hot) posteriors the beam result equals greedy decoding |
| Property-based | hypothesis | Normalisation is invariant to translation/scale/roll of the input; augmentation never produces NaN outside masked regions |
| Model regression | pytest + golden files | Fixed inputs → outputs must match the stored N-best within tolerance after any refactor |
| Integration | pytest-asyncio, testcontainers (Postgres, Redis, MinIO) | Full WS flow: two clients join a room, one sends text, the other receives `utterance.final` |
| Load | locust (WS user class) | 500 concurrent rooms × 2 clients, landmarks at 30 fps; measure p95 broadcast latency and GPU batch sizes |
| Chaos | kill a worker mid-utterance | XAUTOCLAIM recovery; text relay keeps working |
| Security | bandit, pip-audit, ZAP baseline | Token reuse revokes the family; non-member can't join a room; RLS blocks cross-venue KB reads |

```python
# tests/unit/test_normalize.py
import numpy as np
from hypothesis import given, settings, strategies as st
from ksl_core.normalize import normalize_sequence


def _random_body(T=20, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.3, 0.7, size=(T, 107, 3)).astype(np.float32)
    x[:, 42 + 11, :2] = [0.4, 0.5]; x[:, 42 + 12, :2] = [0.6, 0.5]   # shoulders
    return x


@settings(max_examples=50, deadline=None)
@given(dx=st.floats(-0.2, 0.2), dy=st.floats(-0.2, 0.2), s=st.floats(0.5, 2.0))
def test_translation_scale_invariance(dx, dy, s):
    x = _random_body()
    y = x.copy(); y[..., 0] = (y[..., 0] + dx) * s; y[..., 1] = (y[..., 1] + dy) * s; y[..., 2] *= s
    assert np.allclose(normalize_sequence(x), normalize_sequence(y), atol=1e-4)
```

```python
# tests/unit/test_landmark_codec.py
import numpy as np
from ksl_core.landmarks import LandmarkBatch, pack, unpack

def test_roundtrip():
    rng = np.random.default_rng(1)
    xyz = rng.random((4, 107, 3)).astype(np.float32)
    mask = rng.random((4, 107)) > 0.2
    xyz[~mask] = np.nan
    b = unpack(pack(LandmarkBatch(10, 1234, 30.0, xyz, mask)))
    assert b.seq_start == 10 and b.t0_ms == 1234 and (b.mask == mask).all()
    assert np.allclose(b.xyz[mask], xyz[mask], atol=1e-3)
```

## 5. MLOps lifecycle

```
 data build vN ─► train (MLflow run: config, commit, dataset hash, energy) ─► eval suite ─► registry: "staging"
     ▲                                                                                            │ gates (§3)
     │                                                                                            ▼
 feedback/annotations ◄── production metrics, drift ◄── canary 10% ◄── "production" ◄── manual approval
```

* **Reproducibility:** fixed seeds, `torch.use_deterministic_algorithms(True)` for evaluation, pinned dependencies (`uv.lock`), Docker image digests recorded per run.
* **Model artefact layout:** `models/{task}/{version}/{model.pt, model.onnx, config.yaml, vocab.json, calibration.json, model_card.md, metrics.json}`.
* **Shadow mode:** a new CSLR version can run in *shadow* on live traffic (only with consent), logging its N-best without serving it, then compared offline.
* **Drift monitoring:** track the weekly distribution of mean confidence, the clarification rate and the `<unk>` rate. A sustained shift signals new vocabulary or a new user population.

## 6. Performance engineering and CUDA

### 6.1 Order of optimisation (do these first, before writing kernels)

1. **Batching** across sessions (Part V §2). Usually the biggest win.
2. **Mixed precision** (`bf16`/`fp16` autocast) for 1.5–3× throughput.
3. **`torch.compile`** (`mode="reduce-overhead"` uses CUDA Graphs, which help small-batch latency).
4. **Avoid host↔device ping-pong**: run feature building on the GPU (`torch` version of `build_features`) once the CPU becomes the bottleneck.
5. **ONNX Runtime / TensorRT** export for serving, and INT8 quantisation (post-training, calibrated on 500 val samples). Measure the accuracy drop.
6. **Custom kernels**, only where profiling (`torch.profiler`, Nsight Systems) shows a hot spot.

### 6.2 Where CUDA kernels actually pay off in this project

| Hot spot | Why | Approach |
|---|---|---|
| Landmark normalisation + feature building for many sessions | Many small element-wise ops → launch overhead | One fused kernel (example below) or Triton |
| CTC prefix beam search | The Python loop is slow at beam ≥ 16 or vocab ≥ 2k | C++/CUDA beam decoder (port the algorithm in §V-3.2), or a vectorised torch implementation |
| DTW for the baseline and dictionary medoids | O(T²) per pair, millions of pairs | Anti-diagonal parallel DTW kernel |

### 6.3 Example: fused body-normalisation kernel via `load_inline`

This is a good learning exercise for "exploring CUDA" as the group wants. It fuses centre → scale → rotate for a batch of frames in a single kernel.

```python
# ml/kernels/normalize_cuda.py
import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>

__global__ void normalize_kernel(const float* __restrict__ x, float* __restrict__ y,
                                 const float* __restrict__ center, const float* __restrict__ scale,
                                 const float* __restrict__ cs, int N, int P) {
    // x,y: (N, P, 3); center: (N, 3); scale: (N); cs: (N, 2) = cos, sin of -roll
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * P) return;
    int n = idx / P;
    const float* xi = x + idx * 3;
    float inv = 1.0f / scale[n];
    float px = (xi[0] - center[n*3+0]) * inv;
    float py = (xi[1] - center[n*3+1]) * inv;
    float pz = (xi[2] - center[n*3+2]) * inv;
    float c = cs[n*2+0], s = cs[n*2+1];
    float* yi = y + idx * 3;
    yi[0] = c * px - s * py;
    yi[1] = s * px + c * py;
    yi[2] = pz;                                   // NaN propagates naturally for missing points
}

torch::Tensor normalize_frames(torch::Tensor x, torch::Tensor center, torch::Tensor scale, torch::Tensor cs) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dtype() == torch::kFloat32, "x must be contiguous CUDA float32");
    auto y = torch::empty_like(x);
    int N = x.size(0), P = x.size(1);
    int threads = 256, blocks = (N * P + threads - 1) / threads;
    normalize_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(),
        center.data_ptr<float>(), scale.data_ptr<float>(), cs.data_ptr<float>(), N, P);
    return y;
}
"""
cpp_src = "torch::Tensor normalize_frames(torch::Tensor x, torch::Tensor center, torch::Tensor scale, torch::Tensor cs);"

ext = load_inline(name="ksl_norm", cpp_sources=cpp_src, cuda_sources=cuda_src,
                  functions=["normalize_frames"], verbose=False, extra_cuda_cflags=["-O3"])


def normalize_frames(x, center, scale, roll):
    cs = torch.stack([torch.cos(-roll), torch.sin(-roll)], -1).contiguous()
    return ext.normalize_frames(x.contiguous(), center.contiguous(), scale.contiguous(), cs)

# test: compare against the NumPy reference in ksl_core.normalize on random data (atol 1e-5),
# then benchmark with torch.cuda.Event timing vs. the pure-PyTorch version.
```

Report: correctness test vs the reference, a speed-up table, and the honest conclusion. Often the `torch.compile` version is nearly as fast. Knowing *when* custom CUDA is worth it is the real lesson.

### 6.4 On-device (Kotlin offline mode)

* Export **ISLR (core 300–500 signs)** and the **fingerspelling** model to TFLite or ExecuTorch with INT8 weights. Target < 10 MB and < 30 ms per clip on a mid-range Android phone.
* Ship the **gloss→text rules + translation memory** on the device for the most common phrases.
* The backend exposes `GET /models/mobile/manifest` with version, hash, size and minimum app version. The app downloads over Wi-Fi only.

## 7. Deployment

```yaml
# docker/compose.yaml (dev / single-server pilot)
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment: { POSTGRES_PASSWORD: dev, POSTGRES_DB: ksl }
    volumes: [pg:/var/lib/postgresql/data]
  redis:
    image: redis:7-alpine
    command: ["redis-server", "--appendonly", "yes"]
  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    environment: { MINIO_ROOT_USER: dev, MINIO_ROOT_PASSWORD: devdevdev }
    volumes: [minio:/data]
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.17.2
    command: mlflow server --host 0.0.0.0 --backend-store-uri postgresql://postgres:dev@postgres/ksl --artifacts-destination s3://models
  gateway:
    build: { context: .., dockerfile: docker/gateway.Dockerfile }
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --ws websockets --proxy-headers
    env_file: ../.env
    depends_on: [postgres, redis, minio]
    ports: ["8000:8000"]
  worker-sign:
    build: { context: .., dockerfile: docker/worker-gpu.Dockerfile }
    command: python -m workers.sign_worker
    deploy:
      resources:
        reservations:
          devices: [{ driver: nvidia, count: 1, capabilities: [gpu] }]
  worker-speech:
    build: { context: .., dockerfile: docker/worker-gpu.Dockerfile }
    command: python -m workers.asr_worker
  worker-cpu:
    build: { context: .., dockerfile: docker/gateway.Dockerfile }
    command: python -m workers.translate_worker   # g2t/t2g/nlu/production on CPU is fine
  prometheus: { image: prom/prometheus }
  grafana: { image: grafana/grafana }
volumes: { pg: {}, minio: {} }
```

Pin image tags to specific versions (and digests) in production. The tags above are examples to check before use.

* **Pilot hosting:** one server with a single consumer GPU (e.g. 16–24 GB) handles a pilot of dozens of concurrent rooms. Prefer a data centre in or near Kenya for latency and data residency (Part IX-3).
* **Reverse proxy:** Caddy or Nginx for TLS. WebSocket upgrade headers, `proxy_read_timeout 120s`.
* **Horizontal scaling:** gateways are stateless (scale on connection count). Workers scale on Redis Streams lag (`XPENDING` / stream length). Run Redis with AOF persistence.
* **Backups:** nightly `pg_dump` + MinIO bucket replication. Test restores every month.


---

# Part IX — Ethics, bias, privacy and sustainability

## 1. Principles

1. **Nothing about us without us.** Deaf KSL users are co-designers, annotators, evaluators and paid contributors, not just test subjects. Partner with a Deaf association, for example the Kenya National Association of the Deaf, and with KSL interpreters.
2. **Assist, don't replace.** The system supports communication. It is **not** a certified interpreter. In medical, legal and emergency contexts the app shows a banner recommending a human interpreter, and the agent offers `call_staff`.
3. **Fail safe, not silent.** Low confidence leads to clarification, captions or escalation, never a confident wrong output (Part V §4).
4. **Privacy by design.** Landmarks instead of video in real time. Nothing stored without explicit consent.

## 2. Bias detection and mitigation (Brookings framework applied)

The Brookings paper recommends **algorithmic bias impact statements**, regular audits, diverse design teams, and engagement with affected communities. Applied here:

### 2.1 Bias types from the *Sustainable AI* topic, mapped

| Bias type | How it appears in KSL Bridge | Mitigation |
|---|---|---|
| **Sample bias** | Training signers mostly young, urban (Nairobi), right-handed, from a few schools for the Deaf | Recruitment quotas (§III-6.2); report the dataset composition |
| **Prejudicial bias** | Gloss translations encoding stereotypes (gendered defaults: DOCTOR → "he", NURSE → "she") | Gender-neutral translation by default; MT test set with counter-stereotypical sentences |
| **Exclusion bias** | Dropping "noisy" clips that actually come from regional variants or older signers | Cluster-and-review instead of delete (§III-6.3) |
| **Measurement bias** | Landmark detector worse for darker skin in poor lighting, or for certain sleeve/clothing colours | Evaluate landmark detection rate by lighting and skin-tone groups (Monk Skin Tone scale, self-reported or annotator-estimated **with consent**); add low-light data; report it |
| **Algorithmic bias** | Context biasing favours common restaurant phrases, so unusual requests get "corrected" into common ones | Cap bias strength; adversarial test set; clarification instead of silent correction |

### 2.2 Fairness evaluation

```python
# ml/eval/fairness.py
from collections import defaultdict
from .metrics import wer


def group_wer(items, hyps, group_key):
    by = defaultdict(lambda: ([], []))
    for it, h in zip(items, hyps):
        refs, hs = by[it["meta"][group_key]]
        refs.append(it["glosses"]); hs.append(h)
    res = {g: wer(r, h)["wer"] for g, (r, h) in by.items()}
    return res, max(res.values()) - min(res.values())      # per-group WER and worst-case gap

# groups: gender, age_band, region, dominant_hand, hearing_status (Deaf / HoH / CODA),
#         device_tier, lighting, skin_tone_band (only with consent)
```

Publish per-group results in every **model card**, even (especially) when the gaps are uncomfortable.

### 2.3 Bias impact statement (template, adapted from Brookings)

1. What will the system decide or produce, and who is affected?
2. Which groups could be harmed by errors, and how severely (hospital vs restaurant)?
3. What data was used, and whose signing does it represent? Whose doesn't it represent?
4. What were the per-group results? What was done about the gaps?
5. How can affected users contest or correct outputs (feedback button, human escalation)?
6. Who reviews this statement, and when (each major release)?

## 3. Privacy and data protection

The project is in Kenya, so it falls under the **Data Protection Act, 2019**, overseen by the Office of the Data Protection Commissioner (ODPC). Verify the current requirements, such as registration as a data controller/processor and data protection impact assessments for sensitive data, with your supervisor or the university's legal office. The design below supports compliance:

| Requirement | Implementation |
|---|---|
| Lawful basis + consent | `consents` table with versioned text; KSL consent video; granular scopes |
| Data minimisation | Landmarks only in real time; face landmarks limited to 40 points; video only for contribution |
| Purpose limitation | Training only uses samples whose consent includes `contribute_*` |
| Right of access and erasure | `GET /me/export`, `DELETE /me` (cascade + S3 deletion job + exclusion of the user's samples from future dataset builds; document that already-trained models aren't retroactively changed) |
| Sensitive data | Disability-related data counts as sensitive personal data. Encrypt at rest (Postgres TDE/disk encryption, MinIO SSE), restrict access by role, keep an audit log |
| Retention | Real-time landmarks: not stored. Transcripts: off by default, 30-day default when on. Dataset clips: as long as consent lasts |
| Pseudonymisation | `signer_code` in datasets, never names. Annotators see face-blurred video unless the task needs faces (non-manual annotation) |
| Children | Exclude minors from data collection unless a dedicated ethics approval and guardian-consent process exists |

## 4. Sustainability of AI (van Wynsberghe; Wu et al.; Ren & Wierman)

The course distinguishes **AI for sustainability** (our project serves SDG 10 *Reduced inequalities*, SDG 8 *Decent work* through the employment RQ, and SDG 4/3 for education and health access) from the **sustainability of AI** (the footprint of building and running it). Address both:

| Lever | Concrete practice |
|---|---|
| Small models | d = 256 encoders (≈ 10–20 M params), not billion-parameter models. Report params and FLOPs |
| Efficient training | Self-supervised pretraining once, many cheap fine-tunes; early stopping; mixed precision; hyperparameter search with small budgets (Bayesian/ASHA), not exhaustive grids |
| Measure | **CodeCarbon** in every training run; log kWh and estimated CO₂e to MLflow; report totals in the thesis |
| Inference efficiency | On-device ISLR (no server round trip), INT8, batching, autoscale workers to zero at night |
| Location awareness (Ren & Wierman, environmental equity) | When renting cloud GPUs, prefer regions with low-carbon grids. Kenya's grid is largely renewable (geothermal, hydro). Note water-use implications of data-centre cooling |
| Hardware lifecycle | Reuse existing lab GPUs; avoid buying hardware for one-off experiments |

```python
# ml/train/carbon.py
from codecarbon import EmissionsTracker
import mlflow

def tracked(fn, project="ksl-bridge"):
    def wrap(*a, **kw):
        tr = EmissionsTracker(project_name=project, log_level="error", save_to_file=False)
        tr.start()
        try:
            return fn(*a, **kw)
        finally:
            kg = tr.stop()
            mlflow.log_metric("emissions_kgco2e", kg or 0.0)
    return wrap
```

## 5. Generative signing and AI-and-art concerns

Sign production is **generative**. Lessons from the AI-and-art discussion apply:

* **Provenance:** produced signing comes from real signers' clips. Credit contributors in the app, if they wish, and get consent for this specific use (dictionary and avatar).
* **No impersonation:** never animate a real person's likeness without consent. Use a stylised avatar.
* **Validation:** generated signing can be fluent-looking nonsense. Every dictionary entry and production rule is validated by Deaf reviewers before release.

---

# Part X — Roadmap, team split and Kotlin contract

## 1. Phased roadmap (adjust to your semester calendar)

| Phase | Weeks | Backend deliverables | Exit criterion |
|---|---|---|---|
| **0. Foundations** | 1–2 | Repo, CI, compose stack, auth, rooms, WS text relay, landmark codec shared with Kotlin | Two phones chat by text through a room |
| **1. Data + baselines** | 2–5 | Dataset access requests, landmark extraction pipeline, normalisation, splits, k-NN DTW + RF baselines, ethics approval, recording protocol | Baseline ISLR accuracy reported on signer-independent split |
| **2. Core recognition** | 4–9 | MPM pretraining, ISLR Transformer, first own-data recording sessions, CSLR v0 on synthetic + early real data, KN LM, beam decoder | Live **sign → text** captions in a room (partial + final) |
| **3. Two-way** | 7–11 | Text→Gloss (rules + v0 neural), sign production (dictionary + blending), bootstrap ASR/TTS adapters, speech → sign path | Full scenario 1 (one phone, both directions) |
| **4. Robustness + context** | 9–13 | Calibration, clarification policy, venue RAG + biasing, robustness matrix, fairness eval, personalisation v1 | RQ2/RQ4 results tables |
| **5. Agent + robot** | 11–14 | Dialogue manager, demo restaurant adapter, device API, kiosk mode | Scenario 3 (mock restaurant) works end-to-end |
| **6. Own speech models + evaluation** | 12–16 | Trained ASR (and TTS if time allows) vs bootstrap comparison; user study (SUS, task success) with Deaf participants | RQ3/RQ5 results; thesis figures |
| **7. Hardening** | 15–17 | Load tests, security review, model cards, bias impact statement, deployment docs | Demo-day ready |

The **critical path** is data. Request dataset access and ethics approval in **week 1**.

## 2. Team split (example for a 4–5 person group)

| Role | Owns | Backend modules |
|---|---|---|
| ML lead: recognition | ISLR, CSLR, MPM, decoder, robustness | `ml/models/{pose_encoder,islr,cslr,mpm}`, `ml/decoding`, `ml/eval` |
| ML: language | LM, Gloss⇄Text, NLU, RAG, production | `ml/models/{seq2seq,production}`, `app/rag`, `app/pipeline/text2gloss_*` |
| Backend lead | Gateway, auth, rooms, WS protocol, workers, deployment | `app/{auth,rooms,ws,routers}`, `workers/`, `docker/` |
| Kotlin lead | App, on-device landmarks, avatar renderer, offline models | (contract below) |
| Research/community lead | Ethics, recruitment, annotation, user studies, report | `tools/annotate`, `docs/`, fairness reports |

Everyone writes tests for their own modules. The backend lead reviews all protocol changes. The Kotlin and backend leads **co-own `landmark_spec_v1.json` and the WS message catalogue**.

## 3. Kotlin ⇄ backend contract checklist

- [ ] `landmark_spec_v1.json`: point order, `FACE_IDX`, slices, `SPEC_VERSION`. Kotlin code is generated from it.
- [ ] Coordinates: MediaPipe image-normalised x, y ∈ [0, 1] (origin top-left, y down), z relative. **No mirroring on the client** for front cameras (send unmirrored; the server handles handedness). Document this, because it's a classic bug.
- [ ] Frame batching: 4 frames per binary message, `t0_ms` from a monotonic clock, `fps` measured (not assumed).
- [ ] Audio: PCM16 LE mono 16 kHz, 20 ms frames (640 bytes).
- [ ] WS envelope with `seq`/`ack`, `hello` first, heartbeat 15 s, resume logic.
- [ ] Partial-caption rendering rule: replace by `id`, apply only a higher `revision`.
- [ ] Pose stream format for the avatar (`0x03`): same 107-point layout, float16, `fps` in the preceding JSON message. Kotlin retargets points onto the avatar skeleton (IK for arms, direct finger angles from bone vectors).
- [ ] Clarification UI: options as large tappable buttons + avatar signing the question.
- [ ] Feedback buttons on every final utterance (👍 / ✏️).
- [ ] Error-code handling table (§VI-4.4), especially 4009 backpressure → reduce fps.
- [ ] Offline model manifest and fallback when the server is unreachable.
- [ ] Accessibility: high-contrast captions, adjustable caption size, vibration alerts for incoming messages (Deaf users won't hear notification sounds).

---

# Appendices

## A. Glossary

| Term | Meaning |
|---|---|
| **Gloss** | Written label for a sign (CAPITALS), e.g. `WATER`. Not a translation |
| **ISLR / CSLR** | Isolated / Continuous Sign Language Recognition |
| **CTC** | Connectionist Temporal Classification: alignment-free sequence loss with a blank token |
| **Non-manual markers** | Facial expressions, head and body movements that carry grammatical meaning |
| **Co-articulation** | How adjacent signs influence each other's form in continuous signing |
| **Fingerspelling** | Spelling words letter by letter with handshapes |
| **Classifier (CL)** | Handshapes representing classes of objects and their movement/location |
| **Interlingua / IR** | Our shared internal meaning representation (`UtteranceIR`) |
| **Shallow fusion** | Adding weighted LM log-probabilities during beam search |
| **ECE** | Expected Calibration Error |
| **MPM** | Masked Pose Modelling: self-supervised pretraining on unlabelled signing |
| **CODA** | Child of Deaf Adult, a hearing native signer |

## B. Reading list mapped to modules

| Module | Read |
|---|---|
| Agent framing | AIMA Ch. 2 (agents, PEAS, environments); course *Intelligent Agents* slides |
| Beam search / decoding | AIMA Ch. 3–4 (search, beam search); course *Problem Solving Agents* |
| Temporal models, Kalman, HMM | AIMA Ch. 15 |
| Clarification policy | AIMA Ch. 16 (utility, value of information), Ch. 17 (POMDPs) |
| ML methodology | AIMA Ch. 18; course *Theories of ML*, *ML Cheat Sheet*, CRISP-DM slides |
| LMs and smoothing | AIMA Ch. 22 (n-grams); Chen & Goodman (1999) on Kneser–Ney |
| MT and speech | AIMA Ch. 23; Vaswani et al. (2017) *Attention Is All You Need*; Graves et al. (2006) CTC; Gulati et al. (2020) Conformer |
| Pretraining | Han et al. (2021) *Pre-trained models: Past, present and future* (AI Open) |
| RAG | Guest lecture notebook + slides (naive RAG, chunking, failure modes) |
| Sign language AI | KSL word-based pose dataset (*Data in Brief*, 2025); surveys of CSLR (e.g. on RWTH-PHOENIX-2014T); Saunders et al. *Progressive Transformers for Sign Language Production* (2020) |
| Ethics and bias | Brookings, *Algorithmic bias detection and mitigation*; course *Sustainable AI* topic |
| Sustainability | van Wynsberghe (2021) *Sustainable AI* (AI and Ethics); Wu et al. (2022) *Sustainable AI: Environmental Implications…*; Ren & Wierman (2024, HBR) *The Uneven Distribution of AI's Environmental Impacts*; the sustainable development literature review (three pillars) |

## C. Open decisions for the group (decide and record in `docs/decisions/`)

1. Exact interpretation of "no APIs" (Part I §7). Is MediaPipe-as-sensor accepted? Are bootstrap ASR/TTS acceptable until our own models are trained?
2. Target vocabulary size for v1 (recommendation: 300–500 glosses focused on 2 service domains).
3. Which two service domains to focus on (recommendation: restaurant + hospital reception).
4. Avatar style (2D stylised vs 3D rigged) and who builds the retargeting (Kotlin side).
5. Swahili scope: speech-in only, or also text/speech out?
6. Hosting for the pilot (lab server vs cloud), and data residency.
7. Whether transcripts are stored for research, which needs consent flows and retention rules.

## D. Week-1 checklist

- [ ] Send dataset access requests (KSL pose dataset on Zenodo; Kaggle download; contact the authors).
- [ ] Submit the ethics application; contact a Deaf association and KSL interpreters.
- [ ] Create the repo with the layout from Part II §3, CI (ruff, mypy, pytest), and the docker compose stack.
- [ ] Implement `ksl_core/landmarks.py` + codec tests; agree `landmark_spec_v1.json` with the Kotlin lead.
- [ ] Implement auth + rooms + WS text relay (Phase 0 exit criterion).
- [ ] Extract landmarks for the Kaggle KSL videos; run the k-NN DTW baseline. **First number on the board.**
