# ADR-0002: Clients stream landmarks, not video, in real time

* **Status:** accepted
* **Date:** 2026-10-27

## Context
Kenyan mobile data is expensive and variable, faces are sensitive personal data, and server-side video decoding adds latency and cost.

## Decision
The Kotlin app runs MediaPipe Holistic on-device and streams a **107-point subset** (hands 2 × 21, upper pose 25, face 40) in the `KSLB` binary format (`contracts/landmarks/landmark_spec_v1.json`): float16, 4 frames per message, about 20 KB/s at 30 fps. Raw video is uploaded only in consented dataset-contribution mode.

## Consequences
+ Privacy (no faces leave the phone), low bandwidth, lower latency, smaller models.
− Recognition quality is capped by the on-device landmark detector. Detector failures (e.g. low light) become measurement bias that we must evaluate (`docs/bias_impact_statement.md`).
− Kotlin and Python must share index maps exactly. Hence `contracts/` and code generation.
