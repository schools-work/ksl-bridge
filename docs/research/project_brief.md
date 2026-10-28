# Project brief

## Origin

A group project at Strathmore University (School of Computing and Engineering Sciences), framed by the hackathon title:

> **Develop a technology-based solution to bridge communication barriers for sign language users.**

The starting observations:

* Service robots and kiosks are appearing in restaurants and public spaces, but they're not designed for Deaf and hard-of-hearing people.
* Few Kenyans know Kenyan Sign Language (KSL), so everyday interactions often fail: ordering food, asking for directions, reception desks, employment.

## The problem, stated precisely

> How can a Deaf or hard-of-hearing KSL user communicate naturally and in real time with a hearing person, or an intelligent service machine, that does not know KSL, without a human interpreter, and with enough robustness that ordinary variation in signing ("small mistakes") doesn't break the conversation?

**Terminology:** *Deaf* and *hard of hearing* are not interchangeable, and many hard-of-hearing people don't use sign language. Target population: **Deaf and hard-of-hearing people, particularly KSL users**, who face barriers with people and services that don't understand KSL. The system also supports users who prefer **text** both ways.

## What the system is (and is not)

* **It is** a multimodal communication bridge. Sign, speech and text all go into one shared meaning representation (`UtteranceIR`), and any of the three comes out. On top sits a goal-based agent so robots and kiosks can act on what a signer asks.
* **It is not** a certified interpreter. In medical, legal and emergency contexts it recommends a human interpreter and can call staff.
* **It is not** "nine translators". With a hub representation, N inputs and M outputs need N + M components, not N × M.

## The five hidden problems

1. **Perception:** what are the hands, body and face doing? (landmarks)
2. **Sign recognition:** which signs, in which order? (ISLR/CSLR)
3. **Language understanding:** what does the sequence mean in context? (LM, NLU, dialogue context)
4. **Translation:** how is it expressed in English/Swahili, or in KSL? (gloss ⇄ text)
5. **Generation:** text, speech (TTS) or signing (pose sequences for an avatar).

## "Small mistakes", technically

| Source | Name | Mitigation |
|---|---|---|
| Slightly wrong handshape | intra-class variation | normalisation, augmentation, metric-learning embeddings |
| Faster or slower movement | temporal variation | time-warp augmentation, CTC |
| Different signer or dominant hand | domain shift | signer-independent splits, mirroring, personalisation |
| Bad angle, lighting, out of frame | sensor noise / occlusion | landmark dropout, masks, interpolation |
| Unsure between two signs | ambiguity | N-best + LM + venue context, clarification |
| Unknown sign | open set | rejection, fingerspelling module |
| Accent, noise, code-switching | ASR robustness | Kenyan data, noise augmentation, custom LM |

## Differentiation

> A context-aware, error-tolerant, multimodal KSL communication engine for real-time interactions between KSL users, non-signers and intelligent public-service interfaces. It uses its own trained models and works with limited connectivity.

Kenyan context × two-way multimodal × robustness research × service-agent integration.

## Canonical scenarios

1. Face to face on one phone.
2. Face to face on two phones joined by QR code (WebSocket rooms).
3. Restaurant robot or kiosk ordering, with the waiter's tablet in the same room.
4. Hospital reception: domain vocabulary, RAG over the FAQ, a critical-domain safety policy.
5. Offline or low connectivity: on-device ISLR core, sync later.
