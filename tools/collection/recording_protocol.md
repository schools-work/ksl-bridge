# Recording protocol (own continuous-signing dataset)

> Requires university ethics approval **before** the first session. Consent is given in KSL (a consent video) and in writing, scopes `contribute_landmarks` / `contribute_video` / `demographics` (optional). Participants are paid.

## Signers
≥ 20 (aim 30+), balanced by gender, age band, region and school for the Deaf, and dominant hand. Hard-of-hearing KSL users and hearing CODAs are labelled separately (`dataset.signers.hearing_status`).

## Conditions matrix (each signer covers ≥ 6 cells)
| Factor | Levels |
|---|---|
| Device tier | low · mid · high (≥ 3 phone models) |
| Lighting | bright · dim · outdoor shade |
| Background | plain · cluttered |
| Posture | seated · standing |
| Distance | 0.8 m · 1.2 m · 2 m |

## Content per signer
1. **Service scripts:** `scripts/restaurant.csv` (60), `scripts/hospital.csv` (60), `scripts/bank.csv` (40), `scripts/transport.csv` (40). Read the English, then sign naturally in KSL (not signed English).
2. **Free role-play** with a hearing participant (10 minutes per domain). Record the hearing participant's audio too (Kenyan English/Swahili ASR test data).
3. **Core vocabulary:** 500 glosses × 3 repetitions.
4. **Fingerspelling:** 30 names and places, plus numbers 0–100.

## After the session
Upload through `POST /datasets/uploads`, then `POST /datasets/samples` (with `session_id`, `consent_id`). Landmarks are extracted server-side; samples enter `annotation.tasks`.
