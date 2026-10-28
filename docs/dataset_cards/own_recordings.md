# Dataset card: KSL Bridge own recordings

| Field | Value |
|---|---|
| Source slug | `ksl-bridge-own` |
| Licence | Proprietary, consented per signer (`privacy.consents`, scope `contribute_*`) |
| Content | Continuous service sentences (restaurant, hospital, bank, transport), free dialogues, a core vocabulary (500 glosses × 3), fingerspelling |
| Protocol | `tools/collection/recording_protocol.md` |
| Signers | Target ≥ 20 (aim 30+), balanced by gender, age band, region, dominant hand; Deaf, hard-of-hearing and CODA signers labelled |

## Guarantees enforced by the database
* No sample without an active contribution consent (`dataset.require_contribution_consent`).
* Signer-independent splits (`dataset.enforce_build_member`).
* Withdrawal or erasure → `excluded_at` set → never in new builds.

## Ethics
University ethics approval, a KSL consent video, paid participation, a Deaf association as a partner, face-blurred derivatives for annotators who don't need faces.
