# Data protection impact assessment (Kenya Data Protection Act, 2019)

> Draft. Verify the current obligations (controller/processor registration with the ODPC, DPIA requirements) with the supervisor or the university's legal office.

## Processing activities
| Activity | Data | Basis | Retention (`privacy.retention_policies`) |
|---|---|---|---|
| Real-time translation | Landmarks, audio, text (in memory) | Contract / legitimate interest | 0 days (not stored) |
| Stored transcripts | Utterances | Consent `store_transcripts` | 30 days |
| Dataset contribution | Landmarks, video, audio, demographics | Consent `contribute_*`, `demographics` | 1,825 days or until withdrawal |
| Personalisation | Embeddings, LoRA adapters | Consent `personalisation` | Until withdrawal |
| Security | Audit events, refresh tokens | Legitimate interest | 730 days |

## Sensitive data
Disability-related data and biometric-like signals are **sensitive personal data**. Controls: data minimisation (landmarks, not video), RLS, encryption at rest, least-privilege roles, audit, pseudonymous `signer_code`.

## Rights
Access → `GET /me/export` · Rectification → `PATCH /me` · Erasure → `DELETE /me` (`privacy.erase_user` + object deletion job) · Withdraw consent → `DELETE /me/consents/{scope}`.

## Residual risks
Models trained before an erasure are not retrained retroactively; the data is excluded from all future builds. Users are told this in the consent text.
