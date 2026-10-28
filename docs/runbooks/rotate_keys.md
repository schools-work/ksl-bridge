# Runbook: rotate secrets and keys

* **JWT signing key:** generate a new Ed25519 pair (`make keys`), add it with a new `KSL_JWT_KID`, and publish both public keys in JWKS for one access-token TTL (15 min). Then remove the old private key.
* **Device API key:** `POST /venues/{id}/devices/{device_id}/rotate-key` returns the new key once. Update the device, then revoke the old one.
* **API-key pepper:** rotating it invalidates every device key. Plan a coordinated re-issue.
* **Data-encryption key:** re-encrypt `identity.totp_credentials.secret_enc` and `agent.venue_tools.config_enc` with a migration job, then retire the old key.
