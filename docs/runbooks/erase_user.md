# Runbook: user erasure request

1. Verify the requester's identity (logged-in session or verified support channel).
2. `DELETE /me`, or as an admin: `SELECT privacy.erase_user('<user_id>', '<admin_id>');`
3. That queues the job `erase_user_objects`, which deletes MinIO objects (clips, landmarks, adapters) for the user's `signer_code`.
4. Confirm `privacy.erasure_requests.status = 'succeeded'` and keep the report.
5. Tell the user what was removed, and that already-trained models are not retrained retroactively.
