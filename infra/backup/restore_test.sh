#!/usr/bin/env bash
# Monthly restore test into a scratch database; fails loudly if the invariants don't hold after restore.
set -euo pipefail
: "${RESTORE_URL:?postgresql://… of a scratch database}"
LATEST=$(aws --endpoint-url "${KSL_S3_ENDPOINT}" s3 ls s3://backups/postgres/ | sort | tail -1 | awk '{print $4}')
aws --endpoint-url "${KSL_S3_ENDPOINT}" s3 cp "s3://backups/postgres/${LATEST}" /tmp/restore.dump
pg_restore --clean --if-exists --no-owner --dbname "$RESTORE_URL" /tmp/restore.dump
psql "$RESTORE_URL" -v ON_ERROR_STOP=1 -f db/checks/schema_invariants.sql
echo "restore test ok (${LATEST})"
