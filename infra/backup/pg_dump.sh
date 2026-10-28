#!/usr/bin/env bash
# Nightly logical backup (custom format) → MinIO/S3 bucket "backups".
set -euo pipefail
: "${KSL_DATABASE_URL_SYNC:?}"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="/tmp/ksl-${STAMP}.dump"
pg_dump --format=custom --no-owner --file "$OUT" "$KSL_DATABASE_URL_SYNC"
aws --endpoint-url "${KSL_S3_ENDPOINT}" s3 cp "$OUT" "s3://backups/postgres/ksl-${STAMP}.dump"
rm -f "$OUT"
echo "backup ok: ksl-${STAMP}.dump"
