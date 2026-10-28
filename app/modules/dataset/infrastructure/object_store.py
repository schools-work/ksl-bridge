"""Object-store helpers for dataset artefacts (checksums, existence)."""

from __future__ import annotations

from app.platform.s3 import s3_client, split_uri


def exists(uri: str) -> bool:
    bucket, key = split_uri(uri)
    try:
        s3_client().head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def delete_prefix(bucket: str, prefix: str) -> int:
    n = 0
    paginator = s3_client().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        objs = [{"Key": o["Key"]} for o in page.get("Contents", [])]
        if objs:
            s3_client().delete_objects(Bucket=bucket, Delete={"Objects": objs})
            n += len(objs)
    return n
