"""MinIO/S3 helpers: presigned URLs only, so large files never pass through the API process."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from app.settings import get_settings


@lru_cache
def s3_client() -> Any:
    import boto3  # imported lazily: unit tests don't need it

    s = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=s.s3_endpoint,
        aws_access_key_id=s.s3_access_key,
        aws_secret_access_key=s.s3_secret_key,
        region_name=s.s3_region,
    )


def presigned_put(bucket: str, key: str, content_type: str, expires: int = 900) -> str:
    return str(
        s3_client().generate_presigned_url(
            "put_object", Params={"Bucket": bucket, "Key": key, "ContentType": content_type}, ExpiresIn=expires
        )
    )


def presigned_get(bucket: str, key: str, expires: int = 900) -> str:
    return str(
        s3_client().generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires)
    )


def split_uri(uri: str) -> tuple[str, str]:
    """'s3://bucket/key/path' → ('bucket', 'key/path')."""
    if not uri.startswith("s3://"):
        raise ValueError(f"not an s3 uri: {uri}")
    bucket, _, key = uri[5:].partition("/")
    return bucket, key
