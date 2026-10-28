"""AES-GCM envelope for small secrets at rest (TOTP secrets, adapter credentials)."""

from __future__ import annotations

import base64
import hashlib
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from app.settings import get_settings


def _key() -> bytes:
    raw = get_settings().data_encryption_key
    if raw:
        key = base64.b64decode(raw)
        if len(key) != 32:
            raise ValueError("KSL_DATA_ENCRYPTION_KEY must be 32 bytes (base64)")
        return key
    if get_settings().is_production:
        raise RuntimeError("KSL_DATA_ENCRYPTION_KEY is required in staging/prod")
    return hashlib.sha256(b"ksl-bridge-dev-only-key").digest()  # dev/test only


def encrypt(plaintext: bytes, aad: bytes = b"ksl") -> bytes:
    nonce = os.urandom(12)
    return nonce + AESGCM(_key()).encrypt(nonce, plaintext, aad)


def decrypt(blob: bytes, aad: bytes = b"ksl") -> bytes:
    return AESGCM(_key()).decrypt(blob[:12], blob[12:], aad)
