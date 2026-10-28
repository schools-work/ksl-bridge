"""Generate a DEV Ed25519 key pair + data-encryption key and print .env lines."""

from __future__ import annotations

import base64
import os
import secrets

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

if __name__ == "__main__":
    k = Ed25519PrivateKey.generate()
    priv = (
        k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
        .decode()
        .replace("\n", "\\n")
    )
    pub = (
        k.public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode()
        .replace("\n", "\\n")
    )
    print(f'KSL_JWT_PRIVATE_KEY_PEM="{priv}"')
    print(f'KSL_JWT_PUBLIC_KEY_PEM="{pub}"')
    print(f"KSL_DATA_ENCRYPTION_KEY={base64.b64encode(os.urandom(32)).decode()}")
    print(f"KSL_API_KEY_PEPPER={secrets.token_urlsafe(32)}")
    print(f"KSL_OTP_PEPPER={secrets.token_urlsafe(32)}")
