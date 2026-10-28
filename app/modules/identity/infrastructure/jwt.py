"""EdDSA (Ed25519) JWTs with kid; JWKS publication. Ephemeral keys in dev/test if none configured."""

from __future__ import annotations

import base64
import uuid
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Any

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from app.platform.principal import Principal
from app.settings import get_settings

ALG = "EdDSA"


class JwtService:
    def __init__(self, private_pem: str, public_pem: str, kid: str, issuer: str, audience: str, ttl: int):
        self.kid, self.issuer, self.audience, self.ttl = kid, issuer, audience, ttl
        if private_pem:
            self._priv = serialization.load_pem_private_key(private_pem.replace("\\n", "\n").encode(), password=None)
            self._pub = serialization.load_pem_public_key(public_pem.replace("\\n", "\n").encode())
        else:
            if get_settings().is_production:
                raise RuntimeError("KSL_JWT_PRIVATE_KEY_PEM is required in staging/prod")
            key = Ed25519PrivateKey.generate()  # dev/test: tokens die with the process
            self._priv, self._pub = key, key.public_key()

    def issue(self, p: Principal, ttl: int | None = None) -> str:
        now = datetime.now(UTC)
        claims: dict[str, Any] = {
            "sub": p.subject,
            "roles": sorted(p.roles),
            "iat": now,
            "nbf": now,
            "exp": now + timedelta(seconds=ttl or self.ttl),
            "iss": self.issuer,
            "aud": self.audience,
            "jti": uuid.uuid4().hex,
            "mfa": p.mfa,
        }
        if p.venue_id:
            claims["venue"] = str(p.venue_id)
        if p.scope:
            claims["scope"] = p.scope
        return jwt.encode(claims, self._priv, algorithm=ALG, headers={"kid": self.kid})  # type: ignore[arg-type]

    def decode(self, token: str) -> Principal:
        c = jwt.decode(
            token,
            self._pub,
            algorithms=[ALG],
            audience=self.audience,
            issuer=self.issuer,  # type: ignore[arg-type]
            options={"require": ["exp", "sub", "iat"]},
            leeway=10,
        )
        kind, _, ident = str(c["sub"]).partition(":")
        uid = uuid.UUID(ident) if ident else None
        return Principal(
            user_id=uid if kind == "user" else None,
            guest_id=uid if kind == "guest" else None,
            device_id=uid if kind == "device" else None,
            venue_id=uuid.UUID(c["venue"]) if c.get("venue") else None,
            roles=frozenset(c.get("roles", [])),
            scope=c.get("scope"),
            mfa=bool(c.get("mfa")),
        )

    def jwks(self) -> dict[str, Any]:
        assert isinstance(self._pub, Ed25519PublicKey)
        raw = self._pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        x = base64.urlsafe_b64encode(raw).rstrip(b"=").decode()
        return {"keys": [{"kty": "OKP", "crv": "Ed25519", "x": x, "kid": self.kid, "alg": ALG, "use": "sig"}]}


@lru_cache
def get_jwt() -> JwtService:
    s = get_settings()
    return JwtService(
        s.jwt_private_key_pem, s.jwt_public_key_pem, s.jwt_kid, s.jwt_issuer, s.jwt_audience, s.access_ttl_seconds
    )
