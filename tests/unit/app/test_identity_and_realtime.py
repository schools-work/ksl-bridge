from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from app.modules.identity.domain.tokens import OpaqueToken, lockout_seconds
from app.modules.identity.domain.user import validate_password, validate_phone
from app.modules.identity.infrastructure.jwt import JwtService
from app.modules.realtime.protocol import ProtocolError, parse_client
from app.modules.venue.infrastructure.api_keys import HmacDeviceKeys
from app.platform.principal import Principal


def test_phone_and_password_rules() -> None:
    assert validate_phone("0712 345678") == "+254712345678"
    with pytest.raises(ValueError):
        validate_password("short")
    assert lockout_seconds(3) == 0 and lockout_seconds(5) == 30


def test_jwt_roundtrip_and_jwks() -> None:
    svc = JwtService("", "", "test", "ksl-bridge", "ksl-bridge-clients", 60)
    p = Principal(user_id=uuid4(), roles=frozenset({"signer"}), mfa=True)
    q = svc.decode(svc.issue(p))
    assert q.user_id == p.user_id and q.roles == p.roles and q.mfa
    g = Principal(guest_id=uuid4(), scope="room:x")
    assert svc.decode(svc.issue(g)).guest_id == g.guest_id
    assert svc.jwks()["keys"][0]["kty"] == "OKP"


def test_device_keys_and_opaque_tokens() -> None:
    k = HmacDeviceKeys("pepper")
    key, prefix, digest = k.new_key()
    assert k.prefix_of(key) == prefix and k.verify(key, digest) and not k.verify(key + "x", digest)
    t = OpaqueToken.new()
    assert OpaqueToken.hash(t.raw) == t.digest


def test_protocol_validation() -> None:
    env, msg = parse_client('{"v":1,"type":"text.send","seq":3,"ts":1,"data":{"text":"hi"}}')
    assert env.seq == 3 and msg.text == "hi"
    for bad in (
        '{"v":1,"type":"text.send","seq":1,"ts":1,"data":{}}',  # missing field
        '{"v":1,"type":"welcome","seq":1,"ts":1,"data":{}}',  # server-only type
        '{"v":2,"type":"ping","seq":1,"ts":1,"data":{"t":1}}',
    ):  # bad version
        with pytest.raises(ProtocolError):
            parse_client(bad)
    assert datetime.now(UTC) - timedelta(seconds=1) < datetime.now(UTC)
