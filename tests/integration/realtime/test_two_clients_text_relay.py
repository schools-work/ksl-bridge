import json
import uuid

import pytest

pytestmark = pytest.mark.integration


def _msg(type_: str, seq: int, **data: object) -> str:
    return json.dumps({"v": 1, "type": type_, "seq": seq, "ts": 0, "data": data})


def _recv_until(ws, type_: str, limit: int = 20) -> dict:  # type: ignore[no-untyped-def,type-arg]
    for _ in range(limit):
        m = json.loads(ws.receive_text())
        if m["type"] == type_:
            return m
    raise AssertionError(f"no {type_}")


def test_signer_and_qr_guest_exchange_text(client) -> None:  # type: ignore[no-untyped-def]
    email = f"s{uuid.uuid4().hex[:6]}@test.ke"
    client.post(
        "/auth/register",
        json={"display_name": "Signer", "email": email, "password": "Strong-pass-123", "role": "signer"},
    )
    tok = client.post("/auth/login", json={"email": email, "password": "Strong-pass-123"}).json()["access_token"]
    room = client.post("/rooms", headers={"Authorization": f"Bearer {tok}"}, json={"role": "signer"}).json()
    guest = client.post("/auth/guest", json={"join_code": room["join_code"], "display_name": "Waiter"}).json()
    with client.websocket_connect("/ws/v1") as a, client.websocket_connect("/ws/v1") as b:
        a.send_text(_msg("hello", 1, token=tok, client={"app": "test"}))
        b.send_text(_msg("hello", 1, token=guest["access_token"], client={"app": "test"}))
        assert _recv_until(a, "welcome")["data"]["session_id"]
        _recv_until(b, "welcome")
        a.send_text(_msg("room.join", 2, room_id=room["room_id"], role="signer"))
        b.send_text(_msg("room.join", 2, room_id=room["room_id"], role="speaker"))
        _recv_until(a, "room.state")
        _recv_until(b, "room.state")
        b.send_text(_msg("text.send", 3, text="Karibu! What would you like?"))
        got = _recv_until(a, "utterance.final")
        assert got["data"]["utterance"]["best"]["text"] == "Karibu! What would you like?"


def test_unauthenticated_socket_is_closed(client) -> None:  # type: ignore[no-untyped-def]
    from starlette.websockets import WebSocketDisconnect

    with client.websocket_connect("/ws/v1") as ws:
        ws.send_text(_msg("hello", 1, token="nope", client={}))
        with pytest.raises(WebSocketDisconnect) as e:
            ws.receive_text()
        assert e.value.code == 4001
