import uuid

import pytest

pytestmark = pytest.mark.integration


def _user(client) -> str:  # type: ignore[no-untyped-def]
    email = f"a{uuid.uuid4().hex[:6]}@test.ke"
    client.post("/auth/register", json={"display_name": "A", "email": email, "password": "Strong-pass-123"})
    return client.post("/auth/login", json={"email": email, "password": "Strong-pass-123"}).json()["access_token"]


def test_room_is_invisible_to_non_members(client) -> None:  # type: ignore[no-untyped-def]
    owner, outsider = _user(client), _user(client)
    room = client.post("/rooms", headers={"Authorization": f"Bearer {owner}"}, json={}).json()
    r = client.get(f"/rooms/{room['room_id']}", headers={"Authorization": f"Bearer {outsider}"})
    assert r.status_code == 404  # RLS: "not found or not yours"
    joined = client.post(
        "/rooms/join",
        headers={"Authorization": f"Bearer {outsider}"},
        json={"join_code": room["join_code"], "role": "observer"},
    )
    assert joined.status_code == 200
    assert client.get(f"/rooms/{room['room_id']}", headers={"Authorization": f"Bearer {outsider}"}).status_code == 200


def test_admin_endpoints_need_role_and_mfa(client) -> None:  # type: ignore[no-untyped-def]
    tok = _user(client)
    assert client.get("/admin/models/live", headers={"Authorization": f"Bearer {tok}"}).status_code == 403
    assert client.get("/admin/audit", headers={"Authorization": f"Bearer {tok}"}).status_code == 403
    assert client.get("/me").status_code == 401
