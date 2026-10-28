import uuid

import pytest

pytestmark = pytest.mark.integration


def _register(client, email: str) -> None:  # type: ignore[no-untyped-def]
    r = client.post(
        "/auth/register",
        json={
            "display_name": "Wanjiku",
            "email": email,
            "password": "Strong-pass-123",
            "role": "signer",
            "preferred_in": "sign",
            "preferred_out": "sign",
        },
    )
    assert r.status_code == 201, r.text


def test_register_login_me_patch(client) -> None:  # type: ignore[no-untyped-def]
    email = f"u{uuid.uuid4().hex[:6]}@test.ke"
    _register(client, email)
    assert (
        client.post(
            "/auth/register", json={"display_name": "x", "email": email, "password": "Strong-pass-123"}
        ).status_code
        == 409
    )
    tok = client.post("/auth/login", json={"email": email, "password": "Strong-pass-123"}).json()
    h = {"Authorization": f"Bearer {tok['access_token']}"}
    me = client.get("/me", headers=h).json()
    assert me["roles"] == ["signer"] and me["preferred_out"] == "sign"
    assert client.patch("/me", headers=h, json={"sign_speed": 0.8}).json()["sign_speed"] == pytest.approx(0.8)
    assert client.post("/auth/login", json={"email": email, "password": "wrong"}).status_code == 401
