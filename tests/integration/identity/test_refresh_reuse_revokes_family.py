import uuid

import pytest

pytestmark = pytest.mark.integration


def test_refresh_rotation_and_reuse_detection(client) -> None:  # type: ignore[no-untyped-def]
    email = f"r{uuid.uuid4().hex[:6]}@test.ke"
    client.post("/auth/register", json={"display_name": "R", "email": email, "password": "Strong-pass-123"})
    t1 = client.post("/auth/login", json={"email": email, "password": "Strong-pass-123"}).json()
    t2 = client.post("/auth/refresh", json={"refresh_token": t1["refresh_token"]}).json()
    assert t2["refresh_token"] != t1["refresh_token"]
    reuse = client.post("/auth/refresh", json={"refresh_token": t1["refresh_token"]})
    assert reuse.status_code == 401 and "reuse" in reuse.json()["detail"]
    # the whole family is revoked: the newest token no longer works either
    assert client.post("/auth/refresh", json={"refresh_token": t2["refresh_token"]}).status_code == 401
