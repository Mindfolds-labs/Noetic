from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from noetic_pawp.api.server import create_app


def test_health():
    client = TestClient(create_app())
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_predict_valid_and_invalid():
    client = TestClient(create_app())
    ok = client.post("/predict", json={"text": "olá mundo", "language": "pt"})
    assert ok.status_code == 200
    assert "tokens" in ok.json()

    bad = client.post("/predict", json={"text": "", "language": "pt"})
    assert bad.status_code == 422
