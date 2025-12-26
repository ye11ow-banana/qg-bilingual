from __future__ import annotations

from fastapi.testclient import TestClient

from qg_bilingual.server.app import app

client = TestClient(app)


def test_generate_safe_positive():
    payload = {
        "context": "Taras Shevchenko was born on March 9, 1814 in Moryntsi. He became a prominent poet and artist whose works are celebrated.",
        "answer": "March 9, 1814",
        "lang": "en",
        "mode": "aware",
    }
    resp = client.post("/generate_safe", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["passed"] is True
    assert data["question"]
    assert data["reasons"] == []
    assert data["metrics"]["qa_f1"] is not None


def test_generate_safe_toxic_blocked():
    payload = {
        "context": "The report discusses workplace behavior and culture in detail, providing numerous examples and historical notes to set the tone for the investigation.",
        "answer": "idiot",
        "lang": "en",
        "mode": "aware",
    }
    resp = client.post("/generate_safe", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["passed"] is False
    assert "lexicon_block" in data["reasons"]
    assert data["question"] is None
    assert data["metrics"]["qa_conf"] <= 0.9
