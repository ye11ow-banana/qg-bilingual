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


def test_generate_safe_how_many_positive():
    payload = {
        "context": "Kyiv is the capital city of Ukraine. Kyiv is a big city with a population of about 4 million people, made up of long-established residents and newcomers.",
        "answer": "4 million",
        "lang": "en",
        "mode": "aware",
        "wh_type": "how_many",
    }
    resp = client.post("/generate_safe", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["passed"] is True
    assert data["question"]
    assert data["debug"]["nli"]["ok"] is True


def test_generate_safe_kyiv_what_positive():
    payload = {
        "context": "Kyiv is the capital city of Ukraine and its largest urban center. It stands on both banks of the Dnipro River, with broad river islands and wide avenues.",
        "answer": "Kyiv",
        "lang": "en",
        "mode": "aware",
        "wh_type": "what",
    }
    resp = client.post("/generate_safe", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["passed"] is True
    assert data["question"]
    assert data["reasons"] == []
    assert data["metrics"]["qa_f1"] is not None


def test_generate_safe_answer_not_in_context_blocked():
    payload = {
        "context": "Kyiv is the capital city of Ukraine and stands on the Dnipro River. The city has a long history, notable architecture, and many cultural institutions that shape daily life.",
        "answer": "Lviv",
        "lang": "en",
        "mode": "aware",
        "wh_type": "what",
    }
    resp = client.post("/generate_safe", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["passed"] is False
    assert data["question"] is None
    assert any(reason.startswith("qg2qa_") for reason in data["reasons"])


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
