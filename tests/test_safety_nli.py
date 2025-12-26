import pytest

from qg_bilingual.safety.nli import NLIService


@pytest.fixture
def heuristic_cfg():
    return {
        "model": "dummy",
        "require_entailment": True,
        "neutral_ok": False,
        "hypothesis_template": "{question_wo_qm}",
        "thresholds": {"entailment_min_prob": 0.5},
    }


def test_entailment_detected(heuristic_cfg):
    svc = NLIService(heuristic_cfg)
    context = "Kyiv is the capital of Ukraine and located on the Dnipro river."
    question = "What is the capital of Ukraine?"
    result = svc.predict([context], [question])[0]
    assert result["label"] == "entailment"
    assert svc.decide(result)


def test_contradiction_detected(heuristic_cfg):
    svc = NLIService(heuristic_cfg)
    context = "The sky is blue and the grass is green."
    question = "Why is the moon made of cheese?"
    result = svc.predict([context], [question])[0]
    assert result["label"] in {"neutral", "contradiction"}
    assert svc.decide(result) is False


def test_neutral_allowed_when_configured():
    cfg = {
        "model": "dummy",
        "require_entailment": False,
        "neutral_ok": True,
        "thresholds": {"entailment_min_prob": 0.2},
    }
    svc = NLIService(cfg)
    res = svc.predict(["Context with partial info"], ["Question unclear"])[0]
    assert svc.decide(res) is True
