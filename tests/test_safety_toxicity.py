from qg_bilingual.safety.toxicity import ToxicityService


def build_service():
    cfg = {
        "classifier_en": "dummy",
        "classifier_multi": "dummy",
        "prob_max": 0.4,
        "use_lexicon_block": True,
        "lexicons": {
            "en": "src/qg_bilingual/safety/lexicons/en_toxic.json",
            "ua": "src/qg_bilingual/safety/lexicons/ua_toxic.json",
        },
        "sensitive_groups": "src/qg_bilingual/safety/lexicons/sensitive_groups.json",
    }
    return ToxicityService(cfg)


def test_toxicity_detects_lexicon_hits_en():
    svc = build_service()
    text = "You are an idiot and I hate you."
    res = svc.score([text], "en")[0]
    assert "idiot" in res["lexicon_hits"]
    assert res["prob"] > 0.05


def test_toxicity_policy_violation_when_unbacked():
    svc = build_service()
    text = "All immigrants are dangerous"
    context = "The article is about migration policy without claims."
    res = svc.score([text], "en", context=context)[0]
    assert "policy_violation" in res["flags"]


def test_neutral_text_passes():
    svc = build_service()
    text = "The weather is sunny and calm today."
    res = svc.score([text], "en")[0]
    assert res["lexicon_hits"] == []
    assert res["prob"] <= 0.4
    assert not res["flags"]
