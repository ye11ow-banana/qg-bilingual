import pytest

from qg_bilingual.generation.prompts import (
    AGNOSTIC_TEMPLATE_EN,
    AWARE_TEMPLATE_EN,
    WH_KEYWORDS,
    build_prompt,
)


@pytest.mark.parametrize("mode, template", [("aware", AWARE_TEMPLATE_EN), ("agnostic", AGNOSTIC_TEMPLATE_EN)])
def test_prompt_includes_expected_tags(mode: str, template: str):
    prompt = build_prompt(
        context=(
            "Kyiv is the capital of Ukraine and sits on the Dnipro river; the city has a long "
            "history, vibrant culture, and many historic landmarks that attract visitors year-round."
        ),
        answer="Kyiv",
        mode=mode,
        lang="en",
    )

    assert "<context>" in prompt and "</context>" in prompt
    assert template.split("\n")[0].startswith("qg:"), "Template should start with instruction"

    if mode == "aware":
        assert "<answer>kyiv</answer>" in prompt.lower()
    else:
        assert "<answer>" not in prompt


@pytest.mark.parametrize("lang, wh_type", [("en", "when"), ("ua", "when"), ("xx", "why")])
def test_wh_type_constraint_is_injected(lang: str, wh_type: str):
    prompt = build_prompt(
        context=(
            "This is a sufficiently long context used only for testing purposes to ensure that "
            "the prompt formatting injects the correct wh-type constraint and does not fail."
        ),
        answer="Answer",
        mode="aware",
        wh_type=wh_type,
        lang=lang,
    )

    constraint_line = "<constraints>" in prompt and "question.</constraints>" in prompt
    assert constraint_line, "Constraint block should be present when wh_type is provided"

    keyword_map = WH_KEYWORDS.get(lang.lower(), WH_KEYWORDS["en"])
    expected_keyword = keyword_map.get(wh_type, WH_KEYWORDS["en"].get(wh_type, wh_type)).lower()
    assert expected_keyword in prompt.lower()


def test_prompt_raises_for_short_context():
    with pytest.raises(ValueError):
        build_prompt(context="Too short", answer="A", mode="aware", lang="en")
