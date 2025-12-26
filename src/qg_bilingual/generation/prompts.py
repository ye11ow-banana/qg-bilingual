"""Prompt scaffolding for answer-aware and answer-agnostic QG.

This module centralizes the XML-style templates for generating questions with
optional wh-type constraints in English and Ukrainian.
"""

from __future__ import annotations

from typing import Optional

from qg_bilingual.data import normalize_text

AWARE_TEMPLATE = """
<question_generation>
  <instruction>Generate a concise factoid wh-question about the highlighted answer from the context.</instruction>
  <context>{context}</context>
  <answer>{answer}</answer>
  {wh_constraint}
</question_generation>
""".strip()

AGNOSTIC_TEMPLATE = """
<question_generation>
  <instruction>Generate a concise factoid wh-question grounded only in the context (no assumptions).</instruction>
  <context>{context}</context>
  {wh_constraint}
</question_generation>
""".strip()

WH_KEYWORDS = {
    "en": {
        "who": "who",
        "when": "when",
        "where": "where",
        "what": "what",
        "why": "why",
        "how": "how",
    },
    "ua": {
        "who": "хто",
        "when": "коли",
        "where": "де",
        "what": "що",
        "why": "чому",
        "how": "як",
    },
}


def _format_constraint(wh_type: Optional[str], lang: str) -> str:
    if wh_type is None:
        return ""

    normalized_lang = lang.lower()
    normalized_wh = (wh_type or "").lower()

    keyword_map = WH_KEYWORDS.get(normalized_lang, WH_KEYWORDS["en"])
    keyword = keyword_map.get(normalized_wh)
    if keyword is None:
        # Fallback to English keyword even if the lang is unexpected.
        keyword = WH_KEYWORDS["en"].get(normalized_wh, normalized_wh)

    return f"<constraints>Produce a {keyword} question.</constraints>"


def build_prompt(
    *,
    context: str,
    answer: Optional[str],
    mode: str = "aware",
    wh_type: Optional[str] = None,
    lang: str = "en",
) -> str:
    """Construct a formatted prompt string for QG.

    Args:
        context: Source context.
        answer: Highlighted answer (ignored for agnostic mode).
        mode: "aware" or "agnostic".
        wh_type: Optional forced wh-type.
        lang: Language code for wh lexemes.
    """

    constraint = _format_constraint(wh_type, lang)
    normalized_context = normalize_text(context)
    normalized_answer = normalize_text(answer) if answer else ""

    if mode == "agnostic":
        return AGNOSTIC_TEMPLATE.format(context=normalized_context, wh_constraint=constraint)

    return AWARE_TEMPLATE.format(
        context=normalized_context,
        answer=normalized_answer,
        wh_constraint=constraint,
    )


__all__ = ["build_prompt", "WH_KEYWORDS", "AWARE_TEMPLATE", "AGNOSTIC_TEMPLATE"]
