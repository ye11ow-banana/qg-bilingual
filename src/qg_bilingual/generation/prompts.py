"""Prompt scaffolding for answer-aware and answer-agnostic QG.

This module centralizes the XML-style templates for generating questions with
optional wh-type constraints in English and Ukrainian.
"""

from __future__ import annotations

import logging
from typing import Optional

from qg_bilingual.data import normalize_text

LOGGER = logging.getLogger(__name__)

AWARE_TEMPLATE_EN = """
qg: generate a concise factual wh-question in English from the context and the given answer.
<context>
{context}
</context>
<answer>{answer}</answer>
{wh_constraint}
""".strip()

AWARE_TEMPLATE_UA = """
qg: згенеруй лаконічне фактологічне wh-питання українською на основі контексту та наведеної відповіді.
<context>
{context}
</context>
<answer>{answer}</answer>
{wh_constraint}
""".strip()

AGNOSTIC_TEMPLATE_EN = """
qg: generate a concise factual wh-question in English grounded only in the context (do not assume missing facts).
<context>
{context}
</context>
{wh_constraint}
""".strip()

AGNOSTIC_TEMPLATE_UA = """
qg: згенеруй лаконічне фактологічне wh-питання українською лише з контексту (без припущень).
<context>
{context}
</context>
{wh_constraint}
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

    if len(normalized_context.split()) < 20:
        raise ValueError("Context too short for question generation (reason=too_short_context)")

    templates = {
        ("aware", "en"): AWARE_TEMPLATE_EN,
        ("aware", "ua"): AWARE_TEMPLATE_UA,
        ("agnostic", "en"): AGNOSTIC_TEMPLATE_EN,
        ("agnostic", "ua"): AGNOSTIC_TEMPLATE_UA,
    }

    if mode == "agnostic":
        template = templates.get(("agnostic", lang.lower()), AGNOSTIC_TEMPLATE_EN)
        return template.format(context=normalized_context, wh_constraint=constraint)

    if not normalized_answer:
        LOGGER.warning("Skipping record without answer in aware mode")
        raise ValueError("Missing answer for aware prompt")

    template = templates.get(("aware", lang.lower()), AWARE_TEMPLATE_EN)
    return template.format(context=normalized_context, answer=normalized_answer, wh_constraint=constraint)


__all__ = [
    "build_prompt",
    "WH_KEYWORDS",
    "AWARE_TEMPLATE_EN",
    "AWARE_TEMPLATE_UA",
    "AGNOSTIC_TEMPLATE_EN",
    "AGNOSTIC_TEMPLATE_UA",
]
