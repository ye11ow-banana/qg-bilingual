"""Safety helpers: lexicons, toxicity stubs, and policy checks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass
class Lexicons:
    en: Sequence[str]
    ua: Sequence[str]

    def find_matches(self, text: str, lang: str) -> list[str]:
        lang = lang.lower()
        vocab = self.ua if lang == "ua" else self.en
        lowered = text.lower()
        hits = []
        for term in vocab:
            if re.search(rf"\b{re.escape(term.lower())}\b", lowered):
                hits.append(term)
        return hits


@dataclass
class ToxicityClassifier:
    """Lightweight toxicity classifier stub.

    The classifier returns a probability in [0, 1]. In absence of a real model
    this implementation simply boosts probability when lexicon matches are
    present. This keeps tests deterministic while preserving the API shape for
    future model-backed replacements.
    """

    base_prob: float = 0.05

    def score(self, text: str, lang: str, lexicon_hits: Iterable[str]) -> float:
        hits = list(lexicon_hits)
        if not hits:
            return self.base_prob
        return min(1.0, self.base_prob + 0.15 * len(hits))


@dataclass
class Policy:
    context_only: bool
    allowed_wh: Sequence[str]
    protected_groups: Sequence[str]

    def check_context_only(self, question: str, context: str) -> bool:
        if not self.context_only:
            return True

        lowered_question = question.lower()
        lowered_context = context.lower()
        generic_patterns = [r"all [a-z]+", r"always [a-z]+", r"typically [a-z]+"]
        if any(re.search(pattern, lowered_question) for pattern in generic_patterns):
            return False

        for group in self.protected_groups:
            if re.search(rf"\b{re.escape(group.lower())}\b", lowered_question) and not re.search(
                rf"\b{re.escape(group.lower())}\b", lowered_context
            ):
                return False
        return True

    def wh_allowed(self, wh_token: str | None) -> bool:
        if wh_token is None:
            return True
        return wh_token.lower() in {w.lower() for w in self.allowed_wh}


DEFAULT_LEXICONS = Lexicons(
    en=["idiot", "stupid", "hate", "racist"],
    ua=["дурень", "ненавиджу", "расист"],
)

DEFAULT_PROTECTED_GROUPS = [
    "women",
    "men",
    "muslim",
    "jew",
    "lgbt",
    "immigrant",
    "біженець",
    "роми",
]

