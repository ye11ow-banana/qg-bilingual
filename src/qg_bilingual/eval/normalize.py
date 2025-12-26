"""Language-aware normalization utilities for EM/F1 scoring."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from string import punctuation
from typing import Iterable
from collections import Counter


@dataclass
class NormalizationConfig:
    strip_punct: bool = True
    lower: bool = True
    unify_quotes: bool = True
    unify_apostrophe: bool = True


class Normalizer:
    def __init__(self, config: NormalizationConfig, lang: str) -> None:
        self.config = config
        self.lang = lang.lower()

    def normalize(self, text: str) -> str:
        if text is None:
            return ""

        normalized = unicodedata.normalize("NFKC", str(text))
        if self.config.unify_quotes:
            normalized = _unify_quotes(normalized)
        if self.config.unify_apostrophe:
            normalized = _unify_apostrophes(normalized)

        normalized = _collapse_whitespace(normalized.strip())

        if self.lang == "en":
            normalized = _fix_en_spacing(normalized)
        elif self.lang in {"ua", "uk", "ukrainian"}:
            normalized = _fix_ua_spacing(normalized)

        if self.config.strip_punct:
            normalized = _strip_punctuation(normalized)

        if self.config.lower:
            normalized = normalized.lower()

        normalized = _collapse_whitespace(normalized.strip())
        return normalized


def _collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def _unify_quotes(text: str) -> str:
    replacements = {
        "“": '"',
        "”": '"',
        "«": '"',
        "»": '"',
        "‟": '"',
        "‹": '"',
        "›": '"',
        "`": "'",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _unify_apostrophes(text: str) -> str:
    for symbol in ["'", "`", "ʼ", "´", "‘", "’"]:
        text = text.replace(symbol, "’")
    return text


def _fix_en_spacing(text: str) -> str:
    # remove stray spaces before punctuation
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    # avoid double spaces after punctuation
    text = re.sub(r"([,.;:!?])\s+", r"\1 ", text)
    return text


def _fix_ua_spacing(text: str) -> str:
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,.;:!?])\s+", r"\1 ", text)
    return text


def _strip_punctuation(text: str) -> str:
    # keep apostrophes inside words, drop other punctuation
    preserved = "’"
    pattern = "".join(ch for ch in punctuation if ch not in preserved)
    text = re.sub(f"[{re.escape(pattern)}]", " ", text)
    return text


def f1_score(prediction: str, ground_truth: str, normalizer: Normalizer) -> float:
    pred_tokens = _tokens(normalizer.normalize(prediction))
    gold_tokens = _tokens(normalizer.normalize(ground_truth))

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    pred_counts = _count_tokens(pred_tokens)
    gold_counts = _count_tokens(gold_tokens)
    overlap = pred_counts & gold_counts
    num_same = sum(overlap.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, ground_truth: str, normalizer: Normalizer) -> float:
    return float(normalizer.normalize(prediction) == normalizer.normalize(ground_truth))


def _tokens(text: str) -> Iterable[str]:
    return [tok for tok in text.split(" ") if tok]


def _count_tokens(tokens: Iterable[str]):
    return Counter(tokens)
