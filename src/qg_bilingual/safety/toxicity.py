from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Sequence

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
except Exception:  # pragma: no cover - handled gracefully
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    pipeline = None

Lang = Literal["en", "ua"]


@dataclass
class ToxicityConfig:
    classifier_en: str = "dummy"
    classifier_multi: str = "dummy"
    prob_max: float = 0.4
    use_lexicon_block: bool = True
    lexicons: Dict[str, str] | None = None
    sensitive_groups: str | None = None
    batch_size: int = 8


_APOSTROPHE = "['’`ʼ]"
_WORD_BOUNDARY = rf"(?:\b|{_APOSTROPHE})"


def load_lexicon(path: str | Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_sensitive_groups(path: str | Path) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class LexiconMatcher:
    def __init__(self, terms: Sequence[str]) -> None:
        self.terms = list(terms)
        self.patterns = [re.compile(rf"{_WORD_BOUNDARY}{re.escape(term.lower())}{_WORD_BOUNDARY}", re.IGNORECASE) for term in self.terms]

    def find(self, text: str) -> List[str]:
        lowered = text.lower()
        hits: List[str] = []
        for term, pattern in zip(self.terms, self.patterns):
            if pattern.search(lowered):
                hits.append(term)
        return hits


class PolicyChecker:
    def __init__(self, sensitive_groups: Dict[str, List[str]]) -> None:
        self.sensitive_groups = sensitive_groups
        self.generalizers = ["all", "always", "typically", "всі", "завжди", "типово"]

    def check(self, text: str, context: str) -> List[str]:
        flags: List[str] = []
        lowered = text.lower()
        lowered_context = context.lower()
        for category, group_terms in self.sensitive_groups.items():
            for term in group_terms:
                pattern = rf"{_WORD_BOUNDARY}(?:{'|'.join(map(re.escape, self.generalizers))})[^\n\r]+?{re.escape(term.lower())}{_WORD_BOUNDARY}"
                if re.search(pattern, lowered):
                    if term.lower() not in lowered_context:
                        flags.append("policy_violation")
                        flags.append("group_attack")
                        return list(dict.fromkeys(flags))
                if re.search(rf"{_WORD_BOUNDARY}{re.escape(term.lower())}{_WORD_BOUNDARY}", lowered) and term.lower() not in lowered_context:
                    # generic mention without support
                    flags.append("group_attack")
        return list(dict.fromkeys(flags))


class ToxicityService:
    def __init__(self, cfg: Dict, device: str | int | None = None) -> None:
        self.cfg = ToxicityConfig(
            classifier_en=cfg.get("classifier_en", "dummy"),
            classifier_multi=cfg.get("classifier_multi", "dummy"),
            prob_max=cfg.get("prob_max", 0.4),
            use_lexicon_block=cfg.get("use_lexicon_block", True),
            lexicons=cfg.get("lexicons"),
            sensitive_groups=cfg.get("sensitive_groups"),
            batch_size=cfg.get("batch_size", 8),
        )
        self._device = device
        self._pipelines: Dict[str, object] = {}
        self.lexicon_matchers: Dict[str, LexiconMatcher] = {}
        self.policy_checker: PolicyChecker | None = None
        self._load_resources()

    def _load_resources(self) -> None:
        if self.cfg.lexicons:
            for lang_key, path in self.cfg.lexicons.items():
                matcher = LexiconMatcher(load_lexicon(path))
                self.lexicon_matchers[lang_key] = matcher
        if self.cfg.sensitive_groups:
            groups = load_sensitive_groups(self.cfg.sensitive_groups)
            self.policy_checker = PolicyChecker(groups)
        for lang, model_name in (("en", self.cfg.classifier_en), ("ua", self.cfg.classifier_multi)):
            if model_name.lower() == "dummy" or pipeline is None:
                continue
            try:  # pragma: no cover - external
                tok = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self._pipelines[lang] = pipeline(
                    "text-classification",
                    model=model,
                    tokenizer=tok,
                    device=self._device,
                    return_all_scores=True,
                )
            except Exception as exc:  # pragma: no cover - fallback
                LOGGER.warning("Failed to load toxicity model %s: %s", model_name, exc)

    def score(self, texts: List[str], lang: Lang, context: str | None = None) -> List[Dict[str, object]]:
        batch_size = max(1, self.cfg.batch_size)
        matcher = self.lexicon_matchers.get(lang)
        policy_checker = self.policy_checker
        pipeline_for_lang = self._pipelines.get(lang)
        results: List[Dict[str, object]] = []
        start_time = time.time()
        for idx in range(0, len(texts), batch_size):
            batch = texts[idx : idx + batch_size]
            probs = self._score_batch(batch, pipeline_for_lang)
            for text, prob in zip(batch, probs):
                lex_hits = matcher.find(text) if matcher else []
                flags: List[str] = []
                if policy_checker and context is not None:
                    flags = policy_checker.check(text, context)
                results.append({"prob": prob, "lexicon_hits": lex_hits, "flags": flags})
        LOGGER.debug("tox_ms=%.2f", (time.time() - start_time) * 1000)
        return results

    def _score_batch(self, batch: List[str], pipe) -> List[float]:
        if pipe is None:
            return [self._heuristic_prob(text) for text in batch]
        try:  # pragma: no cover - depends on external weights
            outputs = pipe(batch)
            probs: List[float] = []
            for output in outputs:
                # flatten model scores to toxicity probability
                score_map = {item["label"].lower(): item["score"] for item in output}
                tox = score_map.get("toxic", score_map.get("LABEL_1", 0.0))
                if len(score_map) == 1:
                    probs.append(tox)
                else:
                    probs.append(max(score_map.values()))
            return probs
        except Exception as exc:  # pragma: no cover - fallback
            LOGGER.warning("Toxicity model failed, using heuristic: %s", exc)
            return [self._heuristic_prob(text) for text in batch]

    def _heuristic_prob(self, text: str) -> float:
        lowered = text.lower()
        intensity = sum(1 for tok in re.split(r"\W+", lowered) if tok in {"hate", "kill", "idiot", "stupid", "дурень", "ненавиджу"})
        return min(1.0, 0.05 + 0.2 * intensity)

    def decide(self, result: Dict[str, object]) -> bool:
        lexicon_block = self.cfg.use_lexicon_block and result.get("lexicon_hits")
        policy_violation = "policy_violation" in (result.get("flags") or [])
        return (not lexicon_block) and (not policy_violation) and float(result.get("prob", 0.0)) <= self.cfg.prob_max


__all__ = ["ToxicityService", "load_lexicon", "load_sensitive_groups", "LexiconMatcher", "PolicyChecker", "ToxicityConfig"]
