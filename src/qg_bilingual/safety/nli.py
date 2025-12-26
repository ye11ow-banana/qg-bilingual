from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
except Exception:  # pragma: no cover - handled gracefully
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    pipeline = None


@dataclass
class NLIConfig:
    model: str = "dummy"
    require_entailment: bool = True
    neutral_ok: bool = False
    hypothesis_template: str | None = None
    premise_source: str = "context"
    entailment_min_prob: float = 0.5
    batch_size: int = 8


class NLIService:
    """Batch NLI classifier with graceful fallback to heuristic scoring."""

    def __init__(self, cfg: Dict, device: str | int | None = None) -> None:
        self.cfg = NLIConfig(
            model=cfg.get("model", "dummy"),
            require_entailment=cfg.get("require_entailment", True),
            neutral_ok=cfg.get("neutral_ok", False),
            hypothesis_template=cfg.get("hypothesis_template"),
            premise_source=cfg.get("premise_source", "context"),
            entailment_min_prob=cfg.get("thresholds", {}).get("entailment_min_prob", 0.5),
            batch_size=cfg.get("batch_size", cfg.get("batch_size", 8)),
        )
        self._device = device
        self._tokenizer = None
        self._pipeline = None
        self._cache: Dict[str, Dict[str, str]] = {}
        self._init_model()

    def _init_model(self) -> None:
        model_name = self.cfg.model
        if model_name.lower() == "dummy" or pipeline is None:
            LOGGER.info("NLIService running in heuristic mode")
            return
        try:  # pragma: no cover - model load is external
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self._pipeline = pipeline(
                "text-classification",
                model=model,
                tokenizer=self._tokenizer,
                device=self._device,
                return_all_scores=True,
            )
            LOGGER.info("Loaded NLI model %s", model_name)
        except Exception as exc:  # pragma: no cover - fallback path
            LOGGER.warning("Falling back to heuristic NLI due to: %s", exc)
            self._pipeline = None
            self._tokenizer = None

    def _build_hypothesis(self, question: str) -> str:
        cleaned = question.rstrip(" ?")
        template = self.cfg.hypothesis_template
        if template:
            return template.format(question_wo_qm=cleaned)
        return cleaned

    def predict(self, premises: List[str], hypotheses: List[str]) -> List[Dict[str, object]]:
        assert len(premises) == len(hypotheses), "premises and hypotheses must align"
        if self._pipeline:
            return self._predict_model(premises, hypotheses)
        return [self._predict_rule(p, h) for p, h in zip(premises, hypotheses)]

    def _predict_rule(self, premise: str, hypothesis: str) -> Dict[str, object]:
        norm_premise = premise.lower()
        tokens = [tok for tok in re.split(r"\W+", hypothesis.lower()) if tok]
        missing = [tok for tok in tokens if tok not in norm_premise]
        coverage = 0.0 if not tokens else 1.0 - len(missing) / len(tokens)
        if coverage > 0.75:
            label = "entailment"
            probs = {"entailment": 0.85, "neutral": 0.1, "contradiction": 0.05}
        elif coverage > 0.35:
            label = "neutral"
            probs = {"entailment": 0.45, "neutral": 0.45, "contradiction": 0.1}
        else:
            label = "neutral"
            probs = {"entailment": 0.15, "neutral": 0.5, "contradiction": 0.35}
        return {"label": label, "probs": probs}

    def _predict_model(self, premises: List[str], hypotheses: List[str]) -> List[Dict[str, object]]:
        inputs = [f"premise: {p} hypothesis: {h}" for p, h in zip(premises, hypotheses)]
        results: List[Dict[str, object]] = []
        batch_size = max(1, self.cfg.batch_size)
        for start in range(0, len(inputs), batch_size):  # pragma: no cover - external
            batch_inputs = inputs[start : start + batch_size]
            model_outputs = self._pipeline(batch_inputs)
            for output in model_outputs:
                label_scores = {item["label"].lower(): item["score"] for item in output}
                label = max(label_scores, key=label_scores.get)
                probs = {
                    "entailment": label_scores.get("entailment", 0.0),
                    "neutral": label_scores.get("neutral", 0.0),
                    "contradiction": label_scores.get("contradiction", 0.0),
                }
                results.append({"label": label, "probs": probs})
        return results

    def decide(self, result: Dict[str, object]) -> bool:
        label = str(result.get("label"))
        probs = result.get("probs", {}) or {}
        entail_prob = float(probs.get("entailment", 0.0))
        if self.cfg.require_entailment:
            return label == "entailment" and entail_prob >= self.cfg.entailment_min_prob
        if not self.cfg.neutral_ok and label == "neutral":
            return False
        return label in {"entailment", "neutral"}


__all__ = ["NLIService", "NLIConfig"]
