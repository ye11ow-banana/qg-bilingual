"""Safety-aware question generation pipeline.

The implementation follows the sequence described in the task: generation,
QA-based filtering, NLI, toxicity, and policy checks. Heavy models are
represented by lightweight stubs to keep the repository testable without
external downloads while preserving the public API surface for future
plug-ins.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, Optional

from qg_bilingual.eval.normalize import NormalizationConfig, Normalizer, exact_match, f1_score

from .safety import DEFAULT_LEXICONS, Lexicons, Policy, ToxicityClassifier
from .schemas import GenerateRequest, GenerateResponse

LOGGER = logging.getLogger(__name__)


@dataclass
class QGDecodingConfig:
    max_new_tokens: int = 32
    min_new_tokens: int = 4
    no_repeat_ngram_size: int = 3
    strategy: str = "beam"
    num_beams: int = 6
    length_penalty: float = 1.1
    top_p: float = 0.9
    temperature: float = 0.8


@dataclass
class Thresholds:
    f1_pass: float = 0.8
    conf_pass: float = 0.35
    require_entailment: bool = True
    neutral_ok: bool = False
    tox_prob_max: float = 0.40
    lexicon_block: bool = True


@dataclass
class RuntimeConfig:
    device: str = "auto"
    batch_size: int = 8


class StubQGModel:
    def __init__(self, decoding: QGDecodingConfig):
        self.decoding = decoding

    def generate(self, context: str, answer: Optional[str], wh_type: Optional[str]) -> str:
        base = answer or "information"
        wh_prefix = self._select_wh(answer, wh_type)
        question = f"{wh_prefix} {base}?".strip()
        cleaned = re.sub(r"\s+", " ", question)
        if not cleaned.endswith("?"):
            cleaned += "?"
        return cleaned

    def _select_wh(self, answer: Optional[str], wh_type: Optional[str]) -> str:
        if wh_type:
            return wh_type.capitalize()
        if answer and any(ch.isdigit() for ch in answer):
            return "When"
        return "What"


class StubQAModel:
    def predict(self, context: str, question: str) -> tuple[str, float]:
        lowered_ctx = context.lower()
        q_tokens = [tok for tok in re.split(r"\W+", question.lower()) if tok]

        for span_len in range(len(q_tokens), 0, -1):
            for start in range(0, len(q_tokens) - span_len + 1):
                candidate = " ".join(q_tokens[start : start + span_len])
                if candidate and candidate in lowered_ctx:
                    return candidate, 0.9

        return "", 0.05


class StubNLIModel:
    def classify(self, premise: str, hypothesis: str) -> str:
        norm_premise = premise.lower()
        tokens = [tok for tok in re.split(r"\W+", hypothesis.lower()) if tok]
        missing = [tok for tok in tokens if tok and tok not in norm_premise]
        if len(missing) > len(tokens) // 2:
            return "neutral"
        return "entailment"


@dataclass
class PipelineConfig:
    decoding: QGDecodingConfig
    thresholds: Thresholds
    policy: Policy
    lexicons: Lexicons = field(default_factory=lambda: DEFAULT_LEXICONS)
    toxicity_classifier: ToxicityClassifier = field(default_factory=ToxicityClassifier)
    normalizer: Normalizer = field(default_factory=lambda: Normalizer(NormalizationConfig(), lang="en"))


class SafeGenerationPipeline:
    def __init__(
        self,
        qg_model: StubQGModel,
        qa_model_en: StubQAModel,
        qa_model_multi: StubQAModel,
        nli_model: StubNLIModel,
        config: PipelineConfig,
    ) -> None:
        self.qg_model = qg_model
        self.qa_model_en = qa_model_en
        self.qa_model_multi = qa_model_multi
        self.nli_model = nli_model
        self.config = config

    def run(self, request: GenerateRequest) -> GenerateResponse:
        self._validate_request(request)
        question = self._generate_question(request)
        qa_result = self._qa_check(request, question)
        nli_label = self._nli_check(request, question)
        lexicon_hits, tox_prob = self._toxicity_check(request, question)
        policy_ok = self.config.policy.check_context_only(question, request.context)

        reasons = []
        thresholds = self.config.thresholds
        if qa_result["f1"] is not None and qa_result["f1"] < thresholds.f1_pass:
            reasons.append("qg2qa_f1_low")
        if qa_result["confidence"] is not None and qa_result["confidence"] < thresholds.conf_pass:
            reasons.append("qg2qa_conf_low")
        if thresholds.require_entailment:
            if nli_label != "entailment":
                reasons.append("nli_neutral" if nli_label == "neutral" else "nli_contradiction")
        elif not thresholds.neutral_ok and nli_label == "neutral":
            reasons.append("nli_neutral")
        if thresholds.lexicon_block and lexicon_hits:
            reasons.append("lexicon_block")
        if tox_prob is not None and tox_prob > thresholds.tox_prob_max:
            reasons.append("tox_high")
        if not self.config.policy.wh_allowed(qa_result["wh_detected"]):
            reasons.append("policy_violation")
        if not policy_ok:
            reasons.append("policy_violation")

        passed = len(reasons) == 0
        response_question = question if passed else None

        metrics: Dict[str, object] = {
            "qa_em": qa_result["em"],
            "qa_f1": qa_result["f1"],
            "qa_conf": qa_result["confidence"],
            "tox_prob": tox_prob,
            "nli": nli_label,
        }
        debug = {
            "decoding": self.config.decoding.__dict__,
            "wh_detected": qa_result["wh_detected"],
            "lengths": {
                "context": len(request.context.split()),
                "question": len(question.split()),
            },
            "lexicon_hits": lexicon_hits,
        }

        return GenerateResponse(
            question=response_question,
            passed=passed,
            reasons=reasons,
            metrics=metrics,
            debug=debug,
        )

    def _validate_request(self, request: GenerateRequest) -> None:
        tokens = request.context.split()
        if len(tokens) < 20:
            raise ValueError("context too short")
        if not request.context.strip():
            raise ValueError("context required")

    def _generate_question(self, request: GenerateRequest) -> str:
        question = self.qg_model.generate(request.context, request.answer, request.wh_type)
        question = question.strip()
        if not question.endswith("?"):
            question += "?"
        return question

    def _qa_check(self, request: GenerateRequest, question: str) -> Dict[str, Optional[float]]:
        qa_model = self.qa_model_en if request.lang == "en" else self.qa_model_multi
        predicted, confidence = qa_model.predict(request.context, question)
        normalizer = Normalizer(NormalizationConfig(), lang=request.lang)
        wh_detected = self._detect_wh(question)

        if request.answer:
            em = exact_match(predicted, request.answer, normalizer)
            f1 = f1_score(predicted, request.answer, normalizer)
        else:
            em = None
            f1 = None

        return {
            "prediction": predicted,
            "confidence": confidence,
            "em": em,
            "f1": f1,
            "wh_detected": wh_detected,
        }

    def _nli_check(self, request: GenerateRequest, question: str) -> str:
        hypothesis = question.rstrip(" ?")
        return self.nli_model.classify(request.context, hypothesis)

    def _toxicity_check(self, request: GenerateRequest, question: str) -> tuple[list[str], float]:
        lexicon_hits = self.config.lexicons.find_matches(question, request.lang)
        tox_prob = self.config.toxicity_classifier.score(question, request.lang, lexicon_hits)
        return lexicon_hits, tox_prob

    @staticmethod
    def _detect_wh(question: str) -> Optional[str]:
        cleaned = question.strip().lower()
        if not cleaned:
            return None
        token = cleaned.split()[0].strip("?.,! ")
        if token in {"who", "when", "where", "what", "why", "how", "хто", "коли", "де", "чому", "як", "що"}:
            return token
        return None


def build_pipeline(config: PipelineConfig) -> SafeGenerationPipeline:
    qg_model = StubQGModel(config.decoding)
    qa_en = StubQAModel()
    qa_multi = StubQAModel()
    nli = StubNLIModel()
    return SafeGenerationPipeline(qg_model, qa_en, qa_multi, nli, config)


__all__ = ["SafeGenerationPipeline", "PipelineConfig", "build_pipeline", "QGDecodingConfig", "Thresholds", "RuntimeConfig"]
