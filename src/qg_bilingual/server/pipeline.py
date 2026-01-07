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
from qg_bilingual.safety import NLIService, ToxicityService
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
        wh_prefix = self._select_wh(answer, wh_type)

        # Keep deterministic templates for some WH types.
        if wh_prefix.lower().startswith("how many"):
            question = "How many people are mentioned in the context?"
        elif wh_prefix.lower() == "when":
            question = "When did this happen?"
        elif wh_prefix.lower() == "where":
            question = "Where did this take place?"
        elif wh_prefix.lower() == "who":
            question = "Who is mentioned in the context?"
        elif answer:
            # Answer-aware stub: anchor the question to a sentence containing the answer.
            # This makes the QA check meaningful and allows common cases like
            # “Kyiv is the capital city of Ukraine …” with answer=Kyiv to pass.
            question = self._question_from_sentence(context, answer, wh_prefix)
        else:
            question = "What is the passage about?"

        cleaned = re.sub(r"\s+", " ", question)
        if not cleaned.endswith("?"):
            cleaned += "?"
        return cleaned

    @staticmethod
    def _question_from_sentence(context: str, answer: str, wh_prefix: str) -> str:
        # Pick the first sentence that contains the answer.
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", context) if s.strip()]
        answer_re = re.compile(re.escape(answer), flags=re.IGNORECASE)
        sent = next((s for s in sentences if answer_re.search(s)), context.strip())

        # Replace the answer with the WH prefix.
        replaced = answer_re.sub(wh_prefix, sent, count=1).strip()
        replaced = re.sub(r"[.!]+$", "?", replaced)
        if not replaced.endswith("?"):
            replaced += "?"
        # Ensure it starts with the WH prefix for consistency.
        if not replaced.lower().startswith(wh_prefix.lower()):
            # If replacement happened in the middle, try to start from the WH token.
            idx = replaced.lower().find(wh_prefix.lower())
            if idx != -1:
                replaced = replaced[idx:]
        return replaced

    def _select_wh(self, answer: Optional[str], wh_type: Optional[str]) -> str:
        if wh_type:
            if wh_type == "how_many":
                return "How many"
            return wh_type.capitalize()
        if answer:
            lowered = answer.lower()
            if re.search(
                r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
                lowered,
            ):
                return "When"
            if re.search(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", lowered):
                return "When"
            if any(ch.isdigit() for ch in answer):
                return "How many"
        return "What"


class StubQAModel:
    def predict(self, context: str, question: str) -> tuple[str, float]:
        lowered_q = question.strip().lower()

        if lowered_q.startswith("how many"):
            match = re.search(r"\b\d+(?:[\d,]*\d)?(?:\.\d+)?\s+(?:million|billion|thousand|people|persons|%)\b", context, flags=re.IGNORECASE)
            if match:
                return match.group(0), 0.9
            match = re.search(r"\b\d+(?:[\d,]*\d)?(?:\.\d+)?\b", context)
            if match:
                return match.group(0), 0.75

        if lowered_q.startswith("when"):
            match = re.search(
                r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}(?:,\s*\d{4})\b",
                context,
                flags=re.IGNORECASE,
            )
            if match:
                return match.group(0), 0.9
            match = re.search(r"\b\d{4}\b", context)
            if match:
                return match.group(0), 0.6

        # Pattern-based extraction for common copula questions.
        # Example: “What is the capital city of Ukraine?” -> capture “Kyiv” from
        # “Kyiv is the capital city of Ukraine …”
        if re.match(r"^(what|who|where)\b", lowered_q) and " is " in lowered_q:
            predicted = self._extract_before_tail(context, question)
            if predicted:
                return predicted, 0.9

        lowered_ctx = context.lower()
        q_tokens = [tok for tok in re.split(r"\W+", question.lower()) if tok]

        for span_len in range(len(q_tokens), 0, -1):
            for start in range(0, len(q_tokens) - span_len + 1):
                candidate = " ".join(q_tokens[start : start + span_len])
                if candidate and candidate in lowered_ctx:
                    return candidate, 0.9

        return "", 0.05

    @staticmethod
    def _extract_before_tail(context: str, question: str) -> str:
        # Normalize both strings to a punctuation-light form for matching.
        def _plain(text: str) -> str:
            text = text.lower().strip().rstrip("?")
            text = re.sub(r"[^\w’'\-]+", " ", text)
            return " ".join(text.split())

        q_plain = _plain(question)
        ctx_plain = _plain(context)

        # Remove the WH token.
        q_plain = re.sub(r"^(what|who|where)\s+", "", q_plain)
        if not q_plain:
            return ""

        # Capture up to 6 tokens immediately before the remaining question tail.
        token = r"[\w’'\-]+"
        ans_group = rf"(?P<ans>{token}(?:\s+{token}){{0,5}})"
        pattern = re.compile(rf"\b{ans_group}\s+{re.escape(q_plain)}\b")
        m = pattern.search(ctx_plain)
        if not m:
            return ""
        return m.group("ans")


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
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    lexicons: Lexicons = field(default_factory=lambda: DEFAULT_LEXICONS)
    toxicity_classifier: ToxicityClassifier = field(default_factory=ToxicityClassifier)
    normalizer: Normalizer = field(default_factory=lambda: Normalizer(NormalizationConfig(), lang="en"))
    safety_config: Dict[str, object] = field(default_factory=dict)


class SafeGenerationPipeline:
    def __init__(
        self,
        qg_model: StubQGModel,
        qa_model_en: StubQAModel,
        qa_model_multi: StubQAModel,
        nli_model: StubNLIModel,
        config: PipelineConfig,
        nli_service: NLIService | None = None,
        toxicity_service: ToxicityService | None = None,
    ) -> None:
        self.qg_model = qg_model
        self.qa_model_en = qa_model_en
        self.qa_model_multi = qa_model_multi
        self.nli_model = nli_model
        self.config = config
        self.nli_service = nli_service
        self.toxicity_service = toxicity_service

    def run(self, request: GenerateRequest) -> GenerateResponse:
        self._validate_request(request)
        question = self._generate_question(request)
        qa_result = self._qa_check(request, question)
        nli_result = self._nli_check(request, question)
        lexicon_hits, tox_prob, tox_flags = self._toxicity_check(request, question)
        policy_ok = self.config.policy.check_context_only(question, request.context)

        reasons = []
        thresholds = self.config.thresholds
        if qa_result["f1"] is not None and qa_result["f1"] < thresholds.f1_pass:
            reasons.append("qg2qa_f1_low")
        if qa_result["confidence"] is not None and qa_result["confidence"] < thresholds.conf_pass:
            reasons.append("qg2qa_conf_low")
        if not nli_result["ok"]:
            nli_label = str(nli_result["label"])
            reasons.append("nli_neutral" if nli_label == "neutral" else "nli_contradiction")
        if thresholds.lexicon_block and lexicon_hits:
            reasons.append("lexicon_block")
        if tox_prob is not None and tox_prob > thresholds.tox_prob_max:
            reasons.append("tox_high")
        if tox_flags:
            reasons.extend(sorted(set(tox_flags)))
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
            "nli": nli_result["label"],
            "lex_hits": len(lexicon_hits),
        }
        debug = {
            "decoding": self.config.decoding.__dict__,
            "wh_detected": qa_result["wh_detected"],
            "nli": {
                "hypothesis": nli_result["hypothesis"],
                "label": nli_result["label"],
                "probs": nli_result.get("probs"),
                "ok": nli_result["ok"],
            },
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

    @staticmethod
    def _extract_phrase_around_answer(context: str, answer: str, window: int = 6) -> Optional[str]:
        answer_tokens = [tok.strip("\"'“”‘’.,!?;:()[]{}").lower() for tok in answer.split() if tok.strip()]
        if not answer_tokens:
            return None

        ctx_tokens = context.split()
        norm_ctx = [tok.strip("\"'“”‘’.,!?;:()[]{}").lower() for tok in ctx_tokens]
        for i in range(0, len(norm_ctx) - len(answer_tokens) + 1):
            if norm_ctx[i : i + len(answer_tokens)] == answer_tokens:
                start = max(0, i - window)
                end = min(len(ctx_tokens), i + len(answer_tokens) + window)
                return " ".join(ctx_tokens[start:end]).strip()
        return None

    def _build_nli_hypothesis(self, request: GenerateRequest, question: str) -> str:
        if request.answer:
            phrase = self._extract_phrase_around_answer(request.context, request.answer)
            if phrase:
                return phrase.rstrip(" ?")
            return request.answer.strip().rstrip("?")
        return question.rstrip(" ?")

    def _nli_check(self, request: GenerateRequest, question: str) -> Dict[str, object]:
        hypothesis = self._build_nli_hypothesis(request, question)
        if self.nli_service:
            result = self.nli_service.predict([request.context], [hypothesis])[0]
            ok = self.nli_service.decide(result)
            return {"hypothesis": hypothesis, "label": str(result.get("label")), "probs": result.get("probs"), "ok": ok}

        label = self.nli_model.classify(request.context, hypothesis)
        ok = True
        if self.config.thresholds.require_entailment:
            ok = label == "entailment"
        elif not self.config.thresholds.neutral_ok and label == "neutral":
            ok = False
        return {"hypothesis": hypothesis, "label": label, "probs": None, "ok": ok}

    def _toxicity_check(self, request: GenerateRequest, question: str) -> tuple[list[str], float, list[str]]:
        text_for_screening = question
        if request.mode == "aware" and request.answer:
            text_for_screening = f"{question} {request.answer}".strip()

        if self.toxicity_service:
            tox_res = self.toxicity_service.score([text_for_screening], request.lang, context=request.context)[0]
            lexicon_hits = tox_res.get("lexicon_hits", [])
            tox_prob = tox_res.get("prob")
            flags = tox_res.get("flags", [])
            if not lexicon_hits:
                lexicon_hits = self.config.lexicons.find_matches(text_for_screening, request.lang)
            if tox_prob is None:
                tox_prob = self.config.toxicity_classifier.score(text_for_screening, request.lang, lexicon_hits)
            return lexicon_hits, tox_prob, flags

        lexicon_hits = self.config.lexicons.find_matches(text_for_screening, request.lang)
        tox_prob = self.config.toxicity_classifier.score(text_for_screening, request.lang, lexicon_hits)
        return lexicon_hits, tox_prob, []

    @staticmethod
    def _detect_wh(question: str) -> Optional[str]:
        cleaned = question.strip().lower()
        if not cleaned:
            return None
        if cleaned.startswith("how many"):
            return "how_many"
        token = cleaned.split()[0].strip("?.,! ")
        if token in {"who", "when", "where", "what", "why", "how", "хто", "коли", "де", "чому", "як", "що"}:
            return token
        return None


def build_pipeline(config: PipelineConfig) -> SafeGenerationPipeline:
    qg_model = StubQGModel(config.decoding)
    qa_en = StubQAModel()
    qa_multi = StubQAModel()
    nli = StubNLIModel()
    nli_service = NLIService(config.safety_config.get("nli", {}), device=config.runtime.device)
    tox_service = ToxicityService(config.safety_config.get("toxicity", {}), device=config.runtime.device)
    return SafeGenerationPipeline(qg_model, qa_en, qa_multi, nli, config, nli_service=nli_service, toxicity_service=tox_service)


__all__ = ["SafeGenerationPipeline", "PipelineConfig", "build_pipeline", "QGDecodingConfig", "Thresholds", "RuntimeConfig"]
