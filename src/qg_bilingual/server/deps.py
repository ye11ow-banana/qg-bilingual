"""Dependency wiring for the FastAPI server."""

from __future__ import annotations

import functools
import logging
import os
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ImportError:  # pragma: no cover - fallback for offline tests
    yaml = None

from .pipeline import PipelineConfig, QGDecodingConfig, SafeGenerationPipeline, Thresholds, build_pipeline
from .safety import DEFAULT_LEXICONS, DEFAULT_PROTECTED_GROUPS, Policy, ToxicityClassifier

LOGGER = logging.getLogger(__name__)
CONFIG_ENV = "QG_SERVER_CONFIG"
DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"
FALLBACK_CONFIG = {
    "thresholds": {
        "qg": {
            "max_new_tokens": 32,
            "min_new_tokens": 4,
            "no_repeat_ngram_size": 3,
            "strategy": "beam",
            "num_beams": 6,
            "length_penalty": 1.1,
            "top_p": 0.9,
            "temperature": 0.8,
        },
        "qg2qa": {"f1_pass": 0.8, "conf_pass": 0.35},
        "nli": {"require_entailment": True, "neutral_ok": False},
        "toxicity": {"prob_max": 0.4, "lexicon_block": True},
    },
    "policy": {"context_only": True, "allowed_wh": ["who", "when", "where", "what", "why", "how"]},
}


def load_config(path: Path | str | None = None) -> Dict[str, Any]:
    cfg_path = Path(path or DEFAULT_CONFIG_PATH)
    if yaml:
        with cfg_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    LOGGER.warning("PyYAML unavailable, falling back to defaults for %s", cfg_path)
    return FALLBACK_CONFIG


def _build_policy(raw_policy: Dict[str, Any]) -> Policy:
    return Policy(
        context_only=bool(raw_policy.get("context_only", True)),
        allowed_wh=raw_policy.get("allowed_wh", ["who", "when", "where", "what", "why", "how"]),
        protected_groups=DEFAULT_PROTECTED_GROUPS,
    )


def _build_thresholds(raw: Dict[str, Any]) -> Thresholds:
    nli = raw.get("nli", {})
    tox = raw.get("toxicity", {})
    qg2qa = raw.get("qg2qa", {})
    return Thresholds(
        f1_pass=float(qg2qa.get("f1_pass", 0.8)),
        conf_pass=float(qg2qa.get("conf_pass", 0.35)),
        require_entailment=bool(nli.get("require_entailment", True)),
        neutral_ok=bool(nli.get("neutral_ok", False)),
        tox_prob_max=float(tox.get("prob_max", 0.4)),
        lexicon_block=bool(tox.get("lexicon_block", True)),
    )


def _build_decoding(raw: Dict[str, Any]) -> QGDecodingConfig:
    qg_raw = raw.get("qg", {})
    return QGDecodingConfig(
        max_new_tokens=int(qg_raw.get("max_new_tokens", 32)),
        min_new_tokens=int(qg_raw.get("min_new_tokens", 4)),
        no_repeat_ngram_size=int(qg_raw.get("no_repeat_ngram_size", 3)),
        strategy=str(qg_raw.get("strategy", "beam")),
        num_beams=int(qg_raw.get("num_beams", 6)),
        length_penalty=float(qg_raw.get("length_penalty", 1.1)),
        top_p=float(qg_raw.get("top_p", 0.9)),
        temperature=float(qg_raw.get("temperature", 0.8)),
    )


@functools.lru_cache(maxsize=1)
def get_pipeline() -> SafeGenerationPipeline:
    env_path = os.getenv(CONFIG_ENV)
    cfg = load_config(env_path)
    thresholds = _build_thresholds(cfg.get("thresholds", {}))
    decoding = _build_decoding(cfg.get("thresholds", {}))
    policy = _build_policy(cfg.get("policy", {}))

    pipeline_cfg = PipelineConfig(
        decoding=decoding,
        thresholds=thresholds,
        policy=policy,
        lexicons=DEFAULT_LEXICONS,
        toxicity_classifier=ToxicityClassifier(),
    )
    LOGGER.info("Pipeline built with config at %s", DEFAULT_CONFIG_PATH)
    return build_pipeline(pipeline_cfg)


__all__ = ["get_pipeline", "load_config", "DEFAULT_CONFIG_PATH"]
