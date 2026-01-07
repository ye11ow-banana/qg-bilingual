"""Dependency wiring for the FastAPI server."""

from __future__ import annotations

import functools
import logging
import os
from pathlib import Path
from typing import Any, Dict

try:  # pragma: no cover - optional runtime detection
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    import yaml
except ImportError:  # pragma: no cover - fallback for offline tests
    yaml = None

from .pipeline import (
    PipelineConfig,
    QGDecodingConfig,
    RuntimeConfig,
    SafeGenerationPipeline,
    Thresholds,
    build_pipeline,
)
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
        "qg2qa": {"f1_pass": 0.0, "conf_pass": 0.0},
        "nli": {"require_entailment": False, "neutral_ok": True},
        "toxicity": {"prob_max": 0.4, "lexicon_block": True},
    },
    "policy": {"context_only": True, "allowed_wh": ["who", "when", "where", "what", "why", "how", "how_many"]},
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
        allowed_wh=raw_policy.get("allowed_wh", ["who", "when", "where", "what", "why", "how", "how_many"]),
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

    raw_models = cfg.get("models", {}) or {}
    raw_thresholds = cfg.get("thresholds", {}) or {}
    raw_runtime = cfg.get("runtime", {}) or {}
    raw_device = str(raw_runtime.get("device", "auto")).lower()
    if raw_device == "auto":
        if torch is not None and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            resolved_device = "mps"
        else:
            resolved_device = "cpu"
    else:
        resolved_device = raw_device
    raw_nli = raw_thresholds.get("nli", {}) or {}
    raw_tox = raw_thresholds.get("toxicity", {}) or {}

    safety_config = {
        "nli": {
            "model": raw_models.get("nli", "dummy"),
            "require_entailment": bool(raw_nli.get("require_entailment", thresholds.require_entailment)),
            "neutral_ok": bool(raw_nli.get("neutral_ok", thresholds.neutral_ok)),
            "thresholds": {
                "entailment_min_prob": float(raw_nli.get("entailment_min_prob", 0.5)),
            },
            "batch_size": int(raw_runtime.get("batch_size", 8)),
        },
        "toxicity": {
            "classifier_en": raw_models.get("tox_en", "dummy"),
            "classifier_multi": raw_models.get("tox_ua", "dummy"),
            "prob_max": float(raw_tox.get("prob_max", thresholds.tox_prob_max)),
            "use_lexicon_block": bool(raw_tox.get("lexicon_block", thresholds.lexicon_block)),
            "batch_size": int(raw_runtime.get("batch_size", 8)),
        },
    }

    pipeline_cfg = PipelineConfig(
        decoding=decoding,
        thresholds=thresholds,
        policy=policy,
        lexicons=DEFAULT_LEXICONS,
        toxicity_classifier=ToxicityClassifier(),
        runtime=RuntimeConfig(
            device=resolved_device,
            batch_size=int(raw_runtime.get("batch_size", 8)),
        ),
        safety_config=safety_config,
    )
    LOGGER.info("Pipeline built with config at %s", DEFAULT_CONFIG_PATH)
    return build_pipeline(pipeline_cfg)


__all__ = ["get_pipeline", "load_config", "DEFAULT_CONFIG_PATH"]
