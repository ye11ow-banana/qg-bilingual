"""Client abstraction for running the safety-aware QG demo."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, Literal, Optional

from qg_bilingual.server import schemas
from qg_bilingual.server.deps import (
    DEFAULT_CONFIG_PATH,
    _build_decoding,
    _build_policy,
    _build_thresholds,
    load_config,
)
from qg_bilingual.server.pipeline import PipelineConfig, SafeGenerationPipeline, build_pipeline
from qg_bilingual.server.safety import DEFAULT_LEXICONS, ToxicityClassifier


@dataclass
class ClientConfig:
    decoding: dict
    thresholds: dict


class QGClient:
    """Wrapper that hides local pipeline vs HTTP usage."""

    def __init__(
        self,
        mode: Literal["local", "http"],
        server_url: str | None = None,
        config: str | None = None,
    ) -> None:
        if mode not in {"local", "http"}:
            raise ValueError("mode must be 'local' or 'http'")
        self.mode = mode
        self.server_url = (server_url or "http://localhost:8000").rstrip("/")
        self.config_path = config or str(DEFAULT_CONFIG_PATH)

        raw_cfg = load_config(self.config_path)
        self.decoding_cfg = _build_decoding(raw_cfg.get("thresholds", {}))
        self.thresholds_cfg = _build_thresholds(raw_cfg.get("thresholds", {}))
        self.policy_cfg = _build_policy(raw_cfg.get("policy", {}))
        self._pipeline: Optional[SafeGenerationPipeline] = None

    @property
    def config(self) -> ClientConfig:
        return ClientConfig(
            decoding=self.decoding_cfg.__dict__,
            thresholds=self.thresholds_cfg.__dict__,
        )

    def _get_pipeline(self) -> SafeGenerationPipeline:
        if self._pipeline is None:
            pipeline_cfg = PipelineConfig(
                decoding=self.decoding_cfg,
                thresholds=self.thresholds_cfg,
                policy=self.policy_cfg,
                lexicons=DEFAULT_LEXICONS,
                toxicity_classifier=ToxicityClassifier(),
            )
            self._pipeline = build_pipeline(pipeline_cfg)
        return self._pipeline

    def generate_safe(
        self,
        *,
        context: str,
        answer: Optional[str],
        lang: str,
        mode: str,
        wh_type: Optional[str],
    ) -> Dict:
        payload = {
            "context": context,
            "answer": answer if mode == "aware" else None,
            "lang": lang,
            "mode": mode,
            "wh_type": wh_type or None,
        }
        if self.mode == "http":
            return self._call_http(payload)
        return self._call_local(payload)

    def _call_local(self, payload: Dict) -> Dict:
        request = schemas.GenerateRequest(**payload)
        response = self._get_pipeline().run(request)
        data = response.model_dump()
        return self._normalize_question(data)

    def _call_http(self, payload: Dict) -> Dict:
        endpoint = f"{self.server_url}/generate_safe"
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read().decode("utf-8")
                data = json.loads(body)
        except urllib.error.HTTPError as exc:  # pragma: no cover - runtime error path
            raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode('utf-8')}") from exc
        except urllib.error.URLError as exc:  # pragma: no cover - runtime error path
            raise RuntimeError(f"HTTP request failed: {exc.reason}") from exc
        return self._normalize_question(data)

    @staticmethod
    def _normalize_question(data: Dict) -> Dict:
        question = data.get("question")
        if question and not str(question).strip().endswith("?"):
            data["question"] = f"{str(question).strip()}?"
        return data


__all__ = ["QGClient", "ClientConfig"]
