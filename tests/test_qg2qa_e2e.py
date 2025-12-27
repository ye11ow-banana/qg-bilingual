import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch

from qg_bilingual.eval.qg2qa import (
    IOConfig,
    Prediction,
    QG2QARunConfig,
    run_qg2qa,
    save_outputs,
)


class _StubTokenizer:
    model_max_length = 128

    def num_special_tokens_to_add(self, pair: bool | None = None) -> int:  # pragma: no cover - trivial
        return 2

    def encode(self, text, add_special_tokens: bool = False, truncation: bool = False, max_length: int | None = None):
        tokens = text.split()
        if truncation and max_length is not None:
            tokens = tokens[:max_length]
        return list(range(len(tokens)))

    def decode(self, token_ids, skip_special_tokens: bool = True):
        return "dummy"


class _StubBundle:
    def __init__(self):
        self.tokenizer = _StubTokenizer()
        self.device = torch.device("cpu")
        self.device_label = "cpu"
        self.question_max_tokens = 64


# No pytest fixtures to keep the test self-contained

def test_qg2qa_e2e(tmp_path, monkeypatch):
    sample = {
        "context": "Paris is the capital of France.",
        "question": "What is the capital of France?",
        "gold_answer": "Paris",
        "unanswerable": False,
        "lang": "en",
    }
    input_path = tmp_path / "samples.jsonl"
    input_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

    def _fake_load_model(model_name, device=None):  # pragma: no cover - simple stub
        return _StubBundle()

    def _fake_predict_batch(bundle, examples, config, include_unanswerable):  # pragma: no cover - simple stub
        return [Prediction(pred="Paris", confidence=0.99, used_no_answer=False) for _ in examples]

    monkeypatch.setattr("qg_bilingual.eval.qg2qa.load_qa_model", _fake_load_model)
    monkeypatch.setattr("qg_bilingual.eval.qg2qa._predict_batch", _fake_predict_batch)

    cfg = QG2QARunConfig(
        lang="en",
        qa_model="stub",
        io=IOConfig(input_jsonl=input_path, out_dir=tmp_path),
    )

    summary, details, counts = run_qg2qa(cfg, include_unanswerable=False)
    save_outputs(summary, details, tmp_path)

    output_path = tmp_path / "qg2qa_val.json"
    assert output_path.exists()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["em"] == 1.0
    assert payload["f1"] == 1.0
    assert payload["qa_pass_rate"] == 1.0
    assert payload["included"] == 1
    assert payload["skipped_unanswerable"] == 0
    assert counts["valid"] == 1
