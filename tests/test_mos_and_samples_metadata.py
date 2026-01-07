from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from eval.mos import sample_builder

from qg_bilingual.train import save_samples


@dataclass
class _DummyRecord:
    id: str
    context: str
    question: str
    answer: str
    lang: str = ""
    wh_type: str | None = None


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def test_save_samples_writes_non_empty_lang_and_id(tmp_path: Path) -> None:
    out = tmp_path / "samples_val.jsonl"
    records = [
        _DummyRecord(
            id="abc",
            context="Kyiv is the capital city of Ukraine and stands on the Dnipro River.",
            question="What is the capital city of Ukraine?",
            answer="Kyiv",
            lang="",
        )
    ]
    predictions = ["What is the capital city of Ukraine?"]

    save_samples(out, records, predictions, limit=10, mode="aware", lang="en", model_name="t5_smoke_en")
    rows = _read_jsonl(out)
    assert len(rows) == 1
    assert rows[0]["id"] == "abc"
    assert rows[0]["lang"] == "en"
    assert rows[0]["model"] == "t5_smoke_en"
    assert rows[0]["mode"] == "aware"


def test_sample_builder_infers_defaults_and_id(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "t5_smoke_en"
    run_dir.mkdir(parents=True)
    inp = run_dir / "samples_val.jsonl"
    # Intentionally omit id/lang/model/mode to ensure builder fills them.
    inp.write_text(
        json.dumps(
            {
                "context": "Paris is the capital of France. It is known for the Eiffel Tower.",
                "question": "What is the capital of France?",
                "gold_answer": "Paris",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    rows = sample_builder._load_jsonl(inp)
    assert len(rows) == 1
    assert rows[0]["lang"] == "en"
    assert rows[0]["model"] == "t5_smoke_en"
    # mode cannot be inferred from this run name; it should remain empty here.

    batch = sample_builder.build_batch(rows, size=10, seed=13)
    assert len(batch) == 1
    assert batch[0]["id"]
    assert batch[0]["id"].lower() != "none"
    assert batch[0]["lang"] == "en"
    assert batch[0]["model"] == "t5_smoke_en"
