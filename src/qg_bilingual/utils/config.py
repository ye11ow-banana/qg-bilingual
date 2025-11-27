from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class QG2QAConfig:
    qa_ckpt_en: str = "distilbert-base-uncased-distilled-squad"
    qa_ckpt_multi: str = "deepset/xlm-roberta-large-squad2"
    lang: str = "en"
    f1_thr: float = 0.8
    conf_thr: float = 0.35
    device: str = "auto"
    batch_size: int = 16


@dataclass
class TrainConfig:
    model_name: str
    task: str
    train_file: Path
    val_file: Path
    text_field: str
    answer_field: str
    target_field: str
    max_input_len: int
    max_target_len: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    lr: float
    num_train_epochs: int
    warmup_ratio: float
    lora: bool
    lora_r: int
    fp16: bool
    eval_every_steps: int
    output_dir: Path
    seed: int = 42
    gradient_accumulation_steps: int = 1
    qg2qa: QG2QAConfig = field(default_factory=QG2QAConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainConfig":
        raw = _load_yaml(path)
        return cls(
            model_name=str(raw["model_name"]),
            task=str(raw["task"]),
            train_file=Path(raw["train_file"]),
            val_file=Path(raw["val_file"]),
            text_field=str(raw["text_field"]),
            answer_field=str(raw["answer_field"]),
            target_field=str(raw["target_field"]),
            max_input_len=int(raw["max_input_len"]),
            max_target_len=int(raw["max_target_len"]),
            per_device_train_batch_size=int(raw["per_device_train_batch_size"]),
            per_device_eval_batch_size=int(raw["per_device_eval_batch_size"]),
            lr=float(raw["lr"]),
            num_train_epochs=int(raw["num_train_epochs"]),
            warmup_ratio=float(raw["warmup_ratio"]),
            lora=bool(raw.get("lora", False)),
            lora_r=int(raw.get("lora_r", 8)),
            fp16=bool(raw.get("fp16", False)),
            eval_every_steps=int(raw.get("eval_every_steps", 500)),
            output_dir=Path(raw.get("output_dir", "outputs/train_run")),
            seed=int(raw.get("seed", 42)),
            gradient_accumulation_steps=int(raw.get("gradient_accumulation_steps", 1)),
            qg2qa=_load_qg2qa_config(raw.get("qg2qa", {})),
        )


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_qg2qa_config(raw: Dict[str, Any]) -> QG2QAConfig:
    return QG2QAConfig(
        qa_ckpt_en=str(
            raw.get("qa_ckpt_en", "distilbert-base-uncased-distilled-squad")
        ),
        qa_ckpt_multi=str(raw.get("qa_ckpt_multi", "deepset/xlm-roberta-large-squad2")),
        lang=str(raw.get("lang", "en")),
        f1_thr=float(raw.get("f1_thr", 0.8)),
        conf_thr=float(raw.get("conf_thr", 0.35)),
        device=str(raw.get("device", "auto")),
        batch_size=int(raw.get("batch_size", 16)),
    )

