from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


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
        )


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
