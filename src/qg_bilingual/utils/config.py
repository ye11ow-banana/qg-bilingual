from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class TaskConfig:
    mode: str
    lang: str


@dataclass
class DataConfig:
    train_path: Path
    val_path: Path


@dataclass
class TrainSettings:
    lr: float = 3e-5
    weight_decay: float = 0.01
    epochs: int = 3
    batch_per_device: int = 8
    grad_accum: int = 1
    max_input_len: int = 512
    max_target_len: int = 48
    label_smoothing: float = 0.0
    dropout: float = 0.1
    warmup_ratio: float = 0.05
    eval_every_steps: Optional[int] = None
    early_stopping_patience: int = 3
    seed: int = 42

    def __post_init__(self) -> None:
        """Normalize numeric values loaded from YAML.

        Some YAML parsers may return numbers as strings (for example when the
        values are quoted). Casting here ensures downstream consumers such as
        optimizers receive the expected numeric types.
        """

        self.lr = float(self.lr)
        self.weight_decay = float(self.weight_decay)
        self.epochs = int(self.epochs)
        self.batch_per_device = int(self.batch_per_device)
        self.grad_accum = int(self.grad_accum)
        self.max_input_len = int(self.max_input_len)
        self.max_target_len = int(self.max_target_len)
        self.label_smoothing = float(self.label_smoothing)
        self.dropout = float(self.dropout)
        self.warmup_ratio = float(self.warmup_ratio)
        self.eval_every_steps = (
            None if self.eval_every_steps is None else int(self.eval_every_steps)
        )
        self.early_stopping_patience = int(self.early_stopping_patience)
        self.seed = int(self.seed)


@dataclass
class PeftConfig:
    lora: bool = False
    r: int = 8
    alpha: int = 16
    target_modules: List[str] = field(default_factory=list)
    dropout: float = 0.05


@dataclass
class AmpConfig:
    precision: str = "no"  # one of ["fp16", "bf16", "no"]


@dataclass
class DecodingConfig:
    strategy: str = "beam"  # beam or top_p
    num_beams: int = 6
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 3
    max_new_tokens: int = 32
    min_new_tokens: int = 4
    repetition_penalty: float = 1.0
    top_p: float = 0.9
    temperature: float = 1.0

    def __post_init__(self) -> None:
        def _as_float(x):
            return float(x)

        def _as_int(x):
            return int(x)

        self.max_new_tokens = _as_int(self.max_new_tokens)
        self.min_new_tokens = _as_int(self.min_new_tokens)
        self.num_beams = _as_int(self.num_beams)
        self.no_repeat_ngram_size = _as_int(self.no_repeat_ngram_size)
        self.repetition_penalty = _as_float(self.repetition_penalty)
        self.length_penalty = _as_float(self.length_penalty)
        self.top_p = _as_float(self.top_p)
        self.temperature = _as_float(self.temperature)

        assert self.max_new_tokens >= 8
        assert self.min_new_tokens >= 4
        assert self.max_new_tokens > self.min_new_tokens


@dataclass
class EvalConfig:
    compute: List[str] = field(default_factory=lambda: ["rouge", "bleu"])
    save_metrics_to: Path = Path("runs/experiment/metrics_val.json")
    samples_path: Optional[Path] = None
    qg2qa_save_to: Optional[Path] = None


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
    model: str
    task: TaskConfig
    data: DataConfig
    train: TrainSettings
    peft: PeftConfig
    amp: AmpConfig
    decoding: DecodingConfig
    eval: EvalConfig
    qg2qa: QG2QAConfig = field(default_factory=QG2QAConfig)
    output_dir: Path = Path("runs/experiment")
    config_path: Optional[Path] = None

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainConfig":
        raw = _load_yaml(path)
        task = TaskConfig(**raw.get("task", {}))
        data = DataConfig(
            train_path=Path(raw["data"]["train_path"]),
            val_path=Path(raw["data"]["val_path"]),
        )
        train_settings = TrainSettings(**raw.get("train", {}))
        peft = PeftConfig(**raw.get("peft", {}))
        amp = AmpConfig(**raw.get("amp", {}))
        decoding = DecodingConfig(**raw.get("decoding", {}))
        eval_raw = raw.get("eval", {})
        eval_cfg = EvalConfig(
            compute=eval_raw.get("compute", ["rouge", "bleu"]),
            save_metrics_to=Path(
                eval_raw.get("save_metrics_to", "runs/experiment/metrics_val.json")
            ),
            samples_path=Path(eval_raw["samples_path"]) if eval_raw.get("samples_path") else None,
            qg2qa_save_to=Path(eval_raw["qg2qa_save_to"]) if eval_raw.get("qg2qa_save_to") else None,
        )
        qg2qa_cfg = _load_qg2qa_config(raw.get("qg2qa", {}))

        output_dir = Path(raw.get("output_dir") or eval_cfg.save_metrics_to).parent

        return cls(
            model=str(raw["model"]),
            task=task,
            data=data,
            train=train_settings,
            peft=peft,
            amp=amp,
            decoding=decoding,
            eval=eval_cfg,
            qg2qa=qg2qa_cfg,
            output_dir=output_dir,
            config_path=path,
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

