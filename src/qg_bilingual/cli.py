"""Command-line entry point for bilingual question generation training.

The CLI consumes YAML configuration files to orchestrate dataset loading,
model/PEFT initialization, decoding setup, training with early stopping, and
multi-metric evaluation (ROUGE, BLEU, BERTScore, QG→QA EM/F1). It also
scaffolds MOS data collection templates and logs key experiment metadata for
reproducibility.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import yaml
from datasets import Dataset, DatasetDict, load_dataset
from evaluate import load as load_metric
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    set_seed,
)
from transformers.trainer_utils import EvalPrediction

LOGGER = logging.getLogger(__name__)


@dataclass
class DecodingSettings:
    """Container for decoding parameters used during evaluation/comparison."""

    strategy: str = "beam"
    num_beams: int = 4
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    top_p: float = 0.9
    temperature: float = 1.0

    def to_generation_kwargs(self) -> Dict[str, Any]:
        if self.strategy == "beam":
            beams = min(max(self.num_beams, 4), 8)
            penalty = min(max(self.length_penalty, 1.0), 1.2)
            return {
                "num_beams": beams,
                "length_penalty": penalty,
                "no_repeat_ngram_size": self.no_repeat_ngram_size,
                "do_sample": False,
            }

        if self.strategy == "top_p":
            return {
                "do_sample": True,
                "top_p": self.top_p,
                "temperature": self.temperature,
                "no_repeat_ngram_size": self.no_repeat_ngram_size,
            }

        raise ValueError(f"Unknown decoding strategy: {self.strategy}")


@dataclass
class TrainingConfig:
    """Structure mirroring the expected YAML configuration fields."""

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
    lora: bool = False
    lora_r: int = 8
    fp16: bool = False
    decoding: DecodingSettings = field(default_factory=DecodingSettings)
    seed: int = 42
    output_dir: Path = Path("outputs/experiments")
    early_stopping_patience: int = 3
    gradient_accumulation_steps: int = 1
    eval_steps: Optional[int] = None

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainingConfig":
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        decoding_cfg = DecodingSettings(**raw.get("decoding", {}))
        return cls(
            model_name=raw["model_name"],
            task=raw["task"],
            train_file=Path(raw["train_file"]),
            val_file=Path(raw["val_file"]),
            text_field=raw["text_field"],
            answer_field=raw["answer_field"],
            target_field=raw["target_field"],
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
            decoding=decoding_cfg,
            seed=int(raw.get("seed", 42)),
            output_dir=Path(raw.get("output_dir", "outputs/experiments")),
            early_stopping_patience=int(raw.get("early_stopping_patience", 3)),
            gradient_accumulation_steps=int(raw.get("gradient_accumulation_steps", 1)),
            eval_steps=raw.get("eval_steps"),
        )


@dataclass
class ExperimentLogger:
    """Simple experiment logger that writes metadata and templates to disk."""

    base_dir: Path

    def log_metadata(self, cfg: TrainingConfig, extras: Optional[Dict[str, Any]] = None) -> None:
        metadata_path = self.base_dir / "metadata.json"
        payload: Dict[str, Any] = {
            "seed": cfg.seed,
            "config": cfg.__dict__,
        }
        if extras:
            payload.update(extras)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        LOGGER.info("Logged metadata to %s", metadata_path)

    def scaffold_mos_template(self) -> Path:
        template_path = self.base_dir / "mos_template.jsonl"
        if template_path.exists():
            return template_path
        template_path.parent.mkdir(parents=True, exist_ok=True)
        template = {
            "worker_id": "<annotator-id>",
            "sample_id": "<uuid>",
            "context": "<context>",
            "question": "<generated-question>",
            "reference_answer": "<gold-answer>",
            "fluency": "1-5",
            "relevance": "1-5",
            "answerability": "1-5",
            "comments": "",
        }
        template_path.write_text(json.dumps(template, ensure_ascii=False) + "\n", encoding="utf-8")
        LOGGER.info("Scaffolded MOS template at %s", template_path)
        return template_path


def format_input(example: Mapping[str, Any], cfg: TrainingConfig) -> str:
    context = example.get("highlighted_context") or example[cfg.text_field]
    answer = example.get(cfg.answer_field, "")
    return f"generate question: <context> {context} </context> <answer> {answer} </answer>"


def load_and_tokenize(cfg: TrainingConfig, tokenizer) -> DatasetDict:
    data_files = {"train": str(cfg.train_file), "validation": str(cfg.val_file)}
    raw = load_dataset("json", data_files=data_files)

    def preprocess(batch: Mapping[str, Sequence[Any]]) -> Dict[str, Any]:
        prompts = [format_input(example, cfg) for example in batch]
        targets = batch[cfg.target_field]
        model_inputs = tokenizer(
            prompts,
            max_length=cfg.max_input_len,
            truncation=True,
            padding="max_length",
        )
        labels = tokenizer(
            targets,
            max_length=cfg.max_target_len,
            truncation=True,
            padding="max_length",
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = raw.map(preprocess, batched=True, remove_columns=raw["train"].column_names)
    return tokenized


def build_model(cfg: TrainingConfig, tokenizer) -> Any:
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)
    if cfg.lora:
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=cfg.lora_r,
            lora_alpha=2 * cfg.lora_r,
            lora_dropout=0.05,
        )
        model = get_peft_model(model, lora_config)
        LOGGER.info("Wrapped model with LoRA (r=%s)", cfg.lora_r)
    model.resize_token_embeddings(len(tokenizer))
    return model


def build_callbacks(cfg: TrainingConfig) -> Sequence[TrainerCallback]:
    return [EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)]


def prepare_metrics() -> Dict[str, Any]:
    return {
        "rouge": load_metric("rouge"),
        "bleu": load_metric("sacrebleu"),
        "bertscore": load_metric("bertscore"),
        "qa": load_metric("squad"),
    }


def compute_qg_qa_metrics(
    questions: Sequence[str],
    contexts: Sequence[str],
    references: Sequence[str],
    qa_predict_fn: Optional[Callable[[str, str], str]] = None,
) -> Dict[str, float]:
    if qa_predict_fn is None:
        LOGGER.warning("No QA model provided for QG→QA loop; skipping EM/F1 computation.")
        return {}

    predictions = []
    references_json = []
    for idx, (question, context, answer) in enumerate(zip(questions, contexts, references)):
        pred = qa_predict_fn(question, context)
        predictions.append({"id": str(idx), "prediction_text": pred})
        references_json.append({"id": str(idx), "answers": {"text": [answer], "answer_start": [0]}})

    metric = load_metric("squad")
    return metric.compute(predictions=predictions, references=references_json)


def build_compute_metrics(
    tokenizer,
    cfg: TrainingConfig,
    raw_eval: Dataset,
    qa_predict_fn: Optional[Callable[[str, str], str]] = None,
) -> Callable[[EvalPrediction], Dict[str, float]]:
    metrics = prepare_metrics()

    def compute(eval_preds: EvalPrediction) -> Dict[str, float]:
        preds, labels = eval_preds
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = [[token for token in label if token != -100] for label in labels]
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        results: Dict[str, float] = {}
        rouge_scores = metrics["rouge"].compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )
        results.update(
            {
                "rouge1": rouge_scores["rouge1"].mid.fmeasure,
                "rouge2": rouge_scores["rouge2"].mid.fmeasure,
                "rougeL": rouge_scores["rougeL"].mid.fmeasure,
            }
        )

        bleu = metrics["bleu"].compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
        results["bleu"] = bleu["score"]

        bert = metrics["bertscore"].compute(
            predictions=decoded_preds,
            references=decoded_labels,
            lang="en",
        )
        results["bertscore_f1"] = float(sum(bert["f1"]) / len(bert["f1"])) if bert["f1"] else 0.0

        qa_metrics = compute_qg_qa_metrics(
            questions=decoded_preds,
            contexts=raw_eval["context"] if "context" in raw_eval.column_names else [""] * len(decoded_preds),
            references=raw_eval[cfg.answer_field] if cfg.answer_field in raw_eval.column_names else [""] * len(decoded_preds),
            qa_predict_fn=qa_predict_fn,
        )
        results.update({f"qa_{k}": v for k, v in qa_metrics.items()})
        return results

    return compute


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train bilingual QG models from YAML configs.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config file.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainingConfig.from_yaml(args.config)
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
    LOGGER.info("Loaded config from %s", args.config)

    set_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenized = load_and_tokenize(cfg, tokenizer)
    model = build_model(cfg, tokenizer)

    logger = ExperimentLogger(cfg.output_dir)
    logger.log_metadata(cfg)
    logger.scaffold_mos_template()

    decoding_kwargs = cfg.decoding.to_generation_kwargs()
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(cfg.output_dir),
        evaluation_strategy="epoch",
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        learning_rate=cfg.lr,
        num_train_epochs=cfg.num_train_epochs,
        warmup_ratio=cfg.warmup_ratio,
        predict_with_generate=True,
        generation_max_length=cfg.max_target_len,
        generation_num_beams=decoding_kwargs.get("num_beams", None),
        logging_strategy="steps",
        logging_steps=cfg.eval_steps or 50,
        save_strategy="epoch",
        fp16=cfg.fp16,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        seed=cfg.seed,
    )

    compute_metrics = build_compute_metrics(tokenizer, cfg, raw_eval=tokenized["validation"])

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=build_callbacks(cfg),
    )

    LOGGER.info("Starting training...")
    trainer.train()
    LOGGER.info("Training finished. Running evaluation with decoding settings: %s", decoding_kwargs)
    metrics = trainer.evaluate(max_length=cfg.max_target_len, **decoding_kwargs)
    LOGGER.info("Evaluation metrics: %s", metrics)

    metrics_path = cfg.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    LOGGER.info("Persisted metrics to %s", metrics_path)


if __name__ == "__main__":
    main()
