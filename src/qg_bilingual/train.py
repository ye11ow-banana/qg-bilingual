from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from accelerate import Accelerator
from evaluate import load as load_metric
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup,
    set_seed,
)
from transformers.trainer_pt_utils import LabelSmoother

from qg_bilingual.eval.qg2qa import qg2qa_metrics
from qg_bilingual.io.jsonl_dataset import QGJsonlDataset, load_jsonl
from qg_bilingual.utils.config import TrainConfig

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Universal trainer for QG models")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    return parser.parse_args()


def _resolve_git_hash() -> str:
    try:
        output = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent)
        return output.decode().strip()
    except Exception:  # pragma: no cover - defensive
        return "unknown"


def build_model(cfg: TrainConfig, tokenizer) -> AutoModelForSeq2SeqLM:
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model)
    model.config.dropout = cfg.train.dropout
    if hasattr(model.config, "attention_dropout"):
        model.config.attention_dropout = cfg.train.dropout

    model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model.config, "decoder_start_token_id", None) is None:
        model.config.decoder_start_token_id = tokenizer.pad_token_id
    if getattr(model.config, "eos_token_id", None) is None and tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id

    if cfg.peft.lora:
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=cfg.peft.r,
            lora_alpha=cfg.peft.alpha,
            lora_dropout=cfg.peft.dropout,
            target_modules=cfg.peft.target_modules or None,
        )
        model = get_peft_model(model, lora_config)
        LOGGER.info("LoRA enabled (r=%s, alpha=%s)", cfg.peft.r, cfg.peft.alpha)

    model.resize_token_embeddings(len(tokenizer))
    return model


def prepare_dataloaders(cfg: TrainConfig, tokenizer) -> Tuple[DataLoader, DataLoader, Sequence, Sequence]:
    train_records = load_jsonl(cfg.data.train_path)
    val_records = load_jsonl(cfg.data.val_path)

    generator = torch.Generator()
    generator.manual_seed(cfg.train.seed)

    train_dataset = QGJsonlDataset(
        train_records,
        tokenizer,
        max_input_len=cfg.train.max_input_len,
        max_target_len=cfg.train.max_target_len,
        mode=cfg.task.mode,
        lang=cfg.task.lang,
    )
    val_dataset = QGJsonlDataset(
        val_records,
        tokenizer,
        max_input_len=cfg.train.max_input_len,
        max_target_len=cfg.train.max_target_len,
        mode=cfg.task.mode,
        lang=cfg.task.lang,
    )

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=None)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_per_device,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_per_device,
        shuffle=False,
        collate_fn=collator,
        generator=generator,
    )
    return train_loader, val_loader, train_records, val_records


def compute_metrics(
    predictions: Sequence[str],
    references: Sequence[str],
    rouge_metric,
    bleu_metric,
) -> Dict[str, float]:
    def _extract_f1(score) -> float:
        """Handle rouge score objects across evaluate versions.

        Newer versions of ``evaluate`` return simple floats while older ones
        expose a ``mid.fmeasure`` attribute. This helper normalises both
        shapes to a plain float.
        """

        if hasattr(score, "mid"):
            return score.mid.fmeasure
        if isinstance(score, dict):
            if "mid" in score and hasattr(score["mid"], "fmeasure"):
                return score["mid"].fmeasure
            if "fmeasure" in score:
                return score["fmeasure"]
        return float(score)

    rouge_scores = rouge_metric.compute(
        predictions=predictions, references=references, use_stemmer=True
    )
    bleu_scores = bleu_metric.compute(
        predictions=predictions, references=[[ref] for ref in references]
    )
    return {
        "rouge1": _extract_f1(rouge_scores["rouge1"]),
        "rouge2": _extract_f1(rouge_scores["rouge2"]),
        "rougeL": _extract_f1(rouge_scores["rougeL"]),
        "bleu": bleu_scores.get("score", 0.0),
    }


def _question_stats(predictions: Sequence[str]) -> Dict[str, object]:
    lengths = [len(p.split()) for p in predictions if p.strip()]
    first_tokens = [p.strip().split()[0].lower() for p in predictions if p.strip()]
    return {
        "avg_question_length": float(sum(lengths) / len(lengths)) if lengths else 0.0,
        "wh_distribution": dict(Counter(first_tokens)),
    }


def evaluate(
    model: AutoModelForSeq2SeqLM,
    dataloader: DataLoader,
    accelerator: Accelerator,
    tokenizer,
    cfg: TrainConfig,
    rouge_metric,
    bleu_metric,
):
    model.eval()
    generated_texts: List[str] = []
    reference_texts: List[str] = []

    generation_kwargs = {
        "max_new_tokens": max(8, cfg.decoding.max_new_tokens),
        "min_new_tokens": max(4, cfg.decoding.min_new_tokens),
        "no_repeat_ngram_size": cfg.decoding.no_repeat_ngram_size,
        "repetition_penalty": cfg.decoding.repetition_penalty,
        "length_penalty": cfg.decoding.length_penalty,
        "early_stopping": True,
    }
    if cfg.decoding.strategy == "beam":
        generation_kwargs.update(
            {
                "num_beams": cfg.decoding.num_beams,
                "do_sample": False,
            }
        )
    else:
        generation_kwargs.update(
            {
                "do_sample": True,
                "top_p": cfg.decoding.top_p,
                "temperature": cfg.decoding.temperature,
            }
        )

    for batch in tqdm(
        dataloader, desc="Evaluating", disable=not accelerator.is_local_main_process
    ):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **generation_kwargs,
            )

        generated_tokens = accelerator.pad_across_processes(
            generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
        )
        labels = accelerator.pad_across_processes(
            batch["labels"], dim=1, pad_index=-100
        )

        generated_tokens, labels = accelerator.gather_for_metrics(
            (generated_tokens, labels)
        )

        generated_tokens = generated_tokens.cpu()
        labels = labels.cpu()
        labels = torch.where(labels == -100, tokenizer.pad_token_id, labels)

        decoded_preds = tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        generated_texts.extend(decoded_preds)
        reference_texts.extend(decoded_labels)

    metrics = compute_metrics(
        predictions=generated_texts,
        references=reference_texts,
        rouge_metric=rouge_metric,
        bleu_metric=bleu_metric,
    )
    metrics.update(_question_stats(generated_texts))
    model.train()
    return metrics, generated_texts, reference_texts


def save_samples(
    path: Path,
    records: Sequence,
    predictions: Sequence[str],
    *,
    limit: int = 100,
    mode: str = "aware",
    lang: str | None = None,
    model_name: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = 0
    def _detect_wh(question: str) -> str | None:
        cleaned = question.strip().lower()
        if not cleaned:
            return None
        if cleaned.startswith("how many"):
            return "how_many"
        first = cleaned.split()[0].strip("?.,! ")
        if first in {"who", "when", "where", "what", "why", "how", "хто", "коли", "де", "чому", "як", "що"}:
            return first
        return None

    with path.open("w", encoding="utf-8") as f:
        for record, prediction in list(zip(records, predictions)):
            answer = getattr(record, "answer", "")
            cleaned_answer = str(answer).strip()
            if mode.lower() == "aware" and not cleaned_answer:
                continue

            cleaned_prediction = prediction.strip()

            record_lang = str(getattr(record, "lang", "") or "").strip().lower()
            if not record_lang:
                record_lang = str(lang or "").strip().lower()
            if not record_lang:
                record_lang = "unk"

            rec_id = getattr(record, "id", None)
            if rec_id is None or str(rec_id).strip() == "":
                rec_id = str(saved)

            wh_type = getattr(record, "wh_type", None) or _detect_wh(cleaned_prediction)
            invalid_generation = len(cleaned_prediction) == 0

            payload = {
                "id": str(rec_id),
                "model": str(model_name or path.parent.name),
                "mode": mode,
                "context": getattr(record, "context", ""),
                "question": cleaned_prediction,
                "gold_answer": cleaned_answer,
                "reference": cleaned_answer,
                "unanswerable": cleaned_answer == "",
                "lang": record_lang,
                "wh_type": wh_type,
                "reference_question": getattr(record, "question", ""),
                "question_len": len(cleaned_prediction.split()),
                "invalid_generation": invalid_generation,
                "passed": (not invalid_generation) and (cleaned_answer != ""),
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            saved += 1
            if saved >= limit:
                break
    LOGGER.info("Saved %s samples to %s", saved, path)


def save_metrics(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Saved metrics to %s", path)


def train(cfg: TrainConfig) -> Dict[str, float]:
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.train.grad_accum,
        mixed_precision=cfg.amp.precision,
    )
    if accelerator.is_local_main_process:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s",
        )
    accelerator.print(f"Loaded config from {cfg.config_path}")

    set_seed(cfg.train.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    (
        train_loader,
        val_loader,
        _train_records,
        val_records,
    ) = prepare_dataloaders(cfg, tokenizer)
    model = build_model(cfg, tokenizer)

    rouge_metric = load_metric("rouge")
    bleu_metric = load_metric("sacrebleu")
    label_smoother = (
        LabelSmoother(epsilon=cfg.train.label_smoothing)
        if cfg.train.label_smoothing > 0
        else None
    )

    optimizer = AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

    num_update_steps_per_epoch = math.ceil(len(train_loader) / cfg.train.grad_accum)
    max_train_steps = cfg.train.epochs * num_update_steps_per_epoch
    num_warmup_steps = int(cfg.train.warmup_ratio * max_train_steps)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )

    total_steps = 0
    best_rouge = -1.0
    epochs_without_improvement = 0
    progress_bar = tqdm(
        range(max_train_steps), disable=not accelerator.is_local_main_process
    )

    for epoch in range(cfg.train.epochs):
        model.train()
        for step, batch in enumerate(train_loader, start=1):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = (
                    label_smoother(outputs, batch["labels"])
                    if label_smoother
                    else outputs.loss
                )
                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            total_steps += 1
            progress_bar.update(1)
            if accelerator.is_local_main_process and total_steps % 10 == 0:
                LOGGER.info(
                    "Epoch %s Step %s Loss %.4f", epoch + 1, total_steps, loss.item()
                )

            if cfg.train.eval_every_steps and total_steps % cfg.train.eval_every_steps == 0:
                eval_metrics, generated_questions, _ = evaluate(
                    model,
                    val_loader,
                    accelerator,
                    tokenizer,
                    cfg,
                    rouge_metric,
                    bleu_metric,
                )
                accelerator.print(f"Step {total_steps}: {eval_metrics}")
                if eval_metrics["rougeL"] > best_rouge:
                    best_rouge = eval_metrics["rougeL"]
                    epochs_without_improvement = 0
                    save_model(accelerator, model, tokenizer, cfg.output_dir, cfg.peft.lora)
                else:
                    epochs_without_improvement += 1

            if total_steps >= max_train_steps:
                break

        eval_metrics, generated_questions, reference_questions = evaluate(
            model, val_loader, accelerator, tokenizer, cfg, rouge_metric, bleu_metric
        )
        accelerator.print(
            f"Epoch {epoch + 1} validation metrics: {json.dumps(eval_metrics, indent=2)}"
        )
        if eval_metrics["rougeL"] > best_rouge:
            best_rouge = eval_metrics["rougeL"]
            epochs_without_improvement = 0
            save_model(accelerator, model, tokenizer, cfg.output_dir, cfg.peft.lora)
            save_metrics(cfg.eval.save_metrics_to, _build_metrics_payload(cfg, eval_metrics))
            _save_generation_artifacts(cfg, val_records, generated_questions)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= cfg.train.early_stopping_patience:
            accelerator.print("Early stopping triggered")
            break

    accelerator.wait_for_everyone()
    final_metrics, generated_questions, reference_questions = evaluate(
        model, val_loader, accelerator, tokenizer, cfg, rouge_metric, bleu_metric
    )
    metadata = _build_metrics_payload(cfg, final_metrics)
    save_metrics(cfg.eval.save_metrics_to, metadata)
    _save_generation_artifacts(cfg, val_records, generated_questions)

    qa_records = [
        {
            "context": record.context,
            "answer": record.answer,
            "question": question,
        }
        for record, question in zip(val_records, generated_questions)
    ]
    qa_metrics = qg2qa_metrics(
        qa_records,
        qa_ckpt_en=cfg.qg2qa.qa_ckpt_en,
        qa_ckpt_multi=cfg.qg2qa.qa_ckpt_multi,
        lang=cfg.qg2qa.lang,
        f1_thr=cfg.qg2qa.f1_thr,
        conf_thr=cfg.qg2qa.conf_thr,
        batch_size=cfg.qg2qa.batch_size,
        device=cfg.qg2qa.device,
    )
    qg2qa_path = cfg.eval.qg2qa_save_to or cfg.output_dir / "qg2qa_val.json"
    save_metrics(qg2qa_path, qa_metrics)
    return final_metrics


def _save_generation_artifacts(cfg: TrainConfig, val_records: Sequence, predictions: Sequence[str]) -> None:
    samples_path = cfg.eval.samples_path or cfg.output_dir / "samples_val.jsonl"
    save_samples(
        samples_path,
        val_records,
        predictions,
        limit=100,
        mode=cfg.task.mode,
        lang=cfg.task.lang,
        model_name=cfg.output_dir.name,
    )


def _build_metrics_payload(cfg: TrainConfig, metrics: Dict[str, float]) -> Dict[str, object]:
    payload = {
        "model": cfg.model,
        "mode": cfg.task.mode,
        "lang": cfg.task.lang,
        "metrics": metrics,
        "metadata": {
            "seed": cfg.train.seed,
            "git_hash": _resolve_git_hash(),
            "config_path": str(cfg.config_path) if cfg.config_path else None,
            "train_path": str(cfg.data.train_path),
            "val_path": str(cfg.data.val_path),
            "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "max_input_len": cfg.train.max_input_len,
            "max_target_len": cfg.train.max_target_len,
            "decoding": cfg.decoding.__dict__,
            "peft": cfg.peft.__dict__,
        },
    }
    return payload


def save_model(
    accelerator: Accelerator,
    model: AutoModelForSeq2SeqLM,
    tokenizer,
    output_dir: Path,
    lora_enabled: bool,
) -> None:
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        if lora_enabled:
            adapter_dir = output_dir / "adapter"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            unwrapped.save_pretrained(adapter_dir)
            LOGGER.info("Saved LoRA adapters to %s", adapter_dir)

            base_dir = output_dir / "base_model"
            base_dir.mkdir(parents=True, exist_ok=True)
            # type: ignore[attr-defined]
            unwrapped.base_model.model.save_pretrained(base_dir)
            LOGGER.info("Saved base model weights to %s", base_dir)
        else:
            unwrapped.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        LOGGER.info("Saved model and tokenizer to %s", output_dir)
    accelerator.wait_for_everyone()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig.from_yaml(args.config)
    cfg.output_dir = cfg.output_dir.expanduser()
    cfg.eval.save_metrics_to = cfg.eval.save_metrics_to.expanduser()
    if cfg.eval.samples_path:
        cfg.eval.samples_path = cfg.eval.samples_path.expanduser()
    if cfg.eval.qg2qa_save_to:
        cfg.eval.qg2qa_save_to = cfg.eval.qg2qa_save_to.expanduser()
    if not cfg.qg2qa.lang:
        cfg.qg2qa.lang = cfg.task.lang

    metrics = train(cfg)
    LOGGER.info("Finished training. Final metrics: %s", metrics)


if __name__ == "__main__":
    main()

