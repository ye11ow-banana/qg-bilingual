from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, Tuple

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

from qg_bilingual.io.jsonl_dataset import QGJsonlDataset, load_jsonl
from qg_bilingual.utils.config import TrainConfig

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train T5-base for answer-aware QG")
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to YAML config file"
    )
    return parser.parse_args()


def build_model(cfg: TrainConfig, tokenizer) -> AutoModelForSeq2SeqLM:
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)
    if cfg.lora:
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=cfg.lora_r,
            lora_alpha=2 * cfg.lora_r,
            lora_dropout=0.05,
        )
        model = get_peft_model(model, lora_config)
        LOGGER.info("LoRA enabled (r=%s)", cfg.lora_r)
    model.resize_token_embeddings(len(tokenizer))
    return model


def prepare_dataloaders(
    cfg: TrainConfig,
    tokenizer,
) -> Tuple[DataLoader, DataLoader]:
    train_records = load_jsonl(cfg.train_file)
    val_records = load_jsonl(cfg.val_file)

    generator = torch.Generator()
    generator.manual_seed(cfg.seed)

    train_dataset = QGJsonlDataset(
        train_records,
        tokenizer,
        max_input_len=cfg.max_input_len,
        max_target_len=cfg.max_target_len,
    )
    val_dataset = QGJsonlDataset(
        val_records,
        tokenizer,
        max_input_len=cfg.max_input_len,
        max_target_len=cfg.max_target_len,
    )

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=None)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        generator=generator,
    )
    return train_loader, val_loader


def evaluate(
    model: AutoModelForSeq2SeqLM,
    dataloader: DataLoader,
    accelerator: Accelerator,
    tokenizer,
    cfg: TrainConfig,
):
    model.eval()
    rouge = load_metric("rouge")
    generated_texts = []
    reference_texts = []

    for batch in tqdm(
        dataloader, desc="Evaluating", disable=not accelerator.is_local_main_process
    ):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=cfg.max_target_len,
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

    rouge_scores = rouge.compute(
        predictions=generated_texts, references=reference_texts
    )
    model.train()
    return {
        "rouge1": rouge_scores["rouge1"].mid.fmeasure,
        "rougeL": rouge_scores["rougeL"].mid.fmeasure,
    }


def train(cfg: TrainConfig) -> Dict[str, float]:
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision="fp16" if cfg.fp16 else "no",
    )
    if accelerator.is_local_main_process:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s",
        )
    accelerator.print(f"Loaded config from {cfg}")

    set_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    train_loader, val_loader = prepare_dataloaders(cfg, tokenizer)
    model = build_model(cfg, tokenizer)

    optimizer = AdamW(model.parameters(), lr=cfg.lr)

    num_update_steps_per_epoch = math.ceil(
        len(train_loader) / cfg.gradient_accumulation_steps
    )
    max_train_steps = cfg.num_train_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(cfg.warmup_ratio * max_train_steps)
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
    progress_bar = tqdm(
        range(max_train_steps), disable=not accelerator.is_local_main_process
    )

    for epoch in range(cfg.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_loader, start=1):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
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

            if total_steps % cfg.eval_every_steps == 0:
                eval_metrics = evaluate(model, val_loader, accelerator, tokenizer, cfg)
                accelerator.print(f"Step {total_steps}: {eval_metrics}")
                if eval_metrics["rougeL"] > best_rouge:
                    best_rouge = eval_metrics["rougeL"]
                    save_model(accelerator, model, tokenizer, cfg.output_dir, cfg.lora)
                    save_metrics(cfg.output_dir, eval_metrics)

            if total_steps >= max_train_steps:
                break

    accelerator.wait_for_everyone()
    final_metrics = evaluate(model, val_loader, accelerator, tokenizer, cfg)
    save_model(accelerator, model, tokenizer, cfg.output_dir, cfg.lora)
    save_metrics(cfg.output_dir, final_metrics)
    return final_metrics


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


def save_metrics(output_dir: Path, metrics: Dict[str, float]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics_val.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    LOGGER.info("Saved validation metrics to %s", metrics_path)


def main() -> None:
    args = parse_args()
    cfg = TrainConfig.from_yaml(args.config)
    cfg.output_dir = cfg.output_dir.expanduser()

    metrics = train(cfg)
    LOGGER.info("Finished training. Final metrics: %s", metrics)


if __name__ == "__main__":
    main()
