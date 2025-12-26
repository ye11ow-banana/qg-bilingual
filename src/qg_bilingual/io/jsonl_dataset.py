from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


@dataclass
class JsonlRecord:
    context: str
    answer: str
    question: str


def load_jsonl(path: Path) -> List[JsonlRecord]:
    records: List[JsonlRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            records.append(
                JsonlRecord(
                    context=str(row.get("context", "")),
                    answer=str(row.get("answer", "")),
                    question=str(row.get("question", "")),
                )
            )
    return records


def format_prompt(answer: str, context: str, *, mode: str, lang: str) -> str:
    normalized_mode = mode.lower()
    normalized_lang = lang.lower()

    if normalized_mode == "agnostic":
        if normalized_lang in {"ua", "uk", "ukrainian"}:
            instruction = (
                "Згенеруйте запитання до уривка. Не використовуйте підказок про відповідь;"
                " запитання має бути конкретним і базуватися лише на контексті."
            )
        else:
            instruction = (
                "Generate a specific, answerable question from the context only."
                " Do not include or assume an answer in the prompt."
            )
        return (
            "<question_generation>\n"
            f"  <instruction>{instruction}</instruction>\n"
            f"  <context>{context}</context>\n"
            "</question_generation>"
        )

    if normalized_lang in {"ua", "uk", "ukrainian"}:
        instruction = (
            "Згенеруйте запитання для заданої відповіді та контексту."
        )
    else:
        instruction = "Generate a question for the provided answer and context."

    return (
        "<question_generation>\n"
        f"  <instruction>{instruction}</instruction>\n"
        f"  <context>{context}</context>\n"
        f"  <answer>{answer}</answer>\n"
        "</question_generation>"
    )


class QGJsonlDataset(Dataset):
    def __init__(
        self,
        records: Sequence[JsonlRecord],
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_input_len: int,
        max_target_len: int,
        mode: str,
        lang: str,
    ) -> None:
        self.records = list(records)
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.mode = mode
        self.lang = lang

    def __len__(self) -> int:  # noqa: D401
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.records[idx]
        source = format_prompt(
            item.answer,
            item.context,
            mode=self.mode,
            lang=self.lang,
        )
        model_inputs = self.tokenizer(
            source,
            max_length=self.max_input_len,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        labels = self.tokenizer(
            text_target=item.question,
            max_length=self.max_target_len,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )

        input_ids = model_inputs["input_ids"].squeeze(0)
        attention_mask = model_inputs["attention_mask"].squeeze(0)
        label_ids = labels["input_ids"].squeeze(0)
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_ids,
        }
