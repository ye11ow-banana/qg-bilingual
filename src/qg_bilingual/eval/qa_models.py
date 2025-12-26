"""Helpers for loading QA models and managing truncation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer


@dataclass
class QAModelBundle:
    model: AutoModelForQuestionAnswering
    tokenizer: AutoTokenizer
    device: torch.device
    device_label: str
    question_max_tokens: int = 64


def resolve_device(device: Optional[object]) -> Tuple[torch.device, str]:
    if device is None or (isinstance(device, str) and device.lower() == "auto"):
        selected = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return selected, selected.type

    if isinstance(device, str):
        normalized = device.lower()
        if normalized == "cuda":
            return torch.device("cuda"), "cuda"
        if normalized == "cpu":
            return torch.device("cpu"), "cpu"
        if normalized.startswith("cuda:"):
            return torch.device(normalized), normalized
        return torch.device(device), device

    if isinstance(device, torch.device):
        label = device.type
        if device.index is not None:
            label = f"{label}:{device.index}"
        return device, label

    return torch.device("cpu"), str(device)


@lru_cache(maxsize=2)
def _load_model(model_name: str) -> Tuple[AutoModelForQuestionAnswering, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    return model, tokenizer


def load_qa_model(model_name: str, device: Optional[object] = "auto") -> QAModelBundle:
    model, tokenizer = _load_model(model_name)
    resolved_device, label = resolve_device(device)
    model.to(resolved_device)
    model.eval()
    return QAModelBundle(model=model, tokenizer=tokenizer, device=resolved_device, device_label=label)


def truncate_context(tokenizer: AutoTokenizer, context: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return context
    token_ids = tokenizer.encode(
        context,
        add_special_tokens=False,
        truncation=True,
        max_length=max_tokens,
    )
    return tokenizer.decode(token_ids, skip_special_tokens=True)
