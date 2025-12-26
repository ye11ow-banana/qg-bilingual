"""Batch generation entry point for bilingual question generation."""

from __future__ import annotations

import logging
import re
from typing import Dict, Iterable, List, Mapping, MutableSequence, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from qg_bilingual.data import normalize_text
from qg_bilingual.generation.prompts import WH_KEYWORDS, build_prompt

LOGGER = logging.getLogger(__name__)


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    normalized = str(device or "auto").lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(normalized)


def _detect_wh_type(question: str, lang: str) -> Optional[str]:
    """Detect WH type from the first token of the question."""

    leading = normalize_text(question).lstrip()
    if not leading:
        return None
    first_token = re.split(r"\s+", leading)[0]
    token = re.sub(r"[^\w']", "", first_token).lower()

    candidates = WH_KEYWORDS.get(lang.lower(), WH_KEYWORDS["en"])
    for key, keyword in candidates.items():
        if token.startswith(keyword):
            return key

    # Handle English contractions like "who's"/"what's".
    for key in WH_KEYWORDS["en"]:
        if token.startswith(key):
            return key

    return None


def _extract_answer(record: Mapping[str, object]) -> str:
    if "answer" in record:
        return str(record.get("answer") or "")
    answers = record.get("answers")
    if isinstance(answers, list) and answers:
        return str(answers[0])
    return ""


def _apply_question_postprocessing(text: str) -> str:
    cleaned = normalize_text(text)
    if cleaned and not cleaned.endswith("?"):
        cleaned = f"{cleaned}?"
    return cleaned


def generate_questions(
    records: Iterable[Mapping[str, object]],
    cfg_model: Mapping[str, object],
    cfg_decoding: Mapping[str, object],
    cfg_task: Mapping[str, object],
    device: str | torch.device = "auto",
) -> List[Dict[str, object]]:
    """Generate questions for provided records.

    Returns a list of dictionaries with id, question, wh_type, prompt_len, and gen_len.
    """

    model_name = cfg_model.get("name") or cfg_model.get("model_name")
    if not model_name:
        raise ValueError("cfg_model must provide 'name' or 'model_name'")

    tokenizer_name = cfg_model.get("tokenizer") or model_name
    max_input_len = int(cfg_model.get("max_input_len", 512))
    batch_size = int(cfg_model.get("batch_size", 8))
    seed = int(cfg_model.get("seed", 42))

    torch.manual_seed(seed)

    resolved_device = _resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(resolved_device)

    records_list = list(records)
    prompts: MutableSequence[str] = []
    prompt_meta: MutableSequence[Dict[str, object]] = []

    mode = str(cfg_task.get("mode", "aware"))
    wh_forced = cfg_task.get("wh_type")
    lang = str(cfg_task.get("lang", "en"))

    for idx, record in enumerate(records_list):
        context = record.get("context")
        if context is None:
            raise ValueError("Each record must include 'context'")
        answer = _extract_answer(record)
        prompt = build_prompt(
            context=str(context),
            answer=answer,
            mode=mode,
            wh_type=wh_forced,
            lang=lang,
        )
        prompts.append(prompt)
        prompt_meta.append({
            "id": record.get("id", idx),
            "answer": answer,
            "raw_lang": record.get("lang", lang),
        })

    generation_kwargs: Dict[str, object] = {
        "max_new_tokens": int(cfg_decoding.get("max_new_tokens", 32)),
        "min_new_tokens": int(cfg_decoding.get("min_new_tokens", 0)),
        "no_repeat_ngram_size": int(cfg_decoding.get("no_repeat_ngram_size", 0)),
        "repetition_penalty": float(cfg_decoding.get("repetition_penalty", 1.0)),
    }

    strategy = str(cfg_decoding.get("strategy", "beam")).lower()
    if strategy == "beam":
        generation_kwargs.update(
            {
                "num_beams": int(cfg_decoding.get("num_beams", 4)),
                "length_penalty": float(cfg_decoding.get("length_penalty", 1.0)),
                "do_sample": False,
            }
        )
    elif strategy in {"topp", "top-p", "top_p"}:
        generation_kwargs.update(
            {
                "do_sample": True,
                "top_p": float(cfg_decoding.get("top_p", 0.9)),
                "temperature": float(cfg_decoding.get("temperature", 1.0)),
            }
        )
    else:
        raise ValueError(f"Unknown decoding strategy: {strategy}")

    LOGGER.info("Decoding strategy: %s | params=%s", strategy, generation_kwargs)

    outputs: List[Dict[str, object]] = []
    for batch_start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_start : batch_start + batch_size]
        batch_meta = prompt_meta[batch_start : batch_start + batch_size]

        tokenized = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=max_input_len,
            return_tensors="pt",
        )
        tokenized = {k: v.to(resolved_device) for k, v in tokenized.items()}
        prompt_lengths = tokenized["attention_mask"].sum(dim=1).tolist()

        generated = model.generate(**tokenized, **generation_kwargs)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)

        for meta, prompt_len, raw_question, decoded_ids in zip(
            batch_meta, prompt_lengths, decoded, generated
        ):
            question = _apply_question_postprocessing(raw_question)
            wh_detected = _detect_wh_type(question, lang)
            gen_len = int((decoded_ids != tokenizer.pad_token_id).sum().item())

            outputs.append(
                {
                    "id": meta["id"],
                    "question": question,
                    "wh_type": wh_detected,
                    "prompt_len": int(prompt_len),
                    "gen_len": gen_len,
                    "decoding": {"strategy": strategy, **generation_kwargs},
                }
            )

    return outputs


__all__ = ["generate_questions"]
