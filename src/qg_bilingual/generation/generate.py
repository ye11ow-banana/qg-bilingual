"""Batch generation entry point for bilingual question generation."""

from __future__ import annotations

import logging
import re
from pathlib import Path
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


def _needs_regeneration(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return True
    return not any(char.isalnum() for char in normalized)


def generate_questions(
    records: Iterable[Mapping[str, object]],
    cfg_model: Mapping[str, object],
    cfg_decoding: Mapping[str, object],
    cfg_task: Mapping[str, object],
    device: str | torch.device = "auto",
    *,
    debug: bool = False,
    run_dir: str | Path | None = None,
) -> List[Dict[str, object]]:
    """Generate questions for provided records.

    Returns a list of dictionaries with id, question, wh_type, prompt_len, and gen_len.
    """

    model_name = cfg_model.get("name") or cfg_model.get("model_name")
    if not model_name:
        raise ValueError("cfg_model must provide 'name' or 'model_name'")

    tokenizer_name = cfg_model.get("tokenizer") or model_name
    max_input_len = int(
        cfg_model.get("max_input_len")
        or cfg_decoding.get("max_input_len")
        or cfg_task.get("max_input_len")
        or 512
    )
    batch_size = int(cfg_model.get("batch_size", 8))
    seed = int(cfg_model.get("seed", 42))

    torch.manual_seed(seed)

    resolved_device = _resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(resolved_device)

    model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model.config, "decoder_start_token_id", None) is None:
        model.config.decoder_start_token_id = tokenizer.pad_token_id
    if getattr(model.config, "eos_token_id", None) is None and tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id

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
        try:
            prompt = build_prompt(
                context=str(context),
                answer=answer,
                mode=mode,
                wh_type=wh_forced,
                lang=lang,
            )
        except ValueError as exc:  # pragma: no cover - defensive skip
            LOGGER.warning("Skipping record %s: %s", record.get("id", idx), exc)
            continue
        prompts.append(prompt)
        prompt_meta.append(
            {
                "id": record.get("id", idx),
                "answer": answer,
                "raw_lang": record.get("lang", lang),
            }
        )

    min_new_tokens = max(8, int(cfg_decoding.get("min_new_tokens", 8)))
    max_new_tokens = max(24, int(cfg_decoding.get("max_new_tokens", 32)))
    if max_new_tokens < min_new_tokens:
        max_new_tokens = min_new_tokens + 8

    generation_kwargs: Dict[str, object] = {
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": min_new_tokens,
        "no_repeat_ngram_size": int(cfg_decoding.get("no_repeat_ngram_size", 3)),
        "repetition_penalty": float(cfg_decoding.get("repetition_penalty", 1.0)),
        "length_penalty": float(cfg_decoding.get("length_penalty", 1.0)),
    }

    strategy = str(cfg_decoding.get("strategy", "beam")).lower()
    if strategy == "beam":
        generation_kwargs.update({"num_beams": int(cfg_decoding.get("num_beams", 4)), "do_sample": False})
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
    debug_payload: Optional[Dict[str, object]] = None
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

        generated = model.generate(
            **tokenized,
            **generation_kwargs,
            return_dict_in_generate=True,
        )
        decoded_raw = tokenizer.batch_decode(
            generated.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        decoded = [_apply_question_postprocessing(text).strip() for text in decoded_raw]

        for meta, prompt_len, question, decoded_ids, raw_text, prompt_text in zip(
            batch_meta, prompt_lengths, decoded, generated.sequences, decoded_raw, batch_prompts
        ):
            regen_used = False
            regen_kwargs: Dict[str, object] | None = None
            regen_decoded_ids = decoded_ids
            regen_raw_text = raw_text

            if _needs_regeneration(question):
                regen_kwargs = {
                    **generation_kwargs,
                    "do_sample": True,
                    "top_p": float(cfg_decoding.get("top_p", 0.9)),
                    "temperature": float(cfg_decoding.get("temperature", 0.9)),
                }
                regen_kwargs.pop("num_beams", None)
                regen_inputs = tokenizer(
                    prompt_text,
                    padding=True,
                    truncation=True,
                    max_length=max_input_len,
                    return_tensors="pt",
                )
                regen_inputs = {k: v.to(resolved_device) for k, v in regen_inputs.items()}
                regen_generated = model.generate(
                    **regen_inputs,
                    **regen_kwargs,
                    return_dict_in_generate=True,
                )
                regen_raw_text = tokenizer.decode(
                    regen_generated.sequences[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                question = _apply_question_postprocessing(regen_raw_text).strip()
                regen_used = True
                regen_decoded_ids = regen_generated.sequences[0]

            wh_detected = _detect_wh_type(question, lang)
            gen_len = int((regen_decoded_ids != tokenizer.pad_token_id).sum().item())
            invalid = _needs_regeneration(question)

            outputs.append(
                {
                    "id": meta["id"],
                    "question": question,
                    "wh_type": wh_detected,
                    "prompt_len": int(prompt_len),
                    "gen_len": gen_len,
                    "invalid_generation": invalid,
                    "decoding": {"strategy": strategy, **generation_kwargs},
                }
            )

            if debug_payload is None:
                debug_payload = {
                    "prompt": prompt_text,
                    "strategy": strategy,
                    "generation_kwargs": generation_kwargs,
                    "raw_output": raw_text,
                    "normalized_output": question,
                    "regen_used": regen_used,
                    "regen_kwargs": regen_kwargs,
                    "regen_raw_output": regen_raw_text,
                    "gen_len": gen_len,
                }

    if debug:
        target_dir = Path(run_dir or cfg_task.get("run_dir") or ".")
        target_dir.mkdir(parents=True, exist_ok=True)
        debug_path = target_dir / "debug_gen.txt"
        with debug_path.open("w", encoding="utf-8") as f:
            if debug_payload:
                f.write(
                    "PROMPT:\n"
                    + debug_payload["prompt"]
                    + "\n\nDECODE PARAMS:\n"
                    + str(debug_payload["generation_kwargs"])
                    + f"\nstrategy={debug_payload['strategy']}"
                    + "\nREGEN USED: "
                    + str(debug_payload["regen_used"])
                    + "\nRAW OUTPUT:\n"
                    + debug_payload["raw_output"]
                    + "\nNORMALIZED OUTPUT:\n"
                    + debug_payload["normalized_output"]
                    + "\nREGEN RAW OUTPUT:\n"
                    + str(debug_payload["regen_raw_output"])
                    + "\nGEN LEN: "
                    + str(debug_payload["gen_len"])
                    + "\n"
                )
            else:
                f.write("No outputs generated.\n")

    return outputs


__all__ = ["generate_questions"]
