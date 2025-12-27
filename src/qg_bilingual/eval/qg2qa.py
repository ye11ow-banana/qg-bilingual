"""Batch QG→QA scoring with language-aware normalization."""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

import torch
import yaml
from transformers import BatchEncoding

from .normalize import NormalizationConfig, Normalizer, exact_match, f1_score
from .qa_models import QAModelBundle, load_qa_model, resolve_device, truncate_context

LOGGER = logging.getLogger(__name__)


@dataclass
class Thresholds:
    f1_pass: float = 0.8
    conf_pass: float = 0.35


@dataclass
class IOConfig:
    input_jsonl: Optional[Path] = None
    out_dir: Optional[Path] = None


@dataclass
class QG2QARunConfig:
    lang: str = "en"
    qa_model: str = "distilbert-base-uncased-distilled-squad"
    device: str = "auto"
    batch_size: int = 16
    max_context_tokens: int = 512
    question_field: str = "question"
    gold_field: str = "gold_answer"
    thresholds: Thresholds = field(default_factory=Thresholds)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    io: IOConfig = field(default_factory=IOConfig)


@dataclass
class QAExample:
    id: str
    question: str
    context: str
    gold_answer: str
    unanswerable: bool = False
    lang: Optional[str] = None
    wh_type: Optional[str] = None


@dataclass
class Prediction:
    pred: str
    confidence: float
    used_no_answer: bool
    conf_type: str = "start_end_prob"


def _load_config(path: Path) -> QG2QARunConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    thresholds = raw.get("thresholds", {})
    normalization = raw.get("normalization", {})
    io_cfg = raw.get("io", {})
    return QG2QARunConfig(
        lang=str(raw.get("lang", "en")),
        qa_model=str(raw.get("qa_model", "distilbert-base-uncased-distilled-squad")),
        device=str(raw.get("device", "auto")),
        batch_size=int(raw.get("batch_size", 16)),
        max_context_tokens=int(raw.get("max_context_tokens", 512)),
        question_field=str(raw.get("question_field", "question")),
        gold_field=str(raw.get("gold_field", "gold_answer")),
        thresholds=Thresholds(
            f1_pass=float(thresholds.get("f1_pass", 0.8)),
            conf_pass=float(thresholds.get("conf_pass", 0.35)),
        ),
        normalization=NormalizationConfig(
            strip_punct=bool(normalization.get("strip_punct", True)),
            lower=bool(normalization.get("lower", True)),
            unify_quotes=bool(normalization.get("unify_quotes", True)),
            unify_apostrophe=bool(normalization.get("unify_apostrophe", True)),
        ),
        io=IOConfig(
            input_jsonl=Path(io_cfg["input_jsonl"]) if io_cfg.get("input_jsonl") else None,
            out_dir=Path(io_cfg["out_dir"]) if io_cfg.get("out_dir") else None,
        ),
    )


def _load_jsonl(path: Path) -> List[MutableMapping[str, object]]:
    rows: List[MutableMapping[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _chunked(seq: Sequence[QAExample], size: int) -> Iterable[Sequence[QAExample]]:
    for idx in range(0, len(seq), size):
        yield seq[idx : idx + size]


def _select_best_span(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    offsets: List[List[int]],
    sequence_ids: List[Optional[int]],
    context: str,
    include_unanswerable: bool,
) -> Prediction:
    # allow CLS index as a no-answer candidate when requested
    candidate_indices: List[int] = []
    cls_index = 0
    for token_idx, (seq_id, offset) in enumerate(zip(sequence_ids, offsets)):
        if seq_id == 1 and offset[0] >= 0 and offset[1] >= 0:
            candidate_indices.append(token_idx)

    if include_unanswerable:
        candidate_indices = [cls_index] + candidate_indices

    if not candidate_indices:
        return Prediction(pred="", confidence=0.0, used_no_answer=False)

    start_masked = torch.full_like(start_logits, -1e9)
    end_masked = torch.full_like(end_logits, -1e9)
    start_masked[candidate_indices] = start_logits[candidate_indices]
    end_masked[candidate_indices] = end_logits[candidate_indices]

    start_probs = torch.softmax(start_masked, dim=-1)
    end_probs = torch.softmax(end_masked, dim=-1)

    start_probs = start_probs[candidate_indices]
    end_probs = end_probs[candidate_indices]
    scores = torch.matmul(start_probs.unsqueeze(1), end_probs.unsqueeze(0))
    scores = torch.triu(scores)

    best = scores.argmax()
    start_idx = best // scores.size(1)
    end_idx = best % scores.size(1)
    best_start = candidate_indices[start_idx]
    best_end = candidate_indices[end_idx]
    confidence = scores[start_idx, end_idx].item()

    if include_unanswerable and best_start == cls_index and best_end == cls_index:
        return Prediction(pred="", confidence=confidence, used_no_answer=True)

    start_char, end_char = offsets[best_start]
    if start_char < 0 or end_char < 0:
        return Prediction(pred="", confidence=confidence, used_no_answer=False)

    return Prediction(pred=context[start_char:end_char], confidence=confidence, used_no_answer=False)


def _question_length(question: str) -> int:
    return len([tok for tok in question.split() if tok])


def _bucket_for_length(length: int) -> str:
    if length <= 8:
        return "q<=8"
    if length <= 16:
        return "8<q<=16"
    return ">16"


def _detect_wh_type(question: str, fallback: Optional[str] = None) -> Optional[str]:
    if fallback:
        return fallback
    cleaned = question.strip().lower()
    if not cleaned:
        return None
    first_token = cleaned.split()[0].strip("?.,! ")
    en_map = {"who", "what", "where", "when", "why", "how"}
    ua_map = {"хто", "що", "де", "коли", "чому", "як"}
    if first_token in en_map or first_token in ua_map:
        return first_token
    return None


def _max_sequence_length(bundle: QAModelBundle, config: QG2QARunConfig) -> int:
    special_tokens = bundle.tokenizer.num_special_tokens_to_add(pair=True)
    desired = config.max_context_tokens + bundle.question_max_tokens + special_tokens
    return min(bundle.tokenizer.model_max_length, desired)


def _prepare_examples(
    raw_rows: Sequence[MutableMapping[str, object]],
    bundle: QAModelBundle,
    config: QG2QARunConfig,
) -> Tuple[List[QAExample], Dict[str, int]]:
    valid: List[QAExample] = []
    counts = {"total": len(raw_rows), "invalid": 0, "lang_mismatch": 0, "unanswerable": 0}
    for idx, row in enumerate(raw_rows):
        question = str(row.get(config.question_field, row.get("question", ""))).strip()
        context = str(row.get("context", "")).strip()
        gold = str(row.get(config.gold_field, row.get("answer", "")))
        unanswerable = bool(row.get("unanswerable", False))
        lang = str(row.get("lang", config.lang)).lower()
        wh_type = row.get("wh_type")

        if lang and lang != config.lang.lower():
            counts["lang_mismatch"] += 1
            LOGGER.warning("Skipping id=%s due to lang mismatch (%s != %s)", row.get("id", idx), lang, config.lang)
            continue

        if not question or not context:
            counts["invalid"] += 1
            continue

        if not any(char.isalpha() for char in question):
            counts["invalid"] += 1
            continue

        if not unanswerable and not gold:
            counts["invalid"] += 1
            continue

        question_ids = bundle.tokenizer.encode(
            question,
            add_special_tokens=False,
            truncation=True,
            max_length=bundle.question_max_tokens,
        )
        if len(question_ids) > bundle.question_max_tokens:
            LOGGER.warning("Question too long; truncating id=%s to %s tokens", row.get("id", idx), bundle.question_max_tokens)
        question = bundle.tokenizer.decode(question_ids, skip_special_tokens=True)

        truncated_context = truncate_context(bundle.tokenizer, context, config.max_context_tokens)

        counts["unanswerable"] += int(unanswerable)
        valid.append(
            QAExample(
                id=str(row.get("id", idx)),
                question=question,
                context=truncated_context,
                gold_answer=gold,
                unanswerable=unanswerable,
                lang=lang,
                wh_type=str(wh_type) if wh_type else None,
            )
        )

    counts["valid"] = len(valid)
    return valid, counts


def _predict_batch(
    bundle: QAModelBundle,
    examples: Sequence[QAExample],
    config: QG2QARunConfig,
    include_unanswerable: bool,
) -> List[Prediction]:
    max_length = _max_sequence_length(bundle, config)
    encoding: BatchEncoding = bundle.tokenizer(
        [ex.question for ex in examples],
        [ex.context for ex in examples],
        return_tensors="pt",
        padding=True,
        truncation="only_second",
        max_length=max_length,
        return_offsets_mapping=True,
    )

    input_tensors = {k: v.to(bundle.device) for k, v in encoding.items() if k != "offset_mapping"}
    with torch.no_grad():
        outputs = bundle.model(**input_tensors)

    predictions: List[Prediction] = []
    offset_mapping = encoding["offset_mapping"].tolist()
    for i, example in enumerate(examples):
        sequence_ids = encoding.encodings[i].sequence_ids()
        start_logits = outputs.start_logits[i]
        end_logits = outputs.end_logits[i]
        offsets = offset_mapping[i]
        pred = _select_best_span(start_logits, end_logits, offsets, sequence_ids, example.context, include_unanswerable)
        predictions.append(pred)
    return predictions


def _build_detail_row(
    example: QAExample,
    prediction: Prediction,
    normalizer: Normalizer,
    thresholds: Thresholds,
    include_unanswerable: bool,
) -> Dict[str, object]:
    include_metrics = include_unanswerable or not example.unanswerable
    if example.unanswerable and not include_unanswerable:
        em = f1 = 0.0
        passed = False
    elif example.unanswerable:
        if not prediction.pred.strip():
            em = f1 = 1.0 if prediction.confidence >= thresholds.conf_pass else 0.0
        else:
            em = f1 = 0.0
        passed = bool(f1 >= thresholds.f1_pass and prediction.confidence >= thresholds.conf_pass)
    else:
        em = exact_match(prediction.pred, example.gold_answer, normalizer)
        f1 = f1_score(prediction.pred, example.gold_answer, normalizer)
        passed = bool(f1 >= thresholds.f1_pass and prediction.confidence >= thresholds.conf_pass)

    wh_type = _detect_wh_type(example.question, example.wh_type)
    detail = {
        "id": example.id,
        "question": example.question,
        "pred": prediction.pred,
        "gold": example.gold_answer,
        "em": em,
        "f1": f1,
        "conf": prediction.confidence,
        "passed": passed,
        "unanswerable": example.unanswerable,
        "include_in_metrics": include_metrics,
        "lang": example.lang,
        "conf_type": prediction.conf_type,
    }
    if wh_type:
        detail["wh_type"] = wh_type
    detail["q_len_tokens"] = _question_length(example.question)
    return detail


def _aggregate(details: Sequence[Dict[str, object]]) -> Dict[str, object]:
    eligible = [row for row in details if row.get("include_in_metrics")]
    if eligible:
        em = mean(row.get("em", 0.0) for row in eligible)
        f1 = mean(row.get("f1", 0.0) for row in eligible)
        pass_rate = mean(float(row.get("passed", False)) for row in eligible)
    else:
        em = f1 = pass_rate = 0.0
    included = len(eligible)

    buckets: Dict[str, List[Dict[str, object]]] = {"q<=8": [], "8<q<=16": [], ">16": []}
    for row in eligible:
        bucket = _bucket_for_length(int(row.get("q_len_tokens", 0)))
        buckets[bucket].append(row)

    bucket_metrics = {
        name: {
            "count": len(rows),
            "em": mean(r.get("em", 0.0) for r in rows) if rows else 0.0,
            "f1": mean(r.get("f1", 0.0) for r in rows) if rows else 0.0,
            "pass_rate": mean(float(r.get("passed", False)) for r in rows) if rows else 0.0,
        }
        for name, rows in buckets.items()
    }

    wh_groups: Dict[str, List[Dict[str, object]]] = {}
    for row in eligible:
        wh = row.get("wh_type")
        if not wh:
            continue
        wh_groups.setdefault(str(wh), []).append(row)

    wh_metrics = {
        wh: {
            "count": len(rows),
            "em": mean(r.get("em", 0.0) for r in rows) if rows else 0.0,
            "f1": mean(r.get("f1", 0.0) for r in rows) if rows else 0.0,
            "pass_rate": mean(float(r.get("passed", False)) for r in rows) if rows else 0.0,
        }
        for wh, rows in wh_groups.items()
    }

    f1_hist_bins = [round(x * 0.1, 2) for x in range(0, 11)]
    f1_hist_counts = [0] * (len(f1_hist_bins) - 1)
    for row in eligible:
        value = float(row.get("f1", 0.0))
        idx = min(len(f1_hist_counts) - 1, int(math.floor(value * 10)))
        f1_hist_counts[idx] += 1

    return {
        "em": em,
        "f1": f1,
        "pass_rate": pass_rate,
        "by_len_bucket": bucket_metrics,
        "by_wh_type": wh_metrics,
        "f1_histogram": {"bins": f1_hist_bins, "counts": f1_hist_counts},
    }


def evaluate_examples(
    examples: Sequence[QAExample],
    bundle: QAModelBundle,
    config: QG2QARunConfig,
    include_unanswerable: bool,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    normalizer = Normalizer(config.normalization, config.lang)
    details: List[Dict[str, object]] = []
    for batch in _chunked(list(examples), config.batch_size):
        preds = _predict_batch(bundle, batch, config, include_unanswerable)
        for example, pred in zip(batch, preds):
            details.append(_build_detail_row(example, pred, normalizer, config.thresholds, include_unanswerable))

    summary = _aggregate(details)
    summary["conf_type"] = "start_end_prob"
    summary["included"] = included
    return summary, details


def run_qg2qa(config: QG2QARunConfig, include_unanswerable: bool) -> Tuple[Dict[str, object], List[Dict[str, object]], Dict[str, int]]:
    if not config.io.input_jsonl:
        raise ValueError("input_jsonl is required in the config or via --input")
    raw_rows = _load_jsonl(config.io.input_jsonl)
    bundle = load_qa_model(config.qa_model, config.device)
    LOGGER.info("Loaded QA model %s on %s", config.qa_model, bundle.device_label)

    examples, counts = _prepare_examples(raw_rows, bundle, config)
    summary, details = evaluate_examples(examples, bundle, config, include_unanswerable)

    skipped_unanswerable = counts.get("unanswerable", 0) if not include_unanswerable else 0
    summary_payload = {
        "em": summary.get("em", 0.0),
        "f1": summary.get("f1", 0.0),
        "qa_pass_rate": summary.get("pass_rate", 0.0),
        "qa_model": config.qa_model,
        "lang": config.lang,
        "f1_thr": config.thresholds.f1_pass,
        "conf_thr": config.thresholds.conf_pass,
        "included": summary.get("included", 0),
        "skipped_unanswerable": skipped_unanswerable,
        "counts": counts,
        "metrics": summary,
        "qa_device": bundle.device_label,
        "include_unanswerable": include_unanswerable,
    }
    return summary_payload, details, counts


def save_outputs(summary: Dict[str, object], details: List[Dict[str, object]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "qg2qa_val.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    details_path = out_dir / "qg2qa_details.jsonl"
    with details_path.open("w", encoding="utf-8") as f:
        for row in details:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QG→QA scoring over generated questions")
    parser.add_argument("--config", type=Path, required=True, help="Path to qg2qa_*.yaml config")
    parser.add_argument("--input", type=Path, help="Override input_jsonl from config")
    parser.add_argument("--out", type=Path, help="Override out_dir from config")
    parser.add_argument(
        "--question-field",
        type=str,
        default="question",
        help="Name of the JSONL field containing the generated question",
    )
    parser.add_argument(
        "--gold-field",
        type=str,
        default="gold_answer",
        help="Name of the JSONL field containing the gold answer",
    )
    parser.add_argument(
        "--include-unanswerable",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Evaluate unanswerable rows as no-answer spans",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    config = _load_config(args.config)
    if args.input:
        config.io.input_jsonl = args.input
    if args.out:
        config.io.out_dir = args.out
    if args.question_field:
        config.question_field = args.question_field
    if args.gold_field:
        config.gold_field = args.gold_field

    if not config.io.out_dir:
        raise ValueError("--out or io.out_dir in config is required")

    summary, details, counts = run_qg2qa(config, args.include_unanswerable)
    LOGGER.info(
        "Finished QG→QA: lang=%s, total=%s, valid=%s, invalid=%s", config.lang, counts.get("total", 0), counts.get("valid", 0), counts.get("invalid", 0)
    )
    save_outputs(summary, details, config.io.out_dir)


def _select_checkpoint(lang: str, qa_ckpt_en: str, qa_ckpt_multi: str) -> str:
    normalized = lang.lower()
    if normalized in {"en", "eng", "english"}:
        return qa_ckpt_en
    return qa_ckpt_multi


def qg2qa_metrics(
    val_records: Sequence[object],
    qa_ckpt_en: str = "distilbert-base-uncased-distilled-squad",
    qa_ckpt_multi: str = "deepset/xlm-roberta-large-squad2",
    lang: str = "en",
    f1_thr: float = 0.8,
    conf_thr: float = 0.35,
    *,
    batch_size: int = 16,
    device: Optional[object] = None,
) -> dict:
    """Backward-compatible entrypoint for training-time eval."""

    qa_model = _select_checkpoint(lang, qa_ckpt_en, qa_ckpt_multi)
    resolved_device, device_label = resolve_device(device or "auto")
    cfg = QG2QARunConfig(
        lang=lang,
        qa_model=qa_model,
        device=device_label,
        batch_size=batch_size,
        thresholds=Thresholds(f1_pass=f1_thr, conf_pass=conf_thr),
    )
    bundle = load_qa_model(qa_model, resolved_device)

    raw_rows: List[MutableMapping[str, object]] = []
    for idx, record in enumerate(val_records):
        raw_rows.append(
            {
                "id": getattr(record, "id", idx),
                "question": getattr(record, "question", getattr(record, "generated_question", "")),
                "context": getattr(record, "context", ""),
                "gold_answer": getattr(record, "gold_answer", getattr(record, "answer", "")),
                "unanswerable": getattr(record, "unanswerable", False),
                "lang": getattr(record, "lang", lang),
            }
        )

    examples, counts = _prepare_examples(raw_rows, bundle, cfg)
    summary, _ = evaluate_examples(examples, bundle, cfg, include_unanswerable=False)
    return {
        "em": summary.get("em", 0.0),
        "f1": summary.get("f1", 0.0),
        "qa_pass_rate": summary.get("pass_rate", 0.0),
        "qa_model": qa_model,
        "lang": lang,
        "qa_device": device_label,
        "f1_thr": f1_thr,
        "conf_thr": conf_thr,
        "included": summary.get("included", 0),
        "skipped_unanswerable": 0,
    }


if __name__ == "__main__":
    main()
