"""Run decoding ablations (beam vs top-p) with bilingual QG metrics."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
from pathlib import Path
from typing import Dict, List, Mapping, MutableSequence, Sequence

import yaml
from datasets import load_dataset
from evaluate import load as load_metric
from transformers import AutoTokenizer

from qg_bilingual.data import normalize_text
from qg_bilingual.eval.qg2qa import qg2qa_metrics
from qg_bilingual.generation import generate_questions

LOGGER = logging.getLogger(__name__)


def _load_config(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_records(path: Path) -> List[Mapping[str, object]]:
    dataset = load_dataset("json", data_files={"data": str(path)})["data"]
    return [dict(item) for item in dataset]


def _distinct_n(generated: Sequence[str], n: int) -> float:
    ngrams = []
    for text in generated:
        tokens = text.split()
        ngrams.extend([tuple(tokens[i : i + n]) for i in range(0, len(tokens) - n + 1)])
    unique = len(set(ngrams))
    total = len(ngrams)
    return unique / total if total else 0.0


def _question_stats(tokenizer, questions: Sequence[str]) -> Dict[str, float]:
    lengths = [len(tokenizer.encode(q, add_special_tokens=False)) for q in questions]
    if not lengths:
        return {"avg_q_len": 0.0, "median_q_len": 0.0}
    return {
        "avg_q_len": sum(lengths) / len(lengths),
        "median_q_len": statistics.median(lengths),
    }


def _detect_wh_match(results: Sequence[Mapping[str, object]], forced_type: str | None) -> float:
    if forced_type is None:
        return 1.0
    if not results:
        return 0.0
    forced_norm = str(forced_type).lower()
    matches = sum(
        1 for r in results if r.get("wh_type") is not None and r.get("wh_type") == forced_norm
    )
    return matches / len(results)


def _mark_invalid(question: str) -> bool:
    token_count = len(question.split())
    return token_count < 3 or "?" not in question


def run_ablation(config_path: Path, data_path: Path, output_csv: Path) -> None:
    cfg = _load_config(config_path)
    cfg_model = cfg.get("model", {})
    cfg_decoding = cfg.get("decoding", {})
    cfg_task = cfg.get("task", {})
    cfg_qg2qa = cfg.get("qg2qa", {})

    beam_grid = [
        {"strategy": "beam", "num_beams": beams, "length_penalty": lp}
        for beams in (1, 4, 6, 8)
        for lp in (1.0, 1.1)
    ]
    topp_grid = [
        {"strategy": "topp", "top_p": top_p, "temperature": temp}
        for top_p in (0.9, 0.95)
        for temp in (0.7, 0.9)
    ]

    shared_params = {
        "no_repeat_ngram_size": 3,
        "repetition_penalty": 1.1,
        "max_new_tokens": 32,
        "min_new_tokens": 4,
    }

    rouge = load_metric("rouge")
    bleu = load_metric("bleu")

    records = _load_records(data_path)
    tokenizer_name = cfg_model.get("tokenizer") or cfg_model.get("name")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name) if tokenizer_name else None

    rows: MutableSequence[Dict[str, object]] = []
    for decoding_cfg in [*beam_grid, *topp_grid]:
        combined_decoding = {**shared_params, **cfg_decoding, **decoding_cfg}
        LOGGER.info("Running decoding config: %s", combined_decoding)

        generations = generate_questions(records, cfg_model, combined_decoding, cfg_task)
        generated_questions = [normalize_text(item["question"]) for item in generations]
        references = [normalize_text(rec.get("question", "")) for rec in records]

        invalid_flags = [_mark_invalid(q) for q in generated_questions]
        cleaned_questions = [q if not bad else "" for q, bad in zip(generated_questions, invalid_flags)]

        rouge_scores = rouge.compute(predictions=cleaned_questions, references=references)
        bleu_scores = bleu.compute(predictions=[q.split() for q in cleaned_questions], references=[[r.split()] for r in references])

        qa_payload = [
            {
                "question": q,
                "context": rec.get("context", ""),
                "answer": rec.get("answer", ""),
            }
            for q, rec in zip(cleaned_questions, records)
        ]
        qa_scores = qg2qa_metrics(
            qa_payload,
            qa_ckpt_en=cfg_qg2qa.get("qa_ckpt_en", "distilbert-base-uncased-distilled-squad"),
            qa_ckpt_multi=cfg_qg2qa.get("qa_ckpt_multi", "deepset/xlm-roberta-large-squad2"),
            lang=cfg_task.get("lang", "en"),
            f1_thr=cfg_qg2qa.get("f1_thr", 0.8),
            conf_thr=cfg_qg2qa.get("conf_thr", 0.35),
            device=cfg_qg2qa.get("device", "auto"),
        )

        distinct1 = _distinct_n(cleaned_questions, 1)
        distinct2 = _distinct_n(cleaned_questions, 2)
        lengths = _question_stats(tokenizer, cleaned_questions) if tokenizer else {"avg_q_len": 0.0, "median_q_len": 0.0}

        forced_type = cfg_task.get("wh_type")
        wh_forced = forced_type is not None
        wh_match = _detect_wh_match(generations, forced_type)

        rows.append(
            {
                "model": cfg_model.get("name"),
                "mode": cfg_task.get("mode"),
                "lang": cfg_task.get("lang"),
                "wh_forced": wh_forced,
                "wh_match": wh_match,
                "decoding": combined_decoding.get("strategy"),
                "strategy_params": json.dumps(decoding_cfg),
                "rouge1": rouge_scores.get("rouge1", 0.0),
                "rouge2": rouge_scores.get("rouge2", 0.0),
                "rougeL": rouge_scores.get("rougeL", 0.0),
                "bleu": bleu_scores.get("bleu", 0.0),
                "em": qa_scores.get("em", 0.0),
                "f1": qa_scores.get("f1", 0.0),
                "pass_rate": qa_scores.get("qa_pass_rate", 0.0),
                "distinct1": distinct1,
                "distinct2": distinct2,
                "avg_q_len": lengths.get("avg_q_len", 0.0),
                "median_q_len": lengths.get("median_q_len", 0.0),
                "qa_model": qa_scores.get("qa_model"),
                "conf_thr": cfg_qg2qa.get("conf_thr", 0.35),
                "f1_thr": cfg_qg2qa.get("f1_thr", 0.8),
                "invalid_rate": sum(invalid_flags) / len(invalid_flags) if invalid_flags else 0.0,
            }
        )

    fieldnames = list(rows[0].keys()) if rows else []
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("Wrote ablation results to %s", output_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config with model/decoding/task")
    parser.add_argument("--data", type=Path, required=True, help="Path to validation/test JSONL")
    parser.add_argument("--output", type=Path, default=Path("eval/ablation_results.csv"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_ablation(args.config, args.data, args.output)
