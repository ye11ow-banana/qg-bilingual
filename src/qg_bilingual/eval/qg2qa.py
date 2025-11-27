"""Utility to score QG outputs via a QA model (QG→QA loop)."""

from __future__ import annotations

import re
from collections import Counter
from string import punctuation
from typing import Mapping, MutableSequence, Optional, Sequence, Tuple

from evaluate import load as load_metric
from transformers import pipeline


def _get_value(record: object, key: str) -> str:
    if isinstance(record, Mapping):
        return str(record.get(key, ""))
    return str(getattr(record, key, ""))


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(f"[{punctuation}]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def _f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_text(prediction).split()
    gold_tokens = _normalize_text(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    overlap = pred_counts & gold_counts
    num_same = sum(overlap.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _find_answer_start(context: str, answer: str) -> Optional[int]:
    """
    Approximate the character start position of ``answer`` inside ``context``.

    SQuAD-style EM/F1 does not rely on ``answer_start``, but populating it helps
    debugging and keeps references well-formed. Returns ``None`` if not found.
    """

    if not answer:
        return None

    lowered_context = context.lower()
    lowered_answer = answer.lower()
    position = lowered_context.find(lowered_answer)
    return position if position >= 0 else None


def qg2qa_metrics(
    val_records: Sequence[object],
    qa_ckpt: str = "distilbert-base-uncased-distilled-squad",
    f1_thr: float = 0.8,
    conf_thr: float = 0.35,
    *,
    batch_size: int = 16,
    device: Optional[int] = None,
) -> dict:
    """Compute EM/F1 by answering generated questions with a QA model."""

    qa_model = pipeline(
        "question-answering", model=qa_ckpt, tokenizer=qa_ckpt, device=device
    )
    qa_metric = load_metric("squad")

    predictions: MutableSequence[dict] = []
    references: MutableSequence[dict] = []
    f1_scores: MutableSequence[float] = []
    confidences: MutableSequence[float] = []

    batched_inputs: Sequence[Tuple[int, dict]] = [
        (
            idx,
            {
                "question": _get_value(record, "question"),
                "context": _get_value(record, "context"),
                "answer": _get_value(record, "answer"),
            },
        )
        for idx, record in enumerate(val_records)
    ]

    results = qa_model(
        [
            {"question": row[1]["question"], "context": row[1]["context"]}
            for row in batched_inputs
        ],
        batch_size=batch_size,
    )

    if isinstance(results, Mapping):
        results = [results]

    for (idx, payload), result in zip(batched_inputs, results):
        question = payload["question"]
        context = payload["context"]
        answer = payload["answer"]

        pred_text = str(result.get("answer", ""))
        confidence = float(result.get("score", 0.0))
        answer_start = _find_answer_start(context, answer)

        predictions.append({"id": str(idx), "prediction_text": pred_text})
        references.append(
            {
                "id": str(idx),
                "answers": {
                    "text": [answer],
                    # SQuAD EM/F1 ignores start, but we keep a best-effort position
                    # for completeness and debugging.
                    "answer_start": [answer_start if answer_start is not None else 0],
                },
            }
        )
        f1_scores.append(_f1_score(pred_text, answer))
        confidences.append(confidence)

    aggregated = qa_metric.compute(predictions=predictions, references=references)

    pass_count = sum(
        1 for f1, conf in zip(f1_scores, confidences) if f1 >= f1_thr and conf >= conf_thr
    )
    pass_rate = pass_count / len(f1_scores) if f1_scores else 0.0

    return {
        "em": aggregated.get("exact_match", 0.0),
        "f1": aggregated.get("f1", 0.0),
        "qa_pass_rate": pass_rate,
        "qa_f1_distribution": list(f1_scores),
        "qa_confidence_distribution": list(confidences),
    }
