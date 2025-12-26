"""Aggregate MOS annotations and compute agreement statistics."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import scipy.stats as stats


@dataclass
class ExampleEntry:
    id: str
    meta: Mapping[str, Any]
    scores: List[int]
    flags: List[str]


BOOTSTRAP_SAMPLES = 5000


def bootstrap_ci(values: Sequence[float], B: int = BOOTSTRAP_SAMPLES, alpha: float = 0.05) -> Tuple[float, float]:
    if not values:
        return (math.nan, math.nan)
    arr = np.array(values)
    rng = np.random.default_rng(13)
    samples = []
    for _ in range(B):
        resample = rng.choice(arr, size=len(arr), replace=True)
        samples.append(resample.mean())
    lower = float(np.quantile(samples, alpha / 2))
    upper = float(np.quantile(samples, 1 - alpha / 2))
    return (lower, upper)


def cohen_kappa(rater1: Sequence[int], rater2: Sequence[int]) -> float:
    if len(rater1) != len(rater2) or not rater1:
        return math.nan
    labels = sorted(set(rater1) | set(rater2))
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=float)
    for a, b in zip(rater1, rater2):
        matrix[label_to_idx[a], label_to_idx[b]] += 1
    total = matrix.sum()
    if total == 0:
        return math.nan
    p0 = np.trace(matrix) / total
    p1 = (matrix.sum(axis=0) * matrix.sum(axis=1)).sum() / (total ** 2)
    if p1 == 1:
        return math.nan
    return (p0 - p1) / (1 - p1)


def cronbach_alpha(matrix_raters_items: np.ndarray) -> float:
    # matrix: shape (n_raters, n_items)
    if matrix_raters_items.size == 0:
        return math.nan
    k, n = matrix_raters_items.shape
    if k < 2 or n < 2:
        return math.nan
    item_variances = matrix_raters_items.var(axis=0, ddof=1)
    total_scores = matrix_raters_items.sum(axis=0)
    total_variance = total_scores.var(ddof=1)
    if total_variance == 0:
        return math.nan
    alpha = (k / (k - 1)) * (1 - item_variances.sum() / total_variance)
    return float(alpha)


def _load_jsonl(path: Path) -> List[MutableMapping[str, Any]]:
    rows: List[MutableMapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _collect_examples(batch_rows: Sequence[MutableMapping[str, Any]], ann_rows: Sequence[Sequence[MutableMapping[str, Any]]]) -> List[ExampleEntry]:
    meta_by_id: Dict[str, Mapping[str, Any]] = {str(row["id"]): row for row in batch_rows if "id" in row}
    scores_by_id: Dict[str, List[int]] = defaultdict(list)
    flags_by_id: Dict[str, List[str]] = defaultdict(list)

    for annotator_rows in ann_rows:
        for row in annotator_rows:
            if "id" not in row or "mos" not in row:
                continue
            ex_id = str(row["id"])
            scores_by_id[ex_id].append(int(row["mos"]))
            flags_by_id[ex_id].extend(row.get("flags", []))

    entries: List[ExampleEntry] = []
    for ex_id, scores in scores_by_id.items():
        if ex_id not in meta_by_id:
            continue
        entries.append(ExampleEntry(id=ex_id, meta=meta_by_id[ex_id], scores=scores, flags=flags_by_id.get(ex_id, [])))
    return entries


def _per_bucket(entries: Sequence[ExampleEntry], field: str) -> Dict[str, Dict[str, Any]]:
    buckets: Dict[str, Dict[str, Any]] = {}
    grouped: Dict[str, List[ExampleEntry]] = defaultdict(list)
    for entry in entries:
        key = str(entry.meta.get(field, "unknown"))
        grouped[key].append(entry)

    for key, ex_list in grouped.items():
        mos_vals = [mean(e.scores) for e in ex_list if e.scores]
        avg = float(mean(mos_vals)) if mos_vals else math.nan
        ci = bootstrap_ci(mos_vals) if mos_vals else (math.nan, math.nan)
        buckets[key] = {
            "count": len(ex_list),
            "mean_mos": avg,
            "ci95": {"low": ci[0], "high": ci[1]},
        }
    return buckets


def _flag_rates(entries: Sequence[ExampleEntry]) -> Dict[str, float]:
    flags = [flag for e in entries for flag in e.flags]
    if not entries:
        return {}
    denom = max(len(entries), 1)
    counts = Counter(flags)
    return {k: v / denom for k, v in counts.items()}


def _agreement(entries: Sequence[ExampleEntry], annotator_rows: Sequence[Sequence[MutableMapping[str, Any]]]) -> Dict[str, Any]:
    # Build aligned matrices
    annotators = len(annotator_rows)
    if annotators < 2:
        return {"num_annotators": annotators}

    # intersection ids present across annotators
    id_sets = [set(str(row["id"]) for row in rows if "id" in row and "mos" in row) for rows in annotator_rows]
    common_ids = set.intersection(*id_sets)
    matrices: List[List[int]] = [[] for _ in range(annotators)]
    row_maps = [{str(r["id"]): r for r in rows if "id" in r} for rows in annotator_rows]
    for ex_id in sorted(common_ids):
        current: List[int] = []
        missing = False
        for row_map in row_maps:
            row = row_map.get(ex_id)
            if not row or "mos" not in row:
                missing = True
                break
            current.append(int(row["mos"]))
        if missing:
            continue
        for idx, score in enumerate(current):
            matrices[idx].append(score)
    if not common_ids:
        return {"num_annotators": annotators}

    if annotators == 2:
        kappa = cohen_kappa(matrices[0], matrices[1])
        return {"num_annotators": annotators, "cohen_kappa": kappa}

    matrix = np.array(matrices)
    alpha = cronbach_alpha(matrix)
    # pairwise average kappa for additional insight
    pairwise = []
    for i in range(annotators):
        for j in range(i + 1, annotators):
            pairwise.append(cohen_kappa(matrices[i], matrices[j]))
    pairwise_kappa = float(np.nanmean(pairwise)) if pairwise else math.nan
    return {"num_annotators": annotators, "cronbach_alpha": alpha, "pairwise_cohen_kappa": pairwise_kappa}


def _correlations(entries: Sequence[ExampleEntry]) -> Dict[str, Dict[str, float]]:
    metrics = {"rougeL", "em", "f1", "pass_rate"}
    results: Dict[str, Dict[str, float]] = {}
    for metric in metrics:
        pairs = [(mean(e.scores), float(e.meta.get(metric))) for e in entries if metric in e.meta and e.meta.get(metric) is not None]
        if len(pairs) < 5:
            continue
        mos_vals, metric_vals = zip(*pairs)
        pearson = stats.pearsonr(mos_vals, metric_vals)[0]
        spearman = stats.spearmanr(mos_vals, metric_vals)[0]
        results[metric] = {"pearson": float(pearson), "spearman": float(spearman), "count": len(pairs)}
    return results


def aggregate(batch_path: Path, ann_paths: Sequence[Path]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    batch_rows = _load_jsonl(batch_path)
    ann_rows = [_load_jsonl(p) for p in ann_paths]

    entries = _collect_examples(batch_rows, ann_rows)
    example_means = [mean(e.scores) for e in entries if e.scores]
    overall_mean = float(mean(example_means)) if example_means else math.nan
    ci_low, ci_high = bootstrap_ci(example_means) if example_means else (math.nan, math.nan)

    summary = {
        "num_examples": len(entries),
        "mean_mos": overall_mean,
        "ci95": {"low": ci_low, "high": ci_high},
        "agreement": _agreement(entries, ann_rows),
        "flag_rates": _flag_rates(entries),
        "correlations": _correlations(entries),
    }

    buckets = {
        "by_lang": _per_bucket(entries, "lang"),
        "by_mode": _per_bucket(entries, "mode"),
        "by_model": _per_bucket(entries, "model"),
        "by_wh_type": _per_bucket(entries, "wh_type"),
        "by_passed": _per_bucket(entries, "passed_filters"),
    }
    return summary, buckets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate MOS annotations")
    parser.add_argument("--batch", type=Path, required=True, help="Path to mos_batch.jsonl")
    parser.add_argument("--ann", nargs="+", type=Path, required=True, help="Annotation JSONL files")
    parser.add_argument("--out-dir", type=Path, default=Path("eval/mos"), help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, buckets = aggregate(args.batch, args.ann)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with (args.out_dir / "mos_agg.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with (args.out_dir / "mos_by_bucket.json").open("w", encoding="utf-8") as f:
        json.dump(buckets, f, ensure_ascii=False, indent=2)

    print(f"Saved aggregate stats to {args.out_dir}")


if __name__ == "__main__":
    main()
