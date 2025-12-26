"""Build a stratified MOS batch from multiple JSONL inputs.

Each input line is expected to be a JSON object with at least the fields:
```
{id, lang, model, mode, context, question, reference?, wh_type?, passed_filters?}
```
The script tries to preserve diversity across language, mode, model, wh_type,
and post-filter pass/fail buckets.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple


BucketKey = Tuple[str, str, str, str, str]


def _load_jsonl(path: Path) -> List[MutableMapping[str, Any]]:
    records: List[MutableMapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _normalize_pass_flag(row: Mapping[str, Any]) -> str:
    for key in ("passed_filters", "pass_filters", "passed", "qa_passed", "pass_rate_flag"):
        if key in row:
            return "passed" if bool(row[key]) else "failed"
    return "unknown"


def _bucket(row: Mapping[str, Any]) -> BucketKey:
    lang = str(row.get("lang", "unk")).lower() or "unk"
    model = str(row.get("model", "model?"))
    mode = str(row.get("mode", "mode?")).lower()
    wh = str(row.get("wh_type", "wh?")).lower() or "wh?"
    passed = _normalize_pass_flag(row)
    return (lang, mode, model, wh, passed)


def _sample_sizes(bucket_counts: Counter, target_size: int) -> Dict[BucketKey, int]:
    total = sum(bucket_counts.values())
    if total == 0:
        return {}

    # initial proportional allocation with minimum 1 per bucket if possible
    allocations: Dict[BucketKey, int] = {}
    for key, count in bucket_counts.items():
        proposed = max(1, round(target_size * count / total))
        allocations[key] = min(proposed, count)

    current = sum(allocations.values())
    # adjust to match target_size by trimming or expanding
    if current > target_size:
        # remove from largest buckets first
        for key, _ in sorted(allocations.items(), key=lambda kv: kv[1], reverse=True):
            if current <= target_size:
                break
            if allocations[key] > 1:
                allocations[key] -= 1
                current -= 1
    elif current < target_size:
        # add where capacity remains
        for key, count in sorted(bucket_counts.items(), key=lambda kv: kv[1], reverse=True):
            while current < target_size and allocations[key] < count:
                allocations[key] += 1
                current += 1
                if current == target_size:
                    break
    return allocations


def build_batch(records: Sequence[MutableMapping[str, Any]], size: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    bucketed: Dict[BucketKey, List[MutableMapping[str, Any]]] = defaultdict(list)
    for row in records:
        bucketed[_bucket(row)].append(row)

    bucket_counts = Counter({k: len(v) for k, v in bucketed.items()})
    allocations = _sample_sizes(bucket_counts, min(size, sum(bucket_counts.values())))

    sampled: List[Dict[str, Any]] = []
    for key, rows in bucketed.items():
        take = allocations.get(key, 0)
        if take <= 0:
            continue
        rng.shuffle(rows)
        for row in rows[:take]:
            sampled.append(
                {
                    "id": str(row.get("id")),
                    "lang": str(row.get("lang", "")),
                    "model": str(row.get("model", "")),
                    "mode": str(row.get("mode", "")),
                    "context": row.get("context", ""),
                    "question": row.get("question", ""),
                    "reference": row.get("reference") or row.get("answer") or row.get("gold_answer", ""),
                    "wh_type": row.get("wh_type"),
                    "passed_filters": _normalize_pass_flag(row),
                }
            )
    rng.shuffle(sampled)
    return sampled[:size]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MOS batch with stratification.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True, help="Input JSONL files")
    parser.add_argument("--size", type=int, default=300, help="Target batch size (default: 300)")
    parser.add_argument("--seed", type=int, default=13, help="Random seed")
    parser.add_argument("--out", type=Path, default=Path("eval/mos/mos_batch.jsonl"), help="Output JSONL path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: List[MutableMapping[str, Any]] = []
    for path in args.inputs:
        if not path.exists():
            raise FileNotFoundError(path)
        rows.extend(_load_jsonl(path))

    batch = build_batch(rows, size=args.size, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in batch:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(batch)} examples to {args.out}")


if __name__ == "__main__":
    main()
