import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from datasets import load_dataset


QUOTE_MAP = {
    "“": '"',
    "”": '"',
    "«": '"',
    "»": '"',
    "\u201d": '"',
}
APOSTROPHE_MAP = {
    "'": "’",
    "`": "’",
    "´": "’",
    "’": "’",
    "\u2019": "’",
}


def normalize_text(text: str) -> str:
    """Normalize text with NFKC, unified quotes/apostrophes, and clean spacing."""
    text = unicodedata.normalize("NFKC", text.replace("\u00a0", " "))
    for src, dst in QUOTE_MAP.items():
        text = text.replace(src, dst)
    for src, dst in APOSTROPHE_MAP.items():
        text = text.replace(src, dst)

    # tighten spaces around punctuation and slashes
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*([,.;:!?])\s*", r"\1 ", text)
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def align_answer(context: str, answer: str) -> Optional[str]:
    if not answer:
        return ""

    context_norm = normalize_text(context)
    answer_norm = normalize_text(answer)
    if answer_norm in context_norm:
        return answer_norm

    candidates = [
        answer_norm.strip(),
        answer_norm.strip().strip(".?!,;:"),
        answer_norm.replace("’", "'"),
        answer_norm.replace("'", "’"),
    ]
    for cand in candidates:
        cand_norm = normalize_text(cand)
        if cand_norm in context_norm:
            return cand_norm
    return None


def length_ok(text: str, min_len: int, max_len: int) -> bool:
    tokens = text.split()
    return min_len <= len(tokens) <= max_len


def compute_stats(records: Dict[str, List[Dict]]) -> Dict:
    def lengths(items: Iterable[str]) -> List[int]:
        return [len(x.split()) for x in items]

    stats = {
        "counts": {},
        "avg_len": {},
        "median_len": {},
        "unanswerable_share": {},
    }

    for split, rows in records.items():
        stats["counts"][split] = len(rows)
        ctx_lens = lengths(r["context"] for r in rows)
        q_lens = lengths(r["question"] for r in rows)
        ans_lens = lengths(r["answer"] for r in rows)
        for label, lens in [("context", ctx_lens), ("question", q_lens), ("answer", ans_lens)]:
            if lens:
                stats["avg_len"].setdefault(label, {})[split] = sum(lens) / len(lens)
                sorted_lens = sorted(lens)
                mid = len(sorted_lens) // 2
                if len(sorted_lens) % 2 == 0:
                    median_val = (sorted_lens[mid - 1] + sorted_lens[mid]) / 2
                else:
                    median_val = sorted_lens[mid]
                stats["median_len"].setdefault(label, {})[split] = median_val
            else:
                stats["avg_len"].setdefault(label, {})[split] = 0
                stats["median_len"].setdefault(label, {})[split] = 0
        if rows:
            unans = sum(1 for r in rows if r.get("unanswerable", False))
            stats["unanswerable_share"][split] = unans / len(rows)
        else:
            stats["unanswerable_share"][split] = 0.0
    return stats


def save_jsonl(path: Path, rows: Iterable[Dict]):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stratified_group_split(
    groups: Dict[str, List[Dict]], seed: int, train_frac: float, val_frac: float, test_frac: float
) -> Dict[str, List[Dict]]:
    group_keys = list(groups.keys())
    rnd = random.Random(seed)
    rnd.shuffle(group_keys)

    total_groups = len(group_keys)
    if total_groups == 0:
        return {"train": [], "val": [], "test": []}

    # Allocate per-group to preserve paragraph integrity and avoid empty splits
    if train_frac + val_frac + test_frac <= 0:
        train_frac, val_frac, test_frac = 0.8, 0.1, 0.1

    total = train_frac + val_frac + test_frac
    train_frac, val_frac, test_frac = (train_frac / total, val_frac / total, test_frac / total)

    train_groups = max(1, int(round(total_groups * train_frac))) if total_groups >= 2 else total_groups
    val_groups = max(1, int(round(total_groups * val_frac))) if total_groups >= 3 else (1 if total_groups == 2 and val_frac > 0 else 0)
    if train_groups + val_groups > total_groups:
        val_groups = max(0, total_groups - train_groups)
    test_groups = max(0, total_groups - train_groups - val_groups)
    if test_groups == 0 and total_groups > train_groups + val_groups:
        test_groups = total_groups - train_groups - val_groups

    splits = {"train": [], "val": [], "test": []}
    for idx, key in enumerate(group_keys):
        if idx < train_groups:
            split = "train"
        elif idx < train_groups + val_groups:
            split = "val"
        else:
            split = "test"
        splits[split].extend(groups[key])
    return splits


def load_squad_rows(args) -> List[Dict]:
    try:
        dataset = load_dataset("rajpurkar/squad_v2")
        return list(dataset["train"]) + list(dataset["validation"])
    except Exception as exc:  # pragma: no cover - defensive fallback
        fallback = args.local_fallback
        if fallback and fallback.exists():
            print(
                f"Falling back to local dataset at {fallback} because loading squad_v2 failed: {exc}",
                file=sys.stderr,
            )
            with fallback.open("r", encoding="utf-8") as f:
                return json.load(f)
        raise


def process_dataset(args) -> Tuple[Dict[str, List[Dict]], Counter, List[Dict]]:
    all_rows = load_squad_rows(args)

    dedup = set()
    dropped = Counter()
    drop_records: List[Dict] = []
    grouped: Dict[str, List[Dict]] = defaultdict(list)

    for raw in all_rows:
        context = normalize_text(raw["context"])
        question = normalize_text(raw["question"])
        title = raw.get("title") or ""
        unanswerable = bool(raw.get("is_impossible", False))

        if unanswerable and args.drop_unanswerable:
            dropped["unanswerable_dropped"] += 1
            drop_records.append({"reason": "unanswerable_dropped", "title": title})
            continue

        answers = raw.get("answers", {}).get("text") or []
        answer_text = normalize_text(answers[0]) if answers else ""
        if unanswerable:
            answer_text = ""

        if not length_ok(context, args.min_context, args.max_context):
            dropped["context_length"] += 1
            drop_records.append({"reason": "len_filter", "field": "context", "title": title})
            continue
        if not length_ok(question, args.min_question, args.max_question):
            dropped["question_length"] += 1
            drop_records.append({"reason": "len_filter", "field": "question", "title": title})
            continue
        if not unanswerable and not length_ok(answer_text, args.min_answer, args.max_answer):
            dropped["answer_length"] += 1
            drop_records.append({"reason": "len_filter", "field": "answer", "title": title})
            continue

        if not unanswerable:
            aligned = align_answer(context, answer_text)
            if aligned is None:
                dropped["answer_not_in_context"] += 1
                drop_records.append({"reason": "answer_not_in_context", "title": title})
                continue
            answer_text = aligned

        key = (context, question)
        if key in dedup:
            dropped["duplicate"] += 1
            drop_records.append({"reason": "duplicate", "title": title})
            continue
        dedup.add(key)

        group_hash = hashlib.md5(context.encode("utf-8")).hexdigest()
        if args.stratify_by == "title":
            group_id = title or "<unknown-title>"
        else:
            group_id = f"{title}::{group_hash}"
        grouped[group_id].append(
            {
                "context": context,
                "question": question,
                "answer": answer_text,
                "unanswerable": unanswerable,
                "title": title,
            }
        )

    splits = stratified_group_split(
        grouped, args.seed, args.train_frac, args.val_frac, args.test_frac
    )
    return splits, dropped, drop_records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--local-fallback",
        type=Path,
        default=Path("data/artifacts/squad_v2_sample.json"),
        help="Local JSON file with squad_v2-like rows used if hub download fails.",
    )
    parser.add_argument("--min-context", type=int, default=20)
    parser.add_argument("--max-context", type=int, default=512)
    parser.add_argument("--min-question", type=int, default=3)
    parser.add_argument("--max-question", type=int, default=40)
    parser.add_argument("--min-answer", type=int, default=1)
    parser.add_argument("--max-answer", type=int, default=15)
    parser.add_argument(
        "--stratify-by",
        choices=["paragraph", "title"],
        default="paragraph",
        help="Stratify splits by paragraph (context hash) or by article title.",
    )
    unans_group = parser.add_mutually_exclusive_group()
    unans_group.add_argument("--keep-unanswerable", dest="drop_unanswerable", action="store_false")
    unans_group.add_argument("--drop-unanswerable", dest="drop_unanswerable", action="store_true")
    parser.set_defaults(drop_unanswerable=False)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prep_logs = Path("data/prep_logs")
    prep_logs.mkdir(parents=True, exist_ok=True)
    dropped_log = prep_logs / "en_dropped.jsonl"

    splits, dropped, drop_records = process_dataset(args)

    for split, rows in splits.items():
        save_jsonl(args.out_dir / f"{split}.jsonl", rows)

    stats = compute_stats(splits)
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    with dropped_log.open("w", encoding="utf-8") as f:
        for rec in drop_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.write(json.dumps({"summary": dropped}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
