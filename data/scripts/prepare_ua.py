import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional


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
    text = text.replace("\u00a0", " ")
    for src, dst in QUOTE_MAP.items():
        text = text.replace(src, dst)
    for src, dst in APOSTROPHE_MAP.items():
        text = text.replace(src, dst)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*([,.;:!?])\s*", r" \1 ", text)
    text = re.sub(r"\s+/\s+", "/", text)
    text = re.sub(r"\s+-\s+", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def translate_en_to_ua(text: str) -> str:
    # Placeholder stub – replace with a real translation model later.
    return text


def align_answer(context: str, answer: str) -> Optional[str]:
    if not answer:
        return ""
    if answer in context:
        return answer
    candidates = [answer.strip(), answer.strip().strip(".?!,;:"), answer.replace("’", "'"), answer.replace("'", "’")]
    for cand in candidates:
        cand_norm = normalize_text(cand)
        if cand_norm in context:
            return cand_norm
    simplified_context = normalize_text(context)
    if normalize_text(answer) in simplified_context:
        return normalize_text(answer)
    return None


def length_ok(text: str, min_len: int, max_len: int) -> bool:
    tokens = text.split()
    return min_len <= len(tokens) <= max_len


def save_jsonl(path: Path, rows: Iterable[Dict]):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def process_split(rows: List[Dict], args, rng: random.Random, dropped: Counter) -> List[Dict]:
    result = []
    dedup = set()
    for row in rows:
        if args.drop_unanswerable and row.get("unanswerable", False):
            dropped["unanswerable_dropped"] += 1
            continue

        context_en = row["context"]
        question_en = row["question"]
        answer_en = row.get("answer", "")
        unanswerable = row.get("unanswerable", False)

        context = normalize_text(translate_en_to_ua(context_en))
        question = normalize_text(translate_en_to_ua(question_en))
        answer = normalize_text(translate_en_to_ua(answer_en)) if not unanswerable else ""

        if not unanswerable:
            aligned = align_answer(context, answer)
            if aligned is None:
                dropped["answer_not_in_context"] += 1
                continue
            answer = aligned

        if not length_ok(context, args.min_context, args.max_context):
            dropped["context_length"] += 1
            continue
        if not length_ok(question, args.min_question, args.max_question):
            dropped["question_length"] += 1
            continue
        if not unanswerable and not length_ok(answer, args.min_answer, args.max_answer):
            dropped["answer_length"] += 1
            continue

        key = (context, question)
        if key in dedup:
            dropped["duplicate"] += 1
            continue
        dedup.add(key)

        result.append(
            {
                "context": context,
                "question": question,
                "answer": answer,
                "unanswerable": unanswerable,
                "title": row.get("title", ""),
            }
        )
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-context", type=int, default=20)
    parser.add_argument("--max-context", type=int, default=512)
    parser.add_argument("--min-question", type=int, default=3)
    parser.add_argument("--max-question", type=int, default=40)
    parser.add_argument("--min-answer", type=int, default=1)
    parser.add_argument("--max-answer", type=int, default=15)
    parser.add_argument("--train-limit", type=int, default=20000)
    parser.add_argument("--val-limit", type=int, default=1000)
    parser.add_argument("--test-limit", type=int, default=1000)
    unans_group = parser.add_mutually_exclusive_group()
    unans_group.add_argument("--keep-unanswerable", dest="drop_unanswerable", action="store_false")
    unans_group.add_argument("--drop-unanswerable", dest="drop_unanswerable", action="store_true")
    parser.set_defaults(drop_unanswerable=False)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prep_logs = Path("data/prep_logs")
    prep_logs.mkdir(parents=True, exist_ok=True)
    dropped_log = prep_logs / "ua_dropped.jsonl"

    splits: Dict[str, List[Dict]] = {}
    dropped = Counter()

    for split_name, limit in [("train", args.train_limit), ("val", args.val_limit), ("test", args.test_limit)]:
        src_path = args.in_dir / f"{split_name}.jsonl"
        rows: List[Dict] = []
        with src_path.open("r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        rng.shuffle(rows)
        if limit:
            rows = rows[:limit]
        processed = process_split(rows, args, rng, dropped)
        splits[split_name] = processed
        save_jsonl(args.out_dir / f"{split_name}.jsonl", processed)

    stats = compute_stats(splits)
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    with dropped_log.open("w", encoding="utf-8") as f:
        for reason, count in dropped.items():
            f.write(json.dumps({"reason": reason, "count": count}) + "\n")


if __name__ == "__main__":
    main()
