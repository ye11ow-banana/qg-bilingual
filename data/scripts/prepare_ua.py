import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List


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


def normalize_text(s: str) -> str:
    """NFKC, unify quotes/apostrophes, squeeze spaces, trim, and fix punctuation spacing."""

    text = unicodedata.normalize("NFKC", s.replace("\u00a0", " "))
    for src, dst in QUOTE_MAP.items():
        text = text.replace(src, dst)
    for src, dst in APOSTROPHE_MAP.items():
        text = text.replace(src, dst)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*([,.;:!?])", r"\1", text)
    text = re.sub(r"\s+/\s+", "/", text)
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def answer_in_context(ctx: str, ans: str) -> bool:
    ctx_norm = normalize_text(ctx)
    ans_norm = normalize_text(ans)
    return bool(ans_norm) and ans_norm in ctx_norm


def hash_key(ctx: str, q: str) -> str:
    basis = f"{normalize_text(ctx).lower()}||{normalize_text(q).lower()}"
    return hashlib.md5(basis.encode("utf-8")).hexdigest()


def translate_en_to_ua(text: str) -> str:
    """Placeholder stub – replace with a real translation model later."""

    return text


def try_align_answer(context: str, answer: str):
    if not answer:
        return ""
    if answer_in_context(context, answer):
        return normalize_text(answer)
    candidates = [
        answer.rstrip(".?!,;: "),
        answer.replace("\u00a0", " "),
        answer.replace("’", "'"),
        answer.replace("'", "’"),
    ]
    for cand in candidates:
        cand_norm = normalize_text(cand)
        if cand_norm and cand_norm in normalize_text(context):
            return cand_norm
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
        "span_fail_rate": {},
    }

    for split, rows in records.items():
        stats["counts"][split] = len(rows)
        ctx_lens = lengths(r["context"] for r in rows)
        q_lens = lengths(r["question"] for r in rows)
        ans_lens = lengths(r["answer"] for r in rows)
        for label, lens in [("context", ctx_lens), ("question", q_lens), ("answer", ans_lens)]:
            avg = sum(lens) / len(lens) if lens else 0
            med = median(lens) if lens else 0
            stats["avg_len"].setdefault(label, {})[split] = avg
            stats["median_len"].setdefault(label, {})[split] = med
        if rows:
            unans = sum(1 for r in rows if r.get("unanswerable", False))
            answerable = len(rows) - unans
            span_fail = sum(
                1
                for r in rows
                if not r.get("unanswerable", False)
                and not answer_in_context(r["context"], r["answer"])
            )
            stats["unanswerable_share"][split] = unans / len(rows)
            stats["span_fail_rate"][split] = span_fail / answerable if answerable else 0.0
        else:
            stats["unanswerable_share"][split] = 0.0
            stats["span_fail_rate"][split] = 0.0
    return stats


def process_split(rows: List[Dict], args, dropped: Counter, drop_records: List[Dict]) -> List[Dict]:
    result = []
    dedup = set()
    for row in rows:
        if args.drop_unanswerable and row.get("unanswerable", False):
            dropped["unanswerable"] += 1
            drop_records.append({"reason": "unanswerable", "title": row.get("title", "")})
            continue

        context_en = row["context"]
        question_en = row["question"]
        answer_en = row.get("answer", "")
        unanswerable = row.get("unanswerable", False)

        context = normalize_text(translate_en_to_ua(context_en))
        question = normalize_text(translate_en_to_ua(question_en))
        answer = normalize_text(translate_en_to_ua(answer_en)) if not unanswerable else ""

        if not unanswerable:
            aligned = try_align_answer(context, answer)
            if aligned is None:
                dropped["span_misaligned"] += 1
                drop_records.append({"reason": "span_misaligned", "title": row.get("title", "")})
                continue
            answer = aligned

        if not length_ok(context, args.min_context, args.max_context):
            dropped["len_filter"] += 1
            drop_records.append({"reason": "len_filter", "field": "context", "title": row.get("title", "")})
            continue
        if not length_ok(question, args.min_question, args.max_question):
            dropped["len_filter"] += 1
            drop_records.append({"reason": "len_filter", "field": "question", "title": row.get("title", "")})
            continue
        if not unanswerable and not length_ok(answer, args.min_answer, args.max_answer):
            dropped["len_filter"] += 1
            drop_records.append({"reason": "len_filter", "field": "answer", "title": row.get("title", "")})
            continue

        key_hash = hash_key(context, question)
        if key_hash in dedup:
            dropped["duplicate"] += 1
            drop_records.append({"reason": "duplicate", "title": row.get("title", "")})
            continue
        dedup.add(key_hash)

        result.append(
            {
                "context": context,
                "question": question,
                "answer": answer,
                "unanswerable": unanswerable,
                "title": row.get("title", ""),
                "paragraph_index": row.get("paragraph_index"),
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
    unans_group = parser.add_mutually_exclusive_group()
    unans_group.add_argument("--keep-unanswerable", dest="drop_unanswerable", action="store_false")
    unans_group.add_argument("--drop-unanswerable", dest="drop_unanswerable", action="store_true")
    parser.set_defaults(drop_unanswerable=False)
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prep_logs = Path("data/prep_logs")
    prep_logs.mkdir(parents=True, exist_ok=True)
    dropped_log = prep_logs / "ua_dropped.jsonl"

    splits: Dict[str, List[Dict]] = {}
    dropped = Counter()
    drop_records: List[Dict] = []

    for split_name in ("train", "val", "test"):
        src_path = args.in_dir / f"{split_name}.jsonl"
        rows: List[Dict] = []
        with src_path.open("r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))

        processed = process_split(rows, args, dropped, drop_records)
        splits[split_name] = processed
        save_jsonl(args.out_dir / f"{split_name}.jsonl", processed)

    stats = compute_stats(splits)
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    with dropped_log.open("w", encoding="utf-8") as f:
        for rec in drop_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.write(json.dumps({"summary": dropped}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
