import argparse
import hashlib
import json
import re
import statistics
import unicodedata
from pathlib import Path
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


def length_ok(text: str, min_len: int, max_len: int) -> bool:
    tokens = text.split()
    return min_len <= len(tokens) <= max_len


def hash_key(ctx: str, q: str) -> str:
    basis = f"{normalize_text(ctx).lower()}||{normalize_text(q).lower()}"
    return hashlib.md5(basis.encode("utf-8")).hexdigest()


DEFAULT_LIMITS = {
    "context": (20, 512),
    "question": (3, 40),
    "answer": (1, 15),
}


class ValidationResult:
    def __init__(self):
        self.records: List[Dict] = []
        self.failures: List[str] = []
        self.span_fail = 0
        self.unanswerable = 0

    def add(self, record: Dict, span_ok: bool):
        self.records.append(record)
        if record.get("unanswerable", False):
            self.unanswerable += 1
        elif not span_ok:
            self.span_fail += 1



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--lang", choices=["en", "ua"], required=True)
    parser.add_argument("--write-stats", type=Path)
    return parser.parse_args()


def _calc_lengths(items: Iterable[str]) -> List[int]:
    return [len(x.split()) for x in items]


def summarize(rows: List[Dict]) -> Dict:
    ctx_lens = _calc_lengths(r["context"] for r in rows)
    q_lens = _calc_lengths(r["question"] for r in rows)
    ans_lens = _calc_lengths(r["answer"] for r in rows)

    stats = {
        "counts": {"total": len(rows)},
        "avg_len": {},
        "median_len": {},
        "unanswerable_share": {},
        "span_fail_rate": {},
    }

    for label, lens in [("context", ctx_lens), ("question", q_lens), ("answer", ans_lens)]:
        stats["avg_len"][label] = sum(lens) / len(lens) if lens else 0
        stats["median_len"][label] = statistics.median(lens) if lens else 0
    unans = sum(1 for r in rows if r.get("unanswerable", False))
    ans_count = len(rows) - unans
    span_fail = sum(
        1
        for r in rows
        if not r.get("unanswerable", False)
        and not answer_in_context(r["context"], r["answer"])
    )
    stats["unanswerable_share"] = {"total": unans / len(rows) if rows else 0.0}
    stats["span_fail_rate"] = {"total": span_fail / ans_count if ans_count else 0.0}
    return stats


def validate_rows(rows: List[Dict]) -> ValidationResult:
    res = ValidationResult()
    seen = set()

    for idx, row in enumerate(rows, start=1):
        missing = {field for field in ["context", "question", "answer", "unanswerable"] if field not in row}
        if missing:
            res.failures.append(f"Row {idx}: missing fields {missing}")
            continue

        context = str(row["context"])
        question = str(row["question"])
        answer = str(row["answer"])
        unanswerable = bool(row.get("unanswerable", False))

        limits = {
            "context": length_ok(context, *DEFAULT_LIMITS["context"]),
            "question": length_ok(question, *DEFAULT_LIMITS["question"]),
            "answer": unanswerable or length_ok(answer, *DEFAULT_LIMITS["answer"]),
        }
        for field, ok in limits.items():
            if not ok:
                res.failures.append(f"Row {idx}: {field} length out of bounds")

        span_ok = unanswerable or answer_in_context(context, answer)
        res.add({"context": context, "question": question, "answer": answer, "unanswerable": unanswerable}, span_ok)
        key = hash_key(context, question)
        if key in seen:
            res.failures.append(f"Row {idx}: duplicate hash {key}")
        seen.add(key)

        if not span_ok and not unanswerable:
            res.failures.append(f"Row {idx}: answer not in context")

    return res


def read_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    if lines and not lines[-1].endswith("\n"):
        raise ValueError("File must end with a newline")

    rows: List[Dict] = []
    for idx, line in enumerate(lines, start=1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {idx}: {exc}") from exc
    return rows


def maybe_read_drop_log(path: Path) -> Dict:
    log_path = path.parent.parent / "prep_logs" / f"{path.parent.name}_dropped.jsonl"
    if not log_path.exists():
        return {}
    summary = {}
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "summary" in rec:
                summary = rec.get("summary", {})
    return summary


def print_report(path: Path, res: ValidationResult, stats: Dict, drop_summary: Dict):
    total = stats["counts"]["total"]
    ans_count = total - res.unanswerable
    drop_total = sum(drop_summary.values()) if drop_summary else 0

    print("OK" if not res.failures else "FAIL")
    print("File:", path)
    print(
        f"count={total} | unanswerable={res.unanswerable} | span_fail_rate={res.span_fail/(ans_count or 1):.4f} | drops={drop_total}"
    )
    print("length avg:", {k: round(v, 2) for k, v in stats["avg_len"].items()})
    print("length median:", stats["median_len"])
    if drop_summary:
        print("drop summary:", drop_summary)
    if res.failures:
        print("Sample failures (up to 5):", res.failures[:5])


def main():
    args = parse_args()
    rows = read_jsonl(args.path)
    validation = validate_rows(rows)
    stats = summarize(rows)
    drop_summary = maybe_read_drop_log(args.path)

    print_report(args.path, validation, stats, drop_summary)

    if args.write_stats:
        args.write_stats.parent.mkdir(parents=True, exist_ok=True)
        args.write_stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    if validation.failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
