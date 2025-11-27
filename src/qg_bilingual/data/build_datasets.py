"""Dataset builder for bilingual Q&A resources.

This script ingests SQuAD 2.0 and a Ukrainian Q&A corpus, applies basic
normalization, deduplication, and filtering, and then writes fixed train/val/test
splits with answer highlight markers suitable for downstream tasks.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence
import random

DEFAULT_HL_START = "<hl>"
DEFAULT_HL_END = "</hl>"
DEFAULT_NO_ANSWER_TOKEN = "<unanswerable>"


@dataclass
class Example:
    """Container for a single QA example."""

    context: str
    question: str
    answer: str
    highlighted_context: str
    source: str
    language: str
    is_unanswerable: bool = False

    def to_json(self) -> Dict[str, object]:
        return {
            "context": self.context,
            "question": self.question,
            "answer": self.answer,
            "highlighted_context": self.highlighted_context,
            "source": self.source,
            "language": self.language,
            "is_unanswerable": self.is_unanswerable,
        }


def normalize_text(text: str) -> str:
    """Collapse whitespace and normalize a handful of common punctuation."""

    replacements = {
        "\u00a0": " ",
        "\u201c": "\"",
        "\u201d": "\"",
        "\u2018": "'",
        "\u2019": "'",
    }
    normalized = text
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    return " ".join(normalized.strip().split())


def inject_highlight(
    context: str,
    answer: str,
    *,
    start_token: str = DEFAULT_HL_START,
    end_token: str = DEFAULT_HL_END,
    no_answer_token: str = DEFAULT_NO_ANSWER_TOKEN,
) -> str:
    """Surround the first answer occurrence with highlight markers.

    If the answer cannot be located, the function appends a highlighted placeholder
    instead of altering the context.
    """

    if not answer:
        return f"{context} {start_token}{no_answer_token}{end_token}"

    lowered_context = context.lower()
    lowered_answer = answer.lower()
    start = lowered_context.find(lowered_answer)
    if start == -1:
        return f"{context} {start_token}{answer}{end_token}"

    end = start + len(answer)
    return f"{context[:start]}{start_token}{context[start:end]}{end_token}{context[end:]}"


def load_squad_v2(path: Path, *, language: str = "en") -> List[Example]:
    """Load SQuAD 2.0 style data from a JSON file."""

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    examples: List[Example] = []
    for article in payload.get("data", []):
        for paragraph in article.get("paragraphs", []):
            raw_context = paragraph.get("context", "")
            context = normalize_text(raw_context)
            for qa in paragraph.get("qas", []):
                question = normalize_text(qa.get("question", ""))
                is_impossible = qa.get("is_impossible", False)
                answers = qa.get("answers", [])

                if is_impossible:
                    answer_text = ""
                    highlighted = inject_highlight(
                        context,
                        answer_text,
                        no_answer_token=DEFAULT_NO_ANSWER_TOKEN,
                    )
                else:
                    if not answers:
                        # Skip malformed entries.
                        continue
                    answer_text = normalize_text(answers[0].get("text", ""))
                    highlighted = inject_highlight(context, answer_text)

                examples.append(
                    Example(
                        context=context,
                        question=question,
                        answer=answer_text,
                        highlighted_context=highlighted,
                        source="squad_v2",
                        language=language,
                        is_unanswerable=is_impossible,
                    )
                )
    return examples


def load_ua_corpus(path: Path, *, language: str = "uk") -> List[Example]:
    """Load Ukrainian QA data from JSONL with context/question/answer fields."""

    examples: List[Example] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            raw_context = record.get("context", "")
            raw_question = record.get("question", "")
            answer_field = record.get("answer")
            answers_field = record.get("answers")
            answer_text: Optional[str]
            if isinstance(answer_field, str):
                answer_text = answer_field
            elif isinstance(answers_field, list) and answers_field:
                answer_text = answers_field[0]
            else:
                answer_text = None

            if answer_text is None:
                continue

            context = normalize_text(raw_context)
            question = normalize_text(raw_question)
            answer = normalize_text(answer_text)
            highlighted = inject_highlight(context, answer)

            examples.append(
                Example(
                    context=context,
                    question=question,
                    answer=answer,
                    highlighted_context=highlighted,
                    source="uk_qa_corpus",
                    language=language,
                    is_unanswerable=False,
                )
            )
    return examples


def deduplicate(examples: Iterable[Example]) -> List[Example]:
    """Remove duplicate (context, question) pairs while preserving order."""

    seen = set()
    unique_examples: List[Example] = []
    for ex in examples:
        key = (ex.context, ex.question)
        if key in seen:
            continue
        seen.add(key)
        unique_examples.append(ex)
    return unique_examples


def filter_by_length(
    examples: Iterable[Example],
    *,
    min_context_chars: int,
    max_context_chars: int,
    min_question_chars: int,
    max_question_chars: int,
) -> List[Example]:
    """Filter out contexts or questions that are too short/long."""

    filtered: List[Example] = []
    for ex in examples:
        if not (min_context_chars <= len(ex.context) <= max_context_chars):
            continue
        if not (min_question_chars <= len(ex.question) <= max_question_chars):
            continue
        filtered.append(ex)
    return filtered


def stratified_split(
    examples: Sequence[Example],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Dict[str, List[Example]]:
    """Split examples into train/val/test with approximate source stratification."""

    rng = random.Random(seed)
    splits: Dict[str, List[Example]] = {"train": [], "val": [], "test": []}
    grouped: MutableMapping[str, List[Example]] = defaultdict(list)
    for ex in examples:
        grouped[ex.source].append(ex)

    for group in grouped.values():
        rng.shuffle(group)

    for source, group in grouped.items():
        total = len(group)
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)
        test_count = total - train_count - val_count
        # Adjust for small groups to ensure non-empty splits when possible.
        if train_count == 0 and total > 0:
            train_count = 1
        if val_count == 0 and total - train_count > 1:
            val_count = 1
        if train_count + val_count > total:
            val_count = max(0, total - train_count)
            test_count = 0
        splits["train"].extend(group[:train_count])
        splits["val"].extend(group[train_count : train_count + val_count])
        splits["test"].extend(group[train_count + val_count : train_count + val_count + test_count])

    for key in splits:
        rng.shuffle(splits[key])
    return splits


def write_jsonl(records: Iterable[Example], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")


def summarise_counts(examples: Iterable[Example]) -> Counter:
    counts = Counter()
    for ex in examples:
        counts[ex.source] += 1
    return counts


def write_data_card(
    output_dir: Path,
    *,
    total_counts: Counter,
    split_counts: Dict[str, Counter],
    min_context_chars: int,
    max_context_chars: int,
    min_question_chars: int,
    max_question_chars: int,
    seed: int,
) -> None:
    """Emit a Markdown data card describing the artifacts."""

    lines = [
        "# Bilingual QA dataset",
        "",
        "## Sources",
        "- **SQuAD 2.0** (CC BY-SA 4.0) — English QA with answerable and unanswerable examples.",
        "- **Ukrainian QA corpus** (project-provided; ensure licensing before redistribution).",
        "",
        "## Preprocessing",
        "- Normalized whitespace and punctuation for contexts, questions, and answers.",
        "- Added answer highlight markers `<hl>` ... `</hl>`; unanswerable items mark `<unanswerable>`.",
        "- Deduplicated by `(context, question)` pairs.",
        "- Filtered contexts to character length [{min_context_chars}, {max_context_chars}] and questions to [{min_question_chars}, {max_question_chars}].",
        f"- Deterministic splitting with seed `{seed}` and stratification by source where possible.",
        "",
        "## Split sizes",
    ]

    for split, counts in split_counts.items():
        total_split = sum(counts.values())
        per_source = ", ".join(f"{src}: {cnt}" for src, cnt in counts.items())
        lines.append(f"- **{split}**: {total_split} ({per_source})")
    lines.extend(
        [
            "",
            "## Known limitations",
            "- Source coverage is limited to the supplied corpora; domain gaps likely remain.",
            "- Context-answer matching relies on surface string search; paraphrased answers may miss highlights.",
            "- Ukrainian corpus schema is assumed to contain context/question/answer fields; malformed rows are skipped.",
        ]
    )

    card_path = output_dir / "data_card.md"
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build bilingual QA datasets.")
    parser.add_argument("--squad-train", type=Path, required=True, help="Path to SQuAD 2.0 train JSON file.")
    parser.add_argument("--squad-dev", type=Path, required=True, help="Path to SQuAD 2.0 dev JSON file.")
    parser.add_argument("--ua-corpus", type=Path, required=True, help="Path to Ukrainian QA JSONL corpus.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/artifacts"), help="Where to write jsonl artifacts.")
    parser.add_argument("--seed", type=int, default=7, help="Seed for deterministic shuffling.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Training split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio; remainder used for test.")
    parser.add_argument("--min-context-chars", type=int, default=80, help="Minimum context length in characters.")
    parser.add_argument("--max-context-chars", type=int, default=1200, help="Maximum context length in characters.")
    parser.add_argument("--min-question-chars", type=int, default=8, help="Minimum question length in characters.")
    parser.add_argument("--max-question-chars", type=int, default=256, help="Maximum question length in characters.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    squad_examples = load_squad_v2(args.squad_train)
    squad_examples.extend(load_squad_v2(args.squad_dev))
    ua_examples = load_ua_corpus(args.ua_corpus)

    merged = deduplicate([*squad_examples, *ua_examples])
    filtered = filter_by_length(
        merged,
        min_context_chars=args.min_context_chars,
        max_context_chars=args.max_context_chars,
        min_question_chars=args.min_question_chars,
        max_question_chars=args.max_question_chars,
    )

    splits = stratified_split(
        filtered,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    for split_name, records in splits.items():
        write_jsonl(records, args.output_dir / f"{split_name}.jsonl")

    total_counts = summarise_counts(filtered)
    split_counts: Dict[str, Counter] = {name: summarise_counts(records) for name, records in splits.items()}

    write_data_card(
        args.output_dir,
        total_counts=total_counts,
        split_counts=split_counts,
        min_context_chars=args.min_context_chars,
        max_context_chars=args.max_context_chars,
        min_question_chars=args.min_question_chars,
        max_question_chars=args.max_question_chars,
        seed=args.seed,
    )

    print("Wrote artifacts to", args.output_dir.resolve())


if __name__ == "__main__":
    main()
