"""Render Markdown templates by substituting placeholders from stats and config files.

Usage:
    uv run python scripts/render_docs.py \
        --in docs/ --out docs/ \
        --stats-en data/stats_en.json --stats-ua data/stats_ua.json \
        --safety configs/safety.yaml

Placeholders use double braces, e.g. `{{stats_en.counts.train}}`.
Values are written with the placeholder preserved in an HTML comment to keep traceability.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import sys

PLACEHOLDER_RE = re.compile(r"(?<!<!--){{\s*([^{}]+?)\s*}}")
COMMENTED_PLACEHOLDER_RE = re.compile(
    r"(?:[\w\.\-]+)?\s*<!--\s*{{\s*([^{}]+?)\s*}}\s*-->"
)


def _parse_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        pass
    if (raw.startswith("\"") and raw.endswith("\"")) or (
        raw.startswith("'") and raw.endswith("'")
    ):
        return raw[1:-1]
    return raw


def _load_simple_yaml(path: Path) -> Dict[str, Any]:
    """Parse a small subset of YAML (mapping-only) without external deps."""

    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    root: Dict[str, Any] = {}
    stack: list[tuple[int, Dict[str, Any]]] = [(0, root)]

    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        key, _, value = stripped.partition(":")
        key = key.strip()
        value = value.strip()

        while stack and indent < stack[-1][0]:
            stack.pop()
        current = stack[-1][1]

        if not value:
            new_map: Dict[str, Any] = {}
            current[key] = new_map
            stack.append((indent + 2, new_map))
        else:
            current[key] = _parse_value(value)

    return root


@dataclass
class Context:
    stats_en: Mapping[str, Any]
    stats_ua: Mapping[str, Any]
    safety: Mapping[str, Any]
    thresholds: Mapping[str, Any]

    @classmethod
    def from_files(
        cls,
        stats_en: Path,
        stats_ua: Path,
        safety: Path,
        qg2qa_en: Path,
        qg2qa_ua: Path,
    ) -> "Context":
        with stats_en.open("r", encoding="utf-8") as f:
            stats_en_data = json.load(f)
        with stats_ua.open("r", encoding="utf-8") as f:
            stats_ua_data = json.load(f)
        safety_data = _load_simple_yaml(safety)

        thresholds: Dict[str, Any] = {}
        for cfg_path in (qg2qa_en, qg2qa_ua):
            if cfg_path.exists():
                cfg = _load_simple_yaml(cfg_path)
                thresholds.update(cfg.get("thresholds", {}))

        return cls(
            stats_en=stats_en_data,
            stats_ua=stats_ua_data,
            safety=safety_data,
            thresholds=thresholds,
        )

    def resolve(self, path: str) -> Any:
        parts = path.split(".")
        value: Any = self.__dict__
        for part in parts:
            if isinstance(value, Mapping) and part in value:
                value = value[part]
            else:
                raise KeyError(f"Placeholder '{path}' not found in context")
        return value


def find_markdown_files(root: Path) -> Iterable[Path]:
    return [p for p in root.rglob("*.md") if p.is_file()]


def render_text(text: str, context: Context, *, label: str | None = None) -> str:
    def replace_match(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        try:
            value = context.resolve(key)
        except KeyError:
            target = f" in {label}" if label else ""
            print(f"[render-docs] Unknown placeholder '{key}'{target}, keeping as-is.", file=sys.stderr)
            return match.group(0)
        return f"{value}<!--{{{{{key}}}}}-->"

    # Refresh already-commented placeholders first to keep reruns idempotent.
    text = COMMENTED_PLACEHOLDER_RE.sub(replace_match, text)

    def repl(match: re.Match[str]) -> str:
        return replace_match(match)

    return PLACEHOLDER_RE.sub(repl, text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render documentation templates")
    parser.add_argument("--in", dest="input_dir", type=Path, required=True)
    parser.add_argument("--out", dest="output_dir", type=Path, required=True)
    parser.add_argument("--stats-en", dest="stats_en", type=Path, required=True)
    parser.add_argument("--stats-ua", dest="stats_ua", type=Path, required=True)
    parser.add_argument("--safety", dest="safety", type=Path, required=True)
    parser.add_argument(
        "--qg2qa-en",
        dest="qg2qa_en",
        type=Path,
        default=Path("configs/qg2qa_en.yaml"),
        help="QA config for EN thresholds (default: configs/qg2qa_en.yaml)",
    )
    parser.add_argument(
        "--qg2qa-ua",
        dest="qg2qa_ua",
        type=Path,
        default=Path("configs/qg2qa_ua.yaml"),
        help="QA config for UA thresholds (default: configs/qg2qa_ua.yaml)",
    )
    args = parser.parse_args()

    ctx = Context.from_files(
        stats_en=args.stats_en,
        stats_ua=args.stats_ua,
        safety=args.safety,
        qg2qa_en=args.qg2qa_en,
        qg2qa_ua=args.qg2qa_ua,
    )

    output_dir = args.output_dir
    input_dir = args.input_dir
    for md_path in find_markdown_files(input_dir):
        rel = md_path.relative_to(input_dir)
        out_path = output_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        content = md_path.read_text(encoding="utf-8")
        rendered = render_text(content, ctx, label=str(rel))
        out_path.write_text(rendered, encoding="utf-8")

        print(f"Rendered {rel} -> {out_path}")


if __name__ == "__main__":
    main()
