# Data preparation guidelines

## Policies

- **Normalization:** Both EN and UA pipelines normalize with NFKC, convert quotes to `"`, apostrophes to `’`, squeeze whitespace, remove spaces before punctuation, and trim.
- **Unanswerable handling:** By default unanswerable rows are kept as `answer=""`, `unanswerable=true`. Passing `--drop-unanswerable` removes them and logs `reason=unanswerable` to `data/prep_logs/en_dropped.jsonl` or `ua_dropped.jsonl`.
- **Stratification:** Splits are paragraph-stratified: all QAs from the same `(title, paragraph_index)` stay in one split. Fractions must sum to ~1.0; non-empty splits are enforced.
- **Duplicates:** Deduplication uses a stable hash of normalized `(context, question)` per language. Any dropped duplicates are recorded with `reason=duplicate`.
- **Length controls:** CLI flags `--min/max-{context,question,answer}` guard against short/long fields and log `reason=len_filter` when triggered.
- **Span alignment:** Answerable rows must pass `answer_in_context`; misalignments are dropped with `reason=span_misaligned`.
- **Determinism:** The `--seed` flag drives shuffling for reproducible grouping.

## Commands

Prepare English artifacts (falls back to `data/artifacts/squad_v2_sample.json` when offline):

```bash
uv run python data/scripts/prepare_en.py --out-dir data/artifacts/en --stats data/stats_en.json
```

Prepare Ukrainian artifacts (re-uses EN splits):

```bash
uv run python data/scripts/prepare_ua.py --in-dir data/artifacts/en --out-dir data/artifacts/ua --stats data/stats_ua.json
```

Validate any split and optionally rewrite stats:

```bash
uv run python data/scripts/validate_jsonl.py --path data/artifacts/en/train.jsonl --lang en --write-stats data/stats_en.json
```
