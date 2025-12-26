# Data preparation guidelines

- **Normalization:** Both EN and UA pipelines normalize text with NFKC, unify quotes/apostrophes, and tighten spacing so artifacts like ` , ` or ` ?` are removed before saving JSONL.
- **Unanswerable handling:** Examples marked as unanswerable keep `answer=""` and `unanswerable=true` by default; pass `--drop-unanswerable` to filter them out early and log the removals into `data/prep_logs/*_dropped.jsonl`.
- **Stratification:** Splits can preserve grouping by article title via `--stratify-by title`; otherwise, paragraph hashes are used. English splitting also supports custom `--train/val/test-frac` values to control the ratio of group allocation.
- **Length controls:** CLI flags `--min/max-{context,question,answer}` guard against overly short or long fields. Dropped records include the offending field and reason in the prep logs.
- **Determinism:** The `--seed` flag drives shuffling for reproducible grouping and sampling in both language pipelines.
