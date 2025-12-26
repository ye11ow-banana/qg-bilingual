# Data preparation guidelines

- **Normalization:** Both EN and UA pipelines normalize text with NFKC, unify quotes/apostrophes, and tighten spacing so artifacts like ` , ` or ` ?` are removed before saving JSONL.
- **Unanswerable handling:** Examples marked as unanswerable keep `answer=""` and `unanswerable=true` by default; pass `--drop-unanswerable` to filter them out early and log the removals into `data/prep_logs/*_dropped.jsonl`. Stats and downstream QA metrics expect this encoding.
- **Stratification:** Splits can preserve grouping by article title via `--stratify-by title`; otherwise, paragraph hashes are used to keep all QAs from a paragraph together. English splitting supports custom `--train/val/test-frac` values, which must sum to 1.0 and will error if a non-zero split would end up empty after grouping.
- **Length controls:** CLI flags `--min/max-{context,question,answer}` guard against overly short or long fields. Dropped records include the offending field and reason in the prep logs.
- **Determinism:** The `--seed` flag drives shuffling for reproducible grouping and sampling in both language pipelines.
