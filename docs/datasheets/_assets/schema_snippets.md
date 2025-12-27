# Schema snippets for datasheets rendering

- Placeholder format: `{{path.to.value}}` — шляхи відповідають ключам у JSON/YAML (наприклад, `stats_en.counts.train`).
- Приклад запису у stats JSON:

```json
{
  "counts": {"train": 3, "val": 4, "test": 3},
  "unanswerable_share": {"val": 0.25}
}
```

- Приклад використання в Markdown:

```
Train size: 3<!3<!--{{stats_en.counts.train}}-->-->
Unanswerable share (val): 0.25<!0.25<!--{{stats_en.unanswerable_share.val}}-->-->
Threshold F1: 0.8<!0.8<!--{{thresholds.f1_pass}}-->-->
```
