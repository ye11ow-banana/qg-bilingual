# Як зібрати MOS через форму або CSV

## Варіант 1: Google Form
1. Експортуйте `eval/mos/mos_batch.jsonl` у CSV для зручності (наприклад, `jq -r '[.id,.lang,.model,.mode,.wh_type,.context,.question,.reference] | @csv' eval/mos/mos_batch.jsonl > eval/mos/mos_batch.csv`).
2. Створіть Google Form із полями:
   - `id` (short answer, обов’язково)
   - `mos` (linear scale 1–5)
   - `flags` (checkboxes: grammar, ambiguous, ooc, wrong_wh, too_long, too_short, toxic)
   - `comment` (optional paragraph)
3. Увімкніть "Download responses (.csv)" після завершення анотацій.
4. Експортуйте CSV, конвертуйте в JSONL за `schema.json` (див. приклад нижче), покладіть у `eval/mos/results/mos_raw_annotator_<id>.jsonl`.

```bash
python - <<'PY'
import csv, json, sys
input_csv = sys.argv[1]
annotator = sys.argv[2]
with open(input_csv, newline='') as f, open(f"eval/mos/results/mos_raw_annotator_{annotator}.jsonl", "w") as out:
    for row in csv.DictReader(f):
        record = {
            "id": row["id"],
            "mos": int(row["mos"]),
            "flags": [flag.strip() for flag in row.get("flags", "").split(';') if flag.strip()],
            "comment": row.get("comment") or None,
        }
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
PY eval/mos/mos_batch_responses.csv annotator_a
```

## Варіант 2: Ручний CSV
1. Відкрийте `eval/mos/mos_batch.csv` у Google Sheets/Excel.
2. Додайте стовпці `mos`, `flags`, `comment` та роздайте файл анотаторам.
3. Після повернення заповнених файлів використовуйте той самий Python-скрипт (вище) для конвертації у JSONL.

> За потреби можна написати простий Streamlit (див. `eval/mos/app.py` як майбутню точку входу) з офлайн-збереженням у JSONL, але для цього тікету достатньо форми/CSV.
