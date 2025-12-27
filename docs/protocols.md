# Протоколи оцінювання

## QG→QA
- **Вхід:** `samples_val.jsonl` з полями `context`, `question`, `gold_answer`, `lang`, `unanswerable`.
- **Модель QA:** EN — `roberta-base-squad2` (приклад), UA — `xlm-roberta-base-squad2`.
- **Вихід:**
  - `qg2qa_details.jsonl` — пострічкові результати (span, conf, EM, F1, pass).
  - `qg2qa_val.json` — агрегати (середні EM/F1/pass-rate, гістограми/бакети).
- **Пороги:** `f1_pass`, `conf_pass` (див. `configs/qg2qa_*.yaml`).
- **Команда:**

```bash
uv run python -m src.qg_bilingual.eval.qg2qa \
  --config configs/qg2qa_en.yaml \
  --input runs/<exp>/samples_val.jsonl \
  --out runs/<exp>/ \
  --question-field question \
  --gold-field gold_answer
```

## MOS
- **Вибірка:** 200–300 прикладів із стратифікацією за EN/UA, aware/agnostic, wh-типами, pass/fail.
- **Анотатори:** 2–3 на приклад; фіксована інструкція та шкала 1–5.
- **Агрегація:** MOS + 95% CI (bootstrap), κ Коена або α Кронбаха для узгодженості.
- **Команди:**

```bash
uv run python eval/mos/sample_builder.py --inputs runs/*/samples_val.jsonl --size 300 --out eval/mos/mos_batch.jsonl
uv run python eval/mos/aggregate.py --batch eval/mos/mos_batch.jsonl --ann eval/mos/mos_raw_annotator_*.jsonl --out-dir eval/mos/
```

## Bootstrap CI
- Paired bootstrap різниць Δ-метрик (ROUGE-L, EM, F1, pass-rate) з 1000–5000 ресемплами.
- 95% довірчий інтервал; якщо CI не перетинає 0 — покращення значущо.
- Перевіряти FDR (Benjamini–Hochberg 5%) при множинних порівняннях.

## NLI та токсичність (фільтри)
- Конфіг: `configs/safety.yaml` (параметри `require_entailment`, `prob_max`, `lexicon-block`, `policy.context_only`).
- Використання: до/після генерації логувати `reasons` для помилок; тримати офлайн-звіти.

## Посилання
[1] T5 – https://arxiv.org/abs/1910.10683  
[2] BART – https://arxiv.org/abs/1910.13461  
[3] mT5 – https://arxiv.org/abs/2010.11934  
[4] Hugging Face Transformers – https://huggingface.co/docs/transformers  
[5] Hugging Face Datasets – https://huggingface.co/docs/datasets  
[6] ROUGE – https://aclanthology.org/W04-1013/
