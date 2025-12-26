# qg-bilingual

Prototype repository for bilingual (EN/UA) answer-aware question generation using transformer models.

## Project plan
See [docs/project_plan.md](docs/project_plan.md) for the current analytical overview, experimental plan, and immediate action items derived from the diploma assignment.

## Status
- Training CLI for answer-aware T5 models is available with optional LoRA and ROUGE-L validation.
- Data artifacts are expected as JSONL with `context`, `answer`, and `question` fields (see example below).

## HTTP API
An experimental FastAPI server exposes a toxicity- and NLI-checked generation endpoint. Run it with the bundled configuration:

```bash
uv run uvicorn qg_bilingual.server.app:app --host 0.0.0.0 --port 8000
```

- Configuration lives in `src/qg_bilingual/server/config.yaml` and controls model names, decoding parameters, and thresholds for QA/NLI/toxicity checks.
- `/healthz` returns `{"status": "ok"}`.
- `/generate_safe` accepts a JSON body described by `GenerateRequest` and returns `GenerateResponse` with metrics, reasons for filtering, and debug fields.

Example requests:

```bash
# aware EN
curl -X POST :8000/generate_safe -H 'Content-Type: application/json' -d '{
  "context":"Taras Shevchenko was born on March 9, 1814 in Moryntsi.",
  "answer":"March 9, 1814", "lang":"en", "mode":"aware"
}'

# agnostic UA with WHEN constraint
curl -X POST :8000/generate_safe -H 'Content-Type: application/json' -d '{
  "context":"Тарас Шевченко народився 9 березня 1814 року в Моринцях.",
  "lang":"ua", "mode":"agnostic", "wh_type":"when"
}'
```

Responses include `question` (or `null` if filtered out), `passed`, `reasons` (e.g. `qg2qa_f1_low`, `nli_neutral`, `lexicon_block`), a `metrics` map (`qa_em`, `qa_f1`, `qa_conf`, `tox_prob`, `nli`), and `debug` fields with decoding options and detected WH tokens.

## Train (T5-base aware)
Run the answer-aware question generation training loop with the provided YAML config:

```
uv run python -m qg_bilingual.train --config configs/train_t5_base.yaml
```

Evaluation now runs as a standalone QG→QA pass (batched extractive QA over
`context + generated question`). Configure it via dedicated YAMLs
(`configs/qg2qa_en.yaml`, `configs/qg2qa_ua.yaml`):

```yaml
lang: ua                  # en|ua
qa_model: xlm-roberta-base-squad2
device: auto              # cuda|cpu|auto
batch_size: 16
max_context_tokens: 512   # context gets truncated to this size
thresholds:
  f1_pass: 0.80           # pass-rate requires >= this F1
  conf_pass: 0.35         # and >= this span confidence
normalization:
  strip_punct: true
  lower: true
  unify_quotes: true
  unify_apostrophe: true
io:
  input_jsonl: runs/t5_base_aware_ua/samples_val.jsonl
  out_dir: runs/t5_base_aware_ua/
```

Run the scorer:

```bash
uv run python -m qg_bilingual.eval.qg2qa --config configs/qg2qa_en.yaml --input runs/t5_base_aware_en/samples_val.jsonl --out runs/t5_base_aware_en/
```

- `qg2qa_val.json` stores aggregates (EM/F1/pass-rate, buckets by question
  length and wh-type, F1 histogram).
- `qg2qa_details.jsonl` stores row-level predictions with confidence and
  pass flags.
- `--include-unanswerable` keeps `unanswerable=true` rows in metrics; by
  default they are counted in `counts` but excluded from EM/F1/pass-rate unless
  the QA model returns an empty span with confidence ≥ `conf_pass`.

Each JSONL row should look like:

```json
{"context": "<passage text>", "answer": "<gold answer>", "question": "<gold question>"}
```

The model is prompted with a unified XML-style template:

```
<question_generation>
  <instruction>Generate a question for the provided answer and context.</instruction>
  <context><passage text></context>
  <answer><gold answer></answer>
</question_generation>
```

Example generated question (toy):

```
Input answer/context -> "Paris" / "France's capital is known for the Eiffel Tower."
Generated question -> "What is the capital city of France famous for the Eiffel Tower?"
```

After validation, `models/.../metrics_val.json` contains combined text and QG→QA metrics, e.g.:

```json
{
  "rouge1": 0.61,
  "rougeL": 0.58,
  "bleu": 24.3,
  "em": 32.1,
  "f1": 48.7,
  "qa_pass_rate": 0.44,
  "qa_model": "distilbert-base-uncased-distilled-squad",
  "lang": "en",
  "qa_device": "cuda",
  "f1_thr": 0.8,
  "conf_thr": 0.35
}
```

## Data preparation
- English split generation: `uv run python data/scripts/prepare_en.py --out-dir data/artifacts/en --stats data/stats_en.json --seed 42 --stratify-by title --train-frac 0.02 --val-frac 0.01 --test-frac 0.97`
- Ukrainian projection: `uv run python data/scripts/prepare_ua.py --in-dir data/artifacts/en --out-dir data/artifacts/ua --stats data/stats_ua.json --seed 42 --stratify-by title`

Both pipelines normalize text (NFKC + unified quotes/apostrophes + clean punctuation spacing), deduplicate by `(context, question)`, enforce length limits, and keep grouping coherent during stratified splits (paragraph-level by default, overridable via `--stratify-by title`). Unanswerable items are **kept by default** with `answer=""` and `unanswerable=true`; pass `--drop-unanswerable` to remove them early and log the reason (stats and QA metrics treat `unanswerable=true` + empty answer consistently). English splitting accepts custom train/val/test fractions **that must sum to 1.0**; the script will refuse setups that would yield empty non-zero splits once paragraphs/titles are grouped. Ukrainian subsampling can follow the same title-aware order when limiting rows. Ukrainian data currently reuses English text via a `translate_en_to_ua` stub until a real translation model is plugged in.

See [docs/data.md](docs/data.md) for a concise checklist covering normalization, unanswerable policy, stratification, and determinism.

## Safety lexicons
The safety filters rely on small manually curated lexicons stored under `src/qg_bilingual/safety/lexicons`. Terms are added conservatively and reviewed for false positives; contributors should expand them incrementally and keep notes about controversial entries in PR descriptions.
