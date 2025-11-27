# qg-bilingual

Prototype repository for bilingual (EN/UA) answer-aware question generation using transformer models.

## Project plan
See [docs/project_plan.md](docs/project_plan.md) for the current analytical overview, experimental plan, and immediate action items derived from the diploma assignment.

## Status
- Training CLI for answer-aware T5 models is available with optional LoRA and ROUGE-L validation.
- Data artifacts are expected as JSONL with `context`, `answer`, and `question` fields (see example below).

## Train (T5-base aware)
Run the answer-aware question generation training loop with the provided YAML config:

```
uv run python -m qg_bilingual.train --config configs/train_t5_base.yaml
```

Evaluation uses a QA model for EM/F1/pass-rate. Configure language-aware QA in
the `qg2qa` block:

```yaml
qg2qa:
  qa_ckpt_en: "distilbert-base-uncased-distilled-squad"
  qa_ckpt_multi: "deepset/xlm-roberta-large-squad2"
  lang: "en"   # en|ua|auto
  f1_thr: 0.8
  conf_thr: 0.35
  device: "auto"  # cuda|cpu|auto
```

For Ukrainian validation, set `lang: "ua"` to pick the multilingual QA model.

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
  "f1_thr": 0.8,
  "conf_thr": 0.35
}
```
