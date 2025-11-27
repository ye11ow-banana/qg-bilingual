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

Evaluation uses a QA model for EM/F1/pass-rate; switch to the multilingual
checkpoint for UA validation by setting `qa_eval_language: "uk"` (the default EN
checkpoint is `distilbert-base-uncased-distilled-squad`, the multilingual one is
`deepset/xlm-roberta-large-squad2`).

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
  "qa_pass_rate": 0.44
}
```
