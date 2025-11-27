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

Each JSONL row should look like:

```json
{"context": "<passage text>", "answer": "<gold answer>", "question": "<gold question>"}
```

The model is prompted with:

```
generate question: answer: <answer> context: <context>
```
