# Bilingual QA data artifacts

## Sources and licensing
- **SQuAD 2.0** — English reading comprehension dataset with answerable and unanswerable items. Licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
- **Ukrainian QA corpus** — Project-supplied corpus with context/question/answer triples. Confirm redistribution rights before sharing derived artifacts.

## Preprocessing pipeline
- Normalize whitespace and standardize a handful of punctuation marks across contexts, questions, and answers.
- Preserve SQuAD unanswerable questions by keeping empty answers and appending `<hl><unanswerable></hl>` markers.
- Insert highlight markers `<hl> ... </hl>` around the first surface match for each answer; fall back to appending a highlighted answer when no match exists.
- Deduplicate by `(context, question)` pairs across merged corpora.
- Filter out contexts and questions outside configured character-length ranges to remove pathological samples.
- Deterministic train/validation/test splits with a configurable seed and stratification by source when group sizes allow.

## Output format
Each JSONL artifact under `data/artifacts/` contains objects with at least the following fields:
- `context`: normalized passage text.
- `question`: normalized question string.
- `answer`: normalized answer text (empty for unanswerable SQuAD examples).
- `highlighted_context`: context text with highlight markers indicating the answer span or an explicit `<unanswerable>` tag.
- `source`: dataset identifier (`squad_v2` or `uk_qa_corpus`).
- `language`: ISO language tag for the sample.
- `is_unanswerable`: boolean flag highlighting SQuAD 2.0 impossible questions.

## Known limitations
- Surface-level string matching may miss answers that require paraphrase detection, leading to appended highlights.
- The Ukrainian corpus schema is assumed; malformed rows without a clear answer are skipped entirely.
- Length filtering is heuristic and may discard edge cases that would otherwise be valid.
- Final split sizes depend on input availability; reruns with the same seed and inputs remain deterministic.

## Reproducibility
Invoke the builder via:
```
python -m qg_bilingual.data.build_datasets \
  --squad-train /path/to/train-v2.0.json \
  --squad-dev /path/to/dev-v2.0.json \
  --ua-corpus /path/to/ua.jsonl \
  --output-dir data/artifacts
```
