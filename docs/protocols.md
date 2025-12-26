# NLI & Toxicity filters

The `/generate_safe` pipeline and offline validators share the same safety configuration stored in `configs/safety.yaml`. Core parameters:

- **device** / **batch_size**: execution device and batching for both services.
- **NLI**: Hugging Face model name, entailment requirements, hypothesis template, and entailment probability threshold.
- **Toxicity**: language-specific classifiers, probability cutoff, lexicon blocking toggle, and file paths for lexicons and protected groups.
- **Policy**: context-only enforcement for statements about protected groups and disallowing unbacked generalizations.

Example response fragment:

```json
{
  "question": "What is the capital of Ukraine?",
  "passed": true,
  "reasons": [],
  "metrics": {
    "nli": "entailment",
    "tox_prob": 0.05,
    "lex_hits": 0
  }
}
```

Both services expose batch APIs so offline scripts can validate many questions at once using the same configuration.
