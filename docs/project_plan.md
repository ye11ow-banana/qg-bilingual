# Bilingual Question Generation Project Plan

## Topic
**Генерація wh-запитань з англомовних та україномовних текстів за допомогою трансформерних мовних моделей.**

## Analytical context (from draft chapters)
- **Relevance:** Transformer-based seq2seq models (T5/BART/mT5) dominate modern QG; answer-aware inputs with highlighted answers yield higher precision and reduce ambiguity. A reproducible bilingual (EN/UA) protocol with fixed splits, hyperparameters, preprocessing, and human evaluation is a key gap.
- **Corpora:** SQuAD 2.0 (~150k EN examples) plus Ukrainian Q&A sets (~200–300k). Require normalization, deduplication, length filtering, and licensed sources with data cards.
- **Evaluation:** Automatic (ROUGE-1/2/L, BLEU, BERTScore) plus QG→QA (EM/F1) and MOS human ratings (≥10 raters, agreement checks). Statistical significance via bootstrap; include latency/throughput and seed sensitivity.
- **Ethics & safety:** Use licensed data, anonymize, filter toxicity/bias, log rejection reasons; enforce context-only questions and human review for low-confidence generations.

## Methods and tooling
- **Models:** Fine-tune T5-base, BART-base, and mT5-base; compare answer-aware vs answer-agnostic. Consider base/large size ablations and light PEFT/LoRA for resource constraints.
- **Input formatting:** Stable prompts with `<context>` and `<answer>` (highlight markers for answer-aware). Train with cross-entropy + label smoothing; early stopping on validation ROUGE-L; small hyperparameter grids (LR 5e-5–2e-4, batch 8–16 with accumulation, max length 512–768).
- **Decoding:** Compare beam search (beam 4–8, length penalty 1.0–1.2, n-gram blocking) against top-p sampling (p≈0.9, temperature 0.7–1.0); keep common settings for fair comparison.
- **Metrics:** ROUGE/BLEU/BERTScore for lexical/semantic quality; QG→QA EM/F1 for answerability; MOS for human fluency/relevance. Track compute (latency, memory) and robustness across seeds.

## Experimental plan & hypotheses
1. **Baselines:** Reproduce T5-base and BART-base on EN (answer-aware) with fixed seeds/splits and shared input template; log decoding settings. Add mT5-base zero-shot/translate-test for UA.
2. **UA adaptation:** Fine-tune mT5-base on UA corpus; optionally fine-tune T5/BART variants if UA data permit.
3. **Answer-aware vs answer-agnostic:** Controlled comparison under identical decoding; measure ΔROUGE-L and QG→QA shifts.
4. **Model size and context length:** Ablate base vs large variants; test alternative context windows and prompt templates.
5. **Decoding sensitivity:** Beam vs sampling sweeps; analyze impact on fluency and answerability.
6. **Human evaluation:** MOS on fluency/relevance/answerability; compute inter-rater agreement (e.g., Cohen’s κ or Krippendorff’s α).

**Hypotheses:**
- Answer-aware improves ROUGE-L by ≥5% and QG→QA by ≥5 pp over answer-agnostic.
- Fine-tuned mT5 on UA performs within |ΔROUGE-L| ≤ 2% of EN-focused models on UA data after adaptation.
- ROUGE/BERTScore correlate with QG→QA utility (r ≥ 0.6) when low-quality samples are filtered.

## Practice outcomes & risks (from draft)
- Agreed topic, objectives, object/subject defined; initial relevance quantified; protocol sketched (train/val/test, metrics ROUGE/BLEU/BERTScore, QG→QA, MOS).
- **Risks:** Limited UA data quality, GPU scarcity, schedule slips. **Mitigations:** Alternate/augmented corpora, PEFT/LoRA and 8/4-bit, concise supervisor check-ins, strict logging/versioning.

## Near-term action items
- Finalize corpus choices and data cards; implement preprocessing to build `data/artifacts/{train,val,test}.jsonl` with highlighted answers.
- Wire training/eval pipeline (config-driven) for T5/BART/mT5 with answer-aware templates and decoding controls.
- Add evaluation harness for ROUGE/BLEU/BERTScore plus QG→QA and MOS data collection workflow.
- Document reproducibility: seeds, hyperparameters, decoding settings, data splits, and experiment logging.
