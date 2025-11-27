# Task Breakdown and Status

## Completed / documented
- Topic, objectives, object/subject defined and mapped to diploma chapters.
- Analytical context compiled (relevance, baseline methods, metrics, risks) in `docs/project_plan.md`.
- Repository scaffold in place (configs directory, package stub) with plan linkage from README.

## Immediate priority — dataset selection (per assignment first step)
1. **Choose English corpus**: default SQuAD 2.0; confirm license and include handling of unanswerable cases.
2. **Choose Ukrainian corpus**: select from available Q&A sets (e.g., translated SQuAD variants, UA educational/encyclopedic sources); document license and coverage.
3. **Define inclusion rules**: deduplication, min/max length thresholds, removal of malformed markup, and explicit handling of impossible-answer items.
4. **Fix train/val/test splits**: seeded, stratified by source to prevent context leakage; store splits list for reproducibility.
5. **Draft data cards**: sources, licensing, preprocessing steps, language stats, known limitations.

## Next implementation tasks (after datasets fixed)
- **Preprocessing pipeline**: script to normalize texts, highlight answers, and emit `data/artifacts/{train,val,test}.jsonl` with fields `context`, `answer`, `question`.
- **Config-driven training/eval CLI**: load YAML configs, prepare tokenized datasets, train T5/BART/mT5 (optionally LoRA), and run evaluation.
- **Metrics and QA loop**: integrate ROUGE/BLEU/BERTScore plus QG→QA (EM/F1) and MOS collection templates; log seeds/decoding parameters.
- **Reproducibility artifacts**: seed registry, split manifests, and instructions in README for running preprocessing and training.

## Correlation with diploma outline
- **Chapter 1 (Relevance):** addressed via analytical context in `docs/project_plan.md`.
- **Chapter 2 (Data/Methods):** dataset selection + preprocessing rules + model/training setup (immediate focus on datasets).
- **Chapter 3 (Plan & hypotheses):** experimental roadmap and hypotheses already drafted; will be updated after dataset decision.
- **Chapter 4 (Results & next steps):** will capture implemented pipeline, metrics, and demo after experiments.
