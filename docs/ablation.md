# Decoding ablation summary

This experiment compares beam search and nucleus sampling (top-p) for bilingual question generation across answer-aware and answer-agnostic modes.

## Grids and setup
- Beam: `num_beams ∈ {1,4,6,8}`, `length_penalty ∈ {1.0, 1.1}`.
- Top-p: `top_p ∈ {0.9, 0.95}`, `temperature ∈ {0.7, 0.9}`.
- Shared: `no_repeat_ngram_size=3`, `repetition_penalty=1.1`, `max_new_tokens=32`, `min_new_tokens=4`.
- WH control is optional; constraints are injected via XML prompts when a `wh_type` is provided.

## Best-performing configurations
| Language | Mode | Strategy | Params | Notes |
| --- | --- | --- | --- | --- |
| EN | aware | beam | `num_beams=6`, `length_penalty=1.1` | Best ROUGE-L/EM/F1 on validation, stable wh-match. |
| EN | agnostic | top-p | `top_p=0.95`, `temperature=0.9` | Higher distinct-2 and better wh coverage; requires QA filter. |
| UA | aware | beam | `num_beams=8`, `length_penalty=1.0` | Strongest EM/F1 with lowest invalid rate. |
| UA | agnostic | top-p | `top_p=0.9`, `temperature=0.7` | Balanced BLEU and diversity, resilient to repetition. |

## Findings
- Beam search consistently improves answer-aware generation fidelity (ROUGE-L/EM/F1) compared to nucleus sampling.
- Top-p sampling boosts question diversity (distinct-2) and wh-type coverage for answer-agnostic mode but needs tighter QG→QA filtering to keep pass_rate steady.
- For Ukrainian, multilingual QA checkpoints remain necessary for reliable QG→QA evaluation; sensitivity to temperature is lower than in English.
