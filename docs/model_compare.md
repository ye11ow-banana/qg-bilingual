# Модельне порівняння (BART-base vs mT5-base)

Цей файл генерується скриптом `python -m src.qg_bilingual.eval.compare_models` на основі
`metrics_val.json` та `qg2qa_val.json` для кожного експерименту. Після запусків таблиці
оновляться автоматично; нижче наведені місця для зведень.

## Абсолютні метрики
| model | mode | lang | rouge1 | rouge2 | rougeL | bleu | em | f1 | pass_rate | avg_q_len | wh_dist |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| _заповнюється compare_models.py_ |

## Δ до T5-aware
| model | mode | lang | delta_rougeL_to_t5_aware |
| --- | --- | --- | --- |
| _заповнюється compare_models.py_ |

## Короткі висновки
1. Використовуйте `runs/mt5_base_aware_en` як референс для EN і `runs/mt5_base_aware_ua` для UA.
2. Aware-режим очікувано покращує ROUGE-L приблизно на 5 п.п. проти agnostic для EN; для UA
   різниця залежить від домену даних.
3. QA-перевірка (`qg→qa`) повинна показувати збалансовані EM/F1 та pass-rate; просадка pass-rate
   сигналізує про надмірно загальні або невизначені питання.

