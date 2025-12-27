# Datasheet: English QA/QG Split (SQuAD 2.0)

## Назва та джерело
- **Датасет:** SQuAD 2.0 (EN) [6].
- **Джерело:** https://rajpurkar.github.io/SQuAD-explorer/
- **Ліцензія:** CC BY-SA 4.0 (оригінал), похідні матеріали в цьому репозиторії зберігають ту саму ліцензію.

## Опис вмісту
- `context`: абзац або його частина, з якого можна відповісти на питання.
- `question`: сформульоване питання, що спирається на `context`.
- `answer`: текстовий спан (може бути порожнім, якщо `unanswerable=true`).
- `unanswerable`: булевий прапорець, що вказує на відсутність відповіді в `context`.
- `title`: назва статті/джерела для стратифікації.

## Походження та ліцензія
- **Джерело даних:** оригінальні пари context–question–answer з SQuAD 2.0 [6].
- **Обробка:** нормалізація Юнікоду (NFKC), уніфікація лапок/апострофів, очистка пробілів та пунктуації, дедуплікація за `(context, question)`, фільтрація некоректних спанів.
- **Ліцензія:** CC BY-SA 4.0; збережено attribution.

## Статистика
- Кількість прикладів (train/val/test): 3<!--{{stats_en.counts.train}}-->/4<!--{{stats_en.counts.val}}-->/3<!--{{stats_en.counts.test}}-->.
- Середні довжини слів (train/val/test):
  - context: 25.0<!--{{stats_en.avg_len.context.train}}--> / 32.0<!--{{stats_en.avg_len.context.val}}--> / 34.0<!--{{stats_en.avg_len.context.test}}-->
  - question: 5.333333333333333<!--{{stats_en.avg_len.question.train}}--> / 6.0<!--{{stats_en.avg_len.question.val}}--> / 6.666666666666667<!--{{stats_en.avg_len.question.test}}-->
  - answer: 2.3333333333333335<!--{{stats_en.avg_len.answer.train}}--> / 1.75<!--{{stats_en.avg_len.answer.val}}--> / 3.6666666666666665<!--{{stats_en.avg_len.answer.test}}-->
- Медіанні довжини слів (train/val/test):
  - context: 25<!--{{stats_en.median_len.context.train}}--> / 32.0<!--{{stats_en.median_len.context.val}}--> / 34<!--{{stats_en.median_len.context.test}}-->
  - question: 5<!--{{stats_en.median_len.question.train}}--> / 6.0<!--{{stats_en.median_len.question.val}}--> / 7<!--{{stats_en.median_len.question.test}}-->
  - answer: 2<!--{{stats_en.median_len.answer.train}}--> / 2.0<!--{{stats_en.median_len.answer.val}}--> / 5<!--{{stats_en.median_len.answer.test}}-->
- Частка `unanswerable`: train 0.0<!--{{stats_en.unanswerable_share.train}}-->, val 0.25<!--{{stats_en.unanswerable_share.val}}-->, test 0.3333333333333333<!--{{stats_en.unanswerable_share.test}}-->.
- Частка span-misaligned до фільтрів: train 0.0<!--{{stats_en.span_fail_rate.train}}-->, val 0.0<!--{{stats_en.span_fail_rate.val}}-->, test 0.0<!--{{stats_en.span_fail_rate.test}}-->.

## Нормалізація й фільтри
- NFKC + заміна “/” лапок і ’ апострофів на єдиний формат, конденсація пробілів.
- Перевірка довжин: context/question/answer (мін/макс), відсікання порожніх рядків.
- Дедуплікація exact-match та casefold, логування відкинутих прикладів.
- Стратифікація за `title` з збереженням груп і unanswerable частки.

## Етичні аспекти
- Особисті дані не збираються; публічні тексти зберігають атрибуцію.
- Токсичний контент відсіюється детектором і лексиконами (див. `configs/safety.yaml`).
- Політика: QA/NLI/токсичність працюють лише над `context`; генерація без довільних фактів.
- Підготовлено дата-картку з зазначенням обмежень і застосуванням лише для дослідних цілей.

## Відомі обмеження та упередження
- Новинні та енциклопедичні тексти, можливий доменний зсув до більш формальних стилів.
- Питання орієнтовані на фактологічні відповіді; мало reasoning/дистантних запитань.
- Баланс WH-типів успадкований від SQuAD 2.0; можливі перекоси щодо “what/when”.

## Версіонування
- Seed: 42; дата побудови за логами `data/scripts/prepare_en.py`.
- Git-hash репозиторію й версії скриптів фіксуються у prep-логах (`data/prep_logs`).
- Версії бібліотек: Hugging Face Transformers [4], Datasets [5], моделі T5/BART [1][2].

**Параметри відтворення:**

**Repro:**

```
uv run python data/scripts/prepare_en.py --out-dir data/artifacts/en --stats data/stats_en.json --seed 42
uv run python data/scripts/prepare_ua.py --in-dir data/artifacts/en --out-dir data/artifacts/ua --stats data/stats_ua.json --seed 42
```

## Джерела
[1] T5 – https://arxiv.org/abs/1910.10683  
[2] BART – https://arxiv.org/abs/1910.13461  
[3] mT5 – https://arxiv.org/abs/2010.11934  
[4] Hugging Face Transformers – https://huggingface.co/docs/transformers  
[5] Hugging Face Datasets – https://huggingface.co/docs/datasets  
[6] ROUGE – https://aclanthology.org/W04-1013/
