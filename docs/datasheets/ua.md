# Datasheet: Ukrainian QA/QG Split (Translation/Collection)

## Назва та джерело
- **Датасет:** Український Q&A (переклад/збір) [3].
- **Джерела:** переклад EN прикладів та ручні доповнення (університетські/новинні тексти).
- **Ліцензія:** похідна робота; використання лише в дослідницьких цілях, поширення відповідно до вихідних ліцензій і перекладних обмежень.

## Опис вмісту
- `context`: український абзац із перекладеного або зібраного джерела.
- `question`: питання українською, узгоджене з `context`.
- `answer`: текстовий спан або порожній рядок, якщо `unanswerable=true`.
- `unanswerable`: прапорець відсутності відповіді в `context`.
- `title`: заголовок статті або тематичний маркер для стратифікації.

## Походження та ліцензія
- **Джерело даних:** машинний або ручний переклад SQuAD 2.0 та локальні українські джерела (новини/енциклопедія).
- **Обробка:** збереження сегментації, нормалізація Юнікоду (NFKC), уніфікація лапок/апострофів, очищення пробілів, дедуплікація та перевірка вирівнювання спанів.
- **Обмеження ліцензії:** використання лише в освітніх цілях; публікація похідних залежить від ліцензій оригінальних текстів.

## Статистика
- Кількість прикладів (train/val/test): 3<!--{{stats_ua.counts.train}}-->/4<!--{{stats_ua.counts.val}}-->/3<!--{{stats_ua.counts.test}}-->.
- Середні довжини слів (train/val/test):
  - context: 25.0<!--{{stats_ua.avg_len.context.train}}--> / 32.0<!--{{stats_ua.avg_len.context.val}}--> / 34.0<!--{{stats_ua.avg_len.context.test}}-->
  - question: 5.333333333333333<!--{{stats_ua.avg_len.question.train}}--> / 6.0<!--{{stats_ua.avg_len.question.val}}--> / 6.666666666666667<!--{{stats_ua.avg_len.question.test}}-->
  - answer: 2.3333333333333335<!--{{stats_ua.avg_len.answer.train}}--> / 1.75<!--{{stats_ua.avg_len.answer.val}}--> / 3.6666666666666665<!--{{stats_ua.avg_len.answer.test}}-->
- Медіанні довжини слів (train/val/test):
  - context: 25<!--{{stats_ua.median_len.context.train}}--> / 32.0<!--{{stats_ua.median_len.context.val}}--> / 34<!--{{stats_ua.median_len.context.test}}-->
  - question: 5<!--{{stats_ua.median_len.question.train}}--> / 6.0<!--{{stats_ua.median_len.question.val}}--> / 7<!--{{stats_ua.median_len.question.test}}-->
  - answer: 2<!--{{stats_ua.median_len.answer.train}}--> / 2.0<!--{{stats_ua.median_len.answer.val}}--> / 5<!--{{stats_ua.median_len.answer.test}}-->
- Частка `unanswerable`: train 0.0<!--{{stats_ua.unanswerable_share.train}}-->, val 0.25<!--{{stats_ua.unanswerable_share.val}}-->, test 0.3333333333333333<!--{{stats_ua.unanswerable_share.test}}-->.
- Частка span-misaligned до фільтрів: train 0.0<!--{{stats_ua.span_fail_rate.train}}-->, val 0.0<!--{{stats_ua.span_fail_rate.val}}-->, test 0.0<!--{{stats_ua.span_fail_rate.test}}-->.

## Нормалізація й фільтри
- NFKC, уніфікація лапок/апострофів (UA-стиль «» та ’ → стандартизовані символи), видалення зайвих пробілів.
- Контроль довжин і порожніх рядків; QA-сумісність спану та контексту.
- Дедуплікація та вирівнювання перекладів з англійськими паралелями.
- Стратифікація за `title` з збереженням частки unanswerable та тематичного балансу.

## Етичні аспекти
- Анонімізація: імена/персональні дані виключаються на етапі підготовки.
- Токсичність/NLI: фільтрація згідно з `configs/safety.yaml`, логування причин.
- Політика контекст-only: генерація та QA використовують лише дані з `context`.
- Документація: дата-картка містить обмеження, призначення — дослідницьке тестування.

## Відомі обмеження та упередження
- Переклад може спотворювати відтінки значень; можливі синтаксичні кальки з англійської.
- Домени змішані (новини/енциклопедія); відсутня розмова та низька представленість діалектів.
- Питання здебільшого фактологічні; мало reasoning/агрегованих запитань.

## Версіонування
- Seed: 42; журнал у `data/scripts/prepare_ua.py` та `data/prep_logs`.
- Git-hash і версії скриптів фіксуються при підготовці; моделі mT5/HF Transformers [3][4].
- Локальні перекладацькі модулі документовані у README перекладного конвеєра.

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
