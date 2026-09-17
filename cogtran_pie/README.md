# CogTran PIE Pipeline

Отдельный пайплайн для обучения и оценки Cognate Transformer на CSV-сплитах PIE.

## 1. Быстрый старт на Linux сервере

Запускать из корня репозитория `PIE-reconstruction`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r cogtran_pie/requirements.txt
python3 cogtran_pie/train.py --setup-repo --output-dir cogtran_pie/runs/bootstrap --dry-run
```

Важно: не ставь `cogtran_pie/repo/requirements.txt` на новом `pip`.
В апстрим-файле пин `lingpy==2.6.9` с битыми метаданными.
Если видишь `NotImplementedError` из `CharacterTokenizer.get_vocab`, обнови репозиторий (`git pull`): в пайплайне есть совместимый tokenizer.
Если уже поставился `transformers>=5`, переустанови зависимости из `cogtran_pie/requirements.txt` (в этом файле зафиксирована совместимая ветка `<5`).

Дальше `train.py --setup-repo` сам клонирует CognateTransformer в `cogtran_pie/repo`.

## 2. Обучение

```bash
python3 cogtran_pie/train.py \
  --setup-repo \
  --train-file dataset/splits/iecor_kaikki_koebler_normalized/train.csv \
  --val-file dataset/splits/iecor_kaikki_koebler_normalized/val.csv \
  --test-file dataset/splits/iecor_kaikki_koebler_normalized/test.csv \
  --output-dir cogtran_pie/runs/pie_msat_run1 \
  --epochs 200 \
  --batch-size 32 \
  --eval-batch-size 64 \
  --lr 0.0005 \
  --dropout 0.2 \
  --min-unique-langs 2 \
  --gpu 0
```

Важно: в bash после `\` не должно быть пробела.

Если нужно только проверить пути/параметры без запуска:

```bash
python3 cogtran_pie/train.py --setup-repo --output-dir cogtran_pie/runs/tmp --dry-run
```

### Где будет модель

Вся модель и результаты лежат в `--output-dir`, например в `cogtran_pie/runs/pie_msat_run1`:

- веса и конфиг модели (`save_model`)
- tokenizer
- `train_config.json`
- `run_info.json`
- `val_predictions.csv`, `val_metrics.json`
- `test_predictions.csv`, `test_metrics.json` (если не указан `--skip-test-eval`)

## 3. Инференс и оценка

Оценка модели на любом CSV с колонками `input` и `output`:

```bash
python3 cogtran_pie/evaluate.py \
  --setup-repo \
  --model-dir cogtran_pie/runs/pie_msat_run1 \
  --input-file dataset/splits/iecor_kaikki_koebler_normalized/test.csv \
  --output-dir cogtran_pie/runs/pie_msat_run1/eval_test \
  --gpu 0
```

Что сохраняется:

- `predictions.csv`
- `metrics.json`
- `prep_stats.json`
- `eval_config.json`

Можно явно задать пути:

```bash
python3 cogtran_pie/evaluate.py \
  --model-dir cogtran_pie/runs/pie_msat_run1 \
  --input-file /path/to/another_test.csv \
  --output-dir cogtran_pie/runs/pie_msat_run1/eval_custom \
  --metrics-output cogtran_pie/runs/pie_msat_run1/eval_custom/my_metrics.json \
  --predictions-output cogtran_pie/runs/pie_msat_run1/eval_custom/my_preds.csv \
  --gpu 0
```

## 4. Метрики

Считаются метрики:

- `exact_match_accuracy`
- `character_accuracy`
- `mean_edit_distance`
- `mean_normalized_edit_distance`

## 5. Как дать другой датасет

Для обучения:

- `--train-file /path/train.csv`
- `--val-file /path/val.csv`
- `--test-file /path/test.csv`

Для оценки/inference после обучения:

- `--input-file /path/any_eval.csv`

Формат CSV: обязательны колонки `input` и `output`.

## 6. Что делает препроцессинг

- Парсит `input` вида `[lang] word [lang] word ...`
- Если язык повторяется, не теряет формы: делает `lang#1`, `lang#2`, ...
- Выравнивает формы через `lingpy` (или fallback-паддингом)
- Формирует маску цели для `PIE`: `"[PIE]|?|?|?..."`
- Нормализует target (`output`) и по умолчанию берет первую форму до запятой
