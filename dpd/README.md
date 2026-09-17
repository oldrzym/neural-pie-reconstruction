# DPD для PIE реконструкции

[DPD (Dual-Path Decoder)](https://github.com/cmu-llab/dpd) — архитектура из ACL 2024.

## Установка

```bash
cd dpd
pip install -r requirements.txt
```

> Важно: `dpd/repo` не хранится в этом Git-репозитории (он в `.gitignore`).
> При первом запуске на сервере можно автоматически подтянуть его через `--setup-repo`.

## Быстрый старт

### 1. Обучение (одна команда)

```bash
cd dpd

# Первый запуск на новом сервере (авто-клон + авто-патч exp.py)
python train.py --split iecor_kaikki_koebler --setup-repo --gpu 0

# Supervised baseline (GRU)
python train.py --split iecor_kaikki_koebler

# Supervised baseline (Transformer)
python train.py --split iecor_kaikki_koebler --arch Transformer --batch-size 32

# Full DPD (лучшая стратегия)
python train.py --split iecor_kaikki_koebler --strat pimodel_bpall_cringe --arch GRU --batch-size 32

# Linux server: зафиксировать конкретный GPU
python train.py --split iecor_kaikki_koebler --gpu 0

# Быстрый тест (CPU, 5 эпох)
python train.py --split iecor_kaikki_koebler --cpu --epochs 5 --dev
```

`train.py` автоматически:
- При `--setup-repo`: клонирует `dpd/repo` (если отсутствует) и патчит `exp.py`
- Конвертирует CSV в DPD pickle формат
- Запускает `exp.py` с правильными параметрами
- Сохраняет чекпоинты в `repo/checkpoints/<dataset>_<strat>_<arch>/`
- Сохраняет конфиг рядом для загрузки потом
- Не требует интерактивного ввода (без `input()`)

### 2. Оценка модели

```bash
cd dpd

# Оценка лучшего чекпоинта на тесте
python evaluate.py --checkpoint-dir repo/checkpoints/pie_iecor_kaikki_koebler_supervised_only_GRU

# Оценка с предсказаниями
python evaluate.py --checkpoint-dir <path> --predictions

# Сохранить предсказания в CSV
python evaluate.py --checkpoint-dir <path> --predictions --output results.csv

# Оценка на валидации
python evaluate.py --checkpoint-dir <path> --split val
```

## Полный пайплайн (пошагово)

Если хочется запустить руками:

### Шаг 1: Конвертация данных

```bash
python ../dataset/scripts/convert_to_dpd.py \
    --input-dir ../dataset/splits/iecor_kaikki_koebler/ \
    --output-dir ./repo/data/pie_iecor_kaikki_koebler/
```

### Шаг 2: Обучение напрямую через exp.py

```bash
cd repo

python exp.py \
    --dataset pie_iecor_kaikki_koebler \
    --architecture GRU \
    --strat supervised_only \
    --proportion_labelled 1.0 \
    --exclude_unlabelled \
    --max_epochs 200 \
    --batch_size 64 \
    --lr 0.0003 \
    --d2p_dropout_p 0.3 \
    --early_stopping_patience 30 \
    --check_val_every_n_epoch 5 \
    --skip_daughter_tone True \
    --skip_protoform_tone True \
    --d2p_inference_decode_max_length 30 \
    --p2d_inference_decode_max_length 30 \
    --nowandb
```

### Шаг 3: Оценка

```bash
cd dpd
python evaluate.py --checkpoint-dir repo/checkpoints/pie_iecor_kaikki_koebler_supervised_only_GRU --predictions
```

## Параметры train.py

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `--split` | (обязательный) | Поддиректория в `dataset/splits/` |
| `--no-convert` | - | Не конвертировать данные, использовать существующие pickle |
| `--force-reconvert` | - | Принудительно пересоздать pickle даже если они уже есть |
| `--setup-repo` | - | Автоклон `dpd/repo` и авто-патч `exp.py` при отсутствии |
| `--repo-url` | `https://github.com/cmu-llab/dpd.git` | URL репозитория для `--setup-repo` |
| `--arch` | GRU | `GRU` или `Transformer` |
| `--strat` | supervised_only | Стратегия обучения |
| `--epochs` | 200 | Макс. эпох |
| `--batch-size` | 64 | Размер батча |
| `--lr` | 0.0003 | Learning rate |
| `--dropout` | 0.3 | Dropout |
| `--proportion-labelled` | 1.0 | Доля размеченных (авто 0.3 для semi-supervised) |
| `--patience` | 30 | Early stopping patience |
| `--val-every` | 5 | Валидация каждые N эпох |
| `--max-decode-len` | 30 | Макс. длина декодирования PIE форм |
| `--cpu` | - | Обучение на CPU |
| `--gpu N` | -1 (auto) | Индекс GPU |
| `--seed` | -1 (random) | Фиксированный seed |
| `--dev` | - | Режим отладки |
| `--dry-run` | - | Показать команду, не запускать |

## Стратегии обучения

| Стратегия | Описание | proportion_labelled |
|-----------|----------|---------------------|
| `supervised_only` | Базовый seq2seq | 1.0 |
| `pimodel` | Pi-model (consistency loss) | 0.1-0.5 |
| `bpall_cringe` | Backprop-all + CRINGE | 0.1-0.5 |
| `pimodel_bpall_cringe` | **Full DPD** (лучшая) | 0.1-0.5 |

Для `supervised_only` используется `proportion_labelled=1.0` + `exclude_unlabelled`.
Для остальных — автоматически ставится 0.3 (можно менять через `--proportion-labelled`).

## Метрики

| Метрика | Описание |
|---------|----------|
| `accuracy` | Exact Match (полное совпадение) |
| `phoneme_edit_distance` | Edit Distance в фонемах |
| `phoneme_error_rate` | PER (= 1 - CharAccuracy) |
| `feature_error_rate` | FER (на уровне фонологических признаков) |
| `bcubed_f_score` | BCubed F-Score |

## Доступные датасеты

| Сплит | Train | Описание |
|-------|-------|----------|
| `iecor_kaikki_koebler` | 2,880 | IE-CoR + Kaikki + Koebler |
| `combined_all` | 5,102 | Все источники |
| `iecor_only` | ~1,276 | Только IE-CoR |

## Структура файлов

```
dpd/
  train.py             # Запуск обучения (wrapper)
  evaluate.py          # Оценка чекпоинта
  requirements.txt     # Зависимости
  README.md            # Эта инструкция
  repo/                # Клон DPD (пропатченный)
    exp.py             # Основной скрипт (пропатчен: checkpoints + config)
    data/              # Pickle-данные (создаются автоматически)
    checkpoints/       # Сохраненные модели
      pie_<split>_<strat>_<arch>/
        best-*.ckpt    # Лучший чекпоинт
        last.ckpt      # Последний чекпоинт
        config.pkl     # Конфиг для загрузки
```

## Формат данных

DPD pickle:
```python
(langs_list, data)
# langs_list = ["PIE", "lat", "grc", "san", ...]
# data = {
#     "cognate_id": {
#         "daughters": {"lat": ["c","a","n","i","s",""], ...},
#         "protoform": {"PIE": ["k","u","o","n",""]}
#     }
# }
```

> Пустая строка `""` в конце — placeholder для тона (PIE не имеет тонов, используем `skip_protoform_tone=True`).

## Troubleshooting

**Обучение завершилось, но нет чекпоинтов**
- Проверь что `exp.py` пропатчен (должен быть `CHECKPOINT_DIR` и `dirpath` в ModelCheckpoint)
- Проверь `repo/checkpoints/` — там должна быть папка с названием датасета

**OOM на GPU**
- Уменьши `--batch-size` (для DPD стратегий нужно 16-32)
- Используй `--arch GRU` вместо Transformer

**getfreegpu ошибка на Windows**
- Используй `--gpu 0` для явного указания GPU, или `--cpu`

**Linux сервер: выбор GPU**
- Рекомендуется явно передавать `--gpu 0` (или нужный индекс), чтобы избежать ошибок auto-detect в нестандартных окружениях.

**Валидация не запускается**
- `--val-every` контролирует частоту, по умолчанию 5 эпох
- Для отладки используй `--dev` (валидация каждую эпоху)
