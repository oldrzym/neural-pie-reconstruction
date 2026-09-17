# PIE Reconstruction Training

Скрипты для обучения моделей реконструкции праиндоевропейских праформ.

## Доступные подходы

1. **ByT5/mT5** — претренированные seq2seq модели (простой baseline)
2. **DPD-BiReconstructor** — специализированная архитектура для прото-языков (CMU, ACL 2024)

---

## Установка на сервере

```bash
# 1. Подключиться к серверу
ssh user@server

# 2. Обновить pip
pip install --upgrade pip

# 3. Установить PyTorch (для CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Или для CUDA 12.1:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Установить остальные зависимости
pip install -r requirements.txt

# 5. (Опционально) Настроить wandb для логирования
wandb login
```

## Структура файлов

```
training/
├── config.yaml        # Конфиг обучения (гиперпараметры)
├── requirements.txt   # Зависимости
├── train.py          # Основной скрипт обучения
├── evaluate.py       # Оценка модели на тесте
├── inference.py      # Инференс (интерактивный режим)
├── data_module.py    # Загрузка данных
├── models.py         # Определения моделей
└── README.md         # Этот файл
```

## Подготовка данных

Перед обучением нужно создать splits:

```bash
cd ../dataset/scripts

# IE-CoR
python make_splits.py --input ../processed/iecor.csv --output-dir ../splits/iecor/

# Kaikki
python make_splits.py --input ../processed/kaikki.csv --output-dir ../splits/kaikki/

# Starling (если нужен)
python starling_parser.py --output ../processed/starling.csv
python make_splits.py --input ../processed/starling.csv --output-dir ../splits/starling/
```

## Обучение

### Базовый запуск

```bash
cd training

# ByT5-base (рекомендуется)
python train.py --config config.yaml

# С другой моделью
python train.py --config config.yaml --model google/byt5-small

# Без wandb
python train.py --config config.yaml --no-wandb
```

### Доступные модели

| Модель | Параметры | VRAM | Команда |
|--------|-----------|------|---------|
| ByT5-small | 300M | ~6GB | `--model google/byt5-small` |
| ByT5-base | 580M | ~12GB | `--model google/byt5-base` |
| mT5-small | 300M | ~6GB | `--model google/mt5-small` |
| mT5-base | 580M | ~12GB | `--model google/mt5-base` |
| T5-small | 60M | ~4GB | `--model t5-small` |
| T5-base | 220M | ~8GB | `--model t5-base` |

### Обучение на разных датасетах

Измени пути в `config.yaml`:

```yaml
data:
  train_file: "../dataset/splits/kaikki/train.csv"  # или iecor, starling
  val_file: "../dataset/splits/kaikki/val.csv"
  test_file: "../dataset/splits/kaikki/test.csv"
```

## Оценка модели

```bash
# Оценка на тестовом сете
python evaluate.py \
    --model ./outputs \
    --test ../dataset/splits/iecor/test.csv \
    --output ./results/iecor_test

# С другим batch size
python evaluate.py --model ./outputs --test test.csv --batch-size 16
```

## Инференс

```bash
# Одиночное предсказание
python inference.py \
    --model ./outputs \
    --input "[lat] canis [grc] kýōn [san] śvā́"

# Интерактивный режим
python inference.py --model ./outputs --interactive

# Несколько предсказаний (beam search)
python inference.py --model ./outputs --input "..." --top-k 5
```

## Конфигурация (config.yaml)

```yaml
# Основные параметры
model:
  name: "google/byt5-base"  # модель

training:
  num_epochs: 15           # эпохи
  batch_size: 8            # batch size
  learning_rate: 3.0e-4    # learning rate
  warmup_steps: 500        # warmup
  fp16: true               # mixed precision (быстрее)
```

## Метрики

- **Exact Match** — точное совпадение праформы
- **Character Accuracy** — посимвольная точность
- **Edit Distance** — расстояние Левенштейна
- **Normalized Edit Distance** — нормализованное расстояние

## Примерное время обучения

На Tesla T4 16GB:

| Модель | Датасет (~1.5k) | Время |
|--------|-----------------|-------|
| ByT5-small | IE-CoR | ~30 мин |
| ByT5-base | IE-CoR | ~1-2 часа |
| mT5-base | IE-CoR | ~1-2 часа |

## Troubleshooting

### CUDA out of memory
- Уменьшить `batch_size` в config.yaml
- Увеличить `gradient_accumulation_steps`
- Использовать меньшую модель (small вместо base)

### Медленное обучение
- Включить `fp16: true`
- Увеличить `batch_size` (если хватает памяти)

### Плохие результаты
- Увеличить `num_epochs`
- Попробовать другой `learning_rate` (1e-4, 5e-4)
- Проверить данные на ошибки

---

## DPD-BiReconstructor (альтернатива)

DPD — специализированная архитектура для реконструкции прото-языков от CMU (ACL 2024).
Показывает лучшие результаты на Romance и Chinese датасетах.

**Репозиторий:** https://github.com/cmu-llab/dpd

### Установка DPD

```bash
# Создать отдельное окружение
conda create --name dpd python=3.10.13 --yes
conda activate dpd

# Установить зависимости
pip install editdistance einops torch transformers pytorch-lightning wandb scikit-learn scipy
pip install panphon@git+https://github.com/dmort27/panphon.git@6acd3833743a49e63941a0b740ee69eae1dafc1c

# Клонировать репозиторий
git clone https://github.com/cmu-llab/dpd.git
cd dpd
```

### Конвертация данных для DPD

DPD ожидает pickle-формат. Используй конвертер:

```bash
cd ../dataset/scripts

# Конвертировать IE-CoR splits
python convert_to_dpd.py \
    --input-dir ../splits/iecor/ \
    --output-dir ../splits/iecor_dpd/

# Или один файл
python convert_to_dpd.py \
    --input ../splits/iecor/train.csv \
    --output ../splits/iecor_dpd/train.pickle
```

### Обучение DPD

```bash
cd dpd

# Скопировать данные
cp -r ../dataset/splits/iecor_dpd data/pie_iecor

# Настроить wandb
cp .env.example .env
# Отредактировать .env: WANDB_ENTITY=your_entity, WANDB_PROJECT=pie-reconstruction

# Запустить обучение (пример команды)
python exp.py \
    --dataset=pie_iecor \
    --architecture=Transformer \
    --n_epochs=100 \
    --batch_size=32 \
    --lr=1e-4 \
    --d_model=256 \
    --n_heads=8 \
    --n_layers=4
```

### Формат данных DPD

DPD ожидает pickle-файл с структурой:

```python
(langs_list, data) = pickle.load(f)

# langs_list = ["PIE", "lat", "grc", "san", ...]
# data = {
#     "cognate_id": {
#         "daughters": {
#             "lat": ["c", "a", "n", "i", "s", ""],  # последний элемент - тон
#             "grc": ["k", "ý", "ō", "n", ""],
#         },
#         "protoform": {
#             "PIE": ["k̑", "u̯", "ó", "n", ""]
#         }
#     }
# }
```

### Сравнение подходов

| Подход | Плюсы | Минусы |
|--------|-------|--------|
| **ByT5/mT5** | Простая установка, претренирован на Unicode | Не специализирован для лингвистики |
| **DPD** | SOTA на прото-реконструкции, учитывает фонетику | Сложнее настройка, нужен wandb |

---

## Ссылки

- **DPD Paper:** [Semisupervised Neural Proto-Language Reconstruction](https://arxiv.org/abs/2406.05930) (ACL 2024)
- **IE-CoR:** [Indo-European Cognate Relationships](https://github.com/lexibank/iecor)
- **ByT5:** [ByT5: Towards a token-free future with pre-trained byte-to-byte models](https://arxiv.org/abs/2105.13626)
