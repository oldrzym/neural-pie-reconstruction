# Synthesis & Validation Pipeline

Standalone скрипты для синтетики и валидации PIE когнатов через Wiktionary + GPT Batch API.

## Структура

```
synthesis_validation/
├── wiktionary_fetcher.py       # 🔧 Standalone Wiktionary парсер с retry логикой
├── extract_cognates.py         # 📊 Извлечение уникальных когнатов из датасета
├── fetch_all_etymologies.py    # 🌐 Парсинг этимологий для всех когнатов
└── README.md
```

## Pipeline: Шаг за шагом

### Шаг 1: Извлечь уникальные когнаты

```bash
cd dataset/scripts/synthesis_validation
python extract_cognates.py
```

**Результат:** `unique_cognates.json` со всеми уникальными когнатами из train_synt.csv

**Что делает:**
- Парсит все строки из train_synt.csv
- Извлекает все когнаты в формате `[lang] word`
- Сохраняет уникальные пары (lang, word)
- Группирует по языкам и PIE корням

### Шаг 2: Спарсить этимологии с Wiktionary

```bash
python fetch_all_etymologies.py
```

**Результат:** `wiktionary_etymologies.json` с этимологиями для каждого когната

**Что делает:**
- Читает unique_cognates.json
- Для каждого когната запрашивает этимологию с Wiktionary
- Обрабатывает HTTP 429 с exponential backoff
- Ретраит до 5 раз с увеличением задержки
- Сохраняет результаты в кэш
- Можно запускать несколько раз - пропускает уже закэшированные

**Особенности:**
- ⏱️ **Медленно:** ~1.5 секунды на слово (чтобы избежать 429)
- 🔄 **Можно прервать:** Кэш сохраняется каждые 50 слов
- ♻️ **Можно перезапускать:** Пропускает уже спарсенные слова
- 🛡️ **Retry логика:** Автоматический retry при 429 с backoff

**Примерное время:**
- 1000 когнатов ≈ 25 минут
- 5000 когнатов ≈ 2 часа
- 10000 когнатов ≈ 4 часа

### Шаг 3: Валидация через GPT Batch API

После парсинга этимологий используй основной скрипт валидации:

```bash
cd ../../..  # возврат в корень
python dataset/scripts/run_chunk.py 0 100
```

## Конфигурация Wiktionary Fetcher

### Rate Limiting (в [wiktionary_fetcher.py](wiktionary_fetcher.py:13))

```python
INITIAL_DELAY = 1.0           # Начальная задержка между запросами
MAX_DELAY = 30.0              # Максимальная задержка
CONCURRENT_REQUESTS = 1       # Количество параллельных запросов
BACKOFF_MULTIPLIER = 2.0      # Множитель для exponential backoff
MAX_RETRIES = 5               # Максимум попыток на каждое слово
```

**Рекомендации:**
- Не трогай эти настройки если не хочешь получить бан от Wiktionary
- Если видишь много HTTP 429 - увеличь INITIAL_DELAY до 2.0
- Если нужно быстрее - уменьши до 0.5 (рискованно)

### Поддерживаемые языки

**Все 40+ языков из датасета включены:**

✅ Germanic: eng, deu, nld, swe, nor, dan, isl, fri, **far** (добавлен), got
✅ Slavic: rus, pol, ces, slk, ukr, bel, bul, hbs, **mkd** (добавлен), slv, **sor** (добавлен)
✅ Romance: fra, ita, spa, por, ron, cat, **sar** (добавлен), **nea** (добавлен), old, lat
✅ Celtic: gle, cym, bre, **gla** (добавлен)
✅ Baltic: lit, lav
✅ Indo-Iranian: fas, hye, san, hin, **kas** (добавлен), **oss** (добавлен), kur
✅ Greek: ell, grc
✅ Albanian: sqi, aln
✅ Other: **wal** (добавлен), xcl, txb, xto

**Новые языки (которые были "[No mapping]"):**
- far = Faroese
- gla = Scottish Gaelic
- kas = Kashmiri
- mkd = Macedonian
- nea = Neapolitan
- oss = Ossetian
- sar = Sardinian
- sor = Sorbian
- wal = Walloon

## Использование Standalone

### Тестирование на нескольких словах

```python
from wiktionary_fetcher import WiktionaryFetcher
import asyncio

async def test():
    fetcher = WiktionaryFetcher()

    pairs = [
        ('ride', 'eng'),
        ('foot', 'eng'),
        ('reiten', 'deu'),
    ]

    results = await fetcher.fetch_batch(pairs)

    for key, etym in results.items():
        print(f"{key}: {etym[:100]}...")

asyncio.run(test())
```

### Получить одну этимологию

```python
from wiktionary_fetcher import WiktionaryFetcher
import asyncio

async def get_one():
    fetcher = WiktionaryFetcher()

    async with aiohttp.ClientSession() as session:
        fetcher.session = session
        etym = await fetcher.fetch_etymology('ride', 'eng')
        print(etym)

asyncio.run(get_one())
```

## Статус

- ✅ Wiktionary fetcher с retry логикой
- ✅ Все языки добавлены в маппинг
- ✅ Exponential backoff для 429
- ✅ Кэширование результатов
- ✅ Скрипт извлечения когнатов
- ✅ Скрипт парсинга всех этимологий
- ⏳ Ожидает запуска на полном датасете

## FAQ

**Q: Сколько времени займет парсинг всего датасета?**
A: Зависит от количества уникальных когнатов. Ориентировочно 2-4 часа на ~5000-10000 слов.

**Q: Можно ли прервать и продолжить?**
A: Да! Кэш сохраняется каждые 50 слов. Просто запусти снова - пропустит уже спарсенные.

**Q: Что делать если много HTTP 429?**
A: Перезапусти скрипт - он автоматически ретраит. Если не помогает, увеличь INITIAL_DELAY.

**Q: Можно ли распараллелить на несколько процессов?**
A: Не рекомендуется - Wiktionary банит за слишком частые запросы. Лучше запустить один процесс и подождать.

**Q: Как проверить прогресс?**
A: Открой wiktionary_etymologies.json и посмотри количество записей.

## Следующие шаги

1. ✅ **Создана структура** synthesis_validation/
2. ⏳ **Запустить:** `python extract_cognates.py`
3. ⏳ **Запустить:** `python fetch_all_etymologies.py` (долго!)
4. ⏳ **После парсинга:** Запустить валидацию чанками через run_chunk.py
