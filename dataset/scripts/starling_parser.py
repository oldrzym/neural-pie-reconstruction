"""
Starling PIET Parser — парсит HTML страницы Tower of Babel.

Выходной формат CSV:
    input: "[san] ájra- [arm] art [grc] agró-s [lat] ager"
    output: "*ag'r-o-"
    meaning: "field"
    num_cognates: 4

ВНИМАНИЕ: Парсинг занимает ~3-5 минут (159 страниц с задержкой 1 сек).
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional
import argparse
import time
import re


# Маппинг названий полей в Starling → код языка
LANG_FIELD_MAP = {
    "Old Indian": "san",
    "Avestan": "ave",
    "Other Iranian": "ira",
    "Armenian": "hye",
    "Old Greek": "grc",
    "Slavic": "sla",
    "Baltic": "bal",
    "Germanic": "gem",
    "Latin": "lat",
    "Other Italic": "itc",
    "Celtic": "cel",
    "Albanian": "sqi",
    "Hittite": "hit",
    "Tokharian": "txb",
}


def fetch_page(page_num: int, delay: float = 1.0) -> str:
    """
    Скачивает страницу Starling.

    Args:
        page_num: номер страницы (1-based)
        delay: задержка перед запросом (вежливый парсинг)

    Returns:
        HTML содержимое страницы
    """
    first = (page_num - 1) * 20 + 1
    url = f"https://starlingdb.org/cgi-bin/response.cgi?root=config&basename=/data/ie/piet&first={first}"

    time.sleep(delay)

    response = requests.get(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) PIE-Reconstruction-Research"
    }, timeout=30)

    return response.text


def clean_cognate(text: str) -> str:
    """
    Очищает когнат от мусора.

    Пример:
        "ájra- m. `field, plain'" → "ájra"
        "ágō `treiben'; dor. agnéō" → "ágō"
    """
    # 1. Убираем перевод в обратных кавычках `...'
    text = re.sub(r"`[^']*'", "", text)
    # 2. Убираем комментарии в фигурных скобках {..}
    text = re.sub(r"\{[^}]*\}", "", text)
    # 3. Убираем скобки с содержимым (...)
    text = re.sub(r"\([^)]*\)", "", text)
    # 4. Берём первое слово до запятой/точки с запятой
    text = re.split(r'[,;]', text)[0]
    # 5. Убираем грамматические пометки (m., f., n., c., pl.)
    text = re.sub(r'\s+[mfncpl]+\.?\s*$', '', text)
    text = re.sub(r'\s+[mfn]\./[mfn]\.\s*$', '', text)
    # 6. Убираем дефис в конце (суффикс-маркер)
    text = re.sub(r'-+$', '', text)
    # 7. Убираем дефис в начале (префикс-маркер)
    text = re.sub(r'^-+', '', text)
    return text.strip()


def parse_record(record_div) -> Optional[Dict]:
    """
    Парсит один <div class="results_record">.

    Returns:
        Dict with {protoform, meaning, cognates} or None if invalid
    """
    fields = {}

    for div in record_div.find_all("div", recursive=False):
        span_fld = div.find("span", class_="fld")
        if not span_fld:
            continue

        field_name = span_fld.get_text(strip=True).rstrip(":")
        span_unicode = div.find("span", class_="unicode")

        if span_unicode:
            field_value = span_unicode.get_text(strip=True)
            fields[field_name] = field_value

    # Извлекаем праформу
    protoform = fields.get("Proto-IE", "")
    if not protoform:
        return None

    # Нормализуем праформу
    if not protoform.startswith("*"):
        protoform = f"*{protoform}"

    # Извлекаем значение
    meaning = fields.get("Meaning", "")

    # Извлекаем когнаты
    cognates = []
    for field_name, lang_code in LANG_FIELD_MAP.items():
        if field_name in fields:
            word = clean_cognate(fields[field_name])

            if word and len(word) > 1:  # минимум 2 символа
                cognates.append({
                    "lang": field_name,
                    "lang_code": lang_code,
                    "word": word[:50],  # ограничиваем длину
                })

    if not cognates:
        return None

    return {
        "protoform": protoform,
        "meaning": meaning,
        "cognates": cognates,
    }


def parse_page(html: str) -> List[Dict]:
    """Парсит все записи на странице."""
    soup = BeautifulSoup(html, "html.parser")
    records = []

    for record_div in soup.find_all("div", class_="results_record"):
        parsed = parse_record(record_div)
        if parsed:
            records.append(parsed)

    return records


def parse_all_starling(
    num_pages: int = 159,
    delay: float = 1.0,
    use_cache: bool = True,
    cache_dir: Optional[Path] = None
) -> List[Dict]:
    """
    Скачивает и парсит все страницы Starling.

    Args:
        num_pages: количество страниц (159 по умолчанию)
        delay: задержка между запросами
        use_cache: использовать кэш
        cache_dir: папка для кэша

    Returns:
        List of parsed records
    """
    all_records = []

    if cache_dir is None:
        cache_dir = Path("../raw/starling_cache")

    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)

    for page_num in tqdm(range(1, num_pages + 1), desc="Fetching Starling pages"):
        cache_file = cache_dir / f"page_{page_num}.html"

        if use_cache and cache_file.exists():
            html = cache_file.read_text(encoding="utf-8")
        else:
            html = fetch_page(page_num, delay=delay)
            if use_cache:
                cache_file.write_text(html, encoding="utf-8")

        records = parse_page(html)
        all_records.extend(records)

    return all_records


def format_input(cognates: List[Dict]) -> str:
    """Форматирует список когнатов в строку."""
    parts = []
    for cog in cognates:
        parts.append(f"[{cog['lang_code']}] {cog['word']}")
    return " ".join(parts)


def to_training_format(
    parsed_data: List[Dict],
    min_cognates: int = 1
) -> pd.DataFrame:
    """
    Конвертирует в DataFrame для обучения.
    """
    rows = []

    for item in tqdm(parsed_data, desc="Formatting training data"):
        cognates = item["cognates"]
        if len(cognates) < min_cognates:
            continue

        input_str = format_input(cognates)

        rows.append({
            "input": input_str,
            "output": item["protoform"],
            "meaning": item["meaning"],
            "num_cognates": len(cognates),
        })

    return pd.DataFrame(rows)


def main():
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    parser = argparse.ArgumentParser(description="Parse Starling PIET database")
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("../processed/starling.csv"),
        help="Output CSV path"
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=159,
        help="Number of pages to parse (default: 159 = all)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between requests in seconds"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Don't use cache"
    )
    parser.add_argument(
        "--min-cognates",
        type=int,
        default=1,
        help="Minimum number of cognates per group"
    )

    args = parser.parse_args()

    print(f"Parsing Starling PIET ({args.pages} pages)...")
    print(f"Delay: {args.delay}s between requests")

    parsed = parse_all_starling(
        num_pages=args.pages,
        delay=args.delay,
        use_cache=not args.no_cache
    )
    print(f"Total records parsed: {len(parsed)}")

    print("Converting to training format...")
    df = to_training_format(parsed, min_cognates=args.min_cognates)

    print(f"Final dataset size: {len(df)}")

    # Статистика
    print(f"\nStatistics:")
    print(f"  Mean cognates per group: {df['num_cognates'].mean():.1f}")
    print(f"  Median cognates per group: {df['num_cognates'].median():.1f}")
    print(f"  Min cognates: {df['num_cognates'].min()}")
    print(f"  Max cognates: {df['num_cognates'].max()}")

    # Сохраняем
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding='utf-8')
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
