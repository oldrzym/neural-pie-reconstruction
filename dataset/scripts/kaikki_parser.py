"""
Kaikki.org / Wiktionary PIE Parser — конвертирует JSONL в формат для обучения.

Выходной формат CSV:
    input: "[lat] māter [grc] mḗtēr [san] mātár- [arm] mayr"
    output: "*méh₂tēr"
    meaning: "mother"
    pos: "noun"
    num_cognates: 4
"""

import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional, Set
import argparse


# Языки которые нас интересуют (современные и древние ИЕ языки)
TARGET_LANGUAGES = {
    # Germanic
    "English", "German", "Dutch", "Danish", "Swedish", "Norwegian", "Icelandic",
    "Gothic", "Old English", "Old High German", "Old Norse", "Old Saxon",
    "Middle English", "Middle High German", "Middle Dutch",
    # Romance
    "Latin", "French", "Italian", "Spanish", "Portuguese", "Romanian",
    "Old French", "Vulgar Latin", "Old Spanish",
    # Slavic
    "Russian", "Polish", "Czech", "Bulgarian", "Serbian", "Croatian",
    "Old Church Slavonic", "Proto-Slavic", "Ukrainian", "Belarusian",
    # Baltic
    "Lithuanian", "Latvian", "Old Prussian",
    # Greek
    "Ancient Greek", "Greek", "Mycenaean Greek",
    # Indo-Iranian
    "Sanskrit", "Vedic", "Hindi", "Persian", "Avestan", "Pashto",
    "Old Persian", "Middle Persian",
    # Armenian
    "Armenian", "Old Armenian",
    # Albanian
    "Albanian",
    # Celtic
    "Irish", "Welsh", "Old Irish", "Breton", "Scottish Gaelic",
    "Middle Welsh", "Old Welsh", "Gaulish",
    # Anatolian
    "Hittite", "Luwian", "Lycian", "Palaic",
    # Tocharian
    "Tocharian A", "Tocharian B",
}


def get_lang_code(lang_name: str) -> str:
    """Конвертирует название языка в короткий код."""
    lang_codes = {
        "English": "eng", "German": "deu", "Dutch": "nld",
        "Danish": "dan", "Swedish": "swe", "Norwegian": "nor",
        "Icelandic": "isl", "Gothic": "got", "Old English": "ang",
        "Old High German": "goh", "Old Norse": "non",
        "French": "fra", "Italian": "ita", "Spanish": "spa",
        "Portuguese": "por", "Romanian": "ron", "Latin": "lat",
        "Russian": "rus", "Polish": "pol", "Czech": "ces",
        "Bulgarian": "bul", "Serbian": "srp", "Croatian": "hrv",
        "Old Church Slavonic": "chu", "Ukrainian": "ukr",
        "Lithuanian": "lit", "Latvian": "lav", "Old Prussian": "prg",
        "Ancient Greek": "grc", "Greek": "ell", "Mycenaean Greek": "gmy",
        "Sanskrit": "san", "Vedic": "san", "Hindi": "hin",
        "Persian": "fas", "Avestan": "ave", "Pashto": "pus",
        "Armenian": "hye", "Old Armenian": "xcl",
        "Albanian": "sqi",
        "Irish": "gle", "Welsh": "cym", "Old Irish": "sga",
        "Breton": "bre", "Scottish Gaelic": "gla", "Gaulish": "cel",
        "Hittite": "hit", "Luwian": "xlu",
        "Tocharian A": "xto", "Tocharian B": "txb",
    }
    return lang_codes.get(lang_name, lang_name[:3].lower())


def flatten_descendants(
    descendants: List[Dict],
    target_langs: Optional[Set[str]] = None,
    depth: int = 0,
    max_depth: int = 10
) -> List[Dict]:
    """
    Рекурсивно извлекает все листовые формы из дерева descendants.

    Args:
        descendants: список потомков
        target_langs: набор языков для фильтрации (None = все)
        depth: текущая глубина рекурсии
        max_depth: максимальная глубина

    Returns:
        List of dicts with {lang, lang_code, word}
    """
    if depth > max_depth:
        return []

    results = []

    for desc in descendants:
        lang = desc.get("lang", "")
        word = desc.get("word", "")
        lang_code = desc.get("lang_code", "")

        # Пропускаем Proto-* языки (они промежуточные)
        is_proto = lang.startswith("Proto-") or "-pro" in lang_code

        # Если есть вложенные descendants — идём глубже
        if "descendants" in desc and desc["descendants"]:
            results.extend(flatten_descendants(
                desc["descendants"],
                target_langs,
                depth + 1,
                max_depth
            ))
        elif word and not is_proto:
            # Это листовая форма
            if target_langs is None or lang in target_langs:
                results.append({
                    "lang": lang,
                    "lang_code": get_lang_code(lang),
                    "word": word,
                })

    return results


def parse_kaikki(jsonl_path: Path, target_langs: Optional[Set[str]] = None) -> List[Dict]:
    """
    Парсит Wiktionary PIE JSONL.

    Args:
        jsonl_path: путь к JSONL файлу
        target_langs: набор языков для фильтрации (None = TARGET_LANGUAGES)

    Returns:
        List of dicts with {protoform, pos, glosses, cognates}
    """
    if target_langs is None:
        target_langs = TARGET_LANGUAGES

    results = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Parsing Kaikki JSONL"):
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        word = data.get("word", "")
        pos = data.get("pos", "")
        senses = data.get("senses", [])
        descendants = data.get("descendants", [])

        # Извлекаем значения
        glosses = []
        for sense in senses:
            glosses.extend(sense.get("glosses", []))

        # Извлекаем когнаты
        cognates = flatten_descendants(descendants, target_langs)

        if cognates:  # только если есть когнаты
            results.append({
                "protoform": f"*{word}" if not word.startswith("*") else word,
                "pos": pos,
                "glosses": glosses,
                "cognates": cognates,
            })

    return results


def format_input(cognates: List[Dict]) -> str:
    """Форматирует список когнатов в строку."""
    parts = []
    seen = set()  # избегаем дубликатов

    for cog in cognates:
        key = (cog["lang_code"], cog["word"])
        if key not in seen:
            seen.add(key)
            parts.append(f"[{cog['lang_code']}] {cog['word']}")

    return " ".join(parts)


def to_training_format(
    parsed_data: List[Dict],
    min_cognates: int = 1
) -> pd.DataFrame:
    """
    Конвертирует в DataFrame для обучения.

    Returns:
        DataFrame с колонками: input, output, meaning, pos, num_cognates
    """
    rows = []

    for item in tqdm(parsed_data, desc="Formatting training data"):
        cognates = item["cognates"]
        if len(cognates) < min_cognates:
            continue

        input_str = format_input(cognates)
        meaning = ", ".join(item["glosses"]) if item["glosses"] else ""

        rows.append({
            "input": input_str,
            "output": item["protoform"],
            "meaning": meaning,
            "pos": item["pos"],
            "num_cognates": len(cognates),
        })

    return pd.DataFrame(rows)


def main():
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    parser = argparse.ArgumentParser(description="Parse Kaikki/Wiktionary PIE JSONL")
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("../raw/kaikki-pie.jsonl"),
        help="Path to Kaikki PIE JSONL file"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("../processed/kaikki.csv"),
        help="Output CSV path"
    )
    parser.add_argument(
        "--min-cognates",
        type=int,
        default=1,
        help="Minimum number of cognates per group"
    )
    parser.add_argument(
        "--all-langs",
        action="store_true",
        help="Include all languages, not just major IE languages"
    )

    args = parser.parse_args()

    print(f"Loading Kaikki from {args.input}...")
    target_langs = None if args.all_langs else TARGET_LANGUAGES

    parsed = parse_kaikki(args.input, target_langs)
    print(f"Entries with cognates: {len(parsed)}")

    print("Converting to training format...")
    df = to_training_format(parsed, min_cognates=args.min_cognates)

    print(f"Final dataset size: {len(df)}")

    # Статистика
    print(f"\nStatistics:")
    print(f"  Mean cognates per group: {df['num_cognates'].mean():.1f}")
    print(f"  Median cognates per group: {df['num_cognates'].median():.1f}")
    print(f"  Min cognates: {df['num_cognates'].min()}")
    print(f"  Max cognates: {df['num_cognates'].max()}")

    # POS distribution
    print(f"\nPOS distribution:")
    print(df['pos'].value_counts().head(10).to_string())

    # Сохраняем
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding='utf-8')
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
