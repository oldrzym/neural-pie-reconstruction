"""
IE-CoR Parser — конвертирует IE-CoR CLDF в формат для обучения seq2seq модели.

Выходной формат CSV:
    input: "[lat] canis [grc] kýōn [san] śvā́ [got] hunds [arm] šun"
    output: "*k̑u̯ón-"
    meaning: "dog"
    num_cognates: 5
    cognate_set_id: 21
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import argparse


def load_iecor_tables(cldf_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Загружает все таблицы IE-CoR."""
    cognatesets = pd.read_csv(cldf_path / "cognatesets.csv")
    forms = pd.read_csv(cldf_path / "forms.csv")
    cognates = pd.read_csv(cldf_path / "cognates.csv")
    languages = pd.read_csv(cldf_path / "languages.csv")
    return cognatesets, forms, cognates, languages


def get_pie_cognatesets(cognatesets: pd.DataFrame) -> pd.DataFrame:
    """Фильтрует только PIE праформы."""
    return cognatesets[cognatesets["Root_Language"] == "Proto-Indo-European"].copy()


def get_language_code(lang_name: str) -> str:
    """Конвертирует название языка в короткий код."""
    # Маппинг основных языков
    lang_codes = {
        "English": "eng",
        "German": "deu",
        "Dutch": "nld",
        "Danish": "dan",
        "Swedish": "swe",
        "Norwegian: Bokmål": "nob",
        "Icelandic": "isl",
        "Gothic": "got",
        "Old English": "ang",
        "Old High German": "goh",
        "French": "fra",
        "Italian": "ita",
        "Spanish": "spa",
        "Portuguese": "por",
        "Romanian": "ron",
        "Latin": "lat",
        "Russian": "rus",
        "Polish": "pol",
        "Czech": "ces",
        "Bulgarian": "bul",
        "Serbo-Croat": "hbs",
        "Old Church Slavonic": "chu",
        "Lithuanian": "lit",
        "Latvian": "lav",
        "Old Prussian": "prg",
        "Greek: Ancient": "grc",
        "Greek: Modern Std": "ell",
        "Sanskrit": "san",
        "Vedic: Early": "san",
        "Hindi": "hin",
        "Persian: Tehran": "fas",
        "Avestan: Younger": "ave",
        "Armenian: Eastern": "hye",
        "Armenian: Western": "hyw",
        "Armenian: Classical": "xcl",
        "Albanian: Standard": "sqi",
        "Albanian: Gheg": "aln",
        "Old Irish": "sga",
        "Welsh: North": "cym",
        "Breton: Treger": "bre",
        "Hittite": "hit",
        "Luvian": "xlu",
        "Tocharian A": "xto",
        "Tocharian B": "txb",
    }

    # Пробуем найти точное совпадение
    if lang_name in lang_codes:
        return lang_codes[lang_name]

    # Пробуем найти по первому слову
    first_word = lang_name.split(":")[0].strip()
    if first_word in lang_codes:
        return lang_codes[first_word]

    # Возвращаем первые 3 буквы в нижнем регистре
    return lang_name[:3].lower()


def build_cognate_groups(
    pie_cognatesets: pd.DataFrame,
    forms: pd.DataFrame,
    cognates: pd.DataFrame,
    languages: pd.DataFrame
) -> List[Dict]:
    """
    Собирает группы когнатов для каждой PIE праформы.

    Returns:
        List of dicts with keys: protoform, cognate_set_id, meaning, cognates
    """
    # Создаём маппинги для быстрого доступа
    lang_map = languages.set_index("ID")["Name"].to_dict()
    form_map = forms.set_index("ID").to_dict("index")

    results = []

    for _, cogset in tqdm(pie_cognatesets.iterrows(), total=len(pie_cognatesets), desc="Processing cognate sets"):
        cogset_id = cogset["ID"]
        protoform = cogset["Root_Form"]

        # Пропускаем если нет праформы
        if pd.isna(protoform):
            continue

        # Получаем все формы для этого cognate set
        cogset_cognates = cognates[cognates["Cognateset_ID"] == cogset_id]

        cognate_list = []
        for _, cog in cogset_cognates.iterrows():
            form_id = cog["Form_ID"]
            if form_id not in form_map:
                continue

            form_data = form_map[form_id]
            lang_id = form_data["Language_ID"]
            lang_name = lang_map.get(lang_id, "unknown")

            cognate_list.append({
                "lang": lang_name,
                "lang_code": get_language_code(lang_name),
                "word": form_data["Value"],
                "form": form_data["Form"],
                "segments": form_data.get("Segments", ""),
                "phon_form": form_data.get("phon_form", ""),
            })

        if cognate_list:
            results.append({
                "protoform": protoform if protoform.startswith("*") else f"*{protoform}",
                "cognate_set_id": cogset_id,
                "meaning": cogset.get("Root_Gloss", ""),
                "cognates": cognate_list,
            })

    return results


def format_input(cognates: List[Dict], use_segments: bool = False) -> str:
    """
    Форматирует список когнатов в строку для seq2seq модели.

    Args:
        cognates: список когнатов
        use_segments: использовать фонемы вместо орфографии

    Returns:
        "[lat] canis [grc] kýōn [san] śvā́"
    """
    parts = []
    for cog in cognates:
        lang_code = cog["lang_code"]
        if use_segments and cog.get("segments"):
            word = cog["segments"]
        else:
            word = cog["word"]
        parts.append(f"[{lang_code}] {word}")

    return " ".join(parts)


def to_training_format(
    cognate_groups: List[Dict],
    use_segments: bool = False,
    min_cognates: int = 1
) -> pd.DataFrame:
    """
    Конвертирует группы когнатов в DataFrame для обучения.

    Args:
        cognate_groups: список групп когнатов
        use_segments: использовать фонемы
        min_cognates: минимальное количество когнатов для включения

    Returns:
        DataFrame с колонками: input, output, meaning, num_cognates, cognate_set_id
    """
    rows = []

    for group in tqdm(cognate_groups, desc="Formatting training data"):
        if len(group["cognates"]) < min_cognates:
            continue

        input_str = format_input(group["cognates"], use_segments=use_segments)

        rows.append({
            "input": input_str,
            "output": group["protoform"],
            "meaning": group["meaning"] if pd.notna(group["meaning"]) else "",
            "num_cognates": len(group["cognates"]),
            "cognate_set_id": group["cognate_set_id"],
        })

    return pd.DataFrame(rows)


def main():
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    parser = argparse.ArgumentParser(description="Parse IE-CoR to training format")
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("../raw/iecor/cldf"),
        help="Path to IE-CoR CLDF folder"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("../processed/iecor.csv"),
        help="Output CSV path"
    )
    parser.add_argument(
        "--use-segments",
        action="store_true",
        help="Use phonemic segments instead of orthographic forms"
    )
    parser.add_argument(
        "--min-cognates",
        type=int,
        default=1,
        help="Minimum number of cognates per group"
    )

    args = parser.parse_args()

    print(f"Loading IE-CoR from {args.input}...")
    cognatesets, forms, cognates, languages = load_iecor_tables(args.input)

    print(f"Total cognate sets: {len(cognatesets)}")
    pie_cognatesets = get_pie_cognatesets(cognatesets)
    print(f"PIE cognate sets: {len(pie_cognatesets)}")

    print("Building cognate groups...")
    cognate_groups = build_cognate_groups(pie_cognatesets, forms, cognates, languages)
    print(f"Valid groups: {len(cognate_groups)}")

    print("Converting to training format...")
    df = to_training_format(
        cognate_groups,
        use_segments=args.use_segments,
        min_cognates=args.min_cognates
    )

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
