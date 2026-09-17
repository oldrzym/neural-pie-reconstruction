"""Extract all unique cognates from the dataset for Wiktionary fetching."""
import csv
import re
import json
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent.parent.parent / "splits" / "iecor_kaikki_koebler_normalized"
TRAIN_FILE = BASE_DIR / "train_synt.csv"  # Use synthetic data (more cognates)
OUTPUT_FILE = BASE_DIR / "unique_cognates.json"


def parse_cognates(cognate_str):
    """
    Extract list of (lang, word) tuples from cognate string.

    Example: "[eng] ride [deu] reiten" -> [('eng', 'ride'), ('deu', 'reiten')]
    """
    pattern = r'\[([a-z]{3})\]\s*([^\[\]]+?)(?=\s*\[|$)'
    matches = re.findall(pattern, cognate_str)
    return [(lang.strip(), word.strip()) for lang, word in matches]


def main():
    print("Extracting unique cognates from dataset...")
    print(f"Reading from: {TRAIN_FILE}\n")

    # Read dataset
    with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Total rows: {len(rows)}")

    # Extract all cognates
    all_cognates = set()
    cognates_by_lang = defaultdict(set)
    cognates_by_pie = defaultdict(list)

    for i, row in enumerate(rows):
        cognate_str = row.get('input', '')
        pie_form = row.get('output', '') or row.get('PIE', '')

        cognates = parse_cognates(cognate_str)
        for lang, word in cognates:
            all_cognates.add((lang, word))
            cognates_by_lang[lang].add(word)
            cognates_by_pie[pie_form].append((lang, word))

    print(f"Total unique cognates: {len(all_cognates)}")
    print(f"Languages found: {len(cognates_by_lang)}")
    print()

    # Show statistics
    print("Top 10 languages by cognate count:")
    lang_counts = [(lang, len(words)) for lang, words in cognates_by_lang.items()]
    lang_counts.sort(key=lambda x: x[1], reverse=True)

    for lang, count in lang_counts[:10]:
        print(f"  {lang}: {count} words")

    # Save to JSON
    output_data = {
        "unique_cognates": sorted(list(all_cognates)),
        "cognates_by_language": {lang: sorted(list(words)) for lang, words in cognates_by_lang.items()},
        "statistics": {
            "total_unique_cognates": len(all_cognates),
            "total_languages": len(cognates_by_lang),
            "cognates_per_language": {lang: len(words) for lang, words in cognates_by_lang.items()},
        }
    }

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Saved unique cognates to: {OUTPUT_FILE}")
    print(f"✓ Ready for Wiktionary fetching!")


if __name__ == "__main__":
    main()
