"""
Build language name to code mapping from wiktionary_codes.csv and existing datasets.
Creates a comprehensive mapping file for converting full language names to 3-letter codes.
"""

import pandas as pd
import json
from pathlib import Path
from collections import defaultdict

def load_wiktionary_codes(path: str = "etymology-db/wiktionary_codes.csv") -> dict:
    """Load Wiktionary language codes."""
    df = pd.read_csv(path, header=None, names=['code', 'name'])
    # Remove rows with NaN values
    df = df.dropna()
    # Create mapping: name -> code
    mapping = dict(zip(df['name'], df['code']))
    print(f"Loaded {len(mapping):,} language codes from Wiktionary")
    return mapping

def extract_languages_from_datasets() -> set:
    """Extract all unique language names from existing datasets."""
    all_langs = set()

    # Check etymology_db outputs
    for csv_file in Path("dataset/processed").glob("etymology_db_*.csv"):
        print(f"Scanning {csv_file.name}...")
        try:
            df = pd.read_csv(csv_file)
            if len(df) == 0:
                print(f"  (empty, skipping)")
                continue

            for input_str in df['input']:
                # Extract language names from [lang] format
                import re
                langs = re.findall(r'\[([^\]]+)\]', str(input_str))
                all_langs.update(langs)
        except pd.errors.EmptyDataError:
            print(f"  (empty file, skipping)")
            continue

    print(f"Found {len(all_langs):,} unique language names in datasets")
    return all_langs

def create_comprehensive_mapping(wikt_mapping: dict, dataset_langs: set) -> dict:
    """Create comprehensive mapping, handling special cases."""
    final_mapping = {}
    unmapped = []

    for lang in sorted(dataset_langs):
        if lang in wikt_mapping:
            # Direct match
            final_mapping[lang] = wikt_mapping[lang]
        else:
            # Try to find partial match or create code
            found = False

            # Check if it's a variant (e.g., "Norwegian Bokmål" -> "Norwegian")
            for wikt_name, code in wikt_mapping.items():
                if lang.lower() in wikt_name.lower() or wikt_name.lower() in lang.lower():
                    final_mapping[lang] = code
                    found = True
                    break

            if not found:
                # Generate a 3-letter code from the name
                if len(lang) >= 3:
                    code = lang[:3].lower()
                else:
                    code = lang.lower()
                final_mapping[lang] = code
                unmapped.append((lang, code))

    if unmapped:
        print(f"\nWarning: {len(unmapped)} languages not found in Wiktionary codes:")
        for lang, code in unmapped[:20]:
            print(f"  {lang:40s} -> {code} (generated)")
        if len(unmapped) > 20:
            print(f"  ... and {len(unmapped) - 20} more")

    return final_mapping

def save_mapping(mapping: dict, output_path: str = "dataset/language_mapping.json"):
    """Save mapping to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False, sort_keys=True)

    print(f"\nSaved mapping with {len(mapping):,} entries to {output_path}")

def analyze_mapping(mapping: dict):
    """Print statistics about the mapping."""
    print("\n" + "="*60)
    print("MAPPING STATISTICS")
    print("="*60)

    # Group by code length
    code_lengths = defaultdict(int)
    for code in mapping.values():
        if isinstance(code, str):
            code_lengths[len(code)] += 1

    print("\nCode length distribution:")
    for length in sorted(code_lengths.keys()):
        print(f"  {length}-character codes: {code_lengths[length]:,}")

    # Sample entries
    print("\nSample mappings:")
    for _, (name, code) in enumerate(sorted(mapping.items())[:10]):
        try:
            print(f"  {name:40s} -> {code}")
        except UnicodeEncodeError:
            print(f"  {repr(name):40s} -> {code}")

    # Check for conflicts (same code, different names)
    code_to_names = defaultdict(list)
    for name, code in mapping.items():
        if isinstance(code, str):  # Skip NaN codes
            code_to_names[code].append(name)

    conflicts = {code: names for code, names in code_to_names.items() if len(names) > 1}
    if conflicts:
        print(f"\nWarning: {len(conflicts)} code conflicts detected:")
        for code, names in list(conflicts.items())[:5]:
            try:
                print(f"  {code}: {', '.join(names)}")
            except UnicodeEncodeError:
                print(f"  {code}: {len(names)} languages (Unicode display error)")

def main():
    """Main function."""
    print("="*60)
    print("Building Language Mapping")
    print("="*60)

    # Load Wiktionary codes
    wikt_mapping = load_wiktionary_codes()

    # Extract languages from datasets
    dataset_langs = extract_languages_from_datasets()

    # Create comprehensive mapping
    final_mapping = create_comprehensive_mapping(wikt_mapping, dataset_langs)

    # Analyze
    analyze_mapping(final_mapping)

    # Save
    save_mapping(final_mapping)

    # Also save as CSV for easy viewing
    csv_path = "dataset/language_mapping.csv"
    df = pd.DataFrame([
        {'language': lang, 'code': code}
        for lang, code in sorted(final_mapping.items())
    ])
    df.to_csv(csv_path, index=False)
    print(f"Also saved as CSV: {csv_path}")

if __name__ == "__main__":
    main()
