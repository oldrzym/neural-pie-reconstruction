"""
Apply language code mapping to etymology-db datasets.
Converts full language names like [English] to codes like [eng].
"""

import pandas as pd
import json
import re
from pathlib import Path
from tqdm import tqdm

def load_mapping(path: str = "dataset/language_mapping.json") -> dict:
    """Load language name to code mapping."""
    with open(path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    print(f"Loaded mapping for {len(mapping):,} languages")

    # Add some manual fixes for common issues
    mapping['Proto-Indo-European'] = 'ine'  # PIE special code

    return mapping

def convert_language_tags(text: str, mapping: dict) -> str:
    """Convert [Language Name] to [code] in text."""
    def replace_tag(match):
        lang_name = match.group(1).strip()
        code = mapping.get(lang_name, lang_name[:3].lower() if len(lang_name) >= 3 else lang_name.lower())
        return f"[{code}]"

    # Replace all [Language Name] tags
    converted = re.sub(r'\[([^\]]+)\]', replace_tag, text)
    return converted

def process_file(input_path: Path, output_path: Path, mapping: dict):
    """Process a single CSV file, converting language names to codes."""
    print(f"\nProcessing {input_path.name}...")

    # Read CSV
    df = pd.read_csv(input_path)

    if len(df) == 0:
        print("  (empty, skipping)")
        return

    # Convert input column
    print(f"  Converting {len(df):,} rows...")
    df['input'] = [
        convert_language_tags(str(text), mapping)
        for text in tqdm(df['input'], desc="  Converting", leave=False)
    ]

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"  OK Saved to {output_path}")

def main():
    """Main function."""
    print("="*60)
    print("Applying Language Codes to Etymology-db Datasets")
    print("="*60)

    # Load mapping
    mapping = load_mapping()

    # Process all etymology_db files
    input_dir = Path("dataset/processed")
    output_dir = Path("dataset/processed_with_codes")

    for csv_file in sorted(input_dir.glob("etymology_db_*.csv")):
        output_file = output_dir / csv_file.name

        try:
            process_file(csv_file, output_file, mapping)
        except pd.errors.EmptyDataError:
            print(f"\n{csv_file.name}: empty file, skipping")

    print("\n" + "="*60)
    print("OK Done!")
    print("="*60)
    print(f"\nConverted files saved to: {output_dir}/")

    # Show example
    if (output_dir / "etymology_db_pie_direct.csv").exists():
        df = pd.read_csv(output_dir / "etymology_db_pie_direct.csv")
        print("\nExample converted entry:")
        print(f"Input:  {df.iloc[0]['input']}")
        print(f"Output: {df.iloc[0]['output']}")

if __name__ == "__main__":
    main()
