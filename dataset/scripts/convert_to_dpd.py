"""
Convert our CSV format to DPD pickle format.

Our format (CSV):
    input: "[lat] canis [grc] kýōn [san] śvā́"
    output: "*k̑u̯ón-"

DPD format (pickle):
    (langs_list, data) where:
    - langs_list[0] = "PIE" (proto-language)
    - langs_list[1:] = daughter languages
    - data = {
        cognate_id: {
            "daughters": {"lat": ["c", "a", "n", "i", "s"], ...},
            "protoform": {"PIE": ["k̑", "u̯", "ó", "n"]}
        }
    }

Usage:
    python convert_to_dpd.py --input ../splits/iecor/train.csv --output ../splits/iecor_dpd/train.pickle
    python convert_to_dpd.py --input-dir ../splits/iecor/ --output-dir ../splits/iecor_dpd/
"""

import argparse
import pickle
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import pandas as pd


def parse_input_string(input_str: str) -> Dict[str, List[str]]:
    """
    Parse input string like "[lat] canis [grc] kýōn" into dict.

    Returns:
        {"lat": ["c", "a", "n", "i", "s"], "grc": ["k", "ý", "ō", "n"]}
    """
    # Pattern: [lang] word
    pattern = r'\[([^\]]+)\]\s*([^\[]+)'
    matches = re.findall(pattern, input_str)

    result = {}
    for lang, word in matches:
        word = word.strip()
        if word:
            # Split into characters (Unicode-aware)
            chars = list(word)
            # Add empty tone placeholder at the end (DPD expects this)
            chars.append('')
            result[lang] = chars

    return result


def parse_protoform(proto_str: str) -> List[str]:
    """
    Parse protoform like "*k̑u̯ón-" into list of phonemes.

    Note: This is a simplified character-level split.
    For better results, could use IPA segmentation.
    """
    # Remove leading asterisk and trailing hyphen
    proto = proto_str.strip()
    if proto.startswith('*'):
        proto = proto[1:]
    proto = proto.rstrip('-')

    # Split into characters (Unicode-aware)
    chars = list(proto)
    # Add empty tone placeholder
    chars.append('')

    return chars


def convert_csv_to_dpd(
    csv_path: str,
    output_path: str,
    proto_lang: str = "PIE"
) -> Tuple[List[str], Dict]:
    """
    Convert CSV to DPD pickle format.

    Returns:
        (langs_list, data) tuple ready for pickling
    """
    df = pd.read_csv(csv_path)

    # Collect all languages
    all_langs: Set[str] = set()

    # First pass: collect languages
    for input_str in df['input']:
        daughters = parse_input_string(str(input_str))
        all_langs.update(daughters.keys())

    # Sort languages for consistency
    daughter_langs = sorted(list(all_langs))
    langs_list = [proto_lang] + daughter_langs

    print(f"Found {len(daughter_langs)} daughter languages")

    # Second pass: build data
    data = {}

    for idx, row in df.iterrows():
        input_str = str(row['input'])
        output_str = str(row['output'])

        # Handle multiple protoforms (take first one)
        if ',' in output_str:
            output_str = output_str.split(',')[0].strip()

        daughters = parse_input_string(input_str)
        protoform = parse_protoform(output_str)

        # Use cognate_set_id if available, else row index
        if 'cognate_set_id' in row and pd.notna(row['cognate_set_id']):
            cognate_id = str(row['cognate_set_id'])
        else:
            cognate_id = str(idx)

        data[cognate_id] = {
            "daughters": daughters,
            "protoform": {proto_lang: protoform}
        }

    print(f"Converted {len(data)} cognate sets")

    return langs_list, data


def save_pickle(langs_list: List[str], data: Dict, output_path: str):
    """Save to pickle file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump((langs_list, data), f)

    print(f"Saved to {output_path}")


def convert_directory(input_dir: str, output_dir: str, proto_lang: str = "PIE"):
    """Convert all CSV files in a directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Collect all languages across all splits first
    all_langs: Set[str] = set()

    for csv_file in ['train.csv', 'val.csv', 'test.csv']:
        csv_path = input_dir / csv_file
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            for input_str in df['input']:
                daughters = parse_input_string(str(input_str))
                all_langs.update(daughters.keys())

    daughter_langs = sorted(list(all_langs))
    langs_list = [proto_lang] + daughter_langs

    print(f"Total languages across all splits: {len(daughter_langs)}")

    # Convert each split
    for csv_file in ['train.csv', 'val.csv', 'test.csv']:
        csv_path = input_dir / csv_file
        if csv_path.exists():
            print(f"\nConverting {csv_file}...")

            # Use 'dev' instead of 'val' (DPD convention)
            output_name = csv_file.replace('.csv', '.pickle').replace('val', 'dev')
            output_path = output_dir / output_name

            _, data = convert_csv_to_dpd(str(csv_path), str(output_path), proto_lang)
            save_pickle(langs_list, data, str(output_path))


def main():
    parser = argparse.ArgumentParser(description="Convert CSV to DPD pickle format")
    parser.add_argument("--input", type=str, help="Input CSV file")
    parser.add_argument("--output", type=str, help="Output pickle file")
    parser.add_argument("--input-dir", type=str, help="Input directory with train/val/test.csv")
    parser.add_argument("--output-dir", type=str, help="Output directory for pickle files")
    parser.add_argument("--proto-lang", type=str, default="PIE", help="Proto-language name")

    args = parser.parse_args()

    if args.input_dir and args.output_dir:
        convert_directory(args.input_dir, args.output_dir, args.proto_lang)
    elif args.input and args.output:
        langs_list, data = convert_csv_to_dpd(args.input, args.output, args.proto_lang)
        save_pickle(langs_list, data, args.output)
    else:
        parser.print_help()
        print("\nError: Provide either --input/--output or --input-dir/--output-dir")


if __name__ == "__main__":
    main()
