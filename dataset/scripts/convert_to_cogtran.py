"""
Convert our CSV format to CognateTransformer format.

CognateTransformer expects aligned cognate sets:
    data = {'Latin': '[Latin]|c|a|n|i|s', 'Greek': '[Greek]|k|y|o|n|-'}
    solns = {'PIE': '[PIE]|k|w|o|n'}

This requires Multiple Sequence Alignment (MSA) of cognates.
We use lingpy for alignment.

Usage:
    python convert_to_cogtran.py --input ../splits/iecor/train.csv --output ../splits/iecor_cogtran/train.tsv
    python convert_to_cogtran.py --input-dir ../splits/iecor/ --output-dir ../splits/iecor_cogtran/

Requirements:
    pip install lingpy
"""

import argparse
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd

try:
    from lingpy import Multiple, basictypes
    LINGPY_AVAILABLE = True
except ImportError:
    LINGPY_AVAILABLE = False
    print("Warning: lingpy not installed. Run: pip install lingpy")


def parse_input_string(input_str: str) -> Dict[str, str]:
    """
    Parse input string like "[lat] canis [grc] kýōn" into dict.

    Returns:
        {"lat": "canis", "grc": "kýōn"}
    """
    pattern = r'\[([^\]]+)\]\s*([^\[]+)'
    matches = re.findall(pattern, input_str)

    result = {}
    for lang, word in matches:
        word = word.strip()
        if word:
            result[lang] = word

    return result


def tokenize_word(word: str) -> List[str]:
    """
    Tokenize word into segments (characters for now).

    For better results, could use IPA segmentation.
    """
    # Simple character-level tokenization
    return list(word)


def align_cognates_lingpy(cognates: Dict[str, str]) -> Dict[str, List[str]]:
    """
    Align cognate set using lingpy's Multiple alignment.

    Args:
        cognates: {"lat": "canis", "grc": "kýōn", ...}

    Returns:
        {"lat": ["c", "a", "n", "i", "s", "-"], "grc": ["k", "ý", "-", "ō", "n", "-"], ...}
    """
    if not LINGPY_AVAILABLE:
        # Fallback: no alignment, just tokenize
        return {lang: tokenize_word(word) for lang, word in cognates.items()}

    if len(cognates) < 2:
        # Can't align single word
        return {lang: tokenize_word(word) for lang, word in cognates.items()}

    # Prepare sequences for lingpy
    seqs = []
    langs = []
    for lang, word in cognates.items():
        tokens = tokenize_word(word)
        seqs.append(tokens)
        langs.append(lang)

    try:
        # Create Multiple alignment object
        msa = Multiple(seqs)
        msa.prog_align()  # Progressive alignment

        # Get aligned sequences
        aligned = {}
        for i, lang in enumerate(langs):
            aligned[lang] = list(msa.alm_matrix[i])

        return aligned

    except Exception as e:
        print(f"Alignment failed: {e}, using unaligned")
        return {lang: tokenize_word(word) for lang, word in cognates.items()}


def align_cognates_simple(cognates: Dict[str, str]) -> Dict[str, List[str]]:
    """
    Simple alignment: pad all sequences to max length.
    Not linguistically correct but works as fallback.
    """
    tokenized = {lang: tokenize_word(word) for lang, word in cognates.items()}
    max_len = max(len(tokens) for tokens in tokenized.values())

    aligned = {}
    for lang, tokens in tokenized.items():
        # Pad with '-' to max length
        padded = tokens + ['-'] * (max_len - len(tokens))
        aligned[lang] = padded

    return aligned


def format_cogtran_entry(lang: str, tokens: List[str]) -> str:
    """
    Format as CognateTransformer expects: '[Lang]|t|o|k|e|n|s'
    """
    return f"[{lang}]|" + "|".join(tokens)


def convert_row_to_cogtran(
    input_str: str,
    output_str: str,
    use_lingpy: bool = True
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Convert single row to CognateTransformer format.

    Returns:
        (data_dict, solns_dict)
    """
    # Parse cognates
    cognates = parse_input_string(input_str)

    if not cognates:
        return {}, {}

    # Align cognates
    if use_lingpy and LINGPY_AVAILABLE:
        aligned = align_cognates_lingpy(cognates)
    else:
        aligned = align_cognates_simple(cognates)

    # Format data entries
    data = {}
    for lang, tokens in aligned.items():
        data[lang] = format_cogtran_entry(lang, tokens)

    # Parse and format protoform
    proto = output_str.strip()
    if proto.startswith('*'):
        proto = proto[1:]
    proto = proto.rstrip('-')

    # Handle multiple protoforms (take first)
    if ',' in proto:
        proto = proto.split(',')[0].strip()

    proto_tokens = tokenize_word(proto)
    solns = {"PIE": format_cogtran_entry("PIE", proto_tokens)}

    return data, solns


def convert_csv_to_cogtran(
    csv_path: str,
    output_path: str,
    use_lingpy: bool = True
) -> List[Dict]:
    """
    Convert CSV to CognateTransformer format.

    Output: JSON lines file with {data: {...}, solns: {...}} per line
    """
    df = pd.read_csv(csv_path)

    records = []
    skipped = 0

    for idx, row in df.iterrows():
        input_str = str(row['input'])
        output_str = str(row['output'])

        data, solns = convert_row_to_cogtran(input_str, output_str, use_lingpy)

        if data and solns:
            records.append({
                "id": row.get('cognate_set_id', idx),
                "data": data,
                "solns": solns
            })
        else:
            skipped += 1

    # Save as JSON lines
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"Converted {len(records)} records, skipped {skipped}")
    print(f"Saved to {output_path}")

    return records


def convert_directory(input_dir: str, output_dir: str, use_lingpy: bool = True):
    """Convert all CSV files in directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    for csv_file in ['train.csv', 'val.csv', 'test.csv']:
        csv_path = input_dir / csv_file
        if csv_path.exists():
            print(f"\nConverting {csv_file}...")
            output_name = csv_file.replace('.csv', '.jsonl')
            output_path = output_dir / output_name
            convert_csv_to_cogtran(str(csv_path), str(output_path), use_lingpy)


def main():
    parser = argparse.ArgumentParser(description="Convert CSV to CognateTransformer format")
    parser.add_argument("--input", type=str, help="Input CSV file")
    parser.add_argument("--output", type=str, help="Output JSONL file")
    parser.add_argument("--input-dir", type=str, help="Input directory")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--no-lingpy", action="store_true", help="Don't use lingpy for alignment")

    args = parser.parse_args()
    use_lingpy = not args.no_lingpy

    if args.input_dir and args.output_dir:
        convert_directory(args.input_dir, args.output_dir, use_lingpy)
    elif args.input and args.output:
        convert_csv_to_cogtran(args.input, args.output, use_lingpy)
    else:
        parser.print_help()
        print("\nError: Provide either --input/--output or --input-dir/--output-dir")


if __name__ == "__main__":
    main()
