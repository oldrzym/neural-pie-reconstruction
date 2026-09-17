"""
Normalize protoforms to Starling notation (simplified).

Converts IPA-like notation (IE-CoR, Kaikki) to simpler Starling format:
- h₁, h₂, h₃ → (removed)      Laryngeals
- ʷ → w                        Labialization
- ʰ → h                        Aspiration
- ḱ, k̑ → k'                    Palatalization
- ǵ, ĝ → g'                    Palatalization
- u̯ → w, i̯ → y                 Semivowels
- ə → ǝ                        Schwa

Usage:
    python normalize_protoforms.py --input-dir ../splits/combined_all/ --output-dir ../splits/combined_all_simplified/
"""

import argparse
import re
from pathlib import Path
import pandas as pd


def normalize_to_starling(proto: str) -> str:
    """Simplify protoform to Starling notation."""
    if pd.isna(proto):
        return proto

    p = str(proto)

    # Laryngeals → remove
    p = re.sub(r'h[₁₂₃]', '', p)

    # Labialization: ʷ → w
    p = p.replace('ʷ', 'w')

    # Aspiration: ʰ → h
    p = p.replace('ʰ', 'h')

    # Palatalization: ḱ/k̑ → k', ǵ/ĝ → g'
    p = p.replace('ḱ', "k'")
    p = p.replace('k̑', "k'")
    p = p.replace('ǵ', "g'")
    p = p.replace('ĝ', "g'")

    # Semivowels: u̯ → w, i̯ → y
    p = p.replace('u̯', 'w')
    p = p.replace('i̯', 'y')

    # Schwa normalization
    p = p.replace('ə', 'ǝ')

    # Clean up double consonants from laryngeal removal
    # e.g., *steh₂- → *ste- (not *st-)
    # Keep as is - let model learn

    return p


def normalize_csv(input_path: str, output_path: str):
    """Normalize protoforms in a CSV file."""
    df = pd.read_csv(input_path)

    original_outputs = df['output'].tolist()
    df['output'] = df['output'].apply(normalize_to_starling)
    normalized_outputs = df['output'].tolist()

    # Count changes
    changed = sum(1 for o, n in zip(original_outputs, normalized_outputs) if o != n)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return len(df), changed


def normalize_directory(input_dir: str, output_dir: str):
    """Normalize all CSV files in a directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    print(f"Normalizing protoforms: {input_dir} -> {output_dir}")
    print()

    for split in ['train', 'val', 'test']:
        input_path = input_dir / f'{split}.csv'
        output_path = output_dir / f'{split}.csv'

        if input_path.exists():
            total, changed = normalize_csv(str(input_path), str(output_path))
            print(f"  {split}: {total} examples, {changed} normalized ({changed/total*100:.1f}%)")

    print()
    print(f"Output saved to: {output_dir}")


def show_examples(input_dir: str, n: int = 10):
    """Show example normalizations."""
    input_dir = Path(input_dir)
    train_path = input_dir / 'train.csv'

    if not train_path.exists():
        print(f"File not found: {train_path}")
        return

    df = pd.read_csv(train_path)

    print("Example normalizations:")
    print("-" * 60)

    count = 0
    for _, row in df.iterrows():
        original = str(row['output'])
        normalized = normalize_to_starling(original)

        if original != normalized:
            print(f"  {original:30} -> {normalized}")
            count += 1
            if count >= n:
                break

    if count == 0:
        print("  (no changes found in first examples)")


def main():
    parser = argparse.ArgumentParser(description="Normalize protoforms to Starling notation")
    parser.add_argument("--input-dir", type=str, required=True, help="Input directory with train/val/test.csv")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--show-examples", action="store_true", help="Show example normalizations")
    parser.add_argument("--examples-count", type=int, default=10, help="Number of examples to show")

    args = parser.parse_args()

    if args.show_examples:
        show_examples(args.input_dir, args.examples_count)
        print()

    normalize_directory(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
