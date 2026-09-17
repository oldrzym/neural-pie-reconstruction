"""
Merge multiple PIE datasets into one.

Usage:
    python merge_datasets.py \
        --inputs ../splits/iecor/train.csv ../splits/kaikki/train.csv \
        --output ../splits/combined/train.csv

    # Or merge all splits at once:
    python merge_datasets.py --merge-dirs \
        --input-dirs ../splits/iecor ../splits/kaikki \
        --output-dir ../splits/combined
"""

import argparse
import pandas as pd
from pathlib import Path


def merge_csv_files(input_files: list, output_file: str, shuffle: bool = True):
    """Merge multiple CSV files into one."""
    dfs = []
    for f in input_files:
        df = pd.read_csv(f)
        # Add source column to track origin
        df['source'] = Path(f).parent.name
        dfs.append(df)
        print(f"  Loaded {f}: {len(df)} examples")

    merged = pd.concat(dfs, ignore_index=True)

    if shuffle:
        merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)

    # Ensure output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    merged.to_csv(output_file, index=False)
    print(f"  Saved {output_file}: {len(merged)} examples total")

    return merged


def merge_directories(input_dirs: list, output_dir: str):
    """Merge train/val/test splits from multiple directories."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ['train', 'val', 'test']:
        print(f"\nMerging {split}...")
        input_files = []
        for d in input_dirs:
            f = Path(d) / f"{split}.csv"
            if f.exists():
                input_files.append(str(f))
            else:
                print(f"  Warning: {f} not found, skipping")

        if input_files:
            output_file = output_dir / f"{split}.csv"
            merge_csv_files(input_files, str(output_file), shuffle=(split == 'train'))


def main():
    parser = argparse.ArgumentParser(description="Merge PIE datasets")
    parser.add_argument("--inputs", nargs="+", help="Input CSV files to merge")
    parser.add_argument("--output", help="Output CSV file")
    parser.add_argument("--merge-dirs", action="store_true", help="Merge entire directories")
    parser.add_argument("--input-dirs", nargs="+", help="Input directories with train/val/test.csv")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--no-shuffle", action="store_true", help="Don't shuffle merged data")

    args = parser.parse_args()

    if args.merge_dirs:
        if not args.input_dirs or not args.output_dir:
            parser.error("--merge-dirs requires --input-dirs and --output-dir")
        merge_directories(args.input_dirs, args.output_dir)
    elif args.inputs and args.output:
        merge_csv_files(args.inputs, args.output, shuffle=not args.no_shuffle)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
