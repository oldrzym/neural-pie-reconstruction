"""
Make train/val/test splits from processed CSV files.

Usage:
    python make_splits.py --input ../processed/iecor.csv --output-dir ../splits/iecor/
    python make_splits.py --input ../processed/kaikki.csv --output-dir ../splits/kaikki/
    python make_splits.py --input ../processed/starling.csv --output-dir ../splits/starling/

Output:
    train.csv, val.csv, test.csv
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import argparse


def make_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
    stratify_col: str = None
) -> tuple:
    """
    Splits DataFrame into train/val/test.

    Args:
        df: input DataFrame
        train_ratio: fraction for training
        val_ratio: fraction for validation
        test_ratio: fraction for test
        random_state: random seed
        stratify_col: column to stratify by (optional)

    Returns:
        (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001

    # First split: train vs (val+test)
    stratify = df[stratify_col] if stratify_col and stratify_col in df.columns else None

    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        random_state=random_state,
        stratify=stratify
    )

    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    stratify_temp = temp_df[stratify_col] if stratify_col and stratify_col in temp_df.columns else None

    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_size,
        random_state=random_state,
        stratify=stratify_temp
    )

    return train_df, val_df, test_df


def main():
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    parser = argparse.ArgumentParser(description="Make train/val/test splits")
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input CSV file (e.g., ../processed/iecor.csv)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        required=True,
        help="Output directory for splits"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training set ratio (default: 0.8)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test set ratio (default: 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--min-cognates",
        type=int,
        default=None,
        help="Filter: minimum cognates per sample"
    )

    args = parser.parse_args()

    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input)
    print(f"Total samples: {len(df)}")

    # Фильтрация
    if args.min_cognates:
        df = df[df['num_cognates'] >= args.min_cognates]
        print(f"After filtering (min_cognates >= {args.min_cognates}): {len(df)}")

    # Создаём splits
    print(f"\nCreating splits ({args.train_ratio}/{args.val_ratio}/{args.test_ratio})...")
    train_df, val_df, test_df = make_splits(
        df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.seed
    )

    print(f"  Train: {len(train_df)}")
    print(f"  Val:   {len(val_df)}")
    print(f"  Test:  {len(test_df)}")

    # Сохраняем
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.output_dir / "train.csv"
    val_path = args.output_dir / "val.csv"
    test_path = args.output_dir / "test.csv"

    train_df.to_csv(train_path, index=False, encoding='utf-8')
    val_df.to_csv(val_path, index=False, encoding='utf-8')
    test_df.to_csv(test_path, index=False, encoding='utf-8')

    print(f"\nSaved to {args.output_dir}/")
    print(f"  {train_path.name}: {len(train_df)} samples")
    print(f"  {val_path.name}: {len(val_df)} samples")
    print(f"  {test_path.name}: {len(test_df)} samples")

    # Статистика по splits
    print(f"\nStatistics:")
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"  {name}:")
        print(f"    Mean cognates: {split_df['num_cognates'].mean():.1f}")
        print(f"    Median cognates: {split_df['num_cognates'].median():.1f}")


if __name__ == "__main__":
    main()
