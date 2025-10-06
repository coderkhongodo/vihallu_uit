#!/usr/bin/env python3
"""
Script to split a dataset into train/val/test sets with stratification.
Maintains the same class distribution across all splits.
"""

import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def split_dataset(
    input_csv: str,
    output_dir: str = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
    stratify_column: str = "label"
):
    """
    Split dataset into train/val/test sets with stratification.
    
    Args:
        input_csv: Path to input CSV file
        output_dir: Directory to save output files (default: same as input)
        train_ratio: Proportion of training data (default: 0.8)
        val_ratio: Proportion of validation data (default: 0.1)
        test_ratio: Proportion of test data (default: 0.1)
        random_seed: Random seed for reproducibility (default: 42)
        stratify_column: Column name to use for stratification (default: "label")
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Read input CSV
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")
    
    print(f"Reading dataset from: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Total samples: {len(df)}")
    
    # Check if stratify column exists
    if stratify_column not in df.columns:
        raise ValueError(f"Stratify column '{stratify_column}' not found in dataset")
    
    # Print label distribution
    print(f"\nOriginal label distribution:")
    print(df[stratify_column].value_counts())
    print(f"\nProportions:")
    print(df[stratify_column].value_counts(normalize=True))
    
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        random_state=random_seed,
        stratify=df[stratify_column]
    )
    
    # Second split: separate train and val from remaining data
    # Adjust val_ratio relative to the remaining data
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_adjusted,
        random_state=random_seed,
        stratify=train_val_df[stratify_column]
    )
    
    # Print split statistics
    print(f"\n{'='*60}")
    print(f"Split Statistics:")
    print(f"{'='*60}")
    print(f"Train set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val set:   {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test set:  {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    # Print label distribution for each split
    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"\n{split_name} set label distribution:")
        print(split_df[stratify_column].value_counts())
        print(f"Proportions:")
        print(split_df[stratify_column].value_counts(normalize=True))
    
    # Determine output directory
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"Saved splits to:")
    print(f"  Train: {train_path}")
    print(f"  Val:   {val_path}")
    print(f"  Test:  {test_path}")
    print(f"{'='*60}")
    
    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(
        description="Split dataset into train/val/test sets with stratification"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save output files (default: same as input)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Proportion of training data (default: 0.8)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Proportion of validation data (default: 0.1)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Proportion of test data (default: 0.1)"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--stratify-column",
        type=str,
        default="label",
        help="Column name to use for stratification (default: 'label')"
    )
    
    args = parser.parse_args()
    
    split_dataset(
        input_csv=args.input,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed,
        stratify_column=args.stratify_column
    )


if __name__ == "__main__":
    main()

