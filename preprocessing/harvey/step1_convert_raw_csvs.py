#!/usr/bin/env python3
"""
Harvey Dataset CSV Conversion Script

Combines high and low polyreactivity CSVs from raw sources
into a single unified dataset for preprocessing.

Source: test_datasets/harvey/raw/
- high_polyreactivity_high_throughput.csv (71,772 sequences)
- low_polyreactivity_high_throughput.csv (69,702 sequences)

Output: test_datasets/harvey/processed/harvey.csv (141,474 sequences)

The official Harvey CSVs contain IMGT-numbered positions (columns 1-128) and
pre-extracted CDR sequences. This script:
1. Extracts full sequences from IMGT position columns
2. Combines with pre-extracted CDRs (CDR1_nogaps, CDR2_nogaps, CDR3_nogaps)
3. Assigns binary labels (0=low polyreactivity, 1=high polyreactivity)

Date: 2025-11-01
Issue: #4 - Harvey dataset preprocessing
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd


def extract_sequence_from_imgt(row: pd.Series, imgt_cols: Sequence[str]) -> str:
    """
    Extract full sequence from IMGT-numbered position columns.

    Args:
        row: DataFrame row with IMGT position columns (1-128)
        imgt_cols: List of column names ['1', '2', ..., '128']

    Returns:
        Full sequence string with gaps removed
    """
    positions = []
    for col in imgt_cols:
        if col in row and pd.notna(row[col]) and row[col] != "-":
            positions.append(row[col])
    return "".join(positions)


def convert_harvey_csvs(
    high_csv_path: str, low_csv_path: str, output_path: str
) -> pd.DataFrame:
    """
    Convert Harvey high/low CSVs to unified format.

    Args:
        high_csv_path: Path to high_polyreactivity_high_throughput.csv
        low_csv_path: Path to low_polyreactivity_high_throughput.csv
        output_path: Path to output harvey.csv

    Returns:
        Combined DataFrame
    """
    print(f"Reading {high_csv_path}...")
    df_high = pd.read_csv(high_csv_path)
    print(f"  High polyreactivity: {len(df_high)} sequences")

    print(f"Reading {low_csv_path}...")
    df_low = pd.read_csv(low_csv_path)
    print(f"  Low polyreactivity: {len(df_low)} sequences")

    # IMGT position columns (1-128)
    imgt_cols = [str(i) for i in range(1, 129)]

    # Extract full sequences from IMGT positions
    print("Extracting sequences from IMGT positions...")
    df_high["seq"] = df_high.apply(
        lambda row: extract_sequence_from_imgt(row, imgt_cols), axis=1
    )
    df_low["seq"] = df_low.apply(
        lambda row: extract_sequence_from_imgt(row, imgt_cols), axis=1
    )

    # Add binary labels
    df_high["label"] = 1  # high polyreactivity
    df_low["label"] = 0  # low polyreactivity

    # Combine datasets with standardized columns
    print("Combining datasets...")
    df_combined = pd.concat(
        [
            df_high[["seq", "CDR1_nogaps", "CDR2_nogaps", "CDR3_nogaps", "label"]],
            df_low[["seq", "CDR1_nogaps", "CDR2_nogaps", "CDR3_nogaps", "label"]],
        ],
        ignore_index=True,
    )

    # Save combined dataset
    print(f"Saving to {output_path}...")
    df_combined.to_csv(output_path, index=False)

    # Statistics
    print(f"\nCombined dataset: {len(df_combined)} sequences")
    print(f"  High polyreactivity (label=1): {(df_combined['label'] == 1).sum()}")
    print(f"  Low polyreactivity (label=0): {(df_combined['label'] == 0).sum()}")
    print(
        f"  Balance: {(df_combined['label'] == 1).sum() / len(df_combined) * 100:.1f}% high"
    )

    # Sequence length stats
    seq_lengths = df_combined["seq"].str.len()
    print(f"\nSequence length range: {seq_lengths.min()}-{seq_lengths.max()} aa")
    print(f"Mean length: {seq_lengths.mean():.1f} aa")

    return df_combined


def main() -> int:
    """Main conversion pipeline."""
    # Paths
    high_csv = Path("test_datasets/harvey/raw/high_polyreactivity_high_throughput.csv")
    low_csv = Path("test_datasets/harvey/raw/low_polyreactivity_high_throughput.csv")
    output_csv = Path("test_datasets/harvey/processed/harvey.csv")

    # Validate inputs
    if not high_csv.exists():
        print(f"Error: {high_csv} not found!")
        print("Please ensure raw files are in test_datasets/harvey/raw/")
        return 1

    if not low_csv.exists():
        print(f"Error: {low_csv} not found!")
        print("Please ensure raw files are in test_datasets/harvey/raw/")
        return 1

    print("=" * 70)
    print("Harvey Dataset: CSV Conversion")
    print("=" * 70)
    print(f"\nInput (high):  {high_csv}")
    print(f"Input (low):   {low_csv}")
    print(f"Output:        {output_csv}")
    print()

    # Convert
    df = convert_harvey_csvs(str(high_csv), str(low_csv), str(output_csv))

    print("\n" + "=" * 70)
    print("[DONE] Harvey CSV Conversion Complete!")
    print("=" * 70)

    print(f"\nOutput file: {output_csv.absolute()}")
    print(f"Total sequences: {len(df)}")
    print("\nNext steps:")
    print(
        "  1. Run preprocessing/harvey/step2_extract_fragments.py to extract fragments"
    )
    print("  2. Validate with scripts/validation/validate_fragments.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
