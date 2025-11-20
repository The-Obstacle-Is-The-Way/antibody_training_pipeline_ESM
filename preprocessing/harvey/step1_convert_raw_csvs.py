#!/usr/bin/env python3
"""
Harvey Dataset CSV Conversion Script

Combines high and low polyreactivity CSVs from raw sources
into a single unified dataset for preprocessing.

Source: data/test/harvey/raw/
- high_polyreactivity_high_throughput.csv (71,772 sequences)
- low_polyreactivity_high_throughput.csv (69,702 sequences)

Output: data/test/harvey/processed/harvey.csv (141,474 sequences)

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

import pandas as pd

from preprocessing.logging_config import setup_logger
from preprocessing.paths import HARVEY_FULL_CSV, HARVEY_HIGH_CSV, HARVEY_LOW_CSV

logger = setup_logger(__name__)


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
    logger.info(f"Reading {high_csv_path}...")
    df_high = pd.read_csv(high_csv_path)
    logger.info(f"  High polyreactivity: {len(df_high)} sequences")

    logger.info(f"Reading {low_csv_path}...")
    df_low = pd.read_csv(low_csv_path)
    logger.info(f"  Low polyreactivity: {len(df_low)} sequences")

    # IMGT position columns (1-128)
    imgt_cols = [str(i) for i in range(1, 129)]

    # Extract full sequences from IMGT positions
    logger.info("Extracting sequences from IMGT positions...")
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
    logger.info("Combining datasets...")
    df_combined = pd.concat(
        [
            df_high[["seq", "CDR1_nogaps", "CDR2_nogaps", "CDR3_nogaps", "label"]],
            df_low[["seq", "CDR1_nogaps", "CDR2_nogaps", "CDR3_nogaps", "label"]],
        ],
        ignore_index=True,
    )

    # Save combined dataset
    logger.info(f"Saving to {output_path}...")
    df_combined.to_csv(output_path, index=False)

    # Statistics
    logger.info(f"\nCombined dataset: {len(df_combined)} sequences")
    logger.info(f"  High polyreactivity (label=1): {(df_combined['label'] == 1).sum()}")
    logger.info(f"  Low polyreactivity (label=0): {(df_combined['label'] == 0).sum()}")
    logger.info(
        f"  Balance: {(df_combined['label'] == 1).sum() / len(df_combined) * 100:.1f}% high"
    )

    # Sequence length stats
    seq_lengths = df_combined["seq"].str.len()
    logger.info(f"\nSequence length range: {seq_lengths.min()}-{seq_lengths.max()} aa")
    logger.info(f"Mean length: {seq_lengths.mean():.1f} aa")

    return df_combined


def main() -> int:
    """Main conversion pipeline."""
    # Paths
    high_csv = HARVEY_HIGH_CSV
    low_csv = HARVEY_LOW_CSV
    output_csv = HARVEY_FULL_CSV

    # Validate inputs
    if not high_csv.exists():
        logger.info(f"Error: {high_csv} not found!")
        logger.info("Please ensure raw files are in data/test/harvey/raw/")
        return 1

    if not low_csv.exists():
        logger.info(f"Error: {low_csv} not found!")
        logger.info("Please ensure raw files are in data/test/harvey/raw/")
        return 1

    logger.info("=" * 70)
    logger.info("Harvey Dataset: CSV Conversion")
    logger.info("=" * 70)
    logger.info(f"\nInput (high):  {high_csv}")
    logger.info(f"Input (low):   {low_csv}")
    logger.info(f"Output:        {output_csv}")

    # Convert
    df = convert_harvey_csvs(str(high_csv), str(low_csv), str(output_csv))

    logger.info("\n" + "=" * 70)
    logger.info("[DONE] Harvey CSV Conversion Complete!")
    logger.info("=" * 70)

    logger.info(f"\nOutput file: {output_csv.absolute()}")
    logger.info(f"Total sequences: {len(df)}")
    logger.info("\nNext steps:")
    logger.info(
        "  1. Run preprocessing/harvey/step2_extract_fragments.py to extract fragments"
    )
    logger.info("  2. Validate with scripts/validation/validate_fragments.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
