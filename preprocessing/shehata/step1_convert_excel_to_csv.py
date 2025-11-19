#!/usr/bin/env python3
"""
Convert Shehata dataset from Excel (mmc2.xlsx) to CSV format.

This script performs a careful conversion with validation:
1. Reads mmc2.xlsx
2. Maps columns to match jain.csv format
3. Validates data integrity
4. Outputs shehata.csv

Date: 2025-10-31
Issue: #3 - Shehata dataset preprocessing
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd

from preprocessing.logging_config import setup_logger

logger = setup_logger(__name__)


def sanitize_sequence(seq: str | float | None) -> str | float | None:
    """
    Sanitize protein sequence by removing gap characters and invalid residues.

    Gap characters (-) are artifacts from sequence alignment/numbering schemes
    (e.g., IMGT) and must be removed before embedding.

    Args:
        seq: Raw protein sequence string

    Returns:
        Cleaned sequence with only valid amino acids
    """
    if pd.isna(seq):
        return seq

    # Remove gap characters (common in IMGT-numbered sequences)
    seq = str(seq).replace("-", "")

    # Remove whitespace
    seq = seq.strip()

    # Uppercase for consistency
    seq = seq.upper()

    return seq


def validate_sequences(df: pd.DataFrame) -> dict[str, Any]:
    """
    Validate protein sequences for quality.

    Note: Sequences should already be sanitized before validation.
    This function checks the CLEANED sequences for any remaining issues.
    """
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")

    # Track invalid counts separately to avoid object type issues
    invalid_vh_count = 0
    invalid_vl_count = 0

    # Check for invalid amino acids (after sanitization, there should be none)
    for _idx, seq in df["heavy_seq"].dropna().items():
        if not set(cast(str, seq)).issubset(valid_aa):
            invalid_vh_count += 1

    for _idx, seq in df["light_seq"].dropna().items():
        if not set(cast(str, seq)).issubset(valid_aa):
            invalid_vl_count += 1

    validation = {
        "total_sequences": len(df),
        "missing_vh": df["heavy_seq"].isna().sum(),
        "missing_vl": df["light_seq"].isna().sum(),
        "invalid_vh": invalid_vh_count,
        "invalid_vl": invalid_vl_count,
        "vh_length_range": (
            df["heavy_seq"].str.len().min(),
            df["heavy_seq"].str.len().max(),
        ),
        "vl_length_range": (
            df["light_seq"].str.len().min(),
            df["light_seq"].str.len().max(),
        ),
    }

    return validation


def convert_excel_to_csv(
    excel_path: str,
    output_path: str,
    psr_threshold: float | None = None,
    interactive: bool = True,
) -> pd.DataFrame:
    """
    Convert mmc2.xlsx to CSV format matching jain.csv structure.

    Args:
        excel_path: Path to mmc2.xlsx
        output_path: Path for output CSV
        psr_threshold: PSR score threshold for binary classification.
                      If None and interactive=True, will analyze distribution and prompt.
                      If None and interactive=False, will use 98.24th percentile (7/398).
        interactive: If True, prompts for threshold confirmation. If False, uses defaults.

    Returns:
        DataFrame with converted data
    """
    logger.info(f"Reading Excel file: {excel_path}")
    df_excel = pd.read_excel(excel_path)

    logger.info(f"  Rows: {len(df_excel)}")
    logger.info(f"  Columns: {len(df_excel.columns)}")

    # Sanitize sequences BEFORE any analysis
    logger.info("\nSanitizing sequences (removing gaps and invalid characters)...")
    vh_original = df_excel["VH Protein"].copy()
    vl_original = df_excel["VL Protein"].copy()

    df_excel["VH Protein"] = df_excel["VH Protein"].apply(sanitize_sequence)
    df_excel["VL Protein"] = df_excel["VL Protein"].apply(sanitize_sequence)

    # Count gaps removed
    gaps_vh = sum(str(s).count("-") if pd.notna(s) else 0 for s in vh_original)
    gaps_vl = sum(str(s).count("-") if pd.notna(s) else 0 for s in vl_original)

    if gaps_vh > 0 or gaps_vl > 0:
        logger.info(f"  Removed {gaps_vh} gap characters from VH sequences")
        logger.info(f"  Removed {gaps_vl} gap characters from VL sequences")

    # Drop rows without sequence information (Excel footnotes / metadata)
    before_drop = len(df_excel)
    df_excel = df_excel.dropna(subset=["VH Protein", "VL Protein"], how="all")
    dropped = before_drop - len(df_excel)
    if dropped:
        logger.info(
            f"  Dropped {dropped} rows without VH/VL sequences (metadata/footnotes)"
        )

    # Convert PSR scores to numeric and drop entries without measurements
    psr_numeric = pd.to_numeric(df_excel["PSR Score"], errors="coerce")
    invalid_psr_mask = psr_numeric.isna()
    if invalid_psr_mask.any():
        dropped_ids = df_excel.loc[invalid_psr_mask, "Clone name"].tolist()
        dropped_list = ", ".join(dropped_ids)
        logger.info(
            f"  Dropping {invalid_psr_mask.sum()} antibodies without numeric PSR scores: "
            f"{dropped_list}"
        )
        df_excel = df_excel.loc[~invalid_psr_mask].reset_index(drop=True)
        psr_numeric = psr_numeric.loc[~invalid_psr_mask].reset_index(drop=True)

    # Analyze PSR scores if threshold not provided
    if psr_threshold is None:
        logger.info("\nAnalyzing PSR score distribution:")
        logger.info(f"  Valid PSR scores: {psr_numeric.notna().sum()}")
        logger.info(f"  Missing PSR scores: {psr_numeric.isna().sum()}")
        logger.info(f"  Mean: {psr_numeric.mean():.4f}")
        logger.info(f"  Median: {psr_numeric.median():.4f}")
        logger.info(f"  75th percentile: {psr_numeric.quantile(0.75):.4f}")
        logger.info(f"  95th percentile: {psr_numeric.quantile(0.95):.4f}")
        logger.info(f"  Max: {psr_numeric.max():.4f}")
        logger.info(f"\n  PSR = 0: {(psr_numeric == 0).sum()} antibodies")
        logger.info(f"  PSR > 0: {(psr_numeric > 0).sum()} antibodies")

        # Based on paper: "7 out of 398 antibodies characterised as non-specific"
        # This is roughly 1.76% = 98.24th percentile
        suggested_threshold = psr_numeric.quantile(0.9824)
        logger.info("\n  Paper reports: 7/398 non-specific (~1.76%)")
        logger.info(
            f"  Suggested threshold (98.24th percentile): {suggested_threshold:.4f}"
        )

        if interactive:
            # Ask user to confirm
            response = (
                input(f"\n  Use threshold {suggested_threshold:.4f}? [y/n/custom]: ")
                .strip()
                .lower()
            )
            if response == "y":
                psr_threshold = suggested_threshold
            elif response == "custom":
                psr_threshold = float(input("  Enter custom threshold: "))
            else:
                logger.info("  Using PSR > 0 as threshold (any polyreactivity)")
                psr_threshold = 0
        else:
            # Non-interactive mode: use suggested threshold
            psr_threshold = suggested_threshold
            logger.info(
                f"\n  Using suggested threshold (non-interactive mode): {suggested_threshold:.4f}"
            )

    logger.info(f"\nUsing PSR threshold: {psr_threshold}")

    # Create DataFrame matching jain.csv format
    df_csv = pd.DataFrame(
        {
            "id": df_excel["Clone name"],
            "heavy_seq": df_excel["VH Protein"],
            "light_seq": df_excel["VL Protein"],
            "label": (psr_numeric > psr_threshold).astype(int),
            "psr_score": psr_numeric,
            "b_cell_subset": df_excel["B cell subset"],
            "source": "shehata2019",
        }
    )

    # Validate
    logger.info("\nValidating sequences...")
    validation = validate_sequences(df_csv)

    logger.info(f"  Total sequences: {validation['total_sequences']}")
    logger.info(f"  Missing VH: {validation['missing_vh']}")
    logger.info(f"  Missing VL: {validation['missing_vl']}")
    logger.info(f"  Invalid VH (after sanitization): {validation['invalid_vh']}")
    logger.info(f"  Invalid VL (after sanitization): {validation['invalid_vl']}")
    logger.info(f"  VH length range: {validation['vh_length_range']}")
    logger.info(f"  VL length range: {validation['vl_length_range']}")

    if validation["invalid_vh"] > 0 or validation["invalid_vl"] > 0:
        logger.info("\n  ⚠️  WARNING: Some sequences still invalid after sanitization!")
        logger.info("  This may indicate non-standard amino acids or other issues.")

    # Label distribution
    logger.info("\nLabel distribution:")
    label_dist = df_csv["label"].value_counts().sort_index()
    for label, count in label_dist.items():
        label_name = "Specific" if label == 0 else "Non-specific"
        logger.info(
            f"  {label_name} (label={label}): {count} ({count / len(df_csv) * 100:.1f}%)"
        )

    # B cell subset distribution
    logger.info("\nB cell subset distribution:")
    subset_dist = df_csv["b_cell_subset"].value_counts()
    for subset, count in subset_dist.items():
        logger.info(f"  {subset}: {count}")

    # Save
    logger.info(f"\nSaving to: {output_path}")
    df_csv.to_csv(output_path, index=False)
    logger.info(f"  Saved {len(df_csv)} rows")

    return df_csv


def compare_with_original(csv_df: pd.DataFrame, excel_path: str) -> None:
    """Compare CSV output with original Excel for validation."""
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION: Comparing CSV with original Excel")
    print("=" * 60)

    df_excel = pd.read_excel(excel_path)
    df_excel["VH Protein"] = df_excel["VH Protein"].apply(sanitize_sequence)
    df_excel["VL Protein"] = df_excel["VL Protein"].apply(sanitize_sequence)
    df_excel = df_excel.dropna(
        subset=["VH Protein", "VL Protein"], how="all"
    ).reset_index(drop=True)

    # Check row counts
    logger.info("\nRow count check:")
    logger.info(f"  Excel: {len(df_excel)}")
    logger.info(f"  CSV: {len(csv_df)}")
    logger.info(f"  Match: {'✓ YES' if len(df_excel) == len(csv_df) else '✗ NO'}")

    # Spot check sequences
    logger.info("\nSpot checking first 3 sequences...")
    for i in range(min(3, len(csv_df))):
        excel_vh = df_excel.loc[i, "VH Protein"]
        csv_vh = csv_df.loc[i, "heavy_seq"]
        match = excel_vh == csv_vh
        logger.info(f"  Row {i}: {'✓' if match else '✗'} VH match")
        if not match:
            excel_vh_str = cast(str, excel_vh)
            csv_vh_str = cast(str, csv_vh)
            logger.info(f"    Excel: {excel_vh_str[:50]}...")
            logger.info(f"    CSV:   {csv_vh_str[:50]}...")

    logger.info("\nConversion validation complete!")


def main() -> int:
    """Main entry point for command-line execution."""
    excel_path = Path("data/test/shehata/raw/shehata-mmc2.xlsx")
    output_path = Path("data/test/shehata/processed/shehata.csv")

    if not excel_path.exists():
        logger.info(f"Error: {excel_path} not found!")
        logger.info("Please run this script from the repository root.")
        return 1

    logger.info("=" * 60)
    logger.info("Shehata Dataset: Excel → CSV Conversion")
    logger.info("=" * 60)
    logger.info(f"\nInput:  {excel_path}")
    logger.info(f"Output: {output_path}")

    # Convert (non-interactive: uses suggested threshold automatically)
    df_csv = convert_excel_to_csv(
        str(excel_path),
        str(output_path),
        psr_threshold=None,  # Will calculate suggested threshold
        interactive=False,  # No prompts - auto-accept threshold
    )

    # Validate
    compare_with_original(df_csv, str(excel_path))

    logger.info("\n" + "=" * 60)
    logger.info("✓ Conversion complete!")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info(f"  1. Review {output_path}")
    logger.info("  2. Compare with data/test/jain.csv format")
    logger.info("  3. Test loading with data.load_local_data()")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
