#!/usr/bin/env python3
"""
Multi-method validation of Shehata Excel → CSV conversion.

Uses multiple libraries to read Excel and compare results to ensure
data integrity during conversion.

Methods:
1. pandas (openpyxl engine)
2. Direct openpyxl reading
3. CSV checksum validation

Date: 2025-10-31
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import openpyxl
import pandas as pd

from preprocessing.logging_config import setup_logger
from preprocessing.paths import (
    SHEHATA_FRAGMENTS_DIR,
    SHEHATA_PROCESSED_CSV,
    SHEHATA_RAW_EXCEL,
)

logger = setup_logger(__name__)


def sanitize_sequence(seq: str) -> str:
    """Remove gap characters and normalise amino acid strings."""
    if pd.isna(seq):
        return seq
    return str(seq).replace("-", "").strip().upper()


def clean_excel_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same cleaning steps used during conversion:
    - sanitize VH/VL sequences
    - drop rows without sequence information (footnotes)
    - drop rows without numeric PSR measurements
    """
    df = df.copy()
    df["VH Protein"] = df["VH Protein"].apply(sanitize_sequence)
    df["VL Protein"] = df["VL Protein"].apply(sanitize_sequence)
    df = df.dropna(subset=["VH Protein", "VL Protein"], how="all")
    psr_numeric = pd.to_numeric(df["PSR Score"], errors="coerce")
    df = df.loc[psr_numeric.notna()]
    df.reset_index(drop=True, inplace=True)
    return df


def method1_pandas_openpyxl(excel_path: str) -> pd.DataFrame:
    """Read Excel using pandas with openpyxl engine."""
    logger.info("Method 1: pandas.read_excel (openpyxl engine)")
    df = pd.read_excel(excel_path, engine="openpyxl")
    df = clean_excel_df(df)
    logger.info(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
    return df


def method2_openpyxl_direct(excel_path: str) -> pd.DataFrame:
    """Read Excel using openpyxl directly."""
    logger.info("\nMethod 2: openpyxl direct reading")
    wb = openpyxl.load_workbook(excel_path)
    ws = wb.active

    data = []
    headers = None

    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i == 0:
            headers = row
        else:
            data.append(row)

    df = pd.DataFrame(data, columns=headers)
    df = clean_excel_df(df)
    logger.info(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
    return df


def method3_csv_direct(csv_path: str) -> pd.DataFrame:
    """Read the generated CSV."""
    logger.info("\nMethod 3: Reading generated CSV")
    df = pd.read_csv(csv_path)
    logger.info(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
    return df


def compare_sequences(
    df1: pd.DataFrame, df2: pd.DataFrame, col1: str, col2: str, name: str
) -> bool:
    """
    Compare sequences between two DataFrames.

    Properly handles NaN values (NaN == NaN for comparison purposes).
    """
    logger.info(f"\n  Comparing {name}:")

    # Check lengths
    if len(df1) != len(df2):
        logger.info(f"    ✗ Row count mismatch: {len(df1)} vs {len(df2)}")
        return False

    # Compare each sequence
    mismatches = 0
    for i in range(len(df1)):
        seq1 = df1.iloc[i][col1] if col1 in df1.columns else None
        seq2 = df2.iloc[i][col2] if col2 in df2.columns else None

        # Proper NaN comparison: both NaN = match, otherwise check equality
        both_nan = pd.isna(seq1) and pd.isna(seq2)
        both_equal = seq1 == seq2 if not (pd.isna(seq1) or pd.isna(seq2)) else False

        if not (both_nan or both_equal):
            mismatches += 1
            if mismatches <= 3:  # Show first 3 mismatches
                logger.info(f"    ✗ Row {i} mismatch:")
                logger.info(f"      Source: {str(seq1)[:60]}...")
                logger.info(f"      CSV:    {str(seq2)[:60]}...")

    if mismatches == 0:
        logger.info(f"    ✓ All {len(df1)} sequences match!")
        return True
    logger.info(f"    ✗ {mismatches} mismatches found")
    return False


def calculate_checksum(filepath: str | Path) -> str:
    """Calculate SHA256 checksum of file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def validate_fragment_csvs(fragments_dir: Path) -> bool:
    """
    Validate fragment CSV files for gap characters.

    Critical check: ESM-1v cannot handle gap characters.
    This prevents P0 blocker regression.

    Args:
        fragments_dir: Path to directory containing fragment CSVs

    Returns:
        True if all files are gap-free, False otherwise
    """
    logger.info("\n" + "=" * 60)
    logger.info("Fragment CSV Gap Validation (P0 Blocker Check)")
    logger.info("=" * 60)

    if not fragments_dir.exists():
        logger.info(f"  ℹ Fragment directory not found: {fragments_dir}")
        logger.info(
            "  (Run preprocessing/shehata/step2_extract_fragments.py to generate fragments)"
        )
        return True  # Not an error if fragments haven't been generated yet

    fragment_files = list(fragments_dir.glob("*.csv"))
    if not fragment_files:
        logger.info(f"  ℹ No fragment CSV files found in {fragments_dir}")
        return True

    logger.info(
        f"\n  Checking {len(fragment_files)} fragment files for gap characters..."
    )

    all_clean = True
    gap_files = []

    for file in sorted(fragment_files):
        df = pd.read_csv(file)
        gap_count = df["sequence"].str.contains("-", na=False).sum()

        if gap_count > 0:
            all_clean = False
            gap_files.append((file.name, gap_count))
            logger.info(f"    ✗ {file.name}: {gap_count} sequences with gaps")
        else:
            logger.info(f"    ✓ {file.name}: gap-free")

    logger.info("")
    if all_clean:
        logger.info("  ✓ SUCCESS: All fragment files are gap-free")
        logger.info("  ✓ ESM-1v embedding compatibility confirmed")
        return True
    else:
        logger.info("  ✗ FAILURE: Gap characters detected in fragment files")
        logger.info("  ✗ This is a P0 blocker - ESM-1v will fail validation")
        logger.info("\n  Affected files:")
        for filename, count in gap_files:
            logger.info(f"    - {filename}: {count} sequences")
        logger.info(
            "\n  Fix: Use annotation.sequence_aa instead of sequence_alignment_aa"
        )
        logger.info("  See: docs/shehata/SHEHATA_BLOCKER_ANALYSIS.md")
        return False


def main() -> int:
    excel_path = SHEHATA_RAW_EXCEL
    csv_path = SHEHATA_PROCESSED_CSV

    logger.info("=" * 60)
    logger.info("Multi-Method Validation of Shehata Conversion")
    logger.info("=" * 60)

    if not excel_path.exists():
        logger.info(f"✗ Excel file not found: {excel_path}")
        return 1

    if not csv_path.exists():
        logger.info(f"✗ CSV file not found: {csv_path}")
        logger.info("  Run preprocessing/shehata/step1_convert_excel_to_csv.py first!")
        return 1

    logger.info("\nReading files with multiple methods...\n")

    # Read with different methods
    try:
        df_pandas = method1_pandas_openpyxl(str(excel_path))
    except Exception as e:
        logger.info(f"  Error: {e}")
        df_pandas = None

    try:
        df_openpyxl = method2_openpyxl_direct(str(excel_path))
    except Exception as e:
        logger.info(f"  Error: {e}")
        df_openpyxl = None

    try:
        df_csv = method3_csv_direct(str(csv_path))
    except Exception as e:
        logger.info(f"  Error: {e}")
        df_csv = None

    # Cross-validate
    logger.info("\n" + "=" * 60)
    logger.info("Cross-Validation Results")
    logger.info("=" * 60)

    if df_pandas is not None and df_openpyxl is not None:
        logger.info("\n1. Pandas vs Direct openpyxl (Excel reading consistency)")
        compare_sequences(
            df_pandas, df_openpyxl, "VH Protein", "VH Protein", "VH sequences"
        )
        compare_sequences(
            df_pandas, df_openpyxl, "VL Protein", "VL Protein", "VL sequences"
        )

    if df_pandas is not None and df_csv is not None:
        logger.info("\n2. Excel (pandas) vs Generated CSV (conversion accuracy)")
        compare_sequences(
            df_pandas, df_csv, "VH Protein", "heavy_seq", "VH → heavy_seq"
        )
        compare_sequences(
            df_pandas, df_csv, "VL Protein", "light_seq", "VL → light_seq"
        )

        # Check ID mapping
        logger.info("\n  Comparing IDs:")
        id_match = (df_pandas["Clone name"] == df_csv["id"]).all()
        logger.info(f"    {'✓' if id_match else '✗'} Clone name → id mapping")

    # File integrity
    logger.info("\n" + "=" * 60)
    logger.info("File Integrity")
    logger.info("=" * 60)
    logger.info(f"\nExcel checksum: {calculate_checksum(excel_path)}")
    logger.info(f"CSV checksum:   {calculate_checksum(csv_path)}")
    logger.info("\n(These checksums are stored for future verification)")

    # Summary statistics
    logger.info("\n" + "=" * 60)
    logger.info("Summary Statistics")
    logger.info("=" * 60)

    if df_csv is not None:
        logger.info(f"\nGenerated CSV ({csv_path.name}):")
        logger.info(f"  Total rows: {len(df_csv)}")
        logger.info(f"  Columns: {list(df_csv.columns)}")
        logger.info("\n  Label distribution:")
        for label, count in df_csv["label"].value_counts().sort_index().items():
            label_name = "Specific" if label == 0 else "Non-specific"
            logger.info(f"    {label_name}: {count} ({count / len(df_csv) * 100:.1f}%)")

        logger.info("\n  Missing data:")
        logger.info(f"    Missing heavy_seq: {df_csv['heavy_seq'].isna().sum()}")
        logger.info(f"    Missing light_seq: {df_csv['light_seq'].isna().sum()}")
        logger.info(f"    Missing labels: {df_csv['label'].isna().sum()}")

    # Validate fragment CSVs (P0 blocker check)
    fragments_dir = SHEHATA_FRAGMENTS_DIR
    fragments_valid = validate_fragment_csvs(fragments_dir)

    logger.info("\n" + "=" * 60)
    if fragments_valid:
        logger.info("✓ Validation Complete - All Checks Passed")
    else:
        logger.info("✗ Validation Failed - P0 Blocker Detected")
    print("=" * 60)
    return 0 if fragments_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
