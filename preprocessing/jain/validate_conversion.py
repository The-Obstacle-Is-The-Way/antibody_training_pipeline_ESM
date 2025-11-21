#!/usr/bin/env python3
"""
Validation harness for the Jain Excel->CSV conversion (ELISA-only SSOT).

Checks performed:
1. Re-runs the conversion pipeline in-memory and compares against SSOT CSV
2. Verifies ELISA flag counts, label distribution, and column integrity
3. Confirms amino acid sequences contain only valid residues
4. Prints SHA256 checksum for provenance tracking

Expected output:
- jain_with_private_elisa_FULL.csv: 137 antibodies (94 specific, 22 non-specific, 21 mild)
- Distribution: ELISA flags (0-6 range), NOT flags_total
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pandas.testing as pdt

# Clean package import (no sys.path manipulation needed)
from preprocessing.jain.step1_convert_excel_to_csv import (
    calculate_flags,
    load_data,
)
from preprocessing.logging_config import setup_logger
from preprocessing.paths import JAIN_FULL_CSV
from preprocessing.validation_utils import (
    calculate_checksum,
    validate_amino_acids,
)

logger = setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the Jain dataset ELISA-only conversion output."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=JAIN_FULL_CSV,
        help="Path to the converted CSV file (ELISA SSOT).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV file not found: {args.csv}")

    csv_df = pd.read_csv(args.csv)

    # Regenerate from source to verify consistency
    logger.info("Regenerating dataset from source Excel files...")
    regenerated = load_data()
    regenerated = calculate_flags(regenerated)

    # Prepare columns to match CSV output
    regenerated = regenerated[
        [
            "Name",
            "VH",
            "VL",
            "elisa_flags",
            "total_flags",
            "flag_category",
            "label",
            "flag_cardiolipin",
            "flag_klh",
            "flag_lps",
            "flag_ssdna",
            "flag_dsdna",
            "flag_insulin",
            "flag_bvp",
            "flag_self_interaction",
            "flag_chromatography",
            "flag_stability",
        ]
    ].copy()

    regenerated = regenerated.rename(
        columns={"Name": "id", "VH": "vh_sequence", "VL": "vl_sequence"}
    )

    # Align dtypes that may have changed during CSV round-trip
    for col in ["elisa_flags", "total_flags", "label"]:
        csv_df[col] = csv_df[col].astype("Int64")
        regenerated[col] = regenerated[col].astype("Int64")

    # Convert flag_category from Categorical to string for comparison
    if "flag_category" in csv_df.columns:
        csv_df["flag_category"] = csv_df["flag_category"].astype(str)
    if "flag_category" in regenerated.columns:
        regenerated["flag_category"] = regenerated["flag_category"].astype(str)

    regenerated_sorted = regenerated.sort_values("id").reset_index(drop=True)
    csv_sorted = csv_df.sort_values("id").reset_index(drop=True)

    pdt.assert_frame_equal(
        regenerated_sorted, csv_sorted, check_dtype=False, check_like=True
    )

    # High-level stats
    logger.info("=" * 60)
    logger.info("Jain Conversion Validation (ELISA-only SSOT)")
    logger.info("=" * 60)
    logger.info(f"Rows: {len(csv_df)}, Columns: {len(csv_df.columns)}")

    logger.info("\nELISA flag distribution (0-6 range):")
    for flag_count in range(7):  # 0-6 inclusive
        count = (csv_df["elisa_flags"] == flag_count).sum()
        pct = count / len(csv_df) * 100
        logger.info(f"  {flag_count} ELISA flags: {count:3d} antibodies ({pct:5.1f}%)")

    logger.info("\nFlag category distribution (ELISA-based):")
    logger.info(csv_df["flag_category"].value_counts().sort_index())

    logger.info("\nLabel distribution (ELISA-based, nullable):")
    logger.info(str(csv_df["label"].value_counts(dropna=False)))

    # Expected counts
    expected = {"specific": 94, "nonspecific": 22, "mild": 21}
    actual_specific = (csv_df["label"] == 0).sum()
    actual_nonspecific = (csv_df["label"] == 1).sum()
    actual_mild = csv_df["label"].isna().sum()

    logger.info(
        f"\nExpected distribution: {expected['specific']}/{expected['nonspecific']}/{expected['mild']}"
    )
    logger.info(
        f"Actual distribution:   {actual_specific}/{actual_nonspecific}/{actual_mild}"
    )

    if (
        actual_specific == expected["specific"]
        and actual_nonspecific == expected["nonspecific"]
        and actual_mild == expected["mild"]
    ):
        logger.info("Distribution matches ELISA SSOT expectations!")
    else:
        logger.warning("WARNING: Distribution mismatch!")

    # Validate sequences using shared utility
    errors_vh = validate_amino_acids(csv_df, "vh_sequence", "Jain (VH)")
    errors_vl = validate_amino_acids(csv_df, "vl_sequence", "Jain (VL)")

    if not errors_vh and not errors_vl:
        logger.info(
            "\nSequence validation: ✅ all VH/VL sequences contain only valid amino acids"
        )
    else:
        logger.info("\nSequence validation: ⚠ issues detected (details logged above)")

    logger.info(f"\nChecksum (SHA256): {calculate_checksum(args.csv)}")
    logger.info("\nValidation complete ✅")


if __name__ == "__main__":
    main()
