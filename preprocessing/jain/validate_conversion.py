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
import hashlib
from pathlib import Path

import pandas as pd
import pandas.testing as pdt

# Clean package import (no sys.path manipulation needed)
from preprocessing.jain.step1_convert_excel_to_csv import (
    VALID_AA,
    calculate_flags,
    load_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the Jain dataset ELISA-only conversion output."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("test_datasets/jain/processed/jain_with_private_elisa_FULL.csv"),
        help="Path to the converted CSV file (ELISA SSOT).",
    )
    return parser.parse_args()


def checksum(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def validate_sequences(df: pd.DataFrame) -> dict[str, int]:
    """Return counts of sequences containing invalid residues."""
    invalid_counts = {"heavy": 0, "light": 0}
    for seq in df["vh_sequence"].dropna():
        if set(seq) - VALID_AA:
            invalid_counts["heavy"] += 1
    for seq in df["vl_sequence"].dropna():
        if set(seq) - VALID_AA:
            invalid_counts["light"] += 1
    return invalid_counts


def main() -> None:
    args = parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV file not found: {args.csv}")

    csv_df = pd.read_csv(args.csv)

    # Regenerate from source to verify consistency
    print("Regenerating dataset from source Excel files...")
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

    regenerated_sorted = regenerated.sort_values("id").reset_index(drop=True)
    csv_sorted = csv_df.sort_values("id").reset_index(drop=True)

    pdt.assert_frame_equal(
        regenerated_sorted, csv_sorted, check_dtype=False, check_like=True
    )

    # High-level stats
    print("=" * 60)
    print("Jain Conversion Validation (ELISA-only SSOT)")
    print("=" * 60)
    print(f"Rows: {len(csv_df)}, Columns: {len(csv_df.columns)}")

    print("\nELISA flag distribution (0-6 range):")
    for flag_count in range(7):  # 0-6 inclusive
        count = (csv_df["elisa_flags"] == flag_count).sum()
        pct = count / len(csv_df) * 100
        print(f"  {flag_count} ELISA flags: {count:3d} antibodies ({pct:5.1f}%)")

    print("\nFlag category distribution (ELISA-based):")
    print(csv_df["flag_category"].value_counts().sort_index())

    print("\nLabel distribution (ELISA-based, nullable):")
    print(csv_df["label"].value_counts(dropna=False))

    # Expected counts
    expected = {"specific": 94, "nonspecific": 22, "mild": 21}
    actual_specific = (csv_df["label"] == 0).sum()
    actual_nonspecific = (csv_df["label"] == 1).sum()
    actual_mild = csv_df["label"].isna().sum()

    print(
        f"\nExpected distribution: {expected['specific']}/{expected['nonspecific']}/{expected['mild']}"
    )
    print(
        f"Actual distribution:   {actual_specific}/{actual_nonspecific}/{actual_mild}"
    )

    if (
        actual_specific == expected["specific"]
        and actual_nonspecific == expected["nonspecific"]
        and actual_mild == expected["mild"]
    ):
        print("✅ Distribution matches ELISA SSOT expectations!")
    else:
        print("⚠️ WARNING: Distribution mismatch!")

    invalid = validate_sequences(csv_df)
    if invalid["heavy"] == 0 and invalid["light"] == 0:
        print(
            "\nSequence validation: ✅ all VH/VL sequences contain only valid amino acids"
        )
    else:
        print("\nSequence validation: ⚠ issues detected")
        print(f"  Heavy chains with invalid residues: {invalid['heavy']}")
        print(f"  Light chains with invalid residues: {invalid['light']}")

    print("\nChecksum (SHA256):", checksum(args.csv))
    print("\nValidation complete ✅")


if __name__ == "__main__":
    main()
