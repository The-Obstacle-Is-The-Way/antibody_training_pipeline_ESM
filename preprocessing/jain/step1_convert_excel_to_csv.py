#!/usr/bin/env python3
"""
Convert Jain Dataset with Private Disaggregated ELISA Data

CORRECTED Novo Nordisk Methodology (ELISA-ONLY):
  - Use ONLY the 6 ELISA antigens for flag calculation (0-6 range)
  - Threshold: >=4 ELISA flags for non-specific (">3" in paper)
  - Exclude mild (ELISA 1-3 flags) from test set

This replaces the WRONG total_flags (0-10) methodology!

Expected output:
  - FULL.csv: 137 antibodies with all flags
  - 116-ELISA-ONLY.csv: 116 antibodies (94 ELISA=0 + 22 ELISA>=4)

Evidence for ELISA-only:
  - Figure S13 shows x-axis "ELISA flag" (singular) with range 0-6
  - Table 2: "ELISA with a panel of 6 ligands"
  - Paper repeatedly refers to "non-specificity ELISA flags"

Date: 2025-11-03
Status: CORRECTED - Using ELISA-only (NOT total flags!)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from preprocessing.logging_config import setup_logger

logger = setup_logger(__name__)

# Valid amino acids for sequence validation
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Assay cluster definitions (placeholder for validation script compatibility)
ASSAY_CLUSTERS: dict[str, Any] = {}


def load_data() -> pd.DataFrame:
    """Load private ELISA + public SD files."""
    logger.info("Loading data files...")

    # Private ELISA (6 individual antigens)
    private = pd.read_excel(
        "data/test/jain/raw/Private_Jain2017_ELISA_indiv.xlsx",
        sheet_name="Individual-ELISA",
    )
    logger.info(f"  Private ELISA: {len(private)} antibodies")

    # Public SD files
    sd01 = pd.read_excel("data/test/jain/raw/jain-pnas.1616408114.sd01.xlsx")
    sd02 = pd.read_excel("data/test/jain/raw/jain-pnas.1616408114.sd02.xlsx")
    sd03 = pd.read_excel(
        "data/test/jain/raw/jain-pnas.1616408114.sd03.xlsx",
        sheet_name="Results-12-assays",
    )

    logger.info(f"  SD01 (metadata): {len(sd01)} antibodies")
    logger.info(f"  SD02 (sequences): {len(sd02)} antibodies")
    logger.info(f"  SD03 (assays): {len(sd03)} rows")

    # Merge all on 'Name'
    df = sd01.merge(sd02[["Name", "VH", "VL"]], on="Name", how="inner")
    df = df.merge(sd03, on="Name", how="inner")
    df = df.merge(private, on="Name", how="inner")

    logger.info(f"  Merged: {len(df)} antibodies\n")

    if len(df) != 137:
        logger.info(f"  WARNING: Expected 137 antibodies, got {len(df)}")

    return df


def calculate_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate flags using ELISA-ONLY methodology (Novo's actual approach).

    ELISA Flags (0-6 range):
      - Flag per antigen: Cardiolipin, KLH, LPS, ssDNA, dsDNA, Insulin
      - Threshold: 1.9 OD per antigen
      - Sum: 0-6 flags

    Non-ELISA flags (for reference, NOT used for labeling):
      - BVP, self-interaction, chromatography, stability
      - These are calculated but NOT used in mild exclusion!

    Labels (ELISA-ONLY):
      - Specific: ELISA = 0
      - Mild (EXCLUDED): ELISA 1-3
      - Non-specific: ELISA >= 4
    """
    logger.info("Calculating flags (CORRECTED: ELISA-ONLY methodology)...")

    # === ELISA FLAGS (0-6 range) ===
    elisa_threshold = 1.9

    df["flag_cardiolipin"] = (df["ELISA Cardiolipin"] > elisa_threshold).astype(int)
    df["flag_klh"] = (df["ELISA KLH"] > elisa_threshold).astype(int)
    df["flag_lps"] = (df["ELISA LPS"] > elisa_threshold).astype(int)
    df["flag_ssdna"] = (df["ELISA ssDNA"] > elisa_threshold).astype(int)
    df["flag_dsdna"] = (df["ELISA dsDNA"] > elisa_threshold).astype(int)
    df["flag_insulin"] = (df["ELISA Insulin"] > elisa_threshold).astype(int)

    df["elisa_flags"] = (
        df["flag_cardiolipin"]
        + df["flag_klh"]
        + df["flag_lps"]
        + df["flag_ssdna"]
        + df["flag_dsdna"]
        + df["flag_insulin"]
    )

    # === NON-ELISA FLAGS (for reference only, NOT used for labeling) ===
    bvp_threshold = 4.3
    df["flag_bvp"] = (df["BVP ELISA"] > bvp_threshold).astype(int)

    df["flag_self_interaction"] = (
        (df["Poly-Specificity Reagent (PSR) SMP Score (0-1)"] > 0.27)
        | (
            df[
                "Affinity-Capture Self-Interaction Nanoparticle Spectroscopy (AC-SINS) ∆λmax (nm) Average"
            ]
            > 11.8
        )
        | (df["CSI-BLI Delta Response (nm)"] > 0.01)
        | (df["CIC Retention Time (Min)"] > 10.1)
    ).astype(int)

    df["flag_chromatography"] = (
        (df["HIC Retention Time (Min)a"] > 11.7)
        | (df["SMAC Retention Time (Min)a"] > 12.8)
        | (df["SGAC-SINS AS100 ((NH4)2SO4 mM)"] < 370)
    ).astype(int)

    df["flag_stability"] = (df["Slope for Accelerated Stability"] > 0.08).astype(int)

    # Total flags (for reference)
    df["total_flags"] = (
        df["elisa_flags"]
        + df["flag_bvp"]
        + df["flag_self_interaction"]
        + df["flag_chromatography"]
        + df["flag_stability"]
    )

    # === LABELING (ELISA-ONLY!) ===
    # CRITICAL: Use ELISA flags for categorization, NOT total flags!
    df["flag_category"] = pd.cut(
        df["elisa_flags"],  # <-- ELISA-ONLY!
        bins=[-0.5, 0.5, 3.5, 6.5],
        labels=["specific", "mild", "non_specific"],
    )

    # Binary label: 0 = specific, 1 = non-specific, NaN = mild (excluded)
    df["label"] = (
        df["flag_category"]
        .map(
            {"specific": 0, "mild": pd.NA, "non_specific": 1}  # Excluded from test set
        )
        .astype("Int64")
    )

    # Print distribution
    logger.info("\n  ELISA flag distribution (0-6 range):")
    for flag_count in range(7):  # 0-6 inclusive
        count = (df["elisa_flags"] == flag_count).sum()
        pct = count / len(df) * 100
        logger.info(
            f"    {flag_count} ELISA flags: {count:3d} antibodies ({pct:5.1f}%)"
        )

    logger.info("\n  Category distribution (ELISA-ONLY):")
    for cat in ["specific", "mild", "non_specific"]:
        count = (df["flag_category"] == cat).sum()
        pct = count / len(df) * 100
        logger.info(f"    {cat}: {count:3d} antibodies ({pct:5.1f}%)")

    logger.info("\n  Label distribution (ELISA-ONLY test set):")
    label_counts = df["label"].value_counts()
    n_specific = label_counts.get(0, 0)
    n_nonspec = label_counts.get(1, 0)
    logger.info(f"    Specific (ELISA=0): {n_specific}")
    logger.info(f"    Non-specific (ELISA>=4): {n_nonspec}")
    logger.info(f"    Test set size: {n_specific + n_nonspec} (target: 116)")

    # Show comparison to total flags
    logger.info("\n  Comparison: ELISA-only vs Total flags:")
    elisa_mild = (df["elisa_flags"].between(1, 3)).sum()
    total_mild = (df["total_flags"].between(1, 3)).sum()
    logger.info(f"    Mild by ELISA (1-3): {elisa_mild} antibodies")
    logger.info(f"    Mild by Total (1-3): {total_mild} antibodies")
    logger.info(f"    Difference: {total_mild - elisa_mild} antibodies")

    return df


def save_outputs(df: pd.DataFrame) -> None:
    """Save conversion outputs."""
    output_dir = Path("data/test/jain/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\nSaving outputs...")

    # 1. FULL 137-antibody dataset
    full_output = df[
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

    full_output = full_output.rename(
        columns={"Name": "id", "VH": "vh_sequence", "VL": "vl_sequence"}
    )
    full_path = output_dir / "jain_with_private_elisa_FULL.csv"
    full_output.to_csv(full_path, index=False)
    logger.info(f"  ✓ Saved: {full_path}")
    logger.info(f"    Total: {len(full_output)} antibodies (all 137)")

    # 2. 116-antibody ELISA-ONLY test set (exclude ELISA mild 1-3)
    test_df = df[df["label"].notna()].copy()
    test_output = test_df[
        ["Name", "VH", "VL", "elisa_flags", "total_flags", "flag_category", "label"]
    ].copy()

    test_output = test_output.rename(
        columns={"Name": "id", "VH": "vh_sequence", "VL": "vl_sequence"}
    )
    test_path = output_dir / "jain_ELISA_ONLY_116.csv"
    test_output.to_csv(test_path, index=False)
    logger.info(f"  ✓ Saved: {test_path}")
    logger.info(f"    Total: {len(test_output)} antibodies (ELISA-only test set)")
    print(
        f"    Distribution: {(test_output['label'] == 0).sum()} specific / {(test_output['label'] == 1).sum()} non-specific"
    )


def convert_jain_dataset(sd01: Path, sd02: Path, sd03: Path) -> pd.DataFrame:  # noqa: ARG001
    """
    Wrapper for validation script compatibility.
    Recreates the dataset from raw files.

    Args are unused because load_data() uses hardcoded paths,
    but signature must match validation script expectations.
    """
    # This is a simplified version that loads and processes the data
    # For validation purposes, it should return the processed dataframe
    df = load_data()
    df = calculate_flags(df)
    return df


def prepare_output(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wrapper for validation script compatibility.
    Returns the dataframe in the expected output format.
    """
    return df


def main() -> int:
    logger.info("=" * 80)
    logger.info("Jain Dataset Conversion - CORRECTED ELISA-ONLY Methodology")
    print("=" * 80)
    logger.info("Using ELISA-ONLY flags (6 antigens, 0-6 range)")
    logger.info("Threshold: >=4 ELISA flags for non-specific")
    logger.info("Exclude: ELISA 1-3 as 'mild' (NOT total_flags 1-3!)")
    logger.info("=" * 80)
    print()

    # Load data
    df = load_data()

    # Calculate flags
    df = calculate_flags(df)

    # Save outputs
    save_outputs(df)

    print("\n" + "=" * 80)
    logger.info("✓ Conversion Complete!")
    logger.info("=" * 80)
    logger.info("\nFiles generated:")
    logger.info("  1. jain_with_private_elisa_FULL.csv - All 137 antibodies")
    logger.info("  2. jain_ELISA_ONLY_116.csv - ELISA-only test set (116 antibodies)")
    logger.info("\nNext step:")
    logger.info("  Investigate what QC Novo applied to get from 116 → 86")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
