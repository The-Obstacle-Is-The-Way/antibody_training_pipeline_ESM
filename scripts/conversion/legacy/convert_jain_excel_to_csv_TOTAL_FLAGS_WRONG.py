#!/usr/bin/env python3
"""
Convert Jain Dataset with Private Disaggregated ELISA Data + Public BVP Data

This script implements Novo Nordisk's exact methodology using:
  - Private ELISA data (6 disaggregated antigens) from Jain et al. authors
  - Public BVP + biophysical assays from Jain et al. 2017 supplementary SD03

CORRECTED Methodology (from Sakhnini et al. 2025):
  - Flags 1-6: Individual ELISA antigens (Cardiolipin, KLH, LPS, ssDNA, dsDNA, Insulin)
  - Flag 7: BVP ELISA (from public SD03)
  - Flag 8: Self-interaction (ANY of 4 assays)
  - Flag 9: Chromatography (ANY of 3 assays)
  - Flag 10: Stability (1 assay)
  - Total flag range: 0-10 (NOT 0-7!)
  - Threshold: >=4 for non-specific (">3" in paper)
  - Exclude mild (1-3 flags) from test set

Expected output:
  - 137 total antibodies
  - ~60-70 specific (0 flags)
  - ~30-40 mild (1-3 flags) - EXCLUDED
  - ~20-30 non-specific (>=4 flags)
  - Test set: ~80-90 antibodies (specific + non-specific, target ~86)

Date: 2025-11-03
Status: CORRECTED - Fixed missing BVP and collapsed flags bugs
"""

from pathlib import Path

import pandas as pd


def load_data():
    """Load private ELISA + public SD files."""
    print("Loading data files...")

    # Private ELISA (6 individual antigens)
    private = pd.read_excel(
        "test_datasets/Private_Jain2017_ELISA_indiv.xlsx", sheet_name="Individual-ELISA"
    )
    print(f"  Private ELISA: {len(private)} antibodies")

    # Public SD files
    sd01 = pd.read_excel("test_datasets/jain-pnas.1616408114.sd01.xlsx")
    sd02 = pd.read_excel("test_datasets/jain-pnas.1616408114.sd02.xlsx")
    sd03 = pd.read_excel(
        "test_datasets/jain-pnas.1616408114.sd03.xlsx", sheet_name="Results-12-assays"
    )

    print(f"  SD01 (metadata): {len(sd01)} antibodies")
    print(f"  SD02 (sequences): {len(sd02)} antibodies")
    print(f"  SD03 (assays): {len(sd03)} rows")

    # Merge all on 'Name'
    df = sd01.merge(sd02[["Name", "VH", "VL"]], on="Name", how="inner")
    df = df.merge(sd03, on="Name", how="inner")
    df = df.merge(private, on="Name", how="inner")

    print(f"  Merged: {len(df)} antibodies\n")

    if len(df) != 137:
        print(f"  WARNING: Expected 137 antibodies, got {len(df)}")

    return df


def calculate_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate flags using CORRECTED Novo methodology (0-10 range).

    Flags:
      - Flags 1-6: Individual ELISA antigens (threshold 1.9 OD each)
      - Flag 7: BVP ELISA (threshold 4.3 fold-over-background)
      - Flag 8: Self-interaction (ANY of 4 assays)
      - Flag 9: Chromatography (ANY of 3 assays)
      - Flag 10: Stability (1 assay)

    Total: 0-10 flags (NO AGGREGATION!)
    """
    print("Calculating flags (CORRECTED Novo methodology: 0-10 range)...")

    # === FLAGS 1-6: ELISA (from private data) ===
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

    # === FLAG 7: BVP (from public SD03) ===
    # CRITICAL BUG FIX: This was missing in original implementation!
    bvp_threshold = 4.3
    df["flag_bvp"] = (df["BVP ELISA"] > bvp_threshold).astype(int)

    # === FLAG 8: Self-interaction (from public SD03) ===
    # ANY of 4 assays from Jain Table 1 exceeds threshold → 1 flag
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

    # === FLAG 9: Chromatography (from public SD03) ===
    # ANY of 3 assays exceeds threshold → 1 flag
    df["flag_chromatography"] = (
        (df["HIC Retention Time (Min)a"] > 11.7)
        | (df["SMAC Retention Time (Min)a"] > 12.8)
        | (df["SGAC-SINS AS100 ((NH4)2SO4 mM)"] < 370)
    ).astype(int)

    # === FLAG 10: Stability (from public SD03) ===
    df["flag_stability"] = (df["Slope for Accelerated Stability"] > 0.08).astype(int)

    # === TOTAL FLAGS (0-10 range) ===
    # CRITICAL BUG FIX: Sum ALL 10 individual flags (NOT aggregated!)
    df["total_flags"] = (
        df["elisa_flags"]  # 0-6
        + df["flag_bvp"]  # 0-1
        + df["flag_self_interaction"]  # 0-1
        + df["flag_chromatography"]  # 0-1
        + df["flag_stability"]  # 0-1
    )

    # === APPLY THRESHOLD ===
    # Novo: "specific (0 flags) and non-specific (>3 flags)"
    # This means: 0 = specific, 1-3 = mild, >=4 = non-specific

    df["flag_category"] = pd.cut(
        df["total_flags"],
        bins=[-0.5, 0.5, 3.5, 10.5],  # Updated upper bound for 0-10 range
        labels=["specific", "mild", "non_specific"],
    )

    # Binary label: 0 = specific, 1 = non-specific, NaN = mild (excluded)
    df["label"] = (
        df["flag_category"]
        .map(
            {"specific": 0, "mild": pd.NA, "non_specific": 1}  # Excluded from test set
        )
        .astype("Int64")
    )  # Nullable integer type

    # Print distribution
    print("\n  Flag distribution (0-10 range):")
    for flag_count in range(11):  # 0-10 inclusive
        count = (df["total_flags"] == flag_count).sum()
        pct = count / len(df) * 100
        print(f"    {flag_count:2d} flags: {count:3d} antibodies ({pct:5.1f}%)")

    print("\n  Category distribution:")
    for cat in ["specific", "mild", "non_specific"]:
        count = (df["flag_category"] == cat).sum()
        pct = count / len(df) * 100
        print(f"    {cat}: {count:3d} antibodies ({pct:5.1f}%)")

    print("\n  Label distribution (test set only):")
    label_counts = df["label"].value_counts()
    n_specific = label_counts.get(0, 0)
    n_nonspec = label_counts.get(1, 0)
    print(f"    Specific (label=0): {n_specific}")
    print(f"    Non-specific (label=1): {n_nonspec}")
    print(f"    Test set size: {n_specific + n_nonspec}")

    return df


def save_outputs(df: pd.DataFrame):
    """Save conversion outputs."""
    output_dir = Path("test_datasets")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Full dataset with all columns
    print("\nSaving outputs...")

    # 1. Full 137-antibody dataset
    full_output = df[
        [
            "Name",
            "VH",
            "VL",
            "total_flags",
            "elisa_flags",
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
    print(f"  ✓ Saved: {full_path}")
    print(f"    Total: {len(full_output)} antibodies (all 137)")

    # 2. Test set only (exclude mild)
    test_df = df[df["label"].notna()].copy()
    test_output = test_df[
        ["Name", "VH", "total_flags", "elisa_flags", "flag_category", "label"]
    ].copy()

    test_output = test_output.rename(columns={"Name": "id", "VH": "vh_sequence"})
    test_path = output_dir / "jain_with_private_elisa_TEST.csv"
    test_output.to_csv(test_path, index=False)
    print(f"  ✓ Saved: {test_path}")
    print(f"    Total: {len(test_output)} antibodies (test set, mild excluded)")


def main():
    print("=" * 80)
    print("Jain Dataset Conversion - CORRECTED Novo Nordisk Methodology")
    print("=" * 80)
    print("Using private disaggregated ELISA data (6 antigens) + public BVP")
    print("Flag range: 0-10 (6 ELISA + BVP + self + chrom + stability)")
    print("Threshold: >=4 for non-specific ('>3')")
    print("BUGS FIXED: Added BVP flag, removed flag aggregation")
    print("=" * 80)
    print()

    # Load data
    df = load_data()

    # Calculate flags
    df = calculate_flags(df)

    # Save outputs
    save_outputs(df)

    print("\n" + "=" * 80)
    print("✓ Conversion Complete!")
    print("=" * 80)
    print("\nFiles generated:")
    print("  1. jain_with_private_elisa_FULL.csv - All 137 antibodies")
    print("  2. jain_with_private_elisa_TEST.csv - Test set only (94 antibodies)")
    print("\nNext step:")
    print("  Run preprocessing/process_jain.py to generate fragments")


if __name__ == "__main__":
    main()
