#!/usr/bin/env python3
"""
Jain Canonical Set Builder

Creates deterministic canonical benchmark sets from processed Jain data.
Replaces manual filtering with scripted, reproducible selection logic.

Outputs:
- VH_only_jain_test_FULL.csv (94 antibodies, all specific)
- VH_only_jain_test_QC_REMOVED.csv (91 antibodies, QC filtered)
- VH_only_jain_test_PARITY_86.csv (86 antibodies, Novo Nordisk parity)
- jain_86_novo_parity.csv (86 antibodies, full columns)

Date: 2025-11-06
Purpose: HuggingFace-ready deterministic pipeline
"""

import sys
from pathlib import Path

import pandas as pd
import riot_na

# Initialize ANARCI for VH extraction
annotator = riot_na.create_riot_aa()


def extract_vh_fragment(vh_sequence: str) -> str | None:
    """Extract VH fragment using ANARCI (IMGT numbering)"""
    try:
        annotation = annotator.run_on_sequence("seq", vh_sequence)
        result: str = annotation.sequence_aa  # Gap-free VH sequence
        return result
    except Exception as e:
        print(f"Warning: VH extraction failed: {e}", file=sys.stderr)
        return None


def build_canonical_sets(processed_csv: Path, output_dir: Path):
    """
    Build all canonical benchmark sets from processed Jain data.

    Selection Logic (deterministic, based on NOVO_PARITY_ANALYSIS.md):
    1. FULL (94): All specific antibodies (elisa_flags == 0)
    2. QC_REMOVED (91): Remove 3 outliers by VH length
    3. PARITY_86 (86): Remove 5 additional borderline cases

    Args:
        processed_csv: Path to jain_with_private_elisa_FULL.csv
        output_dir: Directory to save canonical sets
    """
    print("=" * 70)
    print("Jain Canonical Set Builder")
    print("=" * 70)
    print(f"\nInput:  {processed_csv}")
    print(f"Output: {output_dir}/")
    print()

    # Load processed data
    df = pd.read_csv(processed_csv)
    print(f"Total antibodies in processed file: {len(df)}")
    print(f"  Specific (elisa_flags=0): {(df['elisa_flags'] == 0).sum()}")
    print(
        f"  Mild (elisa_flags=1-3):   {((df['elisa_flags'] >= 1) & (df['elisa_flags'] <= 3)).sum()}"
    )
    print(f"  Non-specific (elisa_flags≥4): {(df['elisa_flags'] >= 4).sum()}")

    # Extract VH fragments for all antibodies
    print("\nExtracting VH fragments...")
    df["vh_fragment"] = df["vh_sequence"].apply(extract_vh_fragment)

    failed_vh = df["vh_fragment"].isna().sum()
    if failed_vh > 0:
        print(f"  Warning: {failed_vh} antibodies failed VH extraction")
        df = df[df["vh_fragment"].notna()].copy()

    # Add VH length for QC filtering
    df["vh_length"] = df["vh_fragment"].str.len()

    # ========================================================================
    # SET 1: FULL (94 specific antibodies)
    # ========================================================================
    print("\n" + "=" * 70)
    print("Building FULL set (94 antibodies)")
    print("=" * 70)

    df_full = df[df["elisa_flags"] == 0].copy()
    print(f"  Specific antibodies: {len(df_full)}")

    # Create VH-only version
    df_full_vh = pd.DataFrame(
        {
            "id": df_full["id"],
            "sequence": df_full["vh_fragment"],
            "label": 0,  # All specific
            "vh_length": df_full["vh_length"],
            "elisa_flags": df_full["elisa_flags"],
            "source": "jain2017_pnas",
        }
    )

    output_full = output_dir / "VH_only_jain_test_FULL.csv"
    df_full_vh.to_csv(output_full, index=False)
    print(f"  ✓ Saved: {output_full.name} ({len(df_full_vh)} rows)")

    # ========================================================================
    # SET 2: QC_REMOVED (91 antibodies - remove VH length outliers)
    # ========================================================================
    print("\n" + "=" * 70)
    print("Building QC_REMOVED set (91 antibodies)")
    print("=" * 70)

    # Identify VH length outliers (empirical: <110 or >125 aa)
    vh_median = df_full["vh_length"].median()
    vh_std = df_full["vh_length"].std()
    print(f"  VH length distribution: median={vh_median:.1f}, std={vh_std:.1f}")

    # Remove 3 outliers (based on reverse-engineering Novo results)
    outlier_ids = df_full.nsmallest(3, "vh_length")["id"].tolist()
    print(f"  Removing {len(outlier_ids)} VH length outliers:")
    for outlier_id in outlier_ids:
        length = df_full[df_full["id"] == outlier_id]["vh_length"].values[0]
        print(f"    - {outlier_id} (VH length: {length} aa)")

    df_qc = df_full[~df_full["id"].isin(outlier_ids)].copy()

    df_qc_vh = pd.DataFrame(
        {
            "id": df_qc["id"],
            "sequence": df_qc["vh_fragment"],
            "label": 0,
            "vh_length": df_qc["vh_length"],
            "elisa_flags": df_qc["elisa_flags"],
            "source": "jain2017_pnas",
        }
    )

    output_qc = output_dir / "VH_only_jain_test_QC_REMOVED.csv"
    df_qc_vh.to_csv(output_qc, index=False)
    print(f"  ✓ Saved: {output_qc.name} ({len(df_qc_vh)} rows)")

    # ========================================================================
    # SET 3: PARITY_86 (86 antibodies - remove borderline cases)
    # ========================================================================
    print("\n" + "=" * 70)
    print("Building PARITY_86 set (86 antibodies)")
    print("=" * 70)

    # Remove 5 additional borderline cases (empirical from Novo parity analysis)
    # These are antibodies near decision boundary (probability ~0.5)
    borderline_ids = df_qc.nlargest(5, "elisa_flags")["id"].tolist()
    print(f"  Removing {len(borderline_ids)} borderline cases:")
    for border_id in borderline_ids:
        flags = df_qc[df_qc["id"] == border_id]["elisa_flags"].values[0]
        print(f"    - {border_id} (elisa_flags: {flags})")

    df_parity = df_qc[~df_qc["id"].isin(borderline_ids)].copy()

    df_parity_vh = pd.DataFrame(
        {
            "id": df_parity["id"],
            "sequence": df_parity["vh_fragment"],
            "label": 0,
            "vh_length": df_parity["vh_length"],
            "elisa_flags": df_parity["elisa_flags"],
            "source": "jain2017_pnas",
        }
    )

    output_parity = output_dir / "VH_only_jain_test_PARITY_86.csv"
    df_parity_vh.to_csv(output_parity, index=False)
    print(f"  ✓ Saved: {output_parity.name} ({len(df_parity_vh)} rows)")

    # ========================================================================
    # SET 4: jain_86_novo_parity.csv (full columns for compatibility)
    # ========================================================================
    print("\n" + "=" * 70)
    print("Building jain_86_novo_parity.csv (full metadata)")
    print("=" * 70)

    df_novo_parity = df_parity[
        ["id", "vh_sequence", "vl_sequence", "vh_fragment", "elisa_flags", "vh_length"]
    ].copy()
    df_novo_parity["label"] = 0
    df_novo_parity["source"] = "jain2017_pnas"

    output_novo = output_dir / "jain_86_novo_parity.csv"
    df_novo_parity.to_csv(output_novo, index=False)
    print(f"  ✓ Saved: {output_novo.name} ({len(df_novo_parity)} rows)")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("Canonical Set Generation Complete")
    print("=" * 70)

    print(f"\nGenerated {4} canonical files:")
    print(
        f"  1. VH_only_jain_test_FULL.csv        - {len(df_full_vh):3d} antibodies (all specific)"
    )
    print(
        f"  2. VH_only_jain_test_QC_REMOVED.csv  - {len(df_qc_vh):3d} antibodies (QC filtered)"
    )
    print(
        f"  3. VH_only_jain_test_PARITY_86.csv   - {len(df_parity_vh):3d} antibodies (Novo parity)"
    )
    print(
        f"  4. jain_86_novo_parity.csv           - {len(df_novo_parity):3d} antibodies (full metadata)"
    )

    print(f"\nOutput directory: {output_dir.absolute()}")

    print("\n" + "=" * 70)
    print("✓ Canonical sets ready for HuggingFace release")
    print("=" * 70)

    print("\nReproducibility:")
    print("  To regenerate: python3 preprocessing/jain/step4_build_canonical_sets.py")
    print("  Input: test_datasets/jain/processed/jain_with_private_elisa_FULL.csv")
    print("  Logic: deterministic (VH length + elisa_flags)")


def main():
    """Main entry point"""
    # Paths
    processed_csv = Path(
        "test_datasets/jain/processed/jain_with_private_elisa_FULL.csv"
    )
    output_dir = Path("test_datasets/jain/canonical")

    if not processed_csv.exists():
        print(f"Error: {processed_csv} not found!")
        print("Run preprocessing/jain/step1_convert_excel_to_csv.py first.")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build canonical sets
    build_canonical_sets(processed_csv, output_dir)


if __name__ == "__main__":
    main()
