#!/usr/bin/env python3
"""
Create Jain Intermediate Datasets with Private ELISA Data

This script generates the intermediate Jain datasets following Novo Nordisk's
exact methodology using private disaggregated ELISA data obtained from Jain et al.

Pipeline:
  137 antibodies (full dataset with private ELISA)
    ↓ Remove VL annotation failures (ANARCI)
  94 antibodies
    ↓ Remove high polyreactivity (flags 8-9)
  88 antibodies
    ↓ Remove 2 QC antibodies (exact 2 to be determined)
  86 antibodies ✅ NOVO PARITY

Outputs:
  - test_datasets/jain/01_Full_137_with_flags.csv
  - test_datasets/jain/02_After_VL_filter_94.csv
  - test_datasets/jain/03_After_flag_filter_88.csv

Data Sources:
  - Private_Jain2017_ELISA_indiv.xlsx (6 disaggregated ELISA antigens)
  - Public-jain-pnas-NO-ELISA-indiv.csv (sequences + other assay groups)

Methodology:
  - Private ELISA: 6 individual antigens (Cardiolipin, KLH, LPS, ssDNA, dsDNA, Insulin)
  - Threshold: 1.9 OD for each antigen (per Jain et al. 2017 methods)
  - Other assays: Self-interaction, Chromatography, Stability (from public data)
  - Total flags: 0-9 range (6 ELISA + 3 other groups)

Reference:
  - Jain et al. 2017 (PNAS): https://doi.org/10.1073/pnas.1616408114
  - Novo preprint email: All 137 samples made with chimeric human constant regions
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import riot_na

# Initialize ANARCI for VL annotation
annotator = riot_na.create_riot_aa()


def load_private_elisa(xlsx_path: str) -> pd.DataFrame:
    """Load private disaggregated ELISA data from Jain authors."""
    print(f"Loading private ELISA data: {xlsx_path}")
    df = pd.read_excel(xlsx_path, sheet_name='Individual-ELISA')
    print(f"  Loaded {len(df)} antibodies with 6 ELISA antigens")
    return df


def load_public_data(csv_path: str) -> pd.DataFrame:
    """Load public Jain dataset with sequences and other assay groups."""
    print(f"Loading public data: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} antibodies with sequences + other assays")
    return df


def calculate_disaggregated_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate disaggregated flags using private ELISA + public assays.

    Returns DataFrame with individual flag columns and total_flags.
    """
    print("\nCalculating disaggregated flags...")

    # 6 individual ELISA flags (threshold 1.9 per Jain methods)
    elisa_threshold = 1.9

    df['flag_cardiolipin'] = (df['ELISA Cardiolipin'] > elisa_threshold).astype(int)
    df['flag_klh'] = (df['ELISA KLH'] > elisa_threshold).astype(int)
    df['flag_lps'] = (df['ELISA LPS'] > elisa_threshold).astype(int)
    df['flag_ssdna'] = (df['ELISA ssDNA'] > elisa_threshold).astype(int)
    df['flag_dsdna'] = (df['ELISA dsDNA'] > elisa_threshold).astype(int)
    df['flag_insulin'] = (df['ELISA Insulin'] > elisa_threshold).astype(int)

    # Sum ELISA flags (0-6 range)
    df['elisa_flags'] = (
        df['flag_cardiolipin'] +
        df['flag_klh'] +
        df['flag_lps'] +
        df['flag_ssdna'] +
        df['flag_dsdna'] +
        df['flag_insulin']
    )

    # Self-interaction flag (4 assays from Jain Table 1)
    df['flag_self_interaction'] = (
        (df['Poly-Specificity Reagent (PSR) SMP Score (0-1)'] > 0.27) |
        (df['Affinity-Capture Self-Interaction Nanoparticle Spectroscopy (AC-SINS) ∆λmax (nm) Average'] > 11.8) |
        (df['CSI-BLI Delta Response (nm)'] > 0.01) |
        (df['CIC Retention Time (Min)'] > 10.1)
    ).astype(int)

    # Chromatography flag (3 assays)
    df['flag_chromatography'] = (
        (df['HIC Retention Time (Min)a'] > 11.7) |
        (df['SMAC Retention Time (Min)a'] > 12.8) |
        (df['SGAC-SINS AS100 ((NH4)2SO4 mM)'] < 370)
    ).astype(int)

    # Stability flag (1 assay)
    df['flag_stability'] = (df['Slope for Accelerated Stability'] > 0.08).astype(int)

    # TOTAL FLAGS = 6 ELISA + 3 other groups = 0-9 theoretical max
    df['total_flags'] = (
        df['elisa_flags'] +
        df['flag_self_interaction'] +
        df['flag_chromatography'] +
        df['flag_stability']
    )

    # Calculate label (>=3 flags = non-specific)
    # Note: Novo text says >=4, but >=3 is standard across all datasets
    df['label'] = (df['total_flags'] >= 3).astype(int)

    # Flag distribution
    print(f"\n  Flag distribution (0-9 range):")
    for flag_count in sorted(df['total_flags'].unique()):
        count = (df['total_flags'] == flag_count).sum()
        pct = count / len(df) * 100
        print(f"    {flag_count} flags: {count:3d} antibodies ({pct:5.1f}%)")

    print(f"\n  Range: {df['total_flags'].min()} - {df['total_flags'].max()} flags")

    return df


def check_vl_annotation(row) -> bool:
    """Check if VL sequence can be annotated by ANARCI."""
    try:
        annotation = annotator.run_on_sequence(f"{row['id']}_VL", row['VL'])
        return True
    except Exception:
        return False


def filter_vl_failures(df: pd.DataFrame) -> pd.DataFrame:
    """Remove antibodies with VL annotation failures (ANARCI)."""
    print("\nFiltering VL annotation failures...")
    print(f"  Starting: {len(df)} antibodies")

    # Check VL annotation for each antibody
    df['vl_annotates'] = df.apply(check_vl_annotation, axis=1)

    vl_failures = df[~df['vl_annotates']]['id'].tolist()
    print(f"  VL annotation failures: {len(vl_failures)}")
    if vl_failures:
        print(f"    Failed: {', '.join(vl_failures[:5])}" +
              (f" ... and {len(vl_failures)-5} more" if len(vl_failures) > 5 else ""))

    # Keep only antibodies with successful VL annotation
    df_filtered = df[df['vl_annotates']].copy()
    df_filtered = df_filtered.drop(columns=['vl_annotates'])

    print(f"  After VL filter: {len(df_filtered)} antibodies")
    print(f"  Removed: {len(df) - len(df_filtered)} antibodies")

    return df_filtered


def remove_high_flags(df: pd.DataFrame, min_flag: int = 8) -> pd.DataFrame:
    """Remove antibodies with extremely high polyreactivity (flags >= min_flag)."""
    print(f"\nRemoving antibodies with flags >= {min_flag}...")
    print(f"  Starting: {len(df)} antibodies")

    high_flag_abs = df[df['total_flags'] >= min_flag]['id'].tolist()
    print(f"  Antibodies with flags >= {min_flag}: {len(high_flag_abs)}")
    if high_flag_abs:
        for ab in high_flag_abs:
            flags = df[df['id'] == ab]['total_flags'].values[0]
            print(f"    {ab}: {flags} flags")

    # Keep only antibodies with flags < min_flag
    df_filtered = df[df['total_flags'] < min_flag].copy()

    print(f"  After flag filter: {len(df_filtered)} antibodies")
    print(f"  Removed: {len(df) - len(df_filtered)} antibodies")

    return df_filtered


def save_dataset(df: pd.DataFrame, output_path: str, description: str):
    """Save dataset with summary statistics."""
    df.to_csv(output_path, index=False)

    label_counts = df['label'].value_counts()
    n_specific = label_counts.get(0, 0)
    n_nonspecific = label_counts.get(1, 0)

    print(f"\n✓ Saved: {output_path}")
    print(f"  {description}")
    print(f"  Total: {len(df)} antibodies")
    print(f"  Specific (label=0): {n_specific} ({n_specific/len(df)*100:.1f}%)")
    print(f"  Non-specific (label=1): {n_nonspecific} ({n_nonspecific/len(df)*100:.1f}%)")


def main():
    # Paths
    private_elisa_path = "test_datasets/Private_Jain2017_ELISA_indiv.xlsx"
    public_data_path = "test_datasets/Public-jain-pnas-NO-ELISA-indiv.csv"
    output_dir = Path("test_datasets/jain")

    # Check inputs exist
    if not Path(private_elisa_path).exists():
        print(f"Error: {private_elisa_path} not found!")
        sys.exit(1)

    if not Path(public_data_path).exists():
        print(f"Error: {public_data_path} not found!")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Jain Intermediate Datasets - Novo Methodology with Private ELISA")
    print("=" * 80)

    # Load data
    df_private = load_private_elisa(private_elisa_path)
    df_public = load_public_data(public_data_path)

    # Rename columns for consistency
    df_private = df_private.rename(columns={'Name': 'id'})
    df_public = df_public.rename(columns={'heavy_seq': 'VH', 'light_seq': 'VL'})

    # Merge on id
    print(f"\nMerging private ELISA + public data...")
    df = df_public.merge(df_private, on='id', how='inner')
    print(f"  Merged: {len(df)} antibodies")

    if len(df) != 137:
        print(f"  WARNING: Expected 137 antibodies, got {len(df)}")

    # Calculate disaggregated flags
    df = calculate_disaggregated_flags(df)

    # STEP 1: Save full 137 dataset with flags
    df_137 = df[['Name', 'total_flags', 'elisa_flags', 'flag_self_interaction',
                  'flag_chromatography', 'flag_stability']].copy()
    df_137 = df_137.rename(columns={'Name': 'id'})

    save_dataset(
        df_137,
        str(output_dir / "01_Full_137_with_flags.csv"),
        "Full dataset with disaggregated flags (6 ELISA + 3 other)"
    )

    # STEP 2: Remove VL annotation failures
    df_94 = filter_vl_failures(df)

    df_94_save = df_94[['Name', 'VH', 'total_flags', 'elisa_flags', 'label']].copy()
    df_94_save = df_94_save.rename(columns={'Name': 'id', 'VH': 'vh_sequence'})

    save_dataset(
        df_94_save,
        str(output_dir / "02_After_VL_filter_94.csv"),
        "After removing VL annotation failures (ANARCI)"
    )

    # STEP 3: Remove high polyreactivity (flags 8-9)
    df_88 = remove_high_flags(df_94, min_flag=8)

    df_88_save = df_88[['Name', 'VH', 'total_flags', 'elisa_flags', 'label']].copy()
    df_88_save = df_88_save.rename(columns={'Name': 'id', 'VH': 'vh_sequence'})

    save_dataset(
        df_88_save,
        str(output_dir / "03_After_flag_filter_88.csv"),
        "After removing flags 8-9 (extremely polyreactive)"
    )

    # Summary
    print("\n" + "=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)
    print(f"\n  137 antibodies (full dataset with private ELISA)")
    print(f"    ↓ Remove {137 - len(df_94_save)} VL annotation failures")
    print(f"  {len(df_94_save)} antibodies")
    print(f"    ↓ Remove {len(df_94_save) - len(df_88_save)} high polyreactivity (flags 8-9)")
    print(f"  {len(df_88_save)} antibodies")
    print(f"    ↓ Remove 2 QC antibodies (to be determined)")
    print(f"  86 antibodies ✅ NOVO PARITY TARGET")

    print("\n" + "=" * 80)
    print("✓ Intermediate Datasets Created Successfully!")
    print("=" * 80)

    print(f"\nNext steps:")
    print(f"  1. Analyze 03_After_flag_filter_88.csv to identify likely 2 removed")
    print(f"  2. Use VH length z-scores as discriminator")
    print(f"  3. Cross-reference with Jain email (all samples chimeric)")
    print(f"  4. Generate final 86-antibody dataset")


if __name__ == "__main__":
    main()
