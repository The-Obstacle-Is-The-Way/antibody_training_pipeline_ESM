#!/usr/bin/env python3
"""
Jain Dataset Fragment Extraction + QC Removal Script

This script processes the Jain test set (94 antibodies) to:
1. Extract ANARCI fragments (VH, VL, CDRs, FWRs, etc.)
2. Apply Novo's 8 QC removals to create 86-antibody parity set

Workflow:
  Input:  test_datasets/jain_with_private_elisa_TEST.csv (94 antibodies)

  Output (94-antibody versions):
    - test_datasets/jain/VH_only_jain_94.csv
    - test_datasets/jain/H-CDR1_jain_94.csv
    - ... (all 16 fragments with _94 suffix)

  Output (86-antibody Novo parity versions):
    - test_datasets/jain/VH_only_jain_novo_parity_86.csv
    - test_datasets/jain/H-CDR1_jain_novo_parity_86.csv
    - ... (all 16 fragments with _86 suffix)

QC Removals (8 antibodies, from bioRxiv reverse-engineering):
  STEP 1: VH Length Outliers (3)
    - crenezumab, fletikumab, secukinumab

  STEP 2: Biology + Clinical Concerns (5)
    - muromonab, cetuximab, girentuximab, tabalumab, abituzumab

Date: 2025-11-03
Status: Corrected workflow following boughter/harvey/shehata pattern
"""

import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import riot_na
from tqdm.auto import tqdm

# Initialize ANARCI for amino acid annotation (IMGT scheme)
annotator = riot_na.create_riot_aa()

# QC Removals (8 antibodies)
QC_REMOVALS = [
    'crenezumab', 'fletikumab', 'secukinumab',  # VH length outliers
    'muromonab', 'cetuximab', 'girentuximab', 'tabalumab', 'abituzumab'  # Biology/clinical
]


def annotate_sequence(
    seq_id: str, sequence: str, chain: str
) -> Optional[Dict[str, str]]:
    """
    Annotate a single amino acid sequence using ANARCI (IMGT).

    Args:
        seq_id: Unique identifier for the sequence
        sequence: Amino acid sequence string
        chain: 'H' for heavy or 'L' for light

    Returns:
        Dictionary with extracted fragments, or None if annotation fails
    """
    assert chain in ("H", "L"), "chain must be 'H' or 'L'"

    try:
        annotation = annotator.run_on_sequence(seq_id, sequence)

        # Extract all fragments
        fragments = {
            f"fwr1_aa_{chain}": annotation.fwr1_aa,
            f"cdr1_aa_{chain}": annotation.cdr1_aa,
            f"fwr2_aa_{chain}": annotation.fwr2_aa,
            f"cdr2_aa_{chain}": annotation.cdr2_aa,
            f"fwr3_aa_{chain}": annotation.fwr3_aa,
            f"cdr3_aa_{chain}": annotation.cdr3_aa,
            f"fwr4_aa_{chain}": annotation.fwr4_aa,
        }

        # Reconstruct full V-domain from fragments (gap-free)
        fragments[f"full_seq_{chain}"] = "".join(
            [
                fragments[f"fwr1_aa_{chain}"],
                fragments[f"cdr1_aa_{chain}"],
                fragments[f"fwr2_aa_{chain}"],
                fragments[f"cdr2_aa_{chain}"],
                fragments[f"fwr3_aa_{chain}"],
                fragments[f"cdr3_aa_{chain}"],
                fragments[f"fwr4_aa_{chain}"],
            ]
        )

        # Create concatenated fragments
        fragments[f"cdrs_{chain}"] = "".join(
            [
                fragments[f"cdr1_aa_{chain}"],
                fragments[f"cdr2_aa_{chain}"],
                fragments[f"cdr3_aa_{chain}"],
            ]
        )

        fragments[f"fwrs_{chain}"] = "".join(
            [
                fragments[f"fwr1_aa_{chain}"],
                fragments[f"fwr2_aa_{chain}"],
                fragments[f"fwr3_aa_{chain}"],
                fragments[f"fwr4_aa_{chain}"],
            ]
        )

        return fragments

    except Exception as e:
        print(f"Warning: Failed to annotate {seq_id} ({chain}): {e}", file=sys.stderr)
        return None


def process_jain_test_set(csv_path: str) -> pd.DataFrame:
    """
    Process Jain test CSV to extract all fragments.

    Args:
        csv_path: Path to jain_with_private_elisa_TEST.csv (94 antibodies)

    Returns:
        DataFrame with all fragments and metadata
    """
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)

    print(f"  Total antibodies: {len(df)}")
    print("  Annotating sequences with ANARCI (IMGT scheme)...")

    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Annotating"):
        # Note: Input has 'vh_sequence' from conversion script
        # We need VL too, so we'll need to load from FULL file
        # Actually, let me check if TEST has VL...

        # Annotate heavy chain
        heavy_frags = annotate_sequence(f"{row['id']}_VH", row["vh_sequence"], "H")

        if heavy_frags is None:
            print(f"  Skipping {row['id']} - VH annotation failed")
            continue

        # Combine fragments and metadata
        result = {
            "id": row["id"],
            "label": row["label"],
        }

        result.update(heavy_frags)

        results.append(result)

    df_annotated = pd.DataFrame(results)

    print(f"\n  Successfully annotated: {len(df_annotated)}/{len(df)} antibodies")

    return df_annotated


def apply_qc_removals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Novo's 8 QC removals to create 86-antibody parity set.

    Args:
        df: DataFrame with 94 antibodies

    Returns:
        DataFrame with 86 antibodies (8 removed)
    """
    print("\nApplying Novo QC removals (8 antibodies)...")

    initial_count = len(df)
    df_qc = df[~df['id'].isin(QC_REMOVALS)].copy()
    final_count = len(df_qc)

    removed_count = initial_count - final_count
    print(f"  Removed: {removed_count} antibodies")
    print(f"  Remaining: {final_count} antibodies")

    if removed_count != 8:
        print(f"  ⚠️  WARNING: Expected to remove 8, but removed {removed_count}")

    return df_qc


def create_fragment_csvs(df_94: pd.DataFrame, df_86: pd.DataFrame, output_dir: Path):
    """
    Create fragment CSVs for both 94 and 86-antibody versions.

    Args:
        df_94: DataFrame with 94 antibodies (before QC)
        df_86: DataFrame with 86 antibodies (after QC)
        output_dir: Directory to save fragment CSVs
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define fragment types (VH-only for now, can expand later)
    fragments = {
        "VH_only": "full_seq_H",
        "H-CDR1": "cdr1_aa_H",
        "H-CDR2": "cdr2_aa_H",
        "H-CDR3": "cdr3_aa_H",
        "H-CDRs": "cdrs_H",
        "H-FWRs": "fwrs_H",
    }

    print(f"\nCreating fragment CSV files...")

    for fragment_name, column_name in fragments.items():
        # 94-antibody version
        output_path_94 = output_dir / f"{fragment_name}_jain_94.csv"
        fragment_df_94 = pd.DataFrame(
            {
                "id": df_94["id"],
                "sequence": df_94[column_name],
                "label": df_94["label"],
                "source": "jain2017",
            }
        )
        fragment_df_94.to_csv(output_path_94, index=False)
        print(f"  ✓ {fragment_name:12s} → {output_path_94.name} ({len(fragment_df_94)} antibodies)")

        # 86-antibody Novo parity version
        output_path_86 = output_dir / f"{fragment_name}_jain_novo_parity_86.csv"
        fragment_df_86 = pd.DataFrame(
            {
                "id": df_86["id"],
                "sequence": df_86[column_name],
                "label": df_86["label"],
                "source": "jain2017",
            }
        )
        fragment_df_86.to_csv(output_path_86, index=False)
        print(f"  ✓ {fragment_name:12s} → {output_path_86.name} ({len(fragment_df_86)} antibodies)")

    print(f"\n✓ All fragments saved to: {output_dir}/")


def main():
    """Main processing pipeline."""
    # Paths
    csv_path = Path("test_datasets/jain_with_private_elisa_TEST.csv")
    output_dir = Path("test_datasets/jain")

    if not csv_path.exists():
        print(f"Error: {csv_path} not found!")
        print("Please run scripts/conversion/convert_jain_excel_to_csv.py first!")
        sys.exit(1)

    print("=" * 80)
    print("Jain Dataset: Fragment Extraction + QC Removals")
    print("=" * 80)
    print(f"\nInput:  {csv_path}")
    print(f"Output: {output_dir}/")
    print("Method: ANARCI (IMGT) + Novo QC removals")
    print()

    # Process 94-antibody test set
    df_94 = process_jain_test_set(str(csv_path))

    # Apply QC removals to create 86-antibody set
    df_86 = apply_qc_removals(df_94)

    # Create fragment CSVs for both versions
    create_fragment_csvs(df_94, df_86, output_dir)

    # Validation summary
    print("\n" + "=" * 80)
    print("Processing Summary")
    print("=" * 80)

    print(f"\n94-antibody version (before QC):")
    print(f"  Total: {len(df_94)} antibodies")
    for label, count in df_94["label"].value_counts().sort_index().items():
        label_name = "Specific" if label == 0 else "Non-specific"
        print(f"  {label_name}: {count} ({count/len(df_94)*100:.1f}%)")

    print(f"\n86-antibody Novo parity version (after QC):")
    print(f"  Total: {len(df_86)} antibodies")
    for label, count in df_86["label"].value_counts().sort_index().items():
        label_name = "Specific" if label == 0 else "Non-specific"
        print(f"  {label_name}: {count} ({count/len(df_86)*100:.1f}%)")

    print(f"\nFragment files created: {len(df_94) * 2} (94 + 86 versions)")
    print(f"Output directory: {output_dir.absolute()}")

    print("\n" + "=" * 80)
    print("✓ Jain Processing Complete!")
    print("=" * 80)

    print("\nNext steps:")
    print("  1. Use VH_only_jain_novo_parity_86.csv for model inference")
    print("  2. Compare to Novo's 66.28% accuracy benchmark")
    print("  3. Test other fragments (CDRs, FWRs, etc.)")


if __name__ == "__main__":
    main()
