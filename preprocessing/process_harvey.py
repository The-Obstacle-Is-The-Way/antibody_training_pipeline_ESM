#!/usr/bin/env python3
"""
Harvey Dataset Fragment Extraction Script

Processes the Harvey nanobody dataset to extract CDR and framework regions
using ANARCI (IMGT numbering scheme) following Sakhnini et al. 2025 methodology.

Key difference from Jain: Nanobodies are VHH (single heavy chain variable domain)
- No light chain (VL)
- Only heavy chain CDRs and frameworks

Fragments extracted:
1. VHH (full nanobody variable domain)
2. H-CDR1
3. H-CDR2
4. H-CDR3
5. H-CDRs (concatenated H-CDR1+2+3)
6. H-FWRs (concatenated H-FWR1+2+3+4)

Dataset: Harvey et al. 2022 Nature Communications
Date: 2025-11-01
Issue: #4 - Harvey dataset preprocessing (ray/learning branch)
"""

import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import riot_na
from tqdm.auto import tqdm

# Initialize ANARCI for amino acid annotation (IMGT scheme)
annotator = riot_na.create_riot_aa()


def annotate_nanobody(seq_id: str, sequence: str) -> Optional[Dict[str, str]]:
    """
    Annotate a nanobody (VHH) sequence using ANARCI (IMGT).

    Args:
        seq_id: Unique identifier for the nanobody
        sequence: Amino acid sequence string

    Returns:
        Dictionary with extracted fragments, or None if annotation fails
    """
    try:
        annotation = annotator.run_on_sequence(seq_id, sequence)

        # Extract all fragments (heavy chain only for nanobodies)
        fragments = {
            "full_seq_H": annotation.sequence_alignment_aa,
            "fwr1_aa_H": annotation.fwr1_aa,
            "cdr1_aa_H": annotation.cdr1_aa,
            "fwr2_aa_H": annotation.fwr2_aa,
            "cdr2_aa_H": annotation.cdr2_aa,
            "fwr3_aa_H": annotation.fwr3_aa,
            "cdr3_aa_H": annotation.cdr3_aa,
            "fwr4_aa_H": annotation.fwr4_aa,
        }

        # Create concatenated fragments
        fragments["cdrs_H"] = "".join(
            [
                fragments["cdr1_aa_H"],
                fragments["cdr2_aa_H"],
                fragments["cdr3_aa_H"],
            ]
        )

        fragments["fwrs_H"] = "".join(
            [
                fragments["fwr1_aa_H"],
                fragments["fwr2_aa_H"],
                fragments["fwr3_aa_H"],
                fragments["fwr4_aa_H"],
            ]
        )

        return fragments

    except Exception as e:
        print(f"Warning: Failed to annotate {seq_id}: {e}", file=sys.stderr)
        return None


def process_harvey_dataset(csv_path: str) -> pd.DataFrame:
    """
    Process Harvey CSV to extract all fragments.

    Args:
        csv_path: Path to harvey.csv

    Returns:
        DataFrame with all fragments and metadata
    """
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)

    print(f"  Total nanobodies: {len(df)}")
    print(f"  Annotating sequences with ANARCI (IMGT scheme)...")

    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Annotating"):
        # Annotate nanobody (VHH)
        fragments = annotate_nanobody(f"{row['id']}_VHH", row["sequence"])

        if fragments is None:
            print(f"  Skipping {row['id']} - annotation failed")
            continue

        # Combine all fragments and metadata
        result = {
            "id": row["id"],
            "label": row["label"],
            "psr_score": row["psr_score"],
            "polyreactivity_category": row["polyreactivity_category"],
            "source": row["source"],
        }

        result.update(fragments)

        results.append(result)

    df_annotated = pd.DataFrame(results)

    print(f"\n  Successfully annotated: {len(df_annotated)}/{len(df)} nanobodies")

    return df_annotated


def create_fragment_csvs(df: pd.DataFrame, output_dir: Path):
    """
    Create separate CSV files for each fragment type.

    Args:
        df: DataFrame with all fragments
        output_dir: Directory to save fragment CSVs
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define fragment types for nanobodies (VHH only)
    fragments = {
        # 1: Full nanobody domain
        "VHH_only": ("full_seq_H", "nanobody_vhh"),
        # 2-4: Heavy CDRs
        "H-CDR1": ("cdr1_aa_H", "h_cdr1"),
        "H-CDR2": ("cdr2_aa_H", "h_cdr2"),
        "H-CDR3": ("cdr3_aa_H", "h_cdr3"),
        # 5: Concatenated CDRs
        "H-CDRs": ("cdrs_H", "h_cdrs"),
        # 6: Concatenated FWRs
        "H-FWRs": ("fwrs_H", "h_fwrs"),
    }

    print(f"\nCreating {len(fragments)} fragment-specific CSV files...")

    for fragment_name, (column_name, sequence_alias) in fragments.items():
        output_path = output_dir / f"{fragment_name}_harvey.csv"

        # Create fragment-specific CSV with standardized column names
        fragment_df = pd.DataFrame(
            {
                "id": df["id"],
                "sequence": df[column_name],
                "label": df["label"],
                "psr_score": df["psr_score"],
                "polyreactivity_category": df["polyreactivity_category"],
                "source": df["source"],
            }
        )

        fragment_df.to_csv(output_path, index=False)

        print(f"  ✓ {fragment_name:12s} → {output_path.name}")

    print(f"\n✓ All fragments saved to: {output_dir}/")


def main():
    """Main processing pipeline."""
    # Paths
    csv_path = Path("test_datasets/harvey.csv")
    output_dir = Path("test_datasets/harvey")

    if not csv_path.exists():
        print(f"Error: {csv_path} not found!")
        print("Please ensure harvey.csv exists in test_datasets/")
        print("Run: python3 scripts/convert_harvey_to_csv.py")
        sys.exit(1)

    print("=" * 60)
    print("Harvey Dataset: Fragment Extraction")
    print("=" * 60)
    print(f"\nInput:  {csv_path}")
    print(f"Output: {output_dir}/")
    print(f"Method: ANARCI (IMGT numbering scheme)")
    print(f"Type:   Nanobodies (VHH - single domain)\n")

    # Process dataset
    df_annotated = process_harvey_dataset(str(csv_path))

    # Create fragment CSVs
    create_fragment_csvs(df_annotated, output_dir)

    # Validation summary
    print("\n" + "=" * 60)
    print("Fragment Extraction Summary")
    print("=" * 60)

    print(f"\nAnnotated nanobodies: {len(df_annotated)}")
    print(f"Label distribution:")
    for label, count in df_annotated["label"].value_counts().sort_index().items():
        label_name = "Specific (low PSR)" if label == 0 else "Polyreactive (high PSR)"
        print(f"  {label_name}: {count} ({count/len(df_annotated)*100:.1f}%)")

    print(f"\nPolyreactivity categories:")
    for cat, count in (
        df_annotated["polyreactivity_category"].value_counts().sort_index().items()
    ):
        print(f"  {cat}: {count} ({count/len(df_annotated)*100:.1f}%)")

    print(f"\nFragment files created: 6")
    print(f"Output directory: {output_dir.absolute()}")

    print("\n" + "=" * 60)
    print("✓ Harvey Fragment Extraction Complete!")
    print("=" * 60)

    print(f"\nNext steps:")
    print(f"  1. Test loading fragments with data.load_local_data()")
    print(f"  2. Run model inference on fragment-specific CSVs")
    print(f"  3. Compare with Sakhnini et al. 2025 methodology")
    print(f"  4. Document in ray/learning branch for Issue #4")


if __name__ == "__main__":
    main()
