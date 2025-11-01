#!/usr/bin/env python3
"""
Harvey Dataset Fragment Extraction Script

Processes the Harvey dataset to extract VHH (nanobody) fragment types
using ANARCI (IMGT numbering scheme) following Sakhnini et al. 2025 methodology.

Fragments extracted (nanobody-specific, no light chain):
1. VHH (full nanobody variable domain)
2. H-CDR1
3. H-CDR2
4. H-CDR3
5. H-CDRs (concatenated H-CDR1+2+3)
6. H-FWRs (concatenated H-FWR1+2+3+4)

Date: 2025-11-01
Issue: #4 - Harvey dataset preprocessing
"""

import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import riot_na
from tqdm.auto import tqdm

# Initialize ANARCI for amino acid annotation (IMGT scheme)
annotator = riot_na.create_riot_aa()


def annotate_sequence(seq_id: str, sequence: str) -> Optional[Dict[str, str]]:
    """
    Annotate a single VHH (nanobody) sequence using ANARCI (IMGT).

    Args:
        seq_id: Unique identifier for the sequence
        sequence: VHH amino acid sequence string

    Returns:
        Dictionary with extracted fragments, or None if annotation fails
    """
    try:
        annotation = annotator.run_on_sequence(seq_id, sequence)

        # Extract all heavy chain fragments
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
    Process Harvey CSV to extract all VHH fragments.

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
    failures = []
    seq_counter = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Annotating"):
        # Generate sequential ID (harvey_000001, harvey_000002, etc.)
        seq_counter += 1
        seq_id = f"harvey_{seq_counter:06d}"

        # Annotate VHH sequence
        frags = annotate_sequence(seq_id, row["seq"])

        if frags is None:
            failures.append(seq_id)
            continue

        # Combine fragments and metadata
        result = {
            "id": seq_id,
            "label": row["label"],
            "source": "harvey2022",
        }

        result.update(frags)
        results.append(result)

    df_annotated = pd.DataFrame(results)

    print(f"\n  Successfully annotated: {len(df_annotated)}/{len(df)} nanobodies")
    if failures:
        print(f"  Failures: {len(failures)}")
        print(f"  Failed IDs (first 10): {failures[:10]}")

        # Write all failed IDs to log file
        failure_log = Path("test_datasets/harvey/failed_sequences.txt")
        failure_log.parent.mkdir(parents=True, exist_ok=True)
        with open(failure_log, "w") as f:
            f.write("\n".join(failures))
        print(f"  All failed IDs written to: {failure_log}")

    return df_annotated


def create_fragment_csvs(df: pd.DataFrame, output_dir: Path):
    """
    Create separate CSV files for each VHH fragment type.

    Following the nanobody-specific methodology from Sakhnini et al. 2025.

    Args:
        df: DataFrame with all fragments
        output_dir: Directory to save fragment CSVs
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define 6 VHH-specific fragment types (no light chain)
    fragments = {
        # 1: Full nanobody variable domain
        "VHH_only": ("full_seq_H", "vhh_sequence"),
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
                "source": df["source"],
                "sequence_length": df[column_name].str.len(),
            }
        )

        fragment_df.to_csv(output_path, index=False)

        # Show stats
        mean_len = fragment_df["sequence"].str.len().mean()
        min_len = fragment_df["sequence"].str.len().min()
        max_len = fragment_df["sequence"].str.len().max()

        print(
            f"  [OK] {fragment_name:12s} -> {output_path.name:30s} "
            f"(len: {min_len}-{max_len} aa, mean: {mean_len:.1f})"
        )

    print(f"\n[OK] All fragments saved to: {output_dir}/")


def main():
    """Main processing pipeline."""
    # Paths
    csv_path = Path("test_datasets/harvey.csv")
    output_dir = Path("test_datasets/harvey")

    if not csv_path.exists():
        print(f"Error: {csv_path} not found!")
        print("Please run scripts/download_harvey_dataset.py first.")
        sys.exit(1)

    print("=" * 70)
    print("Harvey Dataset: VHH Fragment Extraction")
    print("=" * 70)
    print(f"\nInput:  {csv_path}")
    print(f"Output: {output_dir}/")
    print(f"Method: ANARCI (IMGT numbering scheme)")
    print(f"Note:   Nanobodies (VHH) - no light chain fragments")
    print()

    # Process dataset
    df_annotated = process_harvey_dataset(str(csv_path))

    # Create fragment CSVs
    create_fragment_csvs(df_annotated, output_dir)

    # Validation summary
    print("\n" + "=" * 70)
    print("Fragment Extraction Summary")
    print("=" * 70)

    print(f"\nAnnotated nanobodies: {len(df_annotated)}")
    print(f"Label distribution:")
    for label, count in df_annotated["label"].value_counts().sort_index().items():
        label_name = "Low polyreactivity" if label == 0 else "High polyreactivity"
        print(f"  {label_name}: {count} ({count/len(df_annotated)*100:.1f}%)")

    print(f"\nFragment files created: 6 (VHH-specific)")
    print(f"Output directory: {output_dir.absolute()}")

    print("\n" + "=" * 70)
    print("[DONE] Harvey Preprocessing Complete!")
    print("=" * 70)

    print(f"\nNext steps:")
    print(f"  1. Test loading fragments with data.load_local_data()")
    print(f"  2. Run model inference on fragment-specific CSVs")
    print(f"  3. Compare results with paper (Sakhnini et al. 2025)")
    print(f"  4. Create PR to close Issue #4")


if __name__ == "__main__":
    main()
