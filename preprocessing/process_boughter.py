#!/usr/bin/env python3
"""
Boughter Dataset Stage 2: ANARCI Annotation & Fragment Extraction

Processes boughter.csv from Stage 1, annotates with ANARCI using strict IMGT
numbering, and creates 16 fragment-specific CSV files.

Usage:
    python3 preprocessing/process_boughter.py

Inputs:
    test_datasets/boughter.csv - Output from Stage 1

Outputs:
    test_datasets/boughter/*_boughter.csv - 16 fragment-specific files
    test_datasets/boughter/annotation_failures.log - Failed annotations
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import riot_na

# Initialize ANARCI for amino acid annotation (IMGT scheme)
annotator = riot_na.create_riot_aa()


def annotate_sequence(
    seq_id: str, sequence: str, chain: str
) -> Optional[Dict[str, str]]:
    """
    Annotate a single amino acid sequence using ANARCI (IMGT).

    Uses strict IMGT boundaries per Sakhnini et al. 2025:
    "The primary sequences were annotated in the CDRs using ANARCI
    following the IMGT numbering scheme"

    CDR boundaries (strict IMGT):
    - CDR-H3: positions 105-117 (EXCLUDES position 118, which is FR4 J-anchor)
    - CDR-H2: positions 56-65 (fixed IMGT positions, variable lengths OK)
    - CDR-H1: positions 27-38 (fixed IMGT)

    Note: Position 118 (J-Trp/Phe) is conserved FR4, NOT CDR.
    Boughter's published .dat files include position 118, but we use
    strict IMGT for biological correctness and ML best practices.

    Note: CDR2 length naturally varies (8-11 residues typical).
    IMGT positions are fixed (56-65), but sequences can have gaps
    for deletions or insertion codes. Harvey et al. 2022 confirms
    this is expected with ANARCI/IMGT numbering. This is normal
    antibody diversity, not an error.

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
            f"full_seq_{chain}": annotation.sequence_alignment_aa,
            f"fwr1_aa_{chain}": annotation.fwr1_aa,
            f"cdr1_aa_{chain}": annotation.cdr1_aa,
            f"fwr2_aa_{chain}": annotation.fwr2_aa,
            f"cdr2_aa_{chain}": annotation.cdr2_aa,
            f"fwr3_aa_{chain}": annotation.fwr3_aa,
            f"cdr3_aa_{chain}": annotation.cdr3_aa,
            f"fwr4_aa_{chain}": annotation.fwr4_aa,
        }

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
        print(f"  ANARCI failed for {seq_id} ({chain} chain): {e}")
        return None


def process_antibody(row: pd.Series) -> Optional[Dict]:
    """
    Annotate heavy and light chains, create all 16 fragments.

    Args:
        row: DataFrame row with heavy_seq, light_seq, and metadata

    Returns:
        Dictionary with all fragments and metadata, or None if annotation fails
    """
    seq_id = row["id"]

    # Annotate heavy chain
    heavy_frags = annotate_sequence(seq_id, row["heavy_seq"], "H")
    if heavy_frags is None:
        return None

    # Annotate light chain
    light_frags = annotate_sequence(seq_id, row["light_seq"], "L")
    if light_frags is None:
        return None

    # Combine metadata and fragments
    result = {
        "id": row["id"],
        "subset": row["subset"],
        "num_flags": row["num_flags"],
        "flag_category": row["flag_category"],
        "label": row["label"],
        "include_in_training": row["include_in_training"],
        "source": row["source"],
    }

    result.update(heavy_frags)
    result.update(light_frags)

    # Create paired/combined fragments
    result["vh_vl"] = result["full_seq_H"] + result["full_seq_L"]
    result["all_cdrs"] = result["cdrs_H"] + result["cdrs_L"]
    result["all_fwrs"] = result["fwrs_H"] + result["fwrs_L"]

    return result


def annotate_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Annotate all antibodies in the dataset.

    Args:
        df: DataFrame from Stage 1 (boughter.csv)

    Returns:
        DataFrame with all fragments annotated
    """
    print(f"\nAnnotating {len(df)} antibodies with ANARCI (strict IMGT)...")

    results = []
    failures = []

    for idx, row in df.iterrows():
        result = process_antibody(row)

        if result is None:
            failures.append(row["id"])
        else:
            results.append(result)

        # Progress indicator
        if (idx + 1) % 100 == 0:
            print(f"  Progress: {idx + 1}/{len(df)} ({len(failures)} failures)")

    df_annotated = pd.DataFrame(results)

    print(f"\n✓ Successfully annotated: {len(df_annotated)}/{len(df)} antibodies")

    if failures:
        print(f"✗ Failures: {len(failures)}")
        failure_rate = len(failures) / len(df) * 100
        print(f"  Failure rate: {failure_rate:.2f}%")

        # Write failures to log
        failure_log = Path("test_datasets/boughter/annotation_failures.log")
        failure_log.write_text("\n".join(failures))
        print(f"  Failed IDs written to: {failure_log}")

    return df_annotated


def create_fragment_csvs(df: pd.DataFrame, output_dir: Path):
    """
    Create separate CSV files for each of the 16 fragment types.

    Following Sakhnini et al. 2025 Table 4 methodology.

    Args:
        df: DataFrame with all fragments
        output_dir: Directory to save fragment CSVs
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define all 16 fragment types
    fragments = {
        # 1-2: Full variable domains
        "VH": ("full_seq_H", "heavy_variable_domain"),
        "VL": ("full_seq_L", "light_variable_domain"),
        # 3-5: Heavy CDRs
        "H-CDR1": ("cdr1_aa_H", "h_cdr1"),
        "H-CDR2": ("cdr2_aa_H", "h_cdr2"),
        "H-CDR3": ("cdr3_aa_H", "h_cdr3"),
        # 6-8: Light CDRs
        "L-CDR1": ("cdr1_aa_L", "l_cdr1"),
        "L-CDR2": ("cdr2_aa_L", "l_cdr2"),
        "L-CDR3": ("cdr3_aa_L", "l_cdr3"),
        # 9-10: Concatenated CDRs
        "H-CDRs": ("cdrs_H", "h_cdrs_concatenated"),
        "L-CDRs": ("cdrs_L", "l_cdrs_concatenated"),
        # 11-12: Concatenated FWRs
        "H-FWRs": ("fwrs_H", "h_fwrs_concatenated"),
        "L-FWRs": ("fwrs_L", "l_fwrs_concatenated"),
        # 13: Paired variable domains
        "VH+VL": ("vh_vl", "paired_variable_domains"),
        # 14-15: All CDRs/FWRs
        "All-CDRs": ("all_cdrs", "all_cdrs_heavy_light"),
        "All-FWRs": ("all_fwrs", "all_fwrs_heavy_light"),
        # 16: Full (alias for VH+VL)
        "Full": ("vh_vl", "full_sequence"),
    }

    print(f"\nCreating {len(fragments)} fragment-specific CSV files...")

    for fragment_name, (column_name, description) in fragments.items():
        output_path = output_dir / f"{fragment_name}_boughter.csv"

        # Create fragment-specific CSV with standardized column names
        fragment_df = pd.DataFrame(
            {
                "id": df["id"],
                "sequence": df[column_name],
                "label": df["label"],
                "subset": df["subset"],
                "num_flags": df["num_flags"],
                "flag_category": df["flag_category"],
                "include_in_training": df["include_in_training"],
                "source": df["source"],
                "sequence_length": df[column_name].str.len(),
            }
        )

        fragment_df.to_csv(output_path, index=False)
        print(f"  ✓ {fragment_name:12s} -> {output_path.name}")

    print(f"\n✓ All {len(fragments)} fragment files created in: {output_dir}")


def print_annotation_stats(df: pd.DataFrame):
    """Print CDR length distributions and annotation statistics."""
    print("\n" + "=" * 70)
    print("CDR Length Distributions (Strict IMGT)")
    print("=" * 70)

    cdr_columns = {
        "H-CDR1": "cdr1_aa_H",
        "H-CDR2": "cdr2_aa_H",
        "H-CDR3": "cdr3_aa_H",
        "L-CDR1": "cdr1_aa_L",
        "L-CDR2": "cdr2_aa_L",
        "L-CDR3": "cdr3_aa_L",
    }

    for cdr_name, col_name in cdr_columns.items():
        lengths = df[col_name].str.len()
        print(
            f"\n{cdr_name}: min={lengths.min()}, max={lengths.max()}, "
            f"mean={lengths.mean():.1f}, median={lengths.median():.0f}"
        )

        # Show distribution
        length_dist = lengths.value_counts().sort_index()
        if len(length_dist) <= 10:
            for length, count in length_dist.items():
                print(f"  {length:2d} aa: {count:4d} sequences")


def main():
    """Main processing pipeline."""
    # Load Stage 1 output
    input_csv = Path("test_datasets/boughter.csv")

    if not input_csv.exists():
        print(f"ERROR: {input_csv} not found!")
        print("Please run scripts/convert_boughter_to_csv.py first (Stage 1)")
        sys.exit(1)

    print("=" * 70)
    print("Boughter Dataset - Stage 2: ANARCI Annotation")
    print("=" * 70)

    df = pd.read_csv(input_csv)
    print(f"\nLoaded {len(df)} antibodies from Stage 1")

    # Annotate all sequences
    df_annotated = annotate_all(df)

    # Print CDR statistics
    print_annotation_stats(df_annotated)

    # Create 16 fragment CSVs
    output_dir = Path("test_datasets/boughter")
    create_fragment_csvs(df_annotated, output_dir)

    print("\n" + "=" * 70)
    print("Stage 2 Complete - Boughter Dataset Ready!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Verify fragment files in test_datasets/boughter/")
    print("  2. Check annotation_failures.log for any issues")
    print("  3. Use fragment files for ESM embedding and training")


if __name__ == "__main__":
    main()
