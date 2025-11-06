#!/usr/bin/env python3
"""
Boughter Dataset Preprocessing - Stages 2+3: ANARCI Annotation & Quality Control

Processes boughter.csv from Stage 1, annotates with ANARCI using strict IMGT
numbering, applies post-annotation quality control, and creates 16 fragment-specific
CSV files with include_in_training flags.

Pipeline Position: Stages 2+3 of 3
    Stage 1 → boughter.csv (1,117 sequences)
    Stages 2+3 (this script) → 16 fragment CSVs (1,065 sequences each)
    Training subset: VH_only_boughter_training.csv (914 sequences)

Pipeline Flow:
    Stage 2: ANARCI annotation with strict IMGT boundaries
    Stage 3: Post-annotation quality control (filter X in CDRs, empty CDRs)

Usage:
    python3 preprocessing/boughter/stage2_stage3_annotation_qc.py

Inputs:
    train_datasets/boughter/processed/boughter.csv - Output from Stage 1 (1,117 sequences)

Outputs:
    train_datasets/boughter/annotated/*_boughter.csv - 16 fragment CSVs (1,065 rows each)
    train_datasets/boughter/canonical/VH_only_boughter_training.csv - Training subset (914 rows)
    train_datasets/boughter/annotated/annotation_failures.log - Failed annotations (Stage 2)
    train_datasets/boughter/annotated/qc_filtered_sequences.txt - QC-filtered sequences (Stage 3)

Results Summary:
    Stage 1 input:  1,117 sequences (95.4% DNA translation success from 1,171 raw)
    Stage 2 output: 1,110 sequences (99.4% ANARCI annotation success, 7 failures)
    Stage 3 output: 1,065 sequences (95.9% retention after QC, 45 filtered)
    Training data:  914 sequences (443 specific + 471 non-specific, 151 excluded)

Reference: See docs/boughter/boughter_data_sources.md for complete methodology
"""

import sys
from pathlib import Path
from typing import cast

import pandas as pd
import riot_na

# Initialize ANARCI for amino acid annotation (IMGT scheme)
annotator = riot_na.create_riot_aa()


def annotate_sequence(seq_id: str, sequence: str, chain: str) -> dict[str, str] | None:
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

        # Extract all fragments, converting None to empty string
        # ANARCI returns None for regions it cannot identify
        def safe_str(value):
            """Convert None to empty string, preserve actual strings."""
            return value if value is not None else ""

        # Extract individual fragments
        fragments = {
            f"fwr1_aa_{chain}": safe_str(annotation.fwr1_aa),
            f"cdr1_aa_{chain}": safe_str(annotation.cdr1_aa),
            f"fwr2_aa_{chain}": safe_str(annotation.fwr2_aa),
            f"cdr2_aa_{chain}": safe_str(annotation.cdr2_aa),
            f"fwr3_aa_{chain}": safe_str(annotation.fwr3_aa),
            f"cdr3_aa_{chain}": safe_str(annotation.cdr3_aa),
            f"fwr4_aa_{chain}": safe_str(annotation.fwr4_aa),
        }

        # Reconstruct full V-domain from fragments (avoids constant region garbage)
        # This is gap-free and clean (P0 fix + constant region removal)
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

        # Validate that we got at least SOME fragments
        # If all CDRs are empty, annotation failed
        if not any(
            [
                fragments[f"cdr1_aa_{chain}"],
                fragments[f"cdr2_aa_{chain}"],
                fragments[f"cdr3_aa_{chain}"],
            ]
        ):
            print(f"  ANARCI returned no CDRs for {seq_id} ({chain} chain)")
            return None

        # Create concatenated fragments (safe now - no None values)
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


def process_antibody(row: pd.Series) -> dict | None:
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
        idx_int = cast(int, idx)
        if (idx_int + 1) % 100 == 0:
            print(f"  Progress: {idx_int + 1}/{len(df)} ({len(failures)} failures)")

    df_annotated = pd.DataFrame(results)

    print(f"\n✓ Successfully annotated: {len(df_annotated)}/{len(df)} antibodies")

    if failures:
        print(f"✗ Failures: {len(failures)}")
        failure_rate = len(failures) / len(df) * 100
        print(f"  Failure rate: {failure_rate:.2f}%")

        # Write failures to log
        failure_log = Path("train_datasets/boughter/annotated/annotation_failures.log")
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
    # Using standard naming: VH_only, VL_only (matches other datasets)
    fragments = {
        # 1-2: Full variable domains
        "VH_only": ("full_seq_H", "heavy_variable_domain"),
        "VL_only": ("full_seq_L", "light_variable_domain"),
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

        # Write metadata header as comments, then CSV data
        metadata = f"""# Boughter Dataset - {fragment_name} Fragment
# CDR Extraction Method: ANARCI (IMGT numbering, strict)
# CDR-H3 Boundary: positions 105-117 (EXCLUDES position 118 - FR4 J-anchor)
# CDR-H2 Boundary: positions 56-65 (fixed IMGT, variable lengths are normal)
# CDR-H1 Boundary: positions 27-38 (fixed IMGT)
# Boundary Rationale: Position 118 is FR4 J-anchor (conserved W/F), not CDR
# Boughter Note: Original Boughter files include position 118; we use strict IMGT
# Fragment Description: {description}
# Reference: See docs/cdr_boundary_first_principles_audit.md
# Total Sequences: {len(fragment_df)}
# Training Sequences: {len(fragment_df[fragment_df["include_in_training"]])}
"""

        # Write metadata + CSV
        with open(output_path, "w") as f:
            f.write(metadata)
            fragment_df.to_csv(f, index=False)

        print(f"  ✓ {fragment_name:12s} -> {output_path.name}")

    print(f"\n✓ All {len(fragments)} fragment files created in: {output_dir}")


def export_training_subset(df: pd.DataFrame, output_path: Path):
    """
    Export the canonical training subset used by the model pipeline.

    The canonical file contains only [sequence, label] columns for VH domains and
    includes sequences where include_in_training == True.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train_df = df[df["include_in_training"]].copy()
    if len(train_df) == 0:
        print("⚠ No sequences flagged for training; canonical export skipped.")
        return

    canonical_df = (
        train_df[["full_seq_H", "label"]]
        .rename(columns={"full_seq_H": "sequence"})
        .assign(label=lambda d: d["label"].astype(float))
    )

    canonical_df.to_csv(output_path, index=False)

    label_counts = canonical_df["label"].value_counts().sort_index()
    print(f"\n✓ Canonical training file exported: {output_path}")
    print(f"  Total sequences: {len(canonical_df)}")
    for label, count in label_counts.items():
        label_name = "Specific (0)" if label == 0.0 else "Non-specific (1)"
        pct = count / len(canonical_df) * 100
        print(f"  {label_name}: {count} ({pct:.1f}%)")


def filter_quality_issues(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 3: Post-annotation quality control.

    Following Boughter et al. 2020 methodology (seq_loader.py)
    and 2025 industry best practices (Harvey 2022, AbSet 2024).

    Removes:
    1. Sequences with X (unknown amino acid) in ANY CDR
    2. Sequences with empty ("") CDRs

    This is done AFTER ANARCI annotation to maximize
    information extraction from raw data.

    See docs/accuracy_verification_report.md for rationale.
    """
    print("\n" + "=" * 70)
    print("Stage 3: Post-annotation Quality Control")
    print("=" * 70)
    print(f"Input sequences: {len(df)}")
    print()

    cdr_columns = [
        "cdr1_aa_H",
        "cdr2_aa_H",
        "cdr3_aa_H",
        "cdr1_aa_L",
        "cdr2_aa_L",
        "cdr3_aa_L",
    ]

    df_clean = df.copy()

    # Track sequences removed for each reason
    sequences_with_X = set()
    sequences_with_empty = set()

    # First pass: identify sequences with X in ANY CDR
    for col in cdr_columns:
        has_X = df_clean[df_clean[col].str.contains("X", na=False)]
        if len(has_X) > 0:
            sequences_with_X.update(has_X["id"].tolist())

    # Second pass: identify sequences with empty CDRs
    for col in cdr_columns:
        is_empty = df_clean[df_clean[col] == ""]
        if len(is_empty) > 0:
            sequences_with_empty.update(is_empty["id"].tolist())

    # Remove all problematic sequences
    problematic_ids = sequences_with_X | sequences_with_empty
    df_clean = df_clean[~df_clean["id"].isin(problematic_ids)]

    print(f"Sequences with X in ANY CDR: {len(sequences_with_X)}")
    print(f"Sequences with empty CDRs:    {len(sequences_with_empty)}")
    print(f"Total unique sequences removed: {len(problematic_ids)}")

    if problematic_ids:
        qc_log = Path("train_datasets/boughter/annotated/qc_filtered_sequences.txt")
        qc_log.write_text("\n".join(sorted(problematic_ids)))
        print(f"Filtered IDs written to: {qc_log}")

    print()
    print(f"Output sequences: {len(df_clean)}")
    print(f"Retention rate: {len(df_clean) / len(df) * 100:.1f}%")

    return df_clean


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
    input_csv = Path("train_datasets/boughter/processed/boughter.csv")

    if not input_csv.exists():
        print(f"ERROR: {input_csv} not found!")
        print(
            "Please run preprocessing/boughter/stage1_dna_translation.py first (Stage 1)"
        )
        sys.exit(1)

    print("=" * 70)
    print("Boughter Dataset - Stage 2: ANARCI Annotation")
    print("=" * 70)

    df = pd.read_csv(input_csv)
    print(f"\nLoaded {len(df)} antibodies from Stage 1")

    # Stage 2: Annotate all sequences
    df_annotated = annotate_all(df)

    # Stage 3: Quality control filtering
    df_clean = filter_quality_issues(df_annotated)

    # Print CDR statistics (on clean data)
    print_annotation_stats(df_clean)

    # Create 16 fragment CSVs (from clean data)
    output_dir = Path("train_datasets/boughter/annotated")
    create_fragment_csvs(df_clean, output_dir)

    # Export canonical VH-only training subset
    canonical_path = Path(
        "train_datasets/boughter/canonical/VH_only_boughter_training.csv"
    )
    export_training_subset(df_clean, canonical_path)

    print("\n" + "=" * 70)
    print("Boughter Dataset Processing Complete!")
    print("=" * 70)
    print("\nPipeline Summary:")
    print(f"  Stage 1 (Translation):  {len(df)} sequences")
    print(
        f"  Stage 2 (ANARCI):       {len(df_annotated)} sequences ({len(df_annotated) / len(df) * 100:.1f}%)"
    )
    print(
        f"  Stage 3 (Quality QC):   {len(df_clean)} sequences ({len(df_clean) / len(df) * 100:.1f}%)"
    )
    print("\nNext steps:")
    print("  1. Verify fragment files in train_datasets/boughter/annotated/")
    print("  2. Check annotation_failures.log for any issues")
    print("  3. Review quality metrics in validation report")
    print("  4. Use fragment files for ESM embedding and training")


if __name__ == "__main__":
    main()
