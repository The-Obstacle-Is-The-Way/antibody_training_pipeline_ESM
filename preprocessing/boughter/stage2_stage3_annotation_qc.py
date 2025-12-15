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
    data/train/boughter/processed/boughter.csv - Output from Stage 1 (1,117 sequences)

Outputs:
    data/train/boughter/annotated/*_boughter.csv - 16 fragment CSVs (1,065 rows each)
    data/train/boughter/canonical/VH_only_boughter_training.csv - Training subset (914 rows)
    data/train/boughter/annotated/annotation_failures.log - Failed annotations (Stage 2)
    data/train/boughter/annotated/qc_filtered_sequences.txt - QC-filtered sequences (Stage 3)

Results Summary:
    Stage 1 input:  1,117 sequences (95.4% DNA translation success from 1,171 raw)
    Stage 2 output: 1,110 sequences (99.4% ANARCI annotation success, 7 failures)
    Stage 3 output: 1,065 sequences (95.9% retention after QC, 45 filtered)
    Training data:  914 sequences (443 specific + 471 non-specific, 151 excluded)

Reference: See docs/datasets/boughter/data_sources.md for complete methodology
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from preprocessing.boughter.annotation.annotator import annotate_all
from preprocessing.boughter.annotation.qc import filter_quality_issues
from preprocessing.logging_config import setup_logger
from preprocessing.paths import (
    BOUGHTER_ANNOTATED_DIR,
    BOUGHTER_STAGE1_OUTPUT,
    BOUGHTER_TRAINING_SUBSET,
)

logger = setup_logger(__name__)


def create_fragment_csvs(df: pd.DataFrame, output_dir: Path) -> None:
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

    logger.info(f"\nCreating {len(fragments)} fragment-specific CSV files...")

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

        logger.info(f"  ✓ {fragment_name:12s} -> {output_path.name}")

    logger.info(f"\n✓ All {len(fragments)} fragment files created in: {output_dir}")


def export_training_subset(df: pd.DataFrame, output_path: Path) -> None:
    """
    Export the canonical training subset used by the model pipeline.

    The canonical file contains only [sequence, label] columns for VH domains and
    includes sequences where include_in_training == True.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train_df = df[df["include_in_training"]].copy()
    if len(train_df) == 0:
        logger.info("⚠ No sequences flagged for training; canonical export skipped.")
        return

    canonical_df = (
        train_df[["full_seq_H", "label"]]
        .rename(columns={"full_seq_H": "sequence"})
        .assign(label=lambda d: d["label"].astype(float))
    )

    canonical_df.to_csv(output_path, index=False)

    label_counts = canonical_df["label"].value_counts().sort_index()
    logger.info(f"\n✓ Canonical training file exported: {output_path}")
    logger.info(f"  Total sequences: {len(canonical_df)}")
    for label, count in label_counts.items():
        label_name = "Specific (0)" if label == 0.0 else "Non-specific (1)"
        pct = count / len(canonical_df) * 100
        logger.info(f"  {label_name}: {count} ({pct:.1f}%)")


def print_annotation_stats(df: pd.DataFrame) -> None:
    """Print CDR length distributions and annotation statistics."""
    logger.info("=" * 70)
    logger.info("Boughter Stage 2 & 3: ANARCI Annotation & QC")
    logger.info("=" * 70)

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
        logger.info(
            f"\n{cdr_name}: min={lengths.min()}, max={lengths.max()}, "
            f"mean={lengths.mean():.1f}, median={lengths.median():.0f}"
        )

        # Show distribution
        length_dist = lengths.value_counts().sort_index()
        if len(length_dist) <= 10:
            for length, count in length_dist.items():
                logger.info(f"  {length:2d} aa: {count:4d} sequences")


def main() -> int:
    """Main processing pipeline."""
    # Load Stage 1 output
    input_csv = BOUGHTER_STAGE1_OUTPUT

    if not input_csv.exists():
        logger.info(f"ERROR: {input_csv} not found!")
        logger.info(
            "Please run preprocessing/boughter/stage1_dna_translation.py first (Stage 1)"
        )
        return 1

    logger.info("=" * 70)
    logger.info("Boughter Dataset - Stage 2: ANARCI Annotation")
    logger.info("=" * 70)

    df = pd.read_csv(input_csv)
    logger.info(f"\nLoaded {len(df)} antibodies from Stage 1")

    # Stage 2: Annotate all sequences
    df_annotated = annotate_all(df)

    # Stage 3: Quality control filtering
    df_clean = filter_quality_issues(df_annotated)

    # Print CDR statistics (on clean data)
    print_annotation_stats(df_clean)

    # Create 16 fragment CSVs (from clean data)
    output_dir = BOUGHTER_ANNOTATED_DIR
    create_fragment_csvs(df_clean, output_dir)

    # Export canonical VH-only training subset
    canonical_path = BOUGHTER_TRAINING_SUBSET
    export_training_subset(df_clean, canonical_path)

    logger.info("\n" + "=" * 70)
    logger.info("Boughter Dataset Processing Complete!")
    logger.info("=" * 70)
    logger.info("\nPipeline Summary:")
    logger.info(f"  Stage 1 (Translation):  {len(df)} sequences")
    logger.info(
        f"  Stage 2 (ANARCI):       {len(df_annotated)} sequences ({len(df_annotated) / len(df) * 100:.1f}%)"
    )
    logger.info(
        f"  Stage 3 (Quality QC):   {len(df_clean)} sequences ({len(df_clean) / len(df) * 100:.1f}%)"
    )
    logger.info("\nNext steps:")
    logger.info("  1. Verify fragment files in data/train/boughter/annotated/")
    logger.info("  2. Check annotation_failures.log for any issues")
    logger.info("  3. Review quality metrics in validation report")
    logger.info("  4. Use fragment files for ESM embedding and training")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
