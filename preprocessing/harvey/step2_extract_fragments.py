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

from __future__ import annotations

from pathlib import Path

import pandas as pd

from preprocessing.fragment_utils import process_sequences_to_fragments
from preprocessing.logging_config import setup_logger
from preprocessing.paths import HARVEY_FRAGMENTS_DIR, HARVEY_FULL_CSV

logger = setup_logger(__name__)


def process_harvey_dataset(csv_path: str) -> pd.DataFrame:
    """
    Process Harvey CSV to extract all VHH fragments.

    Args:
        csv_path: Path to harvey.csv

    Returns:
        DataFrame with all fragments and metadata
    """
    logger.info(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)

    logger.info(f"  Total nanobodies: {len(df)}")
    logger.info("  Annotating sequences with ANARCI (IMGT scheme)...")

    # Generate sequential IDs (harvey_000001, harvey_000002, etc.)
    df["id"] = [f"harvey_{i + 1:06d}" for i in range(len(df))]
    # Add source metadata
    df["source"] = "harvey2022"

    # Process with shared utility (heavy chain only)
    df_annotated, failures = process_sequences_to_fragments(
        df, heavy_col="seq", light_col=None, id_col="id"
    )

    logger.info(f"\n  Successfully annotated: {len(df_annotated)}/{len(df)} nanobodies")
    if failures:
        logger.info(f"  Failures: {len(failures)}")
        logger.info(f"  Failed IDs (first 10): {failures[:10]}")

        # Write all failed IDs to log file
        failure_log = HARVEY_FRAGMENTS_DIR / "failed_sequences.txt"
        failure_log.parent.mkdir(parents=True, exist_ok=True)
        with open(failure_log, "w") as f:
            f.write("\n".join(failures))
        logger.info(f"  All failed IDs written to: {failure_log}")

    return df_annotated


def create_fragment_csvs(df: pd.DataFrame, output_dir: Path) -> None:
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

    logger.info(f"\nCreating {len(fragments)} fragment-specific CSV files...")

    for fragment_name, (column_name, _sequence_alias) in fragments.items():
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

        logger.info(
            f"  [OK] {fragment_name:12s} -> {output_path.name:30s} "
            f"(len: {min_len}-{max_len} aa, mean: {mean_len:.1f})"
        )

    logger.info(f"\n[OK] All fragments saved to: {output_dir}/")


def main() -> int:
    """Main processing pipeline."""
    # Paths
    csv_path = HARVEY_FULL_CSV
    output_dir = HARVEY_FRAGMENTS_DIR

    if not csv_path.exists():
        logger.info(f"Error: {csv_path} not found!")
        logger.info(
            "Please run preprocessing/harvey/step1_convert_raw_csvs.py to generate from raw Harvey CSVs."
        )
        logger.info("Raw CSVs should be in: data/test/harvey/raw/")
        return 1

    logger.info("=" * 70)
    logger.info("Harvey Dataset: VHH Fragment Extraction")
    logger.info("=" * 70)
    logger.info(f"\nInput:  {csv_path}")
    logger.info(f"Output: {output_dir}/")
    logger.info("Method: ANARCI (IMGT numbering scheme)")
    logger.info("Note:   Nanobodies (VHH) - no light chain fragments")

    # Process dataset
    df_annotated = process_harvey_dataset(str(csv_path))

    # Create fragment CSVs
    create_fragment_csvs(df_annotated, output_dir)

    # Validation summary
    logger.info("\n" + "=" * 70)
    logger.info("Fragment Extraction Summary")
    logger.info("=" * 70)

    logger.info(f"\nAnnotated nanobodies: {len(df_annotated)}")
    logger.info("Label distribution:")
    for label, count in df_annotated["label"].value_counts().sort_index().items():
        label_name = "Low polyreactivity" if label == 0 else "High polyreactivity"
        logger.info(f"  {label_name}: {count} ({count / len(df_annotated) * 100:.1f}%)")

    logger.info("\nFragment files created: 6 (VHH-specific)")
    logger.info(f"Output directory: {output_dir.absolute()}")

    logger.info("\n" + "=" * 70)
    logger.info("[DONE] Harvey Preprocessing Complete!")
    logger.info("=" * 70)

    logger.info("\nNext steps:")
    logger.info("  1. Test loading fragments with data.load_local_data()")
    logger.info("  2. Run model inference on fragment-specific CSVs")
    logger.info("  3. Compare results with paper (Sakhnini et al. 2025)")
    logger.info("  4. Create PR to close Issue #4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
