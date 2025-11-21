#!/usr/bin/env python3
"""
Shehata Dataset Fragment Extraction Script

Processes the Shehata dataset to extract all 16 antibody fragment types
using ANARCI (IMGT numbering scheme) following Sakhnini et al. 2025 methodology.

Fragments extracted:
1. VH (full heavy variable domain)
2. VL (full light variable domain)
3. H-CDR1
4. H-CDR2
5. H-CDR3
6. L-CDR1
7. L-CDR2
8. L-CDR3
9. H-CDRs (concatenated H-CDR1+2+3)
10. L-CDRs (concatenated L-CDR1+2+3)
11. H-FWRs (concatenated H-FWR1+2+3+4)
12. L-FWRs (concatenated L-FWR1+2+3+4)
13. VH+VL (paired variable domains)
14. All-CDRs (H-CDRs + L-CDRs)
15. All-FWRs (H-FWRs + L-FWRs)
16. Full (VH + VL = same as #13 for compatibility)

Date: 2025-10-31
Issue: #3 - Shehata dataset preprocessing (Phase 2)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from preprocessing.fragment_utils import process_sequences_to_fragments
from preprocessing.logging_config import setup_logger
from preprocessing.paths import SHEHATA_FRAGMENTS_DIR, SHEHATA_PROCESSED_CSV

logger = setup_logger(__name__)


def process_shehata_dataset(csv_path: str) -> pd.DataFrame:
    """
    Process Shehata CSV to extract all fragments.

    Args:
        csv_path: Path to shehata.csv

    Returns:
        DataFrame with all fragments and metadata
    """
    logger.info(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)

    logger.info(f"  Total antibodies: {len(df)}")
    logger.info("  Annotating sequences with ANARCI (IMGT scheme)...")

    # Process with shared utility
    df_annotated, failures = process_sequences_to_fragments(
        df, heavy_col="heavy_seq", light_col="light_seq", id_col="id"
    )

    logger.info(f"\n  Successfully annotated: {len(df_annotated)}/{len(df)} antibodies")
    if failures:
        logger.info(f"  Failures: {len(failures)}")
        logger.info(f"  Failed IDs: {failures}")

    return df_annotated


def create_fragment_csvs(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create separate CSV files for each fragment type.

    Following the 16-fragment methodology from Sakhnini et al. 2025.

    Args:
        df: DataFrame with all fragments
        output_dir: Directory to save fragment CSVs
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define all 16 fragment types
    fragments = {
        # 1-2: Full variable domains
        "VH_only": ("full_seq_H", "heavy_seq"),
        "VL_only": ("full_seq_L", "light_seq"),
        # 3-5: Heavy CDRs
        "H-CDR1": ("cdr1_aa_H", "h_cdr1"),
        "H-CDR2": ("cdr2_aa_H", "h_cdr2"),
        "H-CDR3": ("cdr3_aa_H", "h_cdr3"),
        # 6-8: Light CDRs
        "L-CDR1": ("cdr1_aa_L", "l_cdr1"),
        "L-CDR2": ("cdr2_aa_L", "l_cdr2"),
        "L-CDR3": ("cdr3_aa_L", "l_cdr3"),
        # 9-10: Concatenated CDRs
        "H-CDRs": ("cdrs_H", "h_cdrs"),
        "L-CDRs": ("cdrs_L", "l_cdrs"),
        # 11-12: Concatenated FWRs
        "H-FWRs": ("fwrs_H", "h_fwrs"),
        "L-FWRs": ("fwrs_L", "l_fwrs"),
        # 13: Paired variable domains
        "VH+VL": ("vh_vl", "paired_variable_domains"),
        # 14-15: All CDRs/FWRs
        "All-CDRs": ("all_cdrs", "all_cdrs"),
        "All-FWRs": ("all_fwrs", "all_fwrs"),
        # 16: Full (alias for VH+VL for compatibility)
        "Full": ("vh_vl", "full_sequence"),
    }

    logger.info(f"\nCreating {len(fragments)} fragment-specific CSV files...")

    for fragment_name, (column_name, _sequence_alias) in fragments.items():
        output_path = output_dir / f"{fragment_name}_shehata.csv"

        # Create fragment-specific CSV with standardized column names
        fragment_df = pd.DataFrame(
            {
                "id": df["id"],
                "sequence": df[column_name],
                "label": df["label"],
                "psr_score": df["psr_score"],
                "b_cell_subset": df["b_cell_subset"],
                "source": df["source"],
            }
        )

        fragment_df.to_csv(output_path, index=False)

        logger.info(f"  ✓ {fragment_name:12s} → {output_path.name}")

    logger.info(f"\n✓ All fragments saved to: {output_dir}/")


def main() -> int:
    """Main processing pipeline."""
    # Paths
    csv_path = SHEHATA_PROCESSED_CSV
    output_dir = SHEHATA_FRAGMENTS_DIR

    if not csv_path.exists():
        logger.info(f"Error: {csv_path} not found!")
        logger.info(
            "Please run preprocessing/shehata/step1_convert_excel_to_csv.py first."
        )
        return 1

    logger.info("=" * 60)
    logger.info("Shehata Dataset: Fragment Extraction (Phase 2)")
    logger.info("=" * 60)
    logger.info(f"\nInput:  {csv_path}")
    logger.info(f"Output: {output_dir}/")
    logger.info("Method: ANARCI (IMGT numbering scheme)")
    print()

    # Process dataset
    df_annotated = process_shehata_dataset(str(csv_path))

    # Create fragment CSVs
    create_fragment_csvs(df_annotated, output_dir)

    # Validation summary
    logger.info("\n" + "=" * 60)
    logger.info("Fragment Extraction Summary")
    logger.info("=" * 60)

    logger.info(f"\nAnnotated antibodies: {len(df_annotated)}")
    logger.info("Label distribution:")
    for label, count in df_annotated["label"].value_counts().sort_index().items():
        label_name = "Specific" if label == 0 else "Non-specific"
        logger.info(f"  {label_name}: {count} ({count / len(df_annotated) * 100:.1f}%)")

    logger.info("\nFragment files created: 16")
    logger.info(f"Output directory: {output_dir.absolute()}")

    logger.info("\n" + "=" * 60)
    logger.info("✓ Phase 2 Complete!")
    logger.info("=" * 60)

    logger.info("\nNext steps:")
    logger.info("  1. Test loading fragments with data.load_local_data()")
    logger.info("  2. Run model inference on fragment-specific CSVs")
    logger.info("  3. Compare results with paper (Sakhnini et al. 2025)")
    logger.info("  4. Create PR to close Issue #3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
