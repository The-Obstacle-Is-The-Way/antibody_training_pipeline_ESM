#!/usr/bin/env python3
"""
Boughter Dataset Preprocessing - Stage 1: DNA Translation & Novo Flagging

Processes raw Boughter DNA FASTA files, translates to protein sequences, applies
Novo Nordisk flagging strategy, and outputs combined CSV for downstream processing.

Pipeline Position: Stage 1 of 3
    Stage 1 (this script) → boughter.csv (1,117 sequences)
    Stage 2+3 → Fragment CSVs (1,065 sequences, 16 fragments)

Usage:
    python3 preprocessing/boughter/stage1_dna_translation.py

Inputs:
    data/train/boughter/raw/*.txt - Raw DNA FASTA files (1,171 sequences)

Outputs:
    data/train/boughter/processed/boughter.csv - Combined dataset (1,117 sequences, 95.4% translation success)
    data/train/boughter/raw/translation_failures.log - Failed sequences (54 failures)

Novo Nordisk Flagging Strategy:
    - 0 flags: Specific (label=0, include in training)
    - 1-3 flags: Mildly polyreactive (exclude from training)
    - 4+ flags: Non-specific (label=1, include in training)

Reference: See docs/datasets/boughter/data_sources.md for methodology
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypedDict

import pandas as pd

from preprocessing.boughter.translation.readers import (
    parse_fasta_dna,
    parse_numreact_flags,
    parse_yn_flags,
)
from preprocessing.boughter.translation.translator import (
    translate_dna_to_protein,
    validate_translation,
)
from preprocessing.logging_config import setup_logger
from preprocessing.paths import BOUGHTER_PROCESSED_DIR, BOUGHTER_RAW_DIR

logger = setup_logger(__name__)


def apply_novo_flagging(num_flags: int) -> dict[str, Any]:
    """
    Apply Novo Nordisk flagging strategy from Sakhnini et al. 2025.

    Rules:
    - 0 flags → label 0 (specific), INCLUDE in training
    - 1-3 flags → EXCLUDE from training (mild polyreactivity)
    - >3 flags (4-7) → label 1 (non-specific), INCLUDE in training

    Args:
        num_flags: Number of ELISA flags (0-7)

    Returns:
        Dictionary with label, category, and inclusion status
    """
    if num_flags == 0:
        return {"label": 0, "flag_category": "specific", "include_in_training": True}
    elif 1 <= num_flags <= 3:
        return {"label": None, "flag_category": "mild", "include_in_training": False}
    elif num_flags >= 4:
        return {
            "label": 1,
            "flag_category": "non_specific",
            "include_in_training": True,
        }
    else:
        raise ValueError(f"Invalid flag count: {num_flags}")


def process_subset(
    subset_name: str,
    heavy_path: Path,
    light_path: Path,
    flag_path: Path,
    flag_format: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Process a single subset: translate DNA, pair sequences, apply flagging.

    Args:
        subset_name: Name of subset (e.g., 'flu', 'hiv_nat')
        heavy_path: Path to heavy chain DNA FASTA
        light_path: Path to light chain DNA FASTA
        flag_path: Path to flag file
        flag_format: 'numreact' or 'yn'

    Returns:
        List of dictionaries with processed antibody data
    """
    logger.info(f"\nProcessing subset: {subset_name}")

    # Parse input files
    heavy_dna = parse_fasta_dna(heavy_path)
    light_dna = parse_fasta_dna(light_path)

    if flag_format == "numreact":
        flags = parse_numreact_flags(flag_path)
    elif flag_format == "yn":
        flags = parse_yn_flags(flag_path)
    else:
        raise ValueError(f"Unknown flag format: {flag_format}")

    # Validate counts match
    counts = [len(heavy_dna), len(light_dna), len(flags)]
    if len(set(counts)) != 1:
        raise ValueError(
            f"{subset_name}: Sequence count mismatch - "
            f"Heavy: {counts[0]}, Light: {counts[1]}, Flags: {counts[2]}"
        )

    logger.info(f"  Sequences: {len(heavy_dna)}")

    results = []
    failures = []

    for idx in range(len(heavy_dna)):
        # Generate sequential ID
        seq_id = f"{subset_name}_{idx + 1:06d}"

        # Translate DNA → protein
        heavy_protein = translate_dna_to_protein(heavy_dna[idx])
        light_protein = translate_dna_to_protein(light_dna[idx])

        # Validate translations
        if not heavy_protein or not light_protein:
            failures.append(f"{seq_id}: Translation failed")
            continue

        if not validate_translation(heavy_protein) or not validate_translation(
            light_protein
        ):
            failures.append(f"{seq_id}: Invalid protein sequence")
            continue

        # Apply Novo flagging strategy
        flagging = apply_novo_flagging(flags[idx])

        # Create result record
        results.append(
            {
                "id": seq_id,
                "subset": subset_name,
                "heavy_seq": heavy_protein,
                "light_seq": light_protein,
                "num_flags": flags[idx],
                "flag_category": flagging["flag_category"],
                "label": flagging["label"],
                "include_in_training": flagging["include_in_training"],
                "source": "boughter2020",
            }
        )

    logger.info(f"  Successful: {len(results)}")
    logger.info(f"  Failures: {len(failures)}")

    if failures:
        logger.info(f"  Failed IDs: {', '.join(failures[:5])}")
        if len(failures) > 5:
            logger.info(f"    ... and {len(failures) - 5} more")

    return results, failures


def print_dataset_stats(df: pd.DataFrame) -> None:
    """Print comprehensive dataset statistics."""
    logger.info("\n" + "=" * 70)
    logger.info("Boughter Dataset - Stage 1 Complete")
    logger.info("=" * 70)

    logger.info(f"\nTotal sequences across all subsets: {len(df)}")

    logger.info("\nBreakdown by subset:")
    for subset in sorted(df["subset"].unique()):
        subset_df = df[df["subset"] == subset]
        logger.info(f"  {subset:12s}: {len(subset_df):4d} sequences")

    logger.info("\nFlag distribution:")
    for flag in sorted(df["num_flags"].unique()):
        count = len(df[df["num_flags"] == flag])
        pct = count / len(df) * 100
        logger.info(f"  {flag} flags: {count:4d} ({pct:5.2f}%)")

    logger.info("\nNovo flagging strategy results:")
    for category in ["specific", "mild", "non_specific"]:
        cat_df = df[df["flag_category"] == category]
        count = len(cat_df)
        pct = count / len(df) * 100
        included = len(cat_df[cat_df["include_in_training"]])
        logger.info(
            f"  {category:15s}: {count:4d} ({pct:5.2f}%) - "
            f"{included} included in training"
        )

    training_df = df[df["include_in_training"]]
    logger.info(f"\nTraining set size: {len(training_df)} sequences")
    logger.info(
        f"Excluded (mild 1-3 flags): {len(df[~df['include_in_training']])} sequences"
    )

    if len(training_df) > 0:
        label_dist = training_df["label"].value_counts()
        logger.info("\nTraining set label balance:")
        for label in sorted(label_dist.index):
            count = label_dist[label]
            label_name = "Specific (0)" if label == 0 else "Non-specific (1)"
            pct = count / len(training_df) * 100
            logger.info(f"  {label_name}: {count:4d} ({pct:5.2f}%)")


class SubsetPaths(TypedDict):
    heavy_dna: Path
    light_dna: Path
    flags: Path
    flag_format: Literal["numreact", "yn"]


def main() -> int:
    """Main processing pipeline."""
    # Define dataset structure
    # Raw data is in boughter_raw/ (not committed to git)
    base_dir = BOUGHTER_RAW_DIR

    subsets: dict[str, SubsetPaths] = {
        "flu": {
            "heavy_dna": base_dir / "flu_fastaH.txt",
            "light_dna": base_dir / "flu_fastaL.txt",
            "flags": base_dir / "flu_NumReact.txt",
            "flag_format": "numreact",
        },
        "hiv_nat": {
            "heavy_dna": base_dir / "nat_hiv_fastaH.txt",
            "light_dna": base_dir / "nat_hiv_fastaL.txt",
            "flags": base_dir / "nat_hiv_NumReact.txt",
            "flag_format": "numreact",
        },
        "hiv_cntrl": {
            "heavy_dna": base_dir / "nat_cntrl_fastaH.txt",
            "light_dna": base_dir / "nat_cntrl_fastaL.txt",
            "flags": base_dir / "nat_cntrl_NumReact.txt",
            "flag_format": "numreact",
        },
        "hiv_plos": {
            "heavy_dna": base_dir / "plos_hiv_fastaH.txt",
            "light_dna": base_dir / "plos_hiv_fastaL.txt",
            "flags": base_dir / "plos_hiv_YN.txt",
            "flag_format": "yn",
        },
        "gut_hiv": {
            "heavy_dna": base_dir / "gut_hiv_fastaH.txt",
            "light_dna": base_dir / "gut_hiv_fastaL.txt",
            "flags": base_dir / "gut_hiv_NumReact.txt",
            "flag_format": "numreact",
        },
        "mouse_iga": {
            "heavy_dna": base_dir / "mouse_fastaH.dat",
            "light_dna": base_dir / "mouse_fastaL.dat",
            "flags": base_dir / "mouse_YN.txt",
            "flag_format": "yn",
        },
    }

    # Process all subsets
    all_results = []
    all_failures = []

    for subset_name, paths in subsets.items():
        results, failures = process_subset(
            subset_name,
            paths["heavy_dna"],
            paths["light_dna"],
            paths["flags"],
            paths["flag_format"],
        )
        all_results.extend(results)
        all_failures.extend(failures)

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Print statistics
    print_dataset_stats(df)

    # Save output
    output_path = BOUGHTER_PROCESSED_DIR / "boughter.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"\n✓ Output saved to: {output_path}")

    # Save failure log if any
    if all_failures:
        failure_log = BOUGHTER_RAW_DIR / "translation_failures.log"
        failure_log.write_text("\n".join(all_failures))
        logger.info(f"\u2713 Failure log saved to: {failure_log}")

    logger.info("\n" + "=" * 70)
    logger.info("Stage 1 Complete - Ready for Stage 2 (ANARCI annotation)")
    logger.info("=" * 70)
    logger.info("Boughter Stage 1: DNA Translation & Filtering")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
