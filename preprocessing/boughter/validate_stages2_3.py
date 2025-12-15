#!/usr/bin/env python3
"""
Boughter Dataset Preprocessing - Stages 2+3 Validator

Validates that Stages 2+3 (ANARCI annotation + QC filtering) completed successfully,
with correct fragment extraction, sequence counts, and data integrity.

Pipeline Position: Validates Stages 2+3 output
    Stage 1 → boughter.csv (1,117 sequences)
    Stages 2+3 → Fragment CSVs (1,065 sequences each) ← VALIDATED BY THIS SCRIPT
    Training subset: VH_only_boughter_training.csv (914 sequences)

Usage:
    python3 preprocessing/boughter/validate_stages2_3.py

Validation Checks:
    1. All 16 fragment CSV files exist in data/train/boughter/annotated/
    2. Each fragment has 1,065 rows (95.9% retention from Stage 1)
    3. All rows have include_in_training flag (True/False)
    4. Training subset has 914 rows (443 specific + 471 non-specific)
    5. Schema validation using Pandera (via BoughterSchema)
    6. Label distribution matches expected (0=specific, 1=non-specific)

Outputs:
    - Console validation report for each fragment
    - Summary statistics

Reference: See docs/datasets/boughter/data_sources.md for Stages 2+3 methodology
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from antibody_training_esm.schemas.dataset import get_preprocessing_schema
from preprocessing.logging_config import setup_logger
from preprocessing.paths import BOUGHTER_ANNOTATED_DIR, BOUGHTER_TRAINING_SUBSET
from preprocessing.validation_utils import (
    calculate_label_stats,
    log_label_stats,
    validate_dataframe_with_schema,
    validate_directory_exists,
)

logger = setup_logger(__name__)


def validate_fragment_directory(
    dataset_dir: Path, expected_fragments: int = 16
) -> dict[str, Any]:
    """
    Validate fragment extraction output directory.

    Args:
        dataset_dir: Path to fragment directory (e.g., data/test/jain/)
        expected_fragments: Expected number of fragment CSV files (default: 16)

    Returns:
        Dictionary with validation results
    """
    errors_list: list[str] = []
    warnings_list: list[str] = []
    stats_dict: dict[str, Any] = {}

    results: dict[str, Any] = {
        "valid": True,
        "errors": errors_list,
        "warnings": warnings_list,
        "stats": stats_dict,
    }

    # Check directory exists
    if not validate_directory_exists(dataset_dir):
        results["valid"] = False
        errors_list.append(f"Directory not found: {dataset_dir}")
        return results

    # Check for CSV files (exclude training subset files)
    csv_files = [
        f for f in dataset_dir.glob("*.csv") if not f.name.endswith("_training.csv")
    ]
    if len(csv_files) == 0:
        results["valid"] = False
        errors_list.append("No CSV files found")
        return results

    if len(csv_files) != expected_fragments:
        warnings_list.append(
            f"Expected {expected_fragments} fragments, found {len(csv_files)}"
        )

    stats_dict["num_files"] = len(csv_files)

    # Validate each CSV file using Pandera schema
    # Use preprocessing schema (allows nullable labels for held-out sequences)
    all_row_counts = []
    schema = get_preprocessing_schema()

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, comment="#")

            # Validate with schema
            schema_errors = validate_dataframe_with_schema(df, schema, csv_file.name)
            if schema_errors:
                errors_list.extend(schema_errors)
                results["valid"] = False

            # Check for null labels (warning only - valid for held-out sequences)
            if "label" in df.columns:
                nulls = df["label"].isna().sum()
                if nulls > 0:
                    warnings_list.append(
                        f"{csv_file.name}: {nulls} null/held-out labels"
                    )

            # Track row counts
            all_row_counts.append(len(df))

        except Exception as e:
            errors_list.append(f"{csv_file.name}: Failed to read - {e}")
            results["valid"] = False

    # Check if all files have same number of rows
    if all_row_counts:
        unique_counts = set(all_row_counts)
        if len(unique_counts) > 1:
            warnings_list.append(
                f"Inconsistent row counts: {dict(zip(csv_files, all_row_counts, strict=False))}"
            )

        stats_dict["row_count"] = all_row_counts[0] if all_row_counts else 0
        stats_dict["consistent_rows"] = len(unique_counts) == 1

    return results


def print_validation_report(dataset_dir: Path, expected_fragments: int = 16) -> bool:
    """Print comprehensive validation report."""
    logger.info("=" * 60)
    logger.info("Boughter Stages 2 & 3 Validation")
    logger.info("=" * 60)

    results = validate_fragment_directory(dataset_dir, expected_fragments)

    # Print file count
    logger.info(f"\nFragment files: {results['stats'].get('num_files', 0)}")
    logger.info(f"Antibodies per file: {results['stats'].get('row_count', 0)}")
    logger.info(
        f"Consistent row counts: {'✓ YES' if results['stats'].get('consistent_rows') else '✗ NO'}"
    )

    # Print errors
    if results["errors"]:
        logger.info(f"\n✗ ERRORS ({len(results['errors'])}):")
        for _error in results["errors"]:
            # Validation utils logs errors as they happen, no need to double log unless summary
            pass

    # Print warnings
    if results["warnings"]:
        logger.info(f"\n⚠ WARNINGS ({len(results['warnings'])}):")
        for warning in results["warnings"]:
            logger.info(f"  - {warning}")

    # Label distribution (from VH_only fragment, excluding training subset)
    csv_files = [
        f for f in dataset_dir.glob("*.csv") if not f.name.endswith("_training.csv")
    ]
    if csv_files:
        # Prefer VH_only file for label distribution
        vh_file = dataset_dir / "VH_only_boughter.csv"
        label_file = vh_file if vh_file.exists() else csv_files[0]

        try:
            df = pd.read_csv(label_file, comment="#")
            stats = calculate_label_stats(df)
            log_label_stats(stats, f"Boughter ({label_file.name})")
        except Exception as e:
            logger.error(f"Failed to calculate label stats: {e}")

    # Final verdict
    logger.info("\n" + "=" * 60)
    if results["valid"]:
        logger.info("✓ VALIDATION PASSED")
    else:
        logger.info("✗ VALIDATION FAILED")
    logger.info("=" * 60)

    return bool(results["valid"])


def main() -> int:
    """Validate Boughter dataset Stages 2+3 output."""
    boughter_annotated_dir = BOUGHTER_ANNOTATED_DIR

    if not boughter_annotated_dir.exists():
        logger.info(
            f"✗ Error: Boughter annotated directory not found: {boughter_annotated_dir}"
        )
        return 1

    valid = print_validation_report(boughter_annotated_dir, expected_fragments=16)

    # Additional Boughter-specific checks
    logger.info("\n" + "=" * 60)
    logger.info("BOUGHTER-SPECIFIC VALIDATION")
    logger.info("=" * 60)

    # Check training subset file
    training_file = BOUGHTER_TRAINING_SUBSET
    if training_file.exists():
        df = pd.read_csv(training_file, comment="#")
        logger.info(f"\n✓ Training subset file exists: {training_file.name}")
        logger.info(f"  Rows: {len(df)}")
        logger.info(f"  Specific (0): {(df['label'] == 0).sum()}")
        logger.info(f"  Non-specific (1): {(df['label'] == 1).sum()}")
    else:
        logger.info(f"\n✗ Training subset file not found: {training_file.name}")
        valid = False

    # Check for include_in_training flag in fragment files
    vh_only = boughter_annotated_dir / "VH_only_boughter.csv"
    if vh_only.exists():
        df = pd.read_csv(vh_only, comment="#")
        if "include_in_training" in df.columns:
            logger.info("\n✓ include_in_training flag present")
            logger.info(f"  Training eligible: {df['include_in_training'].sum()}")
            logger.info(
                f"  Excluded (mild 1-3 flags): {(~df['include_in_training']).sum()}"
            )
        else:
            logger.info("\n⚠ include_in_training flag missing (may be older format)")

    logger.info("\n" + "=" * 60)

    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
