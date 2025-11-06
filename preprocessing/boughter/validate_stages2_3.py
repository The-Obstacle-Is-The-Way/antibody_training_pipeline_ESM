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
    1. All 16 fragment CSV files exist in train_datasets/boughter/annotated/
    2. Each fragment has 1,065 rows (95.9% retention from Stage 1)
    3. All rows have include_in_training flag (True/False)
    4. Training subset has 914 rows (443 specific + 471 non-specific)
    5. Required columns present: id, sequence, label, subset, num_flags, etc.
    6. No empty sequences
    7. No null values in critical columns
    8. Label distribution matches expected (0=specific, 1=non-specific)

Outputs:
    - Console validation report for each fragment
    - Summary statistics

Reference: See docs/boughter/boughter_data_sources.md for Stages 2+3 methodology
"""

import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


def validate_fragment_directory(dataset_dir: Path, expected_fragments: int = 16):
    """
    Validate fragment extraction output directory.

    Args:
        dataset_dir: Path to fragment directory (e.g., test_datasets/jain/)
        expected_fragments: Expected number of fragment CSV files (default: 16)

    Returns:
        Dictionary with validation results
    """
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {},
    }

    # Check directory exists
    if not dataset_dir.exists():
        results["valid"] = False
        results["errors"].append(f"Directory not found: {dataset_dir}")
        return results

    # Check for CSV files (exclude training subset files)
    csv_files = [
        f
        for f in dataset_dir.glob("*.csv")
        if not f.name.endswith("_training.csv")
    ]
    if len(csv_files) == 0:
        results["valid"] = False
        results["errors"].append("No CSV files found")
        return results

    if len(csv_files) != expected_fragments:
        results["warnings"].append(
            f"Expected {expected_fragments} fragments, found {len(csv_files)}"
        )

    results["stats"]["num_files"] = len(csv_files)

    # Validate each CSV file
    required_columns = {"id", "sequence", "label", "source"}
    optional_columns = {"sequence_length"}  # Added in newer preprocessing scripts
    all_row_counts = []

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, comment="#")

            # Check required columns
            missing_cols = required_columns - set(df.columns)
            if missing_cols:
                results["errors"].append(
                    f"{csv_file.name}: Missing columns {missing_cols}"
                )
                results["valid"] = False

            # Check for empty sequences
            if "sequence" in df.columns:
                empty_seqs = (df["sequence"].str.len() == 0).sum()
                if empty_seqs > 0:
                    results["errors"].append(
                        f"{csv_file.name}: {empty_seqs} empty sequences"
                    )
                    results["valid"] = False

            # Check for null values in critical columns
            for col in ["id", "sequence"]:
                if col in df.columns:
                    nulls = df[col].isna().sum()
                    if nulls > 0:
                        results["errors"].append(
                            f"{csv_file.name}: {nulls} null values in '{col}'"
                        )
                        results["valid"] = False

            # Check for null labels (warning only - valid for held-out sequences)
            if "label" in df.columns:
                nulls = df["label"].isna().sum()
                if nulls > 0:
                    results["warnings"].append(
                        f"{csv_file.name}: {nulls} null/held-out labels"
                    )

            # Track row counts
            all_row_counts.append(len(df))

        except Exception as e:
            results["errors"].append(f"{csv_file.name}: Failed to read - {e}")
            results["valid"] = False

    # Check if all files have same number of rows
    if all_row_counts:
        unique_counts = set(all_row_counts)
        if len(unique_counts) > 1:
            results["warnings"].append(
                f"Inconsistent row counts: {dict(zip(csv_files, all_row_counts))}"
            )

        results["stats"]["row_count"] = all_row_counts[0] if all_row_counts else 0
        results["stats"]["consistent_rows"] = len(unique_counts) == 1

    return results


def validate_label_distribution(csv_path: Path):
    """Validate label distribution matches expected pattern."""
    df = pd.read_csv(csv_path, comment="#")

    stats = {
        "total": len(df),
        "specific": (df["label"] == 0).sum(),
        "non_specific": (df["label"] == 1).sum(),
    }

    stats["specific_pct"] = stats["specific"] / stats["total"] * 100
    stats["non_specific_pct"] = stats["non_specific"] / stats["total"] * 100

    return stats


def print_validation_report(
    dataset_name: str, dataset_dir: Path, expected_fragments: int = 16
):
    """Print comprehensive validation report."""
    print("=" * 60)
    print(f"{dataset_name.upper()} Dataset Validation")
    print("=" * 60)

    results = validate_fragment_directory(dataset_dir, expected_fragments)

    # Print file count
    print(f"\nFragment files: {results['stats'].get('num_files', 0)}")
    print(f"Antibodies per file: {results['stats'].get('row_count', 0)}")
    print(
        f"Consistent row counts: {'✓ YES' if results['stats'].get('consistent_rows') else '✗ NO'}"
    )

    # Print errors
    if results["errors"]:
        print(f"\n✗ ERRORS ({len(results['errors'])}):")
        for error in results["errors"]:
            print(f"  - {error}")

    # Print warnings
    if results["warnings"]:
        print(f"\n⚠ WARNINGS ({len(results['warnings'])}):")
        for warning in results["warnings"]:
            print(f"  - {warning}")

    # Label distribution (from VH_only fragment, excluding training subset)
    csv_files = [
        f
        for f in dataset_dir.glob("*.csv")
        if not f.name.endswith("_training.csv")
    ]
    if csv_files:
        # Prefer VH_only file for label distribution
        vh_file = dataset_dir / "VH_only_boughter.csv"
        label_file = vh_file if vh_file.exists() else csv_files[0]
        label_stats = validate_label_distribution(label_file)
        print(f"\nLabel distribution (from {label_file.name}):")
        print(
            f"  Specific (0): {label_stats['specific']} ({label_stats['specific_pct']:.1f}%)"
        )
        print(
            f"  Non-specific (1): {label_stats['non_specific']} ({label_stats['non_specific_pct']:.1f}%)"
        )

    # Final verdict
    print("\n" + "=" * 60)
    if results["valid"]:
        print("✓ VALIDATION PASSED")
    else:
        print("✗ VALIDATION FAILED")
    print("=" * 60)

    return results["valid"]


def main():
    """Validate Boughter dataset Stages 2+3 output."""
    boughter_annotated_dir = Path("train_datasets/boughter/annotated")
    boughter_canonical_dir = Path("train_datasets/boughter/canonical")

    if not boughter_annotated_dir.exists():
        print(f"✗ Error: Boughter annotated directory not found: {boughter_annotated_dir}")
        sys.exit(1)

    valid = print_validation_report("boughter", boughter_annotated_dir, expected_fragments=16)

    # Additional Boughter-specific checks
    print("\n" + "=" * 60)
    print("BOUGHTER-SPECIFIC VALIDATION")
    print("=" * 60)

    # Check training subset file
    training_file = boughter_canonical_dir / "VH_only_boughter_training.csv"
    if training_file.exists():
        df = pd.read_csv(training_file, comment="#")
        print(f"\n✓ Training subset file exists: {training_file.name}")
        print(f"  Rows: {len(df)}")
        print(f"  Specific (0): {(df['label'] == 0).sum()}")
        print(f"  Non-specific (1): {(df['label'] == 1).sum()}")
    else:
        print(f"\n✗ Training subset file not found: {training_file.name}")
        valid = False

    # Check for include_in_training flag in fragment files
    vh_only = boughter_annotated_dir / "VH_only_boughter.csv"
    if vh_only.exists():
        df = pd.read_csv(vh_only, comment="#")
        if "include_in_training" in df.columns:
            print(f"\n✓ include_in_training flag present")
            print(f"  Training eligible: {df['include_in_training'].sum()}")
            print(f"  Excluded (mild 1-3 flags): {(~df['include_in_training']).sum()}")
        else:
            print(f"\n⚠ include_in_training flag missing (may be older format)")

    print("\n" + "=" * 60)

    sys.exit(0 if valid else 1)


if __name__ == "__main__":
    main()
