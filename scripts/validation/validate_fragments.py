#!/usr/bin/env python3
"""
Validate fragment extraction results.

This script validates that fragment CSV files were correctly generated
from the preprocessing pipeline.

Date: 2025-11-01
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def validate_fragment_directory(
    dataset_dir: Path, expected_fragments: int = 16
) -> dict[str, Any]:
    """
    Validate fragment extraction output directory.

    Args:
        dataset_dir: Path to fragment directory (e.g., test_datasets/jain/)
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
    if not dataset_dir.exists():
        results["valid"] = False
        errors_list.append(f"Directory not found: {dataset_dir}")
        return results

    # Check for CSV files
    csv_files = list(dataset_dir.glob("*.csv"))
    if len(csv_files) == 0:
        results["valid"] = False
        errors_list.append("No CSV files found")
        return results

    if len(csv_files) != expected_fragments:
        results["warnings"].append(
            f"Expected {expected_fragments} fragments, found {len(csv_files)}"
        )

    results["stats"]["num_files"] = len(csv_files)

    # Validate each CSV file
    required_columns = {"id", "sequence", "label", "source"}
    all_row_counts = []

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, comment="#")

            # Check required columns
            missing_cols = required_columns - set(df.columns)
            if missing_cols:
                errors_list.append(f"{csv_file.name}: Missing columns {missing_cols}")
                results["valid"] = False

            # Check for empty sequences
            if "sequence" in df.columns:
                empty_seqs = (df["sequence"].str.len() == 0).sum()
                if empty_seqs > 0:
                    errors_list.append(f"{csv_file.name}: {empty_seqs} empty sequences")
                    results["valid"] = False

            # Check for null values in critical columns
            for col in ["id", "sequence"]:
                if col in df.columns:
                    nulls = df[col].isna().sum()
                    if nulls > 0:
                        errors_list.append(
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
            errors_list.append(f"{csv_file.name}: Failed to read - {e}")
            results["valid"] = False

    # Check if all files have same number of rows
    if all_row_counts:
        unique_counts = set(all_row_counts)
        if len(unique_counts) > 1:
            results["warnings"].append(
                f"Inconsistent row counts: {dict(zip(csv_files, all_row_counts, strict=False))}"
            )

        results["stats"]["row_count"] = all_row_counts[0] if all_row_counts else 0
        results["stats"]["consistent_rows"] = len(unique_counts) == 1

    return results


def validate_label_distribution(csv_path: Path) -> dict[str, float | int]:
    """Validate label distribution matches expected pattern."""
    df = pd.read_csv(csv_path, comment="#")

    total = len(df)
    specific = int((df["label"] == 0).sum())
    non_specific = int((df["label"] == 1).sum())

    stats: dict[str, int | float] = {
        "total": total,
        "specific": specific,
        "non_specific": non_specific,
    }

    stats["specific_pct"] = specific / total * 100
    stats["non_specific_pct"] = non_specific / total * 100

    return stats


def print_validation_report(
    dataset_name: str, dataset_dir: Path, expected_fragments: int = 16
) -> bool:
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

    # Label distribution (from first CSV)
    csv_files = list(dataset_dir.glob("*.csv"))
    if csv_files:
        label_stats = validate_label_distribution(csv_files[0])
        print("\nLabel distribution:")
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

    return bool(results["valid"])


def main() -> int:
    """Validate all processed datasets."""
    datasets = [
        ("jain", Path("test_datasets/jain/fragments"), 16),  # Full antibodies (VH+VL)
        (
            "shehata",
            Path("test_datasets/shehata/fragments"),
            16,
        ),  # Full antibodies (VH+VL)
        ("harvey", Path("test_datasets/harvey/fragments"), 6),  # Nanobodies (VHH only)
        (
            "boughter",
            Path("train_datasets/boughter/annotated"),
            16,
        ),  # Full antibodies (VH+VL)
    ]

    all_valid = True

    for name, path, expected_frags in datasets:
        if path.exists():
            valid = print_validation_report(name, path, expected_frags)
            all_valid = all_valid and valid
            print()
        else:
            print(f"⚠ Skipping {name} - directory not found: {path}\n")

    return 0 if all_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
