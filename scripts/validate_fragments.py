#!/usr/bin/env python3
"""
Validate fragment extraction outputs.

Checks fragment directories for consistent row counts, required columns,
and balanced labels.

Date: 2025-11-01
"""

import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

REQUIRED_COLUMNS = {"id", "sequence", "label", "source"}
OPTIONAL_COLUMNS = {"sequence_length"}


def validate_fragment_directory(dataset_dir: Path, expected_fragments: int) -> Dict:
    """Validate fragment CSV files in a dataset directory."""
    results: Dict[str, object] = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {},
    }

    if not dataset_dir.exists():
        results["valid"] = False
        results["errors"].append(f"Directory not found: {dataset_dir}")
        return results

    csv_files = sorted(dataset_dir.glob("*.csv"))
    if not csv_files:
        results["valid"] = False
        results["errors"].append("No CSV files found")
        return results

    if len(csv_files) != expected_fragments:
        results["warnings"].append(
            f"Expected {expected_fragments} fragments, found {len(csv_files)}"
        )

    results["stats"]["num_files"] = len(csv_files)

    row_counts: List[int] = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            results["errors"].append(
                f"{csv_file.name}: missing columns {sorted(missing)}"
            )
            results["valid"] = False

        for col in REQUIRED_COLUMNS:
            if col in df.columns:
                nulls = df[col].isna().sum()
                if nulls:
                    results["errors"].append(
                        f"{csv_file.name}: {nulls} null values in '{col}'"
                    )
                    results["valid"] = False

        if "sequence" in df.columns:
            empty = (df["sequence"].str.len() == 0).sum()
            if empty:
                results["errors"].append(f"{csv_file.name}: {empty} empty sequences")
                results["valid"] = False

        row_counts.append(len(df))

    if row_counts:
        unique_counts = set(row_counts)
        results["stats"]["row_count"] = row_counts[0]
        results["stats"]["consistent_rows"] = len(unique_counts) == 1
        if len(unique_counts) > 1:
            results["warnings"].append(
                f"Inconsistent row counts across files: {sorted(unique_counts)}"
            )

    return results


def label_stats(csv_path: Path) -> Dict[str, float]:
    """Return basic label distribution statistics."""
    df = pd.read_csv(csv_path)
    total = len(df)
    specific = int((df["label"] == 0).sum())
    nonspecific = int((df["label"] == 1).sum())
    return {
        "total": total,
        "specific": specific,
        "specific_pct": specific / total * 100 if total else 0.0,
        "nonspecific": nonspecific,
        "nonspecific_pct": nonspecific / total * 100 if total else 0.0,
    }


def print_report(dataset_name: str, dataset_dir: Path, expected_fragments: int) -> bool:
    """Print validation summary for a dataset directory."""
    print("=" * 62)
    print(f"{dataset_name.upper()} Dataset Validation")
    print("=" * 62)

    results = validate_fragment_directory(dataset_dir, expected_fragments)
    stats = results["stats"]

    print(f"\nFragment files: {stats.get('num_files', 0)}")
    print(f"Antibodies per file: {stats.get('row_count', 0)}")
    consistent = stats.get("consistent_rows", False)
    print(f"Consistent row counts: {'✓ YES' if consistent else '✗ NO'}")

    if results["errors"]:
        print(f"\n✗ ERRORS ({len(results['errors'])}):")
        for err in results["errors"]:
            print(f"  - {err}")

    if results["warnings"]:
        print(f"\n⚠ WARNINGS ({len(results['warnings'])}):")
        for warn in results["warnings"]:
            print(f"  - {warn}")

    first_csv = next((p for p in sorted(dataset_dir.glob("*.csv"))), None)
    if first_csv:
        stats = label_stats(first_csv)
        print("\nLabel distribution:")
        print(f"  Specific (0): {stats['specific']} ({stats['specific_pct']:.1f}%)")
        print(
            f"  Non-specific (1): {stats['nonspecific']} ({stats['nonspecific_pct']:.1f}%)"
        )

    print("\n" + "=" * 62)
    print("✓ VALIDATION PASSED" if results["valid"] else "✗ VALIDATION FAILED")
    print("=" * 62)
    return bool(results["valid"])


def main() -> None:
    datasets = [
        ("harvey", Path("test_datasets/harvey"), 6),
    ]

    all_valid = True
    for name, path, expected in datasets:
        if path.exists():
            all_valid &= print_report(name, path, expected)
            print()
        else:
            print(f"⚠ Skipping {name} – directory not found: {path}\n")

    sys.exit(0 if all_valid else 1)


if __name__ == "__main__":
    main()
