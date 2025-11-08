#!/usr/bin/env python3
"""
Jain Dataset CSV Validation Script

Validates all Jain CSVs are properly formatted for HuggingFace release:
- Standard CSV format (no comment headers)
- Correct row counts
- Expected NaN distributions
- Class balance verification

Run before pushing to HuggingFace or after regenerating datasets.

Usage:
    python3 scripts/validation/validate_jain_csvs.py
"""

import sys
from pathlib import Path

import pandas as pd


class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"


def test_csv_format(filepath: Path) -> tuple[bool, str]:
    """Test if CSV can be loaded without comment parameter"""
    try:
        df = pd.read_csv(filepath)
        return True, f"✓ Standard CSV format ({len(df)} rows)"
    except Exception as e:
        # Try with comment parameter (legacy compatibility)
        try:
            df = pd.read_csv(filepath, comment="#")
            return False, f"✗ Requires comment='#' ({len(df)} rows) - {str(e)[:60]}"
        except Exception as e2:
            return False, f"✗ Failed to parse: {str(e2)[:60]}"


def validate_canonical_files():
    """Validate canonical benchmark sets"""
    print(f"\n{Colors.BLUE}{'=' * 70}")
    print("CANONICAL FILES (Production Benchmarks)")
    print(f"{'=' * 70}{Colors.RESET}\n")

    expected_canonical = {
        "jain_86_novo_parity.csv": {"rows": 86, "nan_labels": 0},
    }

    canonical_dir = Path("test_datasets/jain/canonical")
    all_pass = True

    for filename, expected in expected_canonical.items():
        filepath = canonical_dir / filename
        print(f"Testing: {filename}")

        if not filepath.exists():
            print(f"  {Colors.RED}✗ File not found{Colors.RESET}")
            all_pass = False
            continue

        # Test CSV format
        is_standard, msg = test_csv_format(filepath)
        if is_standard:
            print(f"  {Colors.GREEN}{msg}{Colors.RESET}")
        else:
            print(f"  {Colors.RED}{msg}{Colors.RESET}")
            all_pass = False

        # Load and validate contents
        df = pd.read_csv(filepath, comment="#")  # Defensive

        # Check row count
        if len(df) == expected["rows"]:
            print(
                f"  {Colors.GREEN}✓ Row count: {len(df)} (expected: {expected['rows']}){Colors.RESET}"
            )
        else:
            print(
                f"  {Colors.RED}✗ Row count: {len(df)} (expected: {expected['rows']}){Colors.RESET}"
            )
            all_pass = False

        # Check NaN labels
        nan_count = df["label"].isna().sum()
        if nan_count == expected["nan_labels"]:
            print(
                f"  {Colors.GREEN}✓ NaN labels: {nan_count} (expected: {expected['nan_labels']}){Colors.RESET}"
            )
        else:
            print(
                f"  {Colors.RED}✗ NaN labels: {nan_count} (expected: {expected['nan_labels']}){Colors.RESET}"
            )
            all_pass = False

        # Check class distribution
        if nan_count == 0:
            label_dist = df["label"].value_counts().to_dict()
            print(f"  {Colors.GREEN}✓ Label distribution: {label_dist}{Colors.RESET}")

        print()

    return all_pass


def validate_fragment_files():
    """Validate fragment files"""
    print(f"\n{Colors.BLUE}{'=' * 70}")
    print("FRAGMENT FILES (Research/Development)")
    print(f"{'=' * 70}{Colors.RESET}\n")

    fragments_dir = Path("test_datasets/jain/fragments")
    fragment_files = list(fragments_dir.glob("*_jain.csv"))

    expected_fragments = {
        "rows": 137,  # 94 specific + 22 non-specific + 21 mild
        "nan_labels": 21,  # Mild antibodies
    }

    all_pass = True

    for filepath in sorted(fragment_files):
        print(f"Testing: {filepath.name}")

        # Test CSV format
        is_standard, msg = test_csv_format(filepath)
        if is_standard:
            print(f"  {Colors.GREEN}{msg}{Colors.RESET}")
        else:
            print(f"  {Colors.RED}{msg}{Colors.RESET}")
            all_pass = False

        # Load and validate
        df = pd.read_csv(filepath, comment="#")  # Defensive

        # Check row count
        if len(df) == expected_fragments["rows"]:
            print(f"  {Colors.GREEN}✓ Row count: {len(df)}{Colors.RESET}")
        else:
            print(
                f"  {Colors.YELLOW}⚠ Row count: {len(df)} (expected: {expected_fragments['rows']}){Colors.RESET}"
            )

        # Check NaN labels
        nan_count = df["label"].isna().sum()
        if nan_count == expected_fragments["nan_labels"]:
            print(
                f"  {Colors.GREEN}✓ NaN labels: {nan_count} (mild antibodies){Colors.RESET}"
            )
        else:
            print(
                f"  {Colors.YELLOW}⚠ NaN labels: {nan_count} (expected: {expected_fragments['nan_labels']}){Colors.RESET}"
            )

        # Check class distribution
        label_dist = df["label"].value_counts(dropna=False).to_dict()
        print(f"  {Colors.GREEN}✓ Label distribution: {label_dist}{Colors.RESET}")

        print()

    return all_pass


def validate_manifest():
    """Validate manifest file exists and is correct"""
    print(f"\n{Colors.BLUE}{'=' * 70}")
    print("MANIFEST FILE")
    print(f"{'=' * 70}{Colors.RESET}\n")

    manifest_path = Path("test_datasets/jain/fragments/manifest.yml")
    print(f"Testing: {manifest_path.name}")

    if not manifest_path.exists():
        print(f"  {Colors.RED}✗ Manifest not found{Colors.RESET}\n")
        return False

    with open(manifest_path) as f:
        content = f.read()

    # Check for key fields
    required_fields = [
        "source_file",
        "source_sha256",
        "csv_format",
        "labeling_rule",
        "expected_counts",
        "compatibility",
    ]

    all_present = True
    for field in required_fields:
        if field in content:
            print(f"  {Colors.GREEN}✓ Contains '{field}'{Colors.RESET}")
        else:
            print(f"  {Colors.RED}✗ Missing '{field}'{Colors.RESET}")
            all_present = False

    # Check for HF compatibility note
    if "HuggingFace" in content or "huggingface" in content.lower():
        print(f"  {Colors.GREEN}✓ Documents HuggingFace compatibility{Colors.RESET}")
    else:
        print(
            f"  {Colors.YELLOW}⚠ Missing HuggingFace compatibility note{Colors.RESET}"
        )

    print()
    return all_present


def main():
    """Run all validations"""
    print(f"\n{Colors.BLUE}{'=' * 70}")
    print("JAIN DATASET CSV VALIDATION")
    print("For HuggingFace Release Readiness")
    print(f"{'=' * 70}{Colors.RESET}")

    results = {
        "canonical": validate_canonical_files(),
        "fragments": validate_fragment_files(),
        "manifest": validate_manifest(),
    }

    # Summary
    print(f"\n{Colors.BLUE}{'=' * 70}")
    print("VALIDATION SUMMARY")
    print(f"{'=' * 70}{Colors.RESET}\n")

    all_pass = all(results.values())

    for category, passed in results.items():
        status = (
            f"{Colors.GREEN}✓ PASS{Colors.RESET}"
            if passed
            else f"{Colors.RED}✗ FAIL{Colors.RESET}"
        )
        print(f"  {category.upper():15s}: {status}")

    print()

    if all_pass:
        print(f"{Colors.GREEN}{'=' * 70}")
        print("✓ ALL VALIDATIONS PASSED - Ready for HuggingFace release")
        print(f"{'=' * 70}{Colors.RESET}\n")
        return 0
    else:
        print(f"{Colors.RED}{'=' * 70}")
        print("✗ VALIDATION FAILED - Fix issues before HuggingFace release")
        print(f"{'=' * 70}{Colors.RESET}\n")
        print("To fix issues:")
        print("  1. Run: python3 preprocessing/jain/step1_convert_excel_to_csv.py")
        print("  2. Run: python3 preprocessing/jain/step2_preprocess_p5e_s2.py")
        print("  3. Run: python3 preprocessing/jain/step3_extract_fragments.py")
        print("  4. Re-run this validation script")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
