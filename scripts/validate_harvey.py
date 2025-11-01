#!/usr/bin/env python3
"""
Validation script for Harvey dataset to ensure data integrity.

Checks:
1. harvey.csv has exactly 48 nanobodies
2. All 48 IDs from harvey.xlsx PSR sheet are present
3. All 6 fragment files have 48 rows
4. No duplicate IDs
5. All sequences have valid lengths (113-130 aa for nanobodies)
6. PSR scores are in expected range

Usage:
    python3 scripts/validate_harvey.py
"""

from pathlib import Path

import pandas as pd


def validate_harvey_dataset():
    """Run all validation checks on Harvey dataset."""
    print("=" * 60)
    print("Harvey Dataset Validation")
    print("=" * 60)

    passed = True

    # Check 1: Main CSV exists and has 48 rows
    harvey_csv = Path("test_datasets/harvey.csv")
    if not harvey_csv.exists():
        print("❌ harvey.csv not found")
        return False

    df = pd.read_csv(harvey_csv)
    if len(df) != 48:
        print(f"❌ harvey.csv has {len(df)} rows, expected 48")
        passed = False
    else:
        print(f"✓ harvey.csv has 48 nanobodies")

    # Check 2: All IDs from Excel are present
    excel_path = Path("test_datasets/harvey.xlsx")
    if excel_path.exists():
        psr_df = pd.read_excel(excel_path, sheet_name="Supp Figure 3A")
        psr_df = psr_df.iloc[1:].reset_index(drop=True)
        expected_ids = set(psr_df.iloc[:, 0].dropna().tolist())

        if len(expected_ids) != 48:
            print(f"⚠️  Excel has {len(expected_ids)} IDs, expected 48")

        actual_ids = set(df["id"].tolist())
        missing = expected_ids - actual_ids
        extra = actual_ids - expected_ids

        if missing:
            print(f"❌ Missing IDs: {sorted(missing)}")
            passed = False
        else:
            print(f"✓ All 48 IDs from Excel are present")

        if extra:
            print(f"⚠️  Extra IDs not in Excel: {sorted(extra)}")

    # Check 3: No duplicates
    if df["id"].duplicated().any():
        print(f"❌ Duplicate IDs found: {df[df['id'].duplicated()]['id'].tolist()}")
        passed = False
    else:
        print(f"✓ No duplicate IDs")

    # Check 4: Sequence lengths
    min_len = df["sequence_length"].min()
    max_len = df["sequence_length"].max()
    if min_len < 100 or max_len > 150:
        print(
            f"❌ Sequence lengths out of range: {min_len}-{max_len} (expected 113-130)"
        )
        passed = False
    else:
        print(f"✓ Sequence lengths in valid range: {min_len}-{max_len} aa")

    # Check 5: PSR scores
    psr_min = df["psr_score"].min()
    psr_max = df["psr_score"].max()
    if psr_min < 0 or psr_max > 100000:
        print(f"⚠️  PSR scores seem unusual: {psr_min:.1f}-{psr_max:.1f}")
    else:
        print(f"✓ PSR scores in expected range: {psr_min:.1f}-{psr_max:.1f}")

    # Check 6: Fragment files
    fragment_dir = Path("test_datasets/harvey")
    expected_fragments = [
        "VHH_only_harvey.csv",
        "H-CDR1_harvey.csv",
        "H-CDR2_harvey.csv",
        "H-CDR3_harvey.csv",
        "H-CDRs_harvey.csv",
        "H-FWRs_harvey.csv",
    ]

    fragment_ok = True
    for fragment_file in expected_fragments:
        fpath = fragment_dir / fragment_file
        if not fpath.exists():
            print(f"❌ Fragment file missing: {fragment_file}")
            fragment_ok = False
            passed = False
        else:
            frag_df = pd.read_csv(fpath)
            if len(frag_df) != 48:
                print(f"❌ {fragment_file} has {len(frag_df)} rows, expected 48")
                fragment_ok = False
                passed = False

    if fragment_ok:
        print(f"✓ All 6 fragment files have 48 rows")

    # Check 7: Critical IDs that were previously missing
    critical_ids = ["E05'", "F02'", "F07'", "G09'"]
    missing_critical = [cid for cid in critical_ids if cid not in df["id"].tolist()]
    if missing_critical:
        print(f"❌ Critical IDs missing (were fixed in this PR): {missing_critical}")
        passed = False
    else:
        print(f"✓ All 4 previously-missing IDs are present: {critical_ids}")

    print("=" * 60)
    if passed:
        print("✅ ALL VALIDATION CHECKS PASSED")
    else:
        print("❌ VALIDATION FAILED")
    print("=" * 60)

    return passed


if __name__ == "__main__":
    success = validate_harvey_dataset()
    exit(0 if success else 1)
