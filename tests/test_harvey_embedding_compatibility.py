#!/usr/bin/env python3
"""
Harvey Dataset ESM Embedding Compatibility Test

Tests that cleaned Harvey fragment sequences are compatible with ESM-1v
embedding pipeline without requiring model download.

P0 Blocker Fix Verification:
- Ensures no gap characters ('-') in any fragment CSV
- Validates annotation.sequence_aa (gap-free) was used instead of
  annotation.sequence_alignment_aa (with gaps)

Test Coverage:
1. Gap character detection (P0 blocker regression check)
2. Amino acid validation (ESM-1v compatible characters only)
3. Previously affected sequences spot-check
4. Model validation logic simulation
5. Data integrity verification

Date: 2025-11-02
Issue: #4 - Harvey dataset preprocessing
P0 Fix: preprocessing/process_harvey.py:48
"""

import sys
from pathlib import Path

import pandas as pd

# Valid amino acids for ESM-1v model (from model.py:86)
VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYX")


def test_gap_characters():
    """
    Test 1: Verify no gap characters in any fragment CSV.

    P0 BLOCKER: ESM-1v crashes on sequences containing '-' gap characters.
    This test ensures the fix (sequence_aa vs sequence_alignment_aa) is applied.
    """
    print("\n" + "=" * 70)
    print("TEST 1: Gap Character Detection (P0 Blocker Check)")
    print("=" * 70)

    harvey_dir = Path("test_datasets/harvey/fragments")
    fragment_files = [
        "VHH_only_harvey.csv",
        "H-CDR1_harvey.csv",
        "H-CDR2_harvey.csv",
        "H-CDR3_harvey.csv",
        "H-CDRs_harvey.csv",
        "H-FWRs_harvey.csv",
    ]

    all_clean = True
    total_sequences = 0

    for file_name in fragment_files:
        file_path = harvey_dir / file_name
        if not file_path.exists():
            print(f"  ✗ MISSING: {file_name}")
            all_clean = False
            continue

        df = pd.read_csv(file_path)
        gap_count = df["sequence"].str.contains("-", na=False).sum()
        total_sequences = len(df)

        if gap_count > 0:
            print(f"  ✗ {file_name}: {gap_count} sequences with gaps")
            all_clean = False
        else:
            print(f"  ✓ {file_name}: gap-free ({len(df)} sequences)")

    if all_clean:
        print("\n  ✓ PASS: All Harvey fragments are gap-free")
        print(f"  ✓ Total sequences validated: {total_sequences}")
        return True
    else:
        print("\n  ✗ FAIL: Gap characters detected - P0 blocker still present!")
        return False


def test_amino_acid_validation():
    """
    Test 2: Verify all sequences contain only valid amino acids.

    ESM-1v model requires sequences with only standard amino acids.
    Invalid characters will cause validation errors.
    """
    print("\n" + "=" * 70)
    print("TEST 2: Amino Acid Validation")
    print("=" * 70)
    print(f"  Valid amino acids: {sorted(VALID_AMINO_ACIDS)}")

    harvey_dir = Path("test_datasets/harvey/fragments")

    # Check critical files for VHH-based model
    test_files = [
        "VHH_only_harvey.csv",
        "H-CDRs_harvey.csv",
        "H-CDR3_harvey.csv",
    ]

    all_valid = True
    total_sequences = 0

    for file_name in test_files:
        file_path = harvey_dir / file_name
        if not file_path.exists():
            print(f"  ✗ MISSING: {file_name}")
            all_valid = False
            continue

        df = pd.read_csv(file_path)

        # Check each sequence for invalid characters
        invalid_count = 0
        for idx, seq in enumerate(df["sequence"]):
            if not set(seq).issubset(VALID_AMINO_ACIDS):
                invalid_chars = set(seq) - VALID_AMINO_ACIDS
                if invalid_count == 0:
                    print(f"  ✗ {file_name}: Invalid characters found: {invalid_chars}")
                invalid_count += 1

        if invalid_count > 0:
            print(f"    {invalid_count} sequences with invalid amino acids")
            all_valid = False
        else:
            print(f"  ✓ {file_name}: all sequences valid ({len(df)} checked)")
            total_sequences += len(df)

    if all_valid:
        print("\n  ✓ PASS: All sequences contain only valid amino acids")
        print(f"  ✓ Total sequences validated: {total_sequences}")
        return True
    else:
        print("\n  ✗ FAIL: Invalid amino acids detected")
        return False


def test_previously_affected_sequences():
    """
    Test 3: Spot-check sequences that previously had gaps.

    Before fix: VHH_only_harvey.csv had 12,116 sequences with gaps (8.6%)
    After fix: Should be 0
    """
    print("\n" + "=" * 70)
    print("TEST 3: Previously Affected Sequences (Spot Check)")
    print("=" * 70)

    vhh_file = Path("test_datasets/harvey/fragments/VHH_only_harvey.csv")

    if not vhh_file.exists():
        print(f"  ✗ FAIL: {vhh_file} not found")
        return False

    df = pd.read_csv(vhh_file)

    # Sample check: First 5 sequences
    print("\n  Checking sample sequences from VHH_only_harvey.csv:")
    all_clean = True

    for idx in range(min(5, len(df))):
        seq = df.iloc[idx]["sequence"]
        seq_id = df.iloc[idx]["id"]
        has_gaps = "-" in seq

        if has_gaps:
            print(f"    ✗ {seq_id}: contains gaps")
            all_clean = False
        else:
            print(f"    ✓ {seq_id}: gap-free (length {len(seq)} aa)")

    # Check total gap count
    total_gaps = df["sequence"].str.contains("-", na=False).sum()

    print(f"\n  Total sequences: {len(df)}")
    print(f"  Sequences with gaps: {total_gaps}")
    print("  Before P0 fix: 12,116 sequences had gaps (8.6%)")

    if total_gaps == 0 and all_clean:
        print("\n  ✓ PASS: P0 fix successfully eliminated all gaps")
        return True
    else:
        print("\n  ✗ FAIL: Gaps still present after fix")
        return False


def test_model_validation_logic():
    """
    Test 4: Simulate model.py validation logic.

    Replicates the exact validation check from model.py:86-90 to ensure
    compatibility with ESM-1v embedding generation.
    """
    print("\n" + "=" * 70)
    print("TEST 4: ESM Model Validation Simulation")
    print("=" * 70)
    print("  Simulating model.py:86-90 validation logic...")

    vhh_file = Path("test_datasets/harvey/fragments/VHH_only_harvey.csv")

    if not vhh_file.exists():
        print(f"  ✗ FAIL: {vhh_file} not found")
        return False

    df = pd.read_csv(vhh_file)

    # Simulate model.py validation (lines 86-90)
    invalid_sequences = []
    for idx, row in df.iterrows():
        sequence = row["sequence"]

        # Check for invalid amino acids (same logic as model.py)
        if not set(sequence).issubset(VALID_AMINO_ACIDS):
            invalid_sequences.append((row["id"], sequence))

    if len(invalid_sequences) == 0:
        print(f"  ✓ All {len(df)} sequences passed model validation")
        print("  ✓ PASS: Dataset is ESM-1v compatible")
        return True
    else:
        print(f"  ✗ {len(invalid_sequences)} sequences failed validation")
        print(f"  ✗ First failed sequence: {invalid_sequences[0][0]}")
        print("  ✗ FAIL: Dataset NOT compatible with ESM-1v")
        return False


def test_data_integrity():
    """
    Test 5: Verify data integrity after regeneration.

    Ensures all expected files exist, have correct row counts, and
    preserve label distribution.
    """
    print("\n" + "=" * 70)
    print("TEST 5: Data Integrity Verification")
    print("=" * 70)

    harvey_dir = Path("test_datasets/harvey/fragments")
    expected_files = [
        "VHH_only_harvey.csv",
        "H-CDR1_harvey.csv",
        "H-CDR2_harvey.csv",
        "H-CDR3_harvey.csv",
        "H-CDRs_harvey.csv",
        "H-FWRs_harvey.csv",
    ]

    all_valid = True
    expected_rows = None

    print(f"\n  Checking {len(expected_files)} fragment files:")

    for file_name in expected_files:
        file_path = harvey_dir / file_name

        if not file_path.exists():
            print(f"    ✗ MISSING: {file_name}")
            all_valid = False
            continue

        df = pd.read_csv(file_path)

        # All files should have same row count
        if expected_rows is None:
            expected_rows = len(df)

        if len(df) != expected_rows:
            print(
                f"    ✗ {file_name}: row count mismatch ({len(df)} vs {expected_rows})"
            )
            all_valid = False
        else:
            print(f"    ✓ {file_name}: {len(df)} rows")

    # Check label distribution in VHH file
    vhh_file = harvey_dir / "VHH_only_harvey.csv"
    if vhh_file.exists():
        df = pd.read_csv(vhh_file)
        label_counts = df["label"].value_counts().sort_index()

        print("\n  Label distribution:")
        print(
            f"    Low polyreactivity (0): {label_counts.get(0, 0)} ({label_counts.get(0, 0)/len(df)*100:.1f}%)"
        )
        print(
            f"    High polyreactivity (1): {label_counts.get(1, 0)} ({label_counts.get(1, 0)/len(df)*100:.1f}%)"
        )

        # Check if balanced (should be ~50/50)
        balance = label_counts.get(1, 0) / len(df)
        if 0.48 <= balance <= 0.52:
            print("    ✓ Dataset is balanced")
        else:
            print("    ⚠ Dataset may be imbalanced")

    if all_valid and expected_rows is not None:
        print(
            f"\n  ✓ PASS: All {len(expected_files)} files present with {expected_rows} rows"
        )
        return True
    else:
        print("\n  ✗ FAIL: Data integrity issues detected")
        return False


def main():
    """Run all tests and report results."""
    print("\n" + "=" * 70)
    print("Harvey Dataset - ESM-1v Embedding Compatibility Test Suite")
    print("=" * 70)
    print("P0 Blocker Fix: annotation.sequence_aa (gap-free)")
    print("Previous issue: annotation.sequence_alignment_aa (with gaps)")
    print("Fix location: preprocessing/process_harvey.py:48")

    # Run all tests
    tests = [
        ("Gap Character Detection", test_gap_characters),
        ("Amino Acid Validation", test_amino_acid_validation),
        ("Previously Affected Sequences", test_previously_affected_sequences),
        ("ESM Model Validation", test_model_validation_logic),
        ("Data Integrity", test_data_integrity),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n  ✗ EXCEPTION: {test_name} - {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")

    print(f"\n  Total: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n  ✓ ALL TESTS PASSED - Harvey dataset is ESM-1v compatible!")
        print("  ✓ P0 blocker successfully resolved")
        return 0
    else:
        print("\n  ✗ SOME TESTS FAILED - Review errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
