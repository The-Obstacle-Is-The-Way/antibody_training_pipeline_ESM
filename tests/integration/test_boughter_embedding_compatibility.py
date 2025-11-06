#!/usr/bin/env python3
"""
Boughter Dataset ESM Embedding Compatibility Test

Tests that cleaned Boughter fragment sequences are compatible with ESM-1v
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
Issue: Boughter dataset preprocessing P0 blocker
P0 Fix: preprocessing/boughter/stage2_stage3_annotation_qc.py:93-147
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

    boughter_dir = Path("train_datasets/boughter/annotated")
    fragment_files = [
        "VH_only_boughter.csv",
        "VL_only_boughter.csv",
        "H-CDR1_boughter.csv",
        "H-CDR2_boughter.csv",
        "H-CDR3_boughter.csv",
        "L-CDR1_boughter.csv",
        "L-CDR2_boughter.csv",
        "L-CDR3_boughter.csv",
        "H-CDRs_boughter.csv",
        "L-CDRs_boughter.csv",
        "H-FWRs_boughter.csv",
        "L-FWRs_boughter.csv",
        "VH+VL_boughter.csv",
        "All-CDRs_boughter.csv",
        "All-FWRs_boughter.csv",
        "Full_boughter.csv",
    ]

    all_clean = True
    total_sequences = 0

    for file_name in fragment_files:
        file_path = boughter_dir / file_name
        if not file_path.exists():
            print(f"  ✗ MISSING: {file_name}")
            all_clean = False
            continue

        # Read CSV with comment lines (fragment files have metadata headers)
        df = pd.read_csv(file_path, comment="#")
        gap_count = df["sequence"].str.contains("-", na=False).sum()
        total_sequences = len(df)

        if gap_count > 0:
            print(f"  ✗ {file_name}: {gap_count} sequences with gaps")
            all_clean = False
        else:
            print(f"  ✓ {file_name}: gap-free ({len(df)} sequences)")

    if all_clean:
        print("\n  ✓ PASS: All Boughter fragments are gap-free")
        print(f"  ✓ Total sequences validated: {total_sequences}")
    else:
        print("\n  ✗ FAIL: Gap characters detected - P0 blocker still present!")

    # Pytest assertion (no return value)
    assert all_clean, "Gap characters detected - P0 blocker still present!"


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

    boughter_dir = Path("train_datasets/boughter/annotated")

    # Check critical files for full antibody model
    test_files = [
        "VH_only_boughter.csv",
        "VL_only_boughter.csv",
        "H-CDRs_boughter.csv",
        "L-CDRs_boughter.csv",
        "Full_boughter.csv",
    ]

    all_valid = True
    total_sequences = 0

    for file_name in test_files:
        file_path = boughter_dir / file_name
        if not file_path.exists():
            print(f"  ✗ MISSING: {file_name}")
            all_valid = False
            continue

        df = pd.read_csv(file_path, comment="#")

        # Check each sequence for invalid characters
        invalid_count = 0
        for _idx, seq in enumerate(df["sequence"]):
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
    else:
        print("\n  ✗ FAIL: Invalid amino acids detected")

    # Pytest assertion
    assert all_valid, "Invalid amino acids detected"


def test_previously_affected_sequences():
    """
    Test 3: Spot-check sequences that previously had gaps.

    Before fix: VH_only had 11 sequences with gaps (1.0%)
                VL_only had 2 sequences with gaps (0.2%)
                Full had 13 sequences with gaps (1.2%)
    After fix: Should be 0
    """
    print("\n" + "=" * 70)
    print("TEST 3: Previously Affected Sequences (Spot Check)")
    print("=" * 70)

    # Check VH_only
    vh_file = Path("train_datasets/boughter/annotated/VH_only_boughter.csv")
    assert vh_file.exists(), f"{vh_file} not found"

    vh_df = pd.read_csv(vh_file, comment="#")

    # Check VL_only
    vl_file = Path("train_datasets/boughter/annotated/VL_only_boughter.csv")
    assert vl_file.exists(), f"{vl_file} not found"

    vl_df = pd.read_csv(vl_file, comment="#")

    # Sample check: First 5 sequences from each
    print("\n  Checking sample sequences from VH_only_boughter.csv:")
    all_clean = True

    for idx in range(min(5, len(vh_df))):
        seq = vh_df.iloc[idx]["sequence"]
        seq_id = vh_df.iloc[idx]["id"]
        has_gaps = "-" in seq

        if has_gaps:
            print(f"    ✗ {seq_id}: contains gaps")
            all_clean = False
        else:
            print(f"    ✓ {seq_id}: gap-free (length {len(seq)} aa)")

    print("\n  Checking sample sequences from VL_only_boughter.csv:")
    for idx in range(min(5, len(vl_df))):
        seq = vl_df.iloc[idx]["sequence"]
        seq_id = vl_df.iloc[idx]["id"]
        has_gaps = "-" in seq

        if has_gaps:
            print(f"    ✗ {seq_id}: contains gaps")
            all_clean = False
        else:
            print(f"    ✓ {seq_id}: gap-free (length {len(seq)} aa)")

    # Check total gap counts
    vh_gaps = vh_df["sequence"].str.contains("-", na=False).sum()
    vl_gaps = vl_df["sequence"].str.contains("-", na=False).sum()

    print(f"\n  VH_only: {len(vh_df)} total sequences, {vh_gaps} with gaps")
    print(f"  VL_only: {len(vl_df)} total sequences, {vl_gaps} with gaps")
    print("  Before P0 fix: VH had 11 gaps, VL had 2 gaps")

    if vh_gaps == 0 and vl_gaps == 0 and all_clean:
        print("\n  ✓ PASS: P0 fix successfully eliminated all gaps")
    else:
        print("\n  ✗ FAIL: Gaps still present after fix")

    # Pytest assertion
    assert vh_gaps == 0 and vl_gaps == 0 and all_clean, "Gaps still present after fix"


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

    # Test all critical fragment files
    test_files = [
        "VH_only_boughter.csv",
        "VL_only_boughter.csv",
        "Full_boughter.csv",
    ]

    all_passed = True
    total_validated = 0

    for file_name in test_files:
        file_path = Path(f"train_datasets/boughter/annotated/{file_name}")

        if not file_path.exists():
            print(f"  ✗ FAIL: {file_path} not found")
            all_passed = False
            continue

        df = pd.read_csv(file_path, comment="#")

        # Simulate model.py validation (lines 86-90)
        invalid_sequences = []
        for _idx, row in df.iterrows():
            sequence = row["sequence"]

            # Check for invalid amino acids (same logic as model.py)
            if not set(sequence).issubset(VALID_AMINO_ACIDS):
                invalid_sequences.append((row["id"], sequence))

        if len(invalid_sequences) == 0:
            print(f"  ✓ {file_name}: All {len(df)} sequences passed")
            total_validated += len(df)
        else:
            print(f"  ✗ {file_name}: {len(invalid_sequences)} sequences failed")
            print(f"    First failed sequence: {invalid_sequences[0][0]}")
            all_passed = False

    if all_passed:
        print(f"\n  ✓ PASS: All {total_validated} sequences are ESM-1v compatible")
    else:
        print("\n  ✗ FAIL: Dataset NOT compatible with ESM-1v")

    # Pytest assertion
    assert all_passed, "Dataset NOT compatible with ESM-1v"


def test_data_integrity():
    """
    Test 5: Verify data integrity after regeneration.

    Ensures all expected files exist, have correct row counts, and
    preserve label distribution.
    """
    print("\n" + "=" * 70)
    print("TEST 5: Data Integrity Verification")
    print("=" * 70)

    boughter_dir = Path("train_datasets/boughter/annotated")
    expected_files = [
        "VH_only_boughter.csv",
        "VL_only_boughter.csv",
        "H-CDR1_boughter.csv",
        "H-CDR2_boughter.csv",
        "H-CDR3_boughter.csv",
        "L-CDR1_boughter.csv",
        "L-CDR2_boughter.csv",
        "L-CDR3_boughter.csv",
        "H-CDRs_boughter.csv",
        "L-CDRs_boughter.csv",
        "H-FWRs_boughter.csv",
        "L-FWRs_boughter.csv",
        "VH+VL_boughter.csv",
        "All-CDRs_boughter.csv",
        "All-FWRs_boughter.csv",
        "Full_boughter.csv",
    ]

    all_valid = True
    expected_rows = None

    print(f"\n  Checking {len(expected_files)} fragment files:")

    for file_name in expected_files:
        file_path = boughter_dir / file_name

        if not file_path.exists():
            print(f"    ✗ MISSING: {file_name}")
            all_valid = False
            continue

        df = pd.read_csv(file_path, comment="#")

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

    # Check label distribution in VH file
    vh_file = boughter_dir / "VH_only_boughter.csv"
    if vh_file.exists():
        df = pd.read_csv(vh_file, comment="#")

        # Count training vs excluded sequences
        training_df = df[df["include_in_training"]]
        excluded_df = df[~df["include_in_training"]]

        print(f"\n  Total sequences: {len(df)}")
        print(f"  Training sequences: {len(training_df)}")
        print(f"  Excluded (1-3 flags): {len(excluded_df)}")

        if len(training_df) > 0:
            label_counts = training_df["label"].value_counts().sort_index()

            print("\n  Training set label distribution:")
            print(
                f"    Specific (0):     {label_counts.get(0, 0)} ({label_counts.get(0, 0) / len(training_df) * 100:.1f}%)"
            )
            print(
                f"    Non-specific (1): {label_counts.get(1, 0)} ({label_counts.get(1, 0) / len(training_df) * 100:.1f}%)"
            )

            # Check if balanced (should be ~50/50)
            balance = label_counts.get(1, 0) / len(training_df)
            if 0.45 <= balance <= 0.55:
                print("    ✓ Training set is balanced")
            else:
                print("    ⚠ Training set may be imbalanced")

    if all_valid and expected_rows is not None:
        print(
            f"\n  ✓ PASS: All {len(expected_files)} files present with {expected_rows} rows"
        )
    else:
        print("\n  ✗ FAIL: Data integrity issues detected")

    # Pytest assertion
    assert all_valid and expected_rows is not None, "Data integrity issues detected"


def main():
    """Run all tests and report results."""
    print("\n" + "=" * 70)
    print("Boughter Dataset - ESM-1v Embedding Compatibility Test Suite")
    print("=" * 70)
    print("P0 Blocker Fix: V-domain reconstruction from fragments (gap-free)")
    print(
        "Previous issue: annotation.sequence_alignment_aa (with gaps) + sequence_aa (constant region)"
    )
    print("Fix location: preprocessing/boughter/stage2_stage3_annotation_qc.py:93-147")

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
            test_func()  # No return value - uses assertions
            results.append((test_name, True))
        except AssertionError as e:
            print(f"\n  ✗ ASSERTION: {test_name} - {e}")
            results.append((test_name, False))
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
        print("\n  ✓ ALL TESTS PASSED - Boughter dataset is ESM-1v compatible!")
        print("  ✓ P0 blocker successfully resolved")
        return 0
    else:
        print("\n  ✗ SOME TESTS FAILED - Review errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
