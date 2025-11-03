#!/usr/bin/env python3
"""
Jain Dataset ESM Embedding Compatibility Test

Tests that cleaned Jain fragment sequences are compatible with ESM-1v
embedding pipeline without requiring model download.

P0 Blocker Verification:
- Ensures no gap characters ('-') in any fragment CSV
- Validates sequences contain only valid amino acids
- Checks for stop codons ('*') that would crash ESM

Test Coverage:
1. Gap character detection (P0 blocker regression check)
2. Amino acid validation (ESM-1v compatible characters only)
3. Stop codon detection (another P0 blocker)
4. Model validation logic simulation
5. Data integrity verification

Date: 2025-11-02
Issue: Jain dataset preprocessing P0 validation
Expected: 137 antibodies (67 specific, 67 mild, 3 non-specific)
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
    This test ensures sequences are properly cleaned.
    """
    print("\n" + "=" * 70)
    print("TEST 1: Gap Character Detection (P0 Blocker Check)")
    print("=" * 70)

    jain_dir = Path("test_datasets/jain")
    fragment_files = [
        "VH_only_jain.csv",
        "VL_only_jain.csv",
        "H-CDR1_jain.csv",
        "H-CDR2_jain.csv",
        "H-CDR3_jain.csv",
        "L-CDR1_jain.csv",
        "L-CDR2_jain.csv",
        "L-CDR3_jain.csv",
        "H-CDRs_jain.csv",
        "L-CDRs_jain.csv",
        "H-FWRs_jain.csv",
        "L-FWRs_jain.csv",
        "VH+VL_jain.csv",
        "All-CDRs_jain.csv",
        "All-FWRs_jain.csv",
        "Full_jain.csv",
    ]

    all_clean = True
    total_sequences = 0

    for file_name in fragment_files:
        file_path = jain_dir / file_name
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
            # Show first few problematic sequences
            gap_seqs = df[df["sequence"].str.contains("-", na=False)].head(3)
            for idx, row in gap_seqs.iterrows():
                print(f"    ID: {row['id']}, gaps in: {row['sequence'][:50]}...")
            all_clean = False
        else:
            print(f"  ✓ {file_name}: gap-free ({len(df)} sequences)")

    if all_clean:
        print("\n  ✓ PASS: All Jain fragments are gap-free")
        print(f"  ✓ Total sequences validated: {total_sequences}")
        return True
    else:
        print("\n  ✗ FAIL: Gap characters detected - P0 blocker present!")
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

    jain_dir = Path("test_datasets/jain")

    # Check critical files for full antibody model
    test_files = [
        "VH_only_jain.csv",
        "VL_only_jain.csv",
        "H-CDRs_jain.csv",
        "L-CDRs_jain.csv",
        "Full_jain.csv",
    ]

    all_valid = True
    total_sequences = 0

    for file_name in test_files:
        file_path = jain_dir / file_name
        if not file_path.exists():
            print(f"  ✗ MISSING: {file_name}")
            all_valid = False
            continue

        df = pd.read_csv(file_path, comment="#")

        # Check each sequence for invalid characters
        invalid_count = 0
        for idx, seq in enumerate(df["sequence"]):
            if not set(seq).issubset(VALID_AMINO_ACIDS):
                invalid_chars = set(seq) - VALID_AMINO_ACIDS
                if invalid_count == 0:
                    print(f"  ✗ {file_name}: Invalid characters found: {invalid_chars}")
                    # Show which antibody has the issue
                    print(f"    First issue in: {df.iloc[idx]['id']}")
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


def test_stop_codons():
    """
    Test 3: Verify no stop codons in sequences.

    P0 BLOCKER: Stop codons ('*') can cause issues with ESM embedding.
    This is common when constant regions are included.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Stop Codon Detection (P0 Blocker Check)")
    print("=" * 70)

    jain_dir = Path("test_datasets/jain")
    fragment_files = [
        "VH_only_jain.csv",
        "VL_only_jain.csv",
        "Full_jain.csv",
    ]

    all_clean = True
    total_sequences = 0

    for file_name in fragment_files:
        file_path = jain_dir / file_name
        if not file_path.exists():
            print(f"  ✗ MISSING: {file_name}")
            all_clean = False
            continue

        df = pd.read_csv(file_path, comment="#")
        stop_count = df["sequence"].str.contains(r"\*", na=False, regex=True).sum()
        total_sequences = len(df)

        if stop_count > 0:
            print(f"  ✗ {file_name}: {stop_count} sequences with stop codons")
            # Show first few problematic sequences
            stop_seqs = df[
                df["sequence"].str.contains(r"\*", na=False, regex=True)
            ].head(3)
            for idx, row in stop_seqs.iterrows():
                print(f"    ID: {row['id']}")
            all_clean = False
        else:
            print(f"  ✓ {file_name}: no stop codons ({len(df)} sequences)")

    if all_clean:
        print("\n  ✓ PASS: All Jain fragments are stop-codon-free")
        print(f"  ✓ Total sequences validated: {total_sequences}")
        return True
    else:
        print("\n  ✗ FAIL: Stop codons detected - P0 blocker present!")
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

    # Test all critical fragment files
    test_files = [
        "VH_only_jain.csv",
        "VL_only_jain.csv",
        "Full_jain.csv",
    ]

    all_passed = True
    total_validated = 0

    for file_name in test_files:
        file_path = Path(f"test_datasets/jain/{file_name}")

        if not file_path.exists():
            print(f"  ✗ FAIL: {file_path} not found")
            all_passed = False
            continue

        df = pd.read_csv(file_path, comment="#")

        # Simulate model.py validation (lines 86-90)
        invalid_sequences = []
        for idx, row in df.iterrows():
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
        return True
    else:
        print("\n  ✗ FAIL: Dataset NOT compatible with ESM-1v")
        return False


def test_data_integrity():
    """
    Test 5: Verify data integrity.

    Ensures all expected files exist, have correct row counts (137), and
    preserve label distribution (67 specific, 67 mild, 3 non-specific).
    """
    print("\n" + "=" * 70)
    print("TEST 5: Data Integrity Verification")
    print("=" * 70)

    jain_dir = Path("test_datasets/jain")
    expected_files = [
        "VH_only_jain.csv",
        "VL_only_jain.csv",
        "H-CDR1_jain.csv",
        "H-CDR2_jain.csv",
        "H-CDR3_jain.csv",
        "L-CDR1_jain.csv",
        "L-CDR2_jain.csv",
        "L-CDR3_jain.csv",
        "H-CDRs_jain.csv",
        "L-CDRs_jain.csv",
        "H-FWRs_jain.csv",
        "L-FWRs_jain.csv",
        "VH+VL_jain.csv",
        "All-CDRs_jain.csv",
        "All-FWRs_jain.csv",
        "Full_jain.csv",
    ]

    all_valid = True
    expected_rows = 137  # Known count from PNAS supplementary files

    print(f"\n  Checking {len(expected_files)} fragment files:")
    print(f"  Expected rows per file: {expected_rows}")

    for file_name in expected_files:
        file_path = jain_dir / file_name

        if not file_path.exists():
            print(f"    ✗ MISSING: {file_name}")
            all_valid = False
            continue

        df = pd.read_csv(file_path, comment="#")

        if len(df) != expected_rows:
            print(
                f"    ✗ {file_name}: row count mismatch ({len(df)} vs {expected_rows})"
            )
            all_valid = False
        else:
            print(f"    ✓ {file_name}: {len(df)} rows")

    # Check label distribution in VH file
    vh_file = jain_dir / "VH_only_jain.csv"
    if vh_file.exists():
        df = pd.read_csv(vh_file, comment="#")

        # Check label distribution
        # label column: 0 (specific), 1 (non-specific), NaN (mild)
        specific_count = (df["label"] == 0).sum()
        nonspecific_count = (df["label"] == 1).sum()
        mild_count = df["label"].isna().sum()

        print(f"\n  Total sequences: {len(df)}")
        print("  Label distribution:")
        print(
            f"    Specific (0):     {specific_count} ({specific_count/len(df)*100:.1f}%)"
        )
        print(
            f"    Non-specific (1): {nonspecific_count} ({nonspecific_count/len(df)*100:.1f}%)"
        )
        print(f"    Mild (NaN):       {mild_count} ({mild_count/len(df)*100:.1f}%)")

        # Expected: 67 specific, 3 non-specific, 67 mild
        print("\n  Expected distribution: 67 specific, 3 non-specific, 67 mild")

        if specific_count == 67 and nonspecific_count == 3 and mild_count == 67:
            print("    ✓ Label distribution matches expected")
        else:
            print("    ⚠ Label distribution differs from expected")
            all_valid = False

        # For model testing, we use only specific (67) + non-specific (3) = 70
        test_count = specific_count + nonspecific_count
        print(f"\n  Test set size (excluding mild): {test_count} sequences")
        print("    Expected: 70 sequences (67 + 3)")

        if test_count == 70:
            print("    ✓ Test set size matches expected")
        else:
            print("    ⚠ Test set size differs from expected")

    if all_valid and expected_rows == 137:
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
    print("Jain Dataset - ESM-1v Embedding Compatibility Test Suite")
    print("=" * 70)
    print("Dataset: 137 clinical antibodies from PNAS 2017")
    print("Test set: 70 sequences (67 specific + 3 non-specific)")
    print("Excluded: 67 mild sequences (1-3 flags)")

    # Run all tests
    tests = [
        ("Gap Character Detection", test_gap_characters),
        ("Amino Acid Validation", test_amino_acid_validation),
        ("Stop Codon Detection", test_stop_codons),
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
            import traceback

            traceback.print_exc()
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
        print("\n  ✓ ALL TESTS PASSED - Jain dataset is ESM-1v compatible!")
        print("  ✓ Ready for model inference and confusion matrix generation")
        return 0
    else:
        print("\n  ✗ SOME TESTS FAILED - P0 blockers detected")
        print("  ✗ Fix required before ESM embedding can proceed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
