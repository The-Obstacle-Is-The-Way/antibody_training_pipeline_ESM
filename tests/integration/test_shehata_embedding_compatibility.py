#!/usr/bin/env python3
"""
Shehata Dataset ESM Embedding Compatibility Test

Tests that cleaned Shehata fragment sequences are compatible with ESM-1v
embedding pipeline without requiring model download.

This validates the P0 blocker fix for gap characters.

Date: 2025-11-02
"""

import sys
from pathlib import Path

import pandas as pd


def test_gap_characters():
    """Test 1: Verify no gap characters in any fragment CSV"""
    print("\n" + "=" * 60)
    print("TEST 1: Gap Character Detection")
    print("=" * 60)

    fragments_dir = Path("test_datasets/shehata/fragments")
    fragment_files = list(fragments_dir.glob("*.csv"))

    assert fragment_files, "No fragment files found"

    all_clean = True
    for file in sorted(fragment_files):
        df = pd.read_csv(file)
        gap_count = df["sequence"].str.contains("-", na=False).sum()

        if gap_count > 0:
            print(f"  ❌ {file.name}: {gap_count} sequences with gaps")
            all_clean = False
        else:
            print(f"  ✅ {file.name}: gap-free ({len(df)} sequences)")

    if all_clean:
        print("\n✅ PASS: All fragment files are gap-free")
    else:
        print("\n❌ FAIL: Gap characters detected")

    # Pytest assertion
    assert all_clean, "Gap characters detected"


def test_amino_acid_validation():
    """Test 2: Verify all sequences contain only valid amino acids"""
    print("\n" + "=" * 60)
    print("TEST 2: Amino Acid Validation")
    print("=" * 60)

    valid_aas = set("ACDEFGHIKLMNPQRSTVWYX")

    fragments_dir = Path("test_datasets/shehata/fragments")

    # Test previously affected files
    critical_files = [
        "VH_only_shehata.csv",  # Had 13 gaps
        "VL_only_shehata.csv",  # Had 4 gaps
        "Full_shehata.csv",  # Had 17 gaps
    ]

    all_valid = True
    total_sequences = 0

    for filename in critical_files:
        file_path = fragments_dir / filename
        if not file_path.exists():
            print(f"  ⚠️  {filename}: not found (skipping)")
            continue

        df = pd.read_csv(file_path)
        invalid_count = 0

        for idx, seq in df["sequence"].items():
            seq_upper = str(seq).upper().strip()
            if not all(aa in valid_aas for aa in seq_upper):
                invalid_chars = set(seq_upper) - valid_aas
                print(f"  ❌ {filename} row {idx}: invalid chars {invalid_chars}")
                invalid_count += 1
                all_valid = False

        if invalid_count == 0:
            print(f"  ✅ {filename}: all {len(df)} sequences valid")
            total_sequences += len(df)
        else:
            print(f"  ❌ {filename}: {invalid_count}/{len(df)} invalid")

    if all_valid:
        print(
            f"\n✅ PASS: All {total_sequences} sequences contain only valid amino acids"
        )
    else:
        print("\n❌ FAIL: Invalid amino acids detected")

    # Pytest assertion
    assert all_valid, "Invalid amino acids detected"


def test_previously_affected_sequences():
    """Test 3: Spot-check sequences that previously had gaps"""
    print("\n" + "=" * 60)
    print("TEST 3: Previously Affected Sequences")
    print("=" * 60)

    # These IDs previously had gap characters
    vh_check = ["ADI-47173", "ADI-47060", "ADI-45440", "ADI-47105", "ADI-47224"]
    vl_check = ["ADI-47223", "ADI-47114", "ADI-47163", "ADI-47211"]

    fragments_dir = Path("test_datasets/shehata/fragments")
    vh_df = pd.read_csv(fragments_dir / "VH_only_shehata.csv")
    vl_df = pd.read_csv(fragments_dir / "VL_only_shehata.csv")

    all_clean = True

    print("\n  VH sequences (previously had gaps):")
    for id in vh_check:
        if id in vh_df["id"].values:
            seq = vh_df[vh_df["id"] == id]["sequence"].iloc[0]
            has_gap = "-" in seq
            if has_gap:
                print(f"    ❌ {id}: STILL HAS GAPS")
                all_clean = False
            else:
                print(f"    ✅ {id}: gap-free (len={len(seq)})")
        else:
            print(f"    ⚠️  {id}: not found")

    print("\n  VL sequences (previously had gaps):")
    for id in vl_check:
        if id in vl_df["id"].values:
            seq = vl_df[vl_df["id"] == id]["sequence"].iloc[0]
            has_gap = "-" in seq
            if has_gap:
                print(f"    ❌ {id}: STILL HAS GAPS")
                all_clean = False
            else:
                print(f"    ✅ {id}: gap-free (len={len(seq)})")
        else:
            print(f"    ⚠️  {id}: not found")

    if all_clean:
        print("\n✅ PASS: All previously affected sequences are now clean")
    else:
        print("\n❌ FAIL: Some sequences still have issues")

    # Pytest assertion
    assert all_clean, "Some sequences still have issues"


def test_model_validation_logic():
    """Test 4: Simulate model.py validation logic"""
    print("\n" + "=" * 60)
    print("TEST 4: ESM Model Validation Simulation")
    print("=" * 60)

    # This is the exact validation logic from model.py:86-90
    valid_aas = set("ACDEFGHIKLMNPQRSTVWYX")

    fragments_dir = Path("test_datasets/shehata/fragments")
    vh_df = pd.read_csv(fragments_dir / "VH_only_shehata.csv")

    # Sample 10 sequences
    sample_sequences = vh_df["sequence"].head(10).tolist()

    print("\n  Simulating model.py validation on 10 sample sequences:")

    all_valid = True
    for i, seq in enumerate(sample_sequences):
        seq_clean = seq.upper().strip()

        # This is the exact check from model.py
        if not all(aa in valid_aas for aa in seq_clean):
            invalid_chars = set(seq_clean) - valid_aas
            print(f"    ❌ Sequence {i + 1}: would FAIL validation")
            print(f"       Invalid characters: {invalid_chars}")
            all_valid = False
        else:
            print(
                f"    ✅ Sequence {i + 1}: would PASS validation (len={len(seq_clean)})"
            )

    if all_valid:
        print("\n✅ PASS: All sequences would pass ESM model validation")
    else:
        print("\n❌ FAIL: Some sequences would fail ESM validation")

    # Pytest assertion
    assert all_valid, "Some sequences would fail ESM validation"


def test_data_integrity():
    """Test 5: Verify data integrity after regeneration"""
    print("\n" + "=" * 60)
    print("TEST 5: Data Integrity")
    print("=" * 60)

    fragments_dir = Path("test_datasets/shehata/fragments")

    # Check all 16 fragment files exist
    expected_files = [
        "VH_only_shehata.csv",
        "VL_only_shehata.csv",
        "H-CDR1_shehata.csv",
        "H-CDR2_shehata.csv",
        "H-CDR3_shehata.csv",
        "L-CDR1_shehata.csv",
        "L-CDR2_shehata.csv",
        "L-CDR3_shehata.csv",
        "H-CDRs_shehata.csv",
        "L-CDRs_shehata.csv",
        "H-FWRs_shehata.csv",
        "L-FWRs_shehata.csv",
        "VH+VL_shehata.csv",
        "Full_shehata.csv",
        "All-CDRs_shehata.csv",
        "All-FWRs_shehata.csv",
    ]

    all_exist = True
    all_398 = True

    for filename in expected_files:
        file_path = fragments_dir / filename
        if not file_path.exists():
            print(f"  ❌ Missing: {filename}")
            all_exist = False
        else:
            df = pd.read_csv(file_path)
            if len(df) != 398:
                print(f"  ❌ {filename}: {len(df)} rows (expected 398)")
                all_398 = False
            else:
                print(f"  ✅ {filename}: 398 rows")

    if all_exist and all_398:
        print("\n✅ PASS: All 16 fragment files exist with 398 rows each")
    else:
        print("\n❌ FAIL: Data integrity issues detected")

    # Pytest assertion
    assert all_exist and all_398, "Data integrity issues detected"


def main():
    """Run all tests"""
    print("=" * 60)
    print("Shehata Dataset ESM Embedding Compatibility Test")
    print("=" * 60)
    print("\nPurpose: Validate P0 blocker fix (gap characters removed)")
    print("Date: 2025-11-02")

    tests = [
        ("Gap Character Detection", test_gap_characters),
        ("Amino Acid Validation", test_amino_acid_validation),
        ("Previously Affected Sequences", test_previously_affected_sequences),
        ("ESM Model Validation Simulation", test_model_validation_logic),
        ("Data Integrity", test_data_integrity),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            test_func()  # No return value - uses assertions
            results.append((test_name, True))
        except AssertionError as e:
            print(f"\n❌ ASSERTION in {test_name}: {e}")
            results.append((test_name, False))
        except Exception as e:
            print(f"\n❌ EXCEPTION in {test_name}: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(passed for _, passed in results)

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        print("\nShehata dataset is READY for ESM embedding:")
        print("  - No gap characters detected")
        print("  - All amino acids valid")
        print("  - Previously affected sequences clean")
        print("  - Would pass model.py validation")
        print("  - Data integrity confirmed")
        print("\nP0 blocker: RESOLVED ✅")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("=" * 60)
        print("\nP0 blocker may still exist - investigate failures above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
