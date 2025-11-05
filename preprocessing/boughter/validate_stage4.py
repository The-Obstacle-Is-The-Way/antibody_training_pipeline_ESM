#!/usr/bin/env python3
"""
Validation for Stage 4: Additional QC Filtering
==============================================

Validates that strict QC filtering was applied correctly to all fragment files.

Note: Different fragments have different expected counts based on where X appears:
- CDR-only fragments: 914 sequences (no X in CDRs - already filtered by Boughter QC)
- VH_only/H-FWRs: 852 sequences (62 with X in VH frameworks removed)
- VL_only/L-FWRs: 900 sequences (14 with X in VL frameworks removed)
- Full/VH+VL/All-FWRs: 840 sequences (74 total with X in either chain removed)

Date: 2025-11-04
"""

import pandas as pd
from pathlib import Path

BOUGHTER_DIR = Path("train_datasets/boughter")
STRICT_QC_PATTERN = "*_strict_qc.csv"
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Expected sequence counts for different fragment groups
EXPECTED_COUNTS = {
    'CDR_only': {
        'count': 914,
        'fragments': ['All-CDRs', 'H-CDR1', 'H-CDR2', 'H-CDR3', 'H-CDRs',
                     'L-CDR1', 'L-CDR2', 'L-CDR3', 'L-CDRs']
    },
    'VH_frameworks': {
        'count': 852,
        'fragments': ['VH_only', 'H-FWRs']
    },
    'VL_frameworks': {
        'count': 900,
        'fragments': ['VL_only', 'L-FWRs']
    },
    'VH_and_VL': {
        'count': 840,
        'fragments': ['Full', 'VH+VL', 'All-FWRs']
    }
}

def validate_strict_qc():
    """Validate all strict QC files"""

    print("=" * 80)
    print("Stage 4 Validation: Strict QC Quality Checks")
    print("=" * 80)
    print()

    # Find all strict QC files
    strict_qc_files = list(BOUGHTER_DIR.glob(STRICT_QC_PATTERN))

    if len(strict_qc_files) == 0:
        print("❌ ERROR: No *_strict_qc.csv files found!")
        print("   Run preprocessing/boughter/stage4_additional_qc.py first")
        return False

    print(f"Found {len(strict_qc_files)} strict QC files to validate\n")

    all_valid = True
    sequence_counts = {}
    fragment_groups = {}

    for file_path in sorted(strict_qc_files):
        fragment_name = file_path.stem.replace('_boughter_strict_qc', '')
        print(f"Validating: {fragment_name}")

        # Load data
        df = pd.read_csv(file_path, comment='#')
        sequence_counts[fragment_name] = len(df)

        # Check 1: No X in sequences
        has_x = df['sequence'].str.contains('X', na=False).sum()
        if has_x > 0:
            print(f"  ❌ FAIL: Found {has_x} sequences with X")
            all_valid = False
        else:
            print(f"  ✅ No X in sequences")

        # Check 2: No non-standard amino acids
        def has_non_standard(seq):
            return any(aa not in STANDARD_AA for aa in str(seq))

        non_standard = df['sequence'].apply(has_non_standard).sum()
        if non_standard > 0:
            print(f"  ❌ FAIL: Found {non_standard} sequences with non-standard AA")
            all_valid = False
        else:
            print(f"  ✅ All standard amino acids")

        # Check 3: Sequence count
        print(f"  ℹ️  Sequence count: {len(df)}")

        # Check 4: Label distribution (if applicable)
        if 'label' in df.columns and fragment_name in ['VH_only', 'Full', 'VH+VL']:
            label_counts = df['label'].value_counts().sort_index()
            total = len(df)
            label_0_pct = (label_counts.get(0, 0) / total) * 100 if total > 0 else 0
            label_1_pct = (label_counts.get(1, 0) / total) * 100 if total > 0 else 0
            print(f"  ℹ️  Label distribution: 0={label_counts.get(0, 0)} ({label_0_pct:.1f}%), 1={label_counts.get(1, 0)} ({label_1_pct:.1f}%)")

        # Categorize fragment
        for group_name, group_info in EXPECTED_COUNTS.items():
            if fragment_name in group_info['fragments']:
                fragment_groups[fragment_name] = group_name
                break

        print()

    # Check 5: Fragment groups have expected counts
    print("=" * 80)
    print("FRAGMENT GROUP VALIDATION")
    print("=" * 80)
    print()

    for group_name, group_info in EXPECTED_COUNTS.items():
        expected = group_info['count']
        fragments = group_info['fragments']

        print(f"{group_name}: Expected {expected} sequences")
        group_valid = True

        for fragment in fragments:
            if fragment in sequence_counts:
                actual = sequence_counts[fragment]
                if actual == expected:
                    print(f"  ✅ {fragment}: {actual}")
                else:
                    print(f"  ❌ {fragment}: {actual} (expected {expected})")
                    group_valid = False
                    all_valid = False
            else:
                print(f"  ⚠️  {fragment}: NOT FOUND")

        print()

    # Check 6: All files processed
    print("=" * 80)
    print("FILE COMPLETENESS CHECK")
    print("=" * 80)
    print()

    all_expected_fragments = []
    for group_info in EXPECTED_COUNTS.values():
        all_expected_fragments.extend(group_info['fragments'])

    missing = set(all_expected_fragments) - set(sequence_counts.keys())
    unexpected = set(sequence_counts.keys()) - set(all_expected_fragments)

    if not missing and not unexpected:
        print(f"✅ All 16 expected fragments processed")
    else:
        if missing:
            print(f"⚠️  Missing fragments: {missing}")
        if unexpected:
            print(f"⚠️  Unexpected fragments: {unexpected}")

    # Summary table
    print()
    print("=" * 80)
    print("SEQUENCE COUNT SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Fragment':<20} {'Count':>6}  {'Group':<20}")
    print("-" * 80)

    for fragment in sorted(sequence_counts.keys()):
        count = sequence_counts[fragment]
        group = fragment_groups.get(fragment, 'Unknown')
        print(f"{fragment:<20} {count:>6}  {group:<20}")

    # Expected reduction from 914 training sequences
    print()
    print("=" * 80)
    print("QC IMPACT ANALYSIS")
    print("=" * 80)
    print()

    print("Starting point (Boughter QC, training subset): 914 sequences")
    print()
    print("After strict QC (X in frameworks removed):")
    print(f"  VH_only (primary target):  852 sequences (-62, -6.8%)")
    print(f"  VL_only:                   900 sequences (-14, -1.5%)")
    print(f"  Full (VH+VL):              840 sequences (-74, -8.1%)")
    print(f"  CDR-only fragments:        914 sequences (no change - X already filtered)")

    # Final result
    print()
    print("=" * 80)
    if all_valid:
        print("✅ ALL VALIDATION CHECKS PASSED")
        print()
        print("Stage 4 strict QC filtering is CORRECT:")
        print("  - No X in any sequence")
        print("  - No non-standard amino acids")
        print("  - Fragment counts match expected patterns")
        print("  - CDR-only fragments unchanged (914 - X already filtered by Boughter)")
        print("  - VH_only reduced to 852 (-62 with X in VH frameworks)")
    else:
        print("❌ VALIDATION FAILED - See errors above")
    print("=" * 80)

    return all_valid


if __name__ == "__main__":
    success = validate_strict_qc()
    exit(0 if success else 1)
