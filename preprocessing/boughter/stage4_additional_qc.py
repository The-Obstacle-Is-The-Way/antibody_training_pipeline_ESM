#!/usr/bin/env python3
"""
Stage 4: Additional QC Filtering for Boughter Dataset
======================================================

Purpose: Apply industry-standard full-sequence QC beyond Boughter's CDR-only checks

Inputs: train_datasets/boughter/annotated/*_boughter.csv (16 fragment files, 1,065 sequences each)
Outputs: train_datasets/boughter/strict_qc/*_boughter_strict_qc.csv (16 files, ~852 sequences each)

Filters:
1. Select include_in_training == True (reduces 1,065 → 914)
2. Remove X anywhere in full VH sequence (not just CDRs)
3. Remove non-standard amino acids (B, Z, J, U, O)

Date: 2025-11-04
"""

import pandas as pd
from pathlib import Path
import sys

# Paths
INPUT_DIR = Path("train_datasets/boughter/annotated")
OUTPUT_DIR = Path("train_datasets/boughter/strict_qc")
INPUT_PATTERN = "*_boughter.csv"
OUTPUT_SUFFIX = "_strict_qc"

# Standard amino acids
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Fragment list (16 total from Novo Table 4)
FRAGMENTS = [
    "VH_only",
    "VL_only",
    "VH+VL",
    "Full",
    "H-CDR1",
    "H-CDR2",
    "H-CDR3",
    "H-CDRs",
    "H-FWRs",
    "L-CDR1",
    "L-CDR2",
    "L-CDR3",
    "L-CDRs",
    "L-FWRs",
    "All-CDRs",
    "All-FWRs"
]

def apply_strict_qc(df, fragment_name, original_seq_col='sequence'):
    """
    Apply industry-standard QC filters:
    1. Filter to training set (include_in_training == True)
    2. Remove X anywhere in sequence
    3. Remove non-standard amino acids (B, Z, J, U, O)

    Returns: (filtered_df, stats_dict)
    """
    stats = {
        'original': len(df),
        'after_training_filter': 0,
        'removed_x': 0,
        'removed_non_standard': 0,
        'final': 0
    }

    # Step 1: Filter to training set
    if 'include_in_training' in df.columns:
        df_filtered = df[df['include_in_training'] == True].copy()
        stats['after_training_filter'] = len(df_filtered)
        print(f"  Step 1: Training filter: {stats['original']} → {stats['after_training_filter']}")
    else:
        print(f"  WARNING: No 'include_in_training' column in {fragment_name}")
        df_filtered = df.copy()
        stats['after_training_filter'] = len(df_filtered)

    # Step 2: Remove X anywhere in sequence
    before = len(df_filtered)
    df_filtered = df_filtered[~df_filtered[original_seq_col].str.contains('X', na=False)]
    removed = before - len(df_filtered)
    stats['removed_x'] = removed
    print(f"  Step 2: Remove X: {before} → {len(df_filtered)} (-{removed})")

    # Step 3: Remove non-standard amino acids
    def has_non_standard(seq):
        if pd.isna(seq):
            return False
        return any(aa not in STANDARD_AA for aa in str(seq))

    before = len(df_filtered)
    df_filtered = df_filtered[~df_filtered[original_seq_col].apply(has_non_standard)]
    removed = before - len(df_filtered)
    stats['removed_non_standard'] = removed
    print(f"  Step 3: Remove non-standard AA: {before} → {len(df_filtered)} (-{removed})")

    stats['final'] = len(df_filtered)

    return df_filtered, stats


def process_all_fragments():
    """Process all 16 fragment CSVs"""

    print("=" * 80)
    print("Stage 4: Additional QC Filtering for Boughter Dataset")
    print("=" * 80)
    print()

    # Check input directory
    if not INPUT_DIR.exists():
        print(f"ERROR: {INPUT_DIR} not found!")
        sys.exit(1)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all fragment files
    input_files = list(INPUT_DIR.glob(INPUT_PATTERN))

    # Exclude the training export file (it's a flattened subset)
    input_files = [f for f in input_files if 'training' not in f.name]

    if len(input_files) == 0:
        print(f"ERROR: No *_boughter.csv files found in {INPUT_DIR}")
        sys.exit(1)

    print(f"Found {len(input_files)} fragment files to process:")
    for f in sorted(input_files):
        print(f"  - {f.name}")
    print()

    # Process each fragment
    all_stats = {}

    for input_path in sorted(input_files):
        fragment_name = input_path.stem.replace('_boughter', '')
        output_name = f"{fragment_name}_boughter_strict_qc.csv"
        output_path = OUTPUT_DIR / output_name

        print(f"Processing: {fragment_name}")
        print(f"  Input:  {input_path.name}")
        print(f"  Output: {output_name}")

        # Load data
        df = pd.read_csv(input_path, comment='#')

        # Apply strict QC
        df_clean, stats = apply_strict_qc(df, fragment_name)
        all_stats[fragment_name] = stats

        # Prepare output with metadata header
        metadata = f"""# Boughter Dataset - {fragment_name} Fragment (Strict QC)
# QC Level: Boughter QC (X in CDRs, empty CDRs) + Industry Standard (X anywhere, non-standard AA)
# Source: Filtered from {input_path.name}
# Original sequences: {stats['original']}
# After training filter: {stats['after_training_filter']}
# After strict QC: {stats['final']} (-{stats['removed_x']} X, -{stats['removed_non_standard']} non-standard AA)
# Reference: See BOUGHTER_ADDITIONAL_QC_PLAN.md
"""

        # Save output
        with open(output_path, 'w') as f:
            f.write(metadata)
            df_clean.to_csv(f, index=False)

        print(f"  ✓ Saved: {len(df_clean)} sequences")
        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Check consistency across fragments
    final_counts = {name: stats['final'] for name, stats in all_stats.items()}
    if len(set(final_counts.values())) == 1:
        print(f"✅ All fragments have same sequence count: {list(final_counts.values())[0]}")
    else:
        print("⚠️  WARNING: Fragments have different sequence counts!")
        for name, count in sorted(final_counts.items()):
            print(f"   {name}: {count}")

    # Overall statistics
    example_stats = all_stats[list(all_stats.keys())[0]]
    print()
    print(f"Pipeline progression (per fragment):")
    print(f"  Original (all flags):       {example_stats['original']:4d} sequences")
    print(f"  Training filter:            {example_stats['after_training_filter']:4d} sequences")
    print(f"  After strict QC:            {example_stats['final']:4d} sequences")
    print()
    print(f"  Removed by X filter:        {example_stats['removed_x']:4d} sequences")
    print(f"  Removed by non-standard AA: {example_stats['removed_non_standard']:4d} sequences")
    print(f"  Total removed:              {example_stats['after_training_filter'] - example_stats['final']:4d} sequences")
    print()

    # Files created
    print(f"Files created: {len(all_stats)}")
    for name in sorted(all_stats.keys()):
        output_name = f"{name}_boughter_strict_qc.csv"
        print(f"  ✓ {output_name}")

    print()
    print("=" * 80)
    print("Stage 4 Complete!")
    print("=" * 80)


if __name__ == "__main__":
    process_all_fragments()
