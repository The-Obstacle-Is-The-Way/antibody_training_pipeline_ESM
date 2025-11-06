# Boughter Dataset: Additional QC Filtering Plan

**Date:** 2025-11-04 (Updated after senior audit)
**Status:** 🔧 **IMPLEMENTATION PLAN - CORRECTED**
**Branch:** leroy-jenkins/boughter-clean

---

## Executive Summary

**Problem:** QC audit found 62/914 sequences (6.8%) with X amino acid in the training set. Of these, 46 have X at position 0 (start of VH - Framework 1), and 16 have X elsewhere. Boughter's QC only checks CDRs, so these pass but likely fail Novo's industry-standard full-sequence filtering.

**Solution:** Add **Stage 4: Additional QC** to filter sequences with X ANYWHERE in full VH sequence (not just CDRs).

**Key Principle:** **DO NOT modify existing files/scripts** - only EXTEND the pipeline with new filtering stage.

---

## Current Pipeline State (VERIFIED 2025-11-04)

### Actual Directory Structure

```
preprocessing/boughter/
├── stage1_dna_translation.py          ✅ Keep as-is
├── stage2_stage3_annotation_qc.py     ✅ Keep as-is
├── validate_stage1.py                 ✅ Keep as-is
└── validate_stages2_3.py              ✅ Keep as-is

train_datasets/boughter/
├── raw/                               ✅ Stage 0: Raw DNA FASTA files (source data)
│   ├── flu_fastaH.txt
│   ├── flu_fastaL.txt
│   ├── mouse_fastaH.dat
│   └── ... (other raw files)
│
├── processed/                         ✅ Stage 1: Translated proteins
│   └── boughter.csv                   (1,117 sequences)
│
├── annotated/                         ✅ Stages 2+3: ANARCI fragments
│   ├── VH_only_boughter.csv           (1,065 seqs, has include_in_training column)
│   ├── H-CDR3_boughter.csv            (1,065 seqs, has include_in_training column)
│   ├── All-CDRs_boughter.csv          (1,065 seqs, has include_in_training column)
│   ├── ... (13 more fragment CSVs)
│   ├── annotation_failures.log
│   ├── qc_filtered_sequences.txt
│   └── validation_report.txt
│
├── canonical/                         ✅ Authoritative training file
│   ├── VH_only_boughter_training.csv  (914 seqs, training subset ONLY)
│   └── README.md                      └── Columns: [sequence, label] - NO id, NO fragments
│
└── strict_qc/                         ✅ Stage 4: Experimental strict QC
    ├── VH_only_boughter_strict_qc.csv (16 strict QC files)
    └── README.md
```

### Current Pipeline Flow (VERIFIED)

```
train_datasets/boughter/raw/ - Raw DNA FASTA files (1,171 sequences)
   ↓
Stage 1: DNA → Protein translation
   ↓ preprocessing/boughter/stage1_dna_translation.py
   ↓ Output: train_datasets/boughter/processed/boughter.csv (1,117 sequences)
   ↓
Stage 2+3: ANARCI annotation + Boughter QC
   ↓ preprocessing/boughter/stage2_stage3_annotation_qc.py
   ↓ QC: X in CDRs only, empty CDRs
   ↓ Outputs (1,065 sequences each):
   ↓   • train_datasets/boughter/annotated/*_boughter.csv (16 fragment files)
   ↓     (with id, sequence, label, subset, num_flags, flag_category,
   ↓      include_in_training, source, sequence_length)
   ↓
   ↓   • train_datasets/boughter/canonical/VH_only_boughter_training.csv
   ↓     (914 sequences where include_in_training==True)
   ↓     (columns: sequence, label ONLY - no id, no fragments)
```

**Status:** ✅ Pipeline is CORRECT and matches Boughter's methodology exactly!

---

## What We Discovered (2025-11-04)

### QC Audit Results

**Script:** `scripts/audit_boughter_training_qc.py`

**Findings:**
- **62 sequences** total with X amino acid anywhere in VH
  - **46 sequences** with X at position 0 (Framework 1 start)
  - **16 sequences** with X at other positions

**Why Boughter's QC didn't filter them:**
```python
# Boughter's seq_loader.py only checks CDRs:
total_abs[~total_abs['cdrH1_aa'].str.contains("X")]  # CDR-H1
total_abs[~total_abs['cdrH2_aa'].str.contains("X")]  # CDR-H2
total_abs[~total_abs['cdrH3_aa'].str.contains("X")]  # CDR-H3
# Does NOT check full VH sequence!

# Position 0 and other framework positions are NOT in CDRs
# Therefore: sequences with X in frameworks PASS Boughter's QC ✅
```

**Why Novo likely filtered them:**
- Industry standard: filter X ANYWHERE in sequence (not just CDRs)
- ESM embedding models expect valid amino acids at all positions
- X at position 0 indicates DNA translation ambiguity (should be E/Q/D)
- Professional QC practice in pharma/biotech

**Impact on accuracy:**
- Our accuracy: 67.5% ± 8.9% (914 sequences)
- Novo accuracy: 71%
- Gap: 3.5 percentage points (within 0.4 standard deviations)
- Hypothesis: Filtering these 62 sequences will close the gap

**Expected post-filtering:**
- 914 - 62 = **852 sequences** with strict QC

---

## Critical Architectural Insight (CORRECTED)

### ❌ WRONG APPROACH (as in original plan):

```python
# Read VH_only_boughter_training.csv
df = pd.read_csv('VH_only_boughter_training.csv')
# Columns: [sequence, label] - NO id, NO fragments!

# Apply filters
df_clean = filter_x(df)

# Try to regenerate fragments... ❌ IMPOSSIBLE!
# Cannot reconstruct H-CDR3, All-CDRs, VH+VL without original data
```

**Problem:** `VH_only_boughter_training.csv` is a **flattened export** with only `[sequence, label]`. You cannot reconstruct:
- Light chain sequences (VL, L-CDR1/2/3)
- Paired VH+VL sequences
- Individual CDR/FWR fragments
- Source metadata (id, subset, flags)

### ✅ CORRECT APPROACH:

```python
# Read fragment CSVs which have ALL metadata
df = pd.read_csv('H-CDR3_boughter.csv', comment='#')
# Columns: [id, sequence, label, subset, num_flags, flag_category,
#           include_in_training, source, sequence_length]

# Filter to training set (where include_in_training == True)
df_train = df[df['include_in_training'] == True]  # 914 sequences

# Apply strict QC
df_clean = filter_x(df_train)  # ~852 sequences

# Save with ALL original columns preserved
df_clean.to_csv('H-CDR3_boughter_strict_qc.csv', index=False)
```

**Why this works:**
- Fragment CSVs have `id` column to track sequences across fragments
- They have all metadata (subset, flags, source)
- They already have `include_in_training` flag from Stage 2+3
- We preserve full data provenance

---

## What We Need to Add

### NEW Stage 4: Additional QC (Industry Standard Full-Sequence Filtering)

**Purpose:** Apply industry-standard QC filters beyond Boughter's CDR-only checks

**Input:** All 16 fragment CSVs (`train_datasets/boughter/annotated/*_boughter.csv`)

**Filters to Apply:**
1. ✅ Filter sequences where `include_in_training == False` (reduces 1,065 → 914)
2. ✅ **Filter X anywhere in VH sequence** (not just CDRs) - removes 62 sequences
3. ✅ **Filter non-standard amino acids** (B, Z, J, U, O) if present
4. ⚠️ **Optional:** Filter extreme length outliers (>2.5 SD from mean) - REVIEW FIRST

**Output:** 16 fragment CSVs with `_strict_qc` suffix (~852 sequences each)

---

## File Naming Convention

### Philosophy

- **Base files** (all sequences, all flags): `*_boughter.csv` (1,065 sequences)
- **Training subset export**: `VH_only_boughter_training.csv` (914 sequences, flattened)
- **Strict QC files**: `*_boughter_strict_qc.csv` (~852 sequences, preserves all columns)

### Naming Pattern

```
{fragment}_boughter_{qc_level}.csv

Where:
- fragment: VH_only, H-CDR3, All-CDRs, Full, VH+VL, etc.
- qc_level: (none) = All sequences, Boughter QC only (1,065)
            strict_qc = Training subset + industry standard (~852)
```

### Examples

**Current files (keep as-is):**
```
train_datasets/boughter/annotated/VH_only_boughter.csv                # 1,065 sequences (all flags)
train_datasets/boughter/canonical/VH_only_boughter_training.csv       # 914 sequences (export, no metadata)
train_datasets/boughter/annotated/H-CDR3_boughter.csv                 # 1,065 sequences (all flags)
train_datasets/boughter/annotated/All-CDRs_boughter.csv               # 1,065 sequences (all flags)
train_datasets/boughter/annotated/Full_boughter.csv                   # 1,065 sequences (all flags)
... (13 more fragment CSVs)
```

**NEW files (to be created):**
```
train_datasets/boughter/strict_qc/VH_only_boughter_strict_qc.csv      # ~852 sequences (X filtered)
train_datasets/boughter/strict_qc/H-CDR3_boughter_strict_qc.csv       # ~852 sequences
train_datasets/boughter/strict_qc/All-CDRs_boughter_strict_qc.csv     # ~852 sequences
train_datasets/boughter/strict_qc/Full_boughter_strict_qc.csv         # ~852 sequences
... (13 more fragment CSVs with strict QC)
```

---

## Implementation Steps

### Step 1: Create Stage 4 QC Script

**File:** `preprocessing/boughter/stage4_additional_qc.py`

**Purpose:** Apply industry-standard full-sequence QC filters to all fragment CSVs

**Inputs:** `train_datasets/boughter/annotated/*_boughter.csv` (16 files, 1,065 sequences each)

**Filters:**
1. Select `include_in_training == True` (reduces to 914)
2. Remove X anywhere in VH sequence
3. Remove non-standard amino acids (B, Z, J, U, O) if present

**Outputs:** `train_datasets/boughter/*_boughter_strict_qc.csv` (16 files, ~852 sequences each)

**Complete Implementation:**
```python
#!/usr/bin/env python3
"""
Stage 4: Additional QC Filtering for Boughter Dataset
======================================================

Purpose: Apply industry-standard full-sequence QC beyond Boughter's CDR-only checks

Inputs: train_datasets/boughter/annotated/*_boughter.csv (16 fragment files, 1,065 sequences each)
Outputs: train_datasets/boughter/*_boughter_strict_qc.csv (16 files, ~852 sequences each)

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
BOUGHTER_DIR = Path("train_datasets/boughter")
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
    if not BOUGHTER_DIR.exists():
        print(f"ERROR: {BOUGHTER_DIR} not found!")
        sys.exit(1)

    # Find all fragment files
    input_files = list(BOUGHTER_DIR.glob(INPUT_PATTERN))

    # Exclude the training export file (it's a flattened subset)
    input_files = [f for f in input_files if 'training' not in f.name]

    if len(input_files) == 0:
        print(f"ERROR: No *_boughter.csv files found in {BOUGHTER_DIR}")
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
        output_path = BOUGHTER_DIR / output_name

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
```

---

### Step 2: Validation Script

**File:** `preprocessing/boughter/validate_stage4.py`

**Purpose:** Verify strict QC filtering worked correctly

**Checks:**
1. Confirm no X in any sequence
2. Confirm no non-standard amino acids
3. Verify sequence count (~852 expected)
4. Check all 16 fragments have same count
5. Verify label distribution remains balanced

**Implementation:**
```python
#!/usr/bin/env python3
"""
Validation for Stage 4: Additional QC Filtering
==============================================

Validates that strict QC filtering was applied correctly to all fragment files.

Date: 2025-11-04
"""

import pandas as pd
from pathlib import Path

BOUGHTER_DIR = Path("train_datasets/boughter")
STRICT_QC_PATTERN = "*_strict_qc.csv"
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

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

        # Check 4: Label distribution (if VH_only or Full)
        if 'label' in df.columns and fragment_name in ['VH_only', 'Full']:
            label_counts = df['label'].value_counts().sort_index()
            print(f"  ℹ️  Label distribution: {dict(label_counts)}")

        print()

    # Check 5: All fragments have same count
    print("=" * 80)
    print("CROSS-FRAGMENT VALIDATION")
    print("=" * 80)

    unique_counts = set(sequence_counts.values())
    if len(unique_counts) == 1:
        count = list(unique_counts)[0]
        print(f"✅ All fragments have same sequence count: {count}")
    else:
        print(f"❌ FAIL: Fragments have different sequence counts:")
        for name, count in sorted(sequence_counts.items()):
            print(f"   {name}: {count}")
        all_valid = False

    # Check 6: Expected count (~852)
    expected_min = 840
    expected_max = 865
    actual_count = list(sequence_counts.values())[0]

    print()
    if expected_min <= actual_count <= expected_max:
        print(f"✅ Sequence count in expected range: {actual_count} (expected ~852)")
    else:
        print(f"⚠️  WARNING: Sequence count {actual_count} outside expected range ({expected_min}-{expected_max})")

    # Final result
    print()
    print("=" * 80)
    if all_valid:
        print("✅ ALL VALIDATION CHECKS PASSED")
    else:
        print("❌ VALIDATION FAILED - See errors above")
    print("=" * 80)

    return all_valid


if __name__ == "__main__":
    success = validate_strict_qc()
    exit(0 if success else 1)
```

---

### Step 3: Update Documentation

**Files to update:**

1. **train_datasets/boughter/README.md**
   - Document two QC levels: Boughter QC (1,065) vs Strict QC (~852)
   - Explain file naming convention
   - List all available variants

2. **train_datasets/BOUGHTER_DATA_PROVENANCE.md**
   - Add Stage 4 to pipeline documentation
   - Explain additional QC rationale
   - Update sequence count tables

3. **preprocessing/boughter/README.md**
   - Add stage4_additional_qc.py documentation
   - Update pipeline diagram
   - Explain when to use strict QC vs base QC

---

## QC Level Comparison

### Boughter QC (Current - 914 training sequences)

**What it includes:**
- ✅ X filtering in CDRs only (L1, L2, L3, H1, H2, H3)
- ✅ Empty CDR filtering
- ✅ Flagging (0 and 4+ flags only, exclude 1-3)

**What it doesn't include:**
- ❌ X filtering in full sequence (Frameworks 1-4)
- ❌ Non-standard AA filtering (B, Z, J, U, O)
- ❌ Length outlier filtering

**Files:**
- `*_boughter.csv` (1,065 sequences, has `include_in_training` column)
- `VH_only_boughter_training.csv` (914 sequences, flattened export)

**Source:** Boughter et al. 2020, seq_loader.py lines 10-33

**Use case:** Exact replication of Boughter's published methodology

---

### Strict QC (NEW - ~852 training sequences)

**What it includes:**
- ✅ All Boughter QC filters (above)
- ✅ **PLUS:** X filtering in FULL sequence (not just CDRs)
- ✅ **PLUS:** Non-standard AA filtering (B, Z, J, U, O)
- ⚠️ **Optional:** Length outlier filtering (needs review)

**Files:**
- `*_boughter_strict_qc.csv` (~852 sequences each)

**Source:** Industry standard practice + Novo Nordisk likely methodology

**Use case:** Matching Novo's likely QC + modern ML best practices

---

## Expected Outcomes

### Sequence Counts

```
Pipeline Stage                        Sequences    Cumulative Reduction
─────────────────────────────────────────────────────────────────────
Raw DNA FASTA (6 subsets)             1,171        -
Stage 1: DNA translation              1,171        0
Stage 2+3: ANARCI + Boughter QC       1,065        -106 (9.0%)
Training filter (0 and 4+ flags)        914        -151 (12.9% of 1,065)
Stage 4: Additional QC                 ~852        -62 (6.8% of 914)
─────────────────────────────────────────────────────────────────────
Total reduction (raw → strict QC)                  -319 (27.2%)
```

### Model Performance Hypothesis

**Current (Boughter QC, 914 sequences):**
- Accuracy: 67.5% ± 8.9% (10-fold CV)

**Expected (Strict QC, ~852 sequences):**
- Accuracy: ~71% (hypothesis: match Novo's reported performance)
- Reasoning: Remove noisy sequences with X in frameworks

**Statistical note:**
- Current gap: 3.5 percentage points (0.4 standard deviations)
- Not statistically significant, but consistent trend
- Novo likely used similar strict QC

---

## Next Steps (Implementation Order)

### Phase 1: Core Filtering (Priority 1)
1. ✅ Create this plan document (DONE)
2. ✅ Senior audit validation (DONE)
3. 🔧 Create `preprocessing/boughter/stage4_additional_qc.py`
4. 🔧 Run Stage 4 on all 16 fragment CSVs
5. 🔧 Create and run `preprocessing/boughter/validate_stage4.py`

### Phase 2: Documentation (Priority 2)
6. 🔧 Update `train_datasets/boughter/README.md`
7. 🔧 Update `train_datasets/BOUGHTER_DATA_PROVENANCE.md`
8. 🔧 Update `preprocessing/boughter/README.md`

### Phase 3: Model Training (Priority 3)
9. 🔧 Train model on strict QC dataset (~852 sequences)
10. 🔧 Compare performance: Boughter QC (914) vs Strict QC (~852)
11. 🔧 Analyze results and update documentation

---

## Key Principles

1. ✅ **DO NOT modify existing files/scripts** - only extend the pipeline
2. ✅ **Use clear naming conventions** - `_strict_qc` suffix for additional QC
3. ✅ **Maintain parallel structure** - all 16 fragments have both variants
4. ✅ **Work from source CSVs** - use `*_boughter.csv` (with metadata), NOT flattened exports
5. ✅ **Preserve all columns** - keep id, subset, flags, source for full provenance
6. ✅ **Validate thoroughly** - confirm filtering worked correctly across all fragments

---

## References

**Discovery:**
- QC audit script: `scripts/audit_boughter_training_qc.py`
- Audit results: 62/914 sequences with X (46 at position 0, 16 elsewhere)

**Methodology clarification:**
- Master doc: `docs/boughter/BOUGHTER_NOVO_METHODOLOGY_CLARIFICATION.md`
- Boughter code: https://github.com/ctboughter/AIMS_manuscripts/blob/main/seq_loader.py

**Related documentation:**
- `docs/boughter/BOUGHTER_NOVO_REPLICATION_ANALYSIS.md`
- `docs/boughter/cdr_boundary_first_principles_audit.md`
- `docs/boughter/boughter_data_sources.md`

---

**Document Status:**
- **Version:** 2.0 (CORRECTED after senior audit)
- **Date:** 2025-11-04
- **Status:** ✅ Ready for implementation
- **Audit:** Validated against actual codebase structure
- **Next Action:** Create `preprocessing/boughter/stage4_additional_qc.py`
