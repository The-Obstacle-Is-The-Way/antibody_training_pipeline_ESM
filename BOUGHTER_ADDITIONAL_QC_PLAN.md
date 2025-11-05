# Boughter Dataset: Additional QC Filtering Plan

**Date:** 2025-11-04
**Status:** 🔧 **IMPLEMENTATION PLAN**
**Branch:** leroy-jenkins/boughter-clean

---

## Executive Summary

**Problem:** Our training set has 62/914 sequences (6.8%) with X at position 0 of full VH sequence. These pass Boughter's QC (which only checks CDRs) but likely fail Novo's additional industry-standard filtering.

**Solution:** Add a **Stage 4: Additional QC** step to filter these sequences AFTER existing pipeline.

**Key Principle:** **DO NOT modify existing files/scripts** - only EXTEND the pipeline with new filtering stage.

---

## Current Pipeline State

### Existing Files (DO NOT MODIFY)

```
preprocessing/boughter/
├── stage1_dna_translation.py          ✅ Keep as-is
├── stage2_stage3_annotation_qc.py     ✅ Keep as-is
├── validate_stage1.py                 ✅ Keep as-is
└── validate_stages2_3.py              ✅ Keep as-is

train_datasets/boughter_raw/
├── stage1_translated.csv              ✅ Keep as-is (1,171 sequences)
└── stage2_stage3_annotated_qc.csv     ✅ Keep as-is (1,065 sequences)

train_datasets/boughter/
├── VH_only_boughter.csv               ✅ Keep as-is (1,065 sequences, all flags)
├── VH_only_boughter_training.csv      ✅ Keep as-is (914 sequences, 0 and 4+ flags)
└── README.md                          ✅ Keep as-is
```

### Current Pipeline Flow

```
Raw DNA (1,171 sequences)
   ↓
Stage 1: DNA → Protein translation
   ↓ preprocessing/boughter/stage1_dna_translation.py
   ↓ Output: train_datasets/boughter_raw/stage1_translated.csv
   ↓
Stage 2+3: ANARCI annotation + Boughter QC
   ↓ preprocessing/boughter/stage2_stage3_annotation_qc.py
   ↓ QC: X in CDRs, empty CDRs
   ↓ Output: train_datasets/boughter_raw/stage2_stage3_annotated_qc.csv (1,065 sequences)
   ↓
Fragment Generation + Flagging
   ↓ (integrated into stage2_stage3_annotation_qc.py)
   ↓ Output: train_datasets/boughter/VH_only_boughter.csv (all 1,065 sequences)
   ↓ Output: train_datasets/boughter/VH_only_boughter_training.csv (914 sequences, 0 and 4+ flags only)
```

**Status:** ✅ This pipeline is CORRECT and matches Boughter's methodology exactly!

---

## What We Discovered (2025-11-04)

### QC Audit Results

**Script:** `scripts/audit_boughter_training_qc.py`

**Finding:** 62/914 sequences (6.8%) have X at position 0 of full VH sequence

**Why Boughter's QC didn't filter them:**
```python
# Boughter's seq_loader.py only checks CDRs:
total_abs[~total_abs['cdrH1_aa'].str.contains("X")]  # CDR-H1
total_abs[~total_abs['cdrH2_aa'].str.contains("X")]  # CDR-H2
total_abs[~total_abs['cdrH3_aa'].str.contains("X")]  # CDR-H3
# Does NOT check full VH sequence!

# Position 0 is in Framework 1 (NOT in any CDR)
# Therefore: sequences with X at position 0 PASS Boughter's QC ✅
```

**Why Novo likely filtered them:**
- Industry standard: filter X ANYWHERE in sequence (not just CDRs)
- ESM embedding models expect valid amino acids at all positions
- Professional QC practice in pharma/biotech

**Impact on accuracy:**
- Our accuracy: 67.5% ± 8.9% (914 sequences)
- Novo accuracy: 71%
- Gap: 3.5 percentage points (within 0.4 standard deviations)
- Hypothesis: Filtering these 62 sequences will close the gap

---

## What We Need to Add

### NEW Stage 4: Additional QC (Industry Standard Full-Sequence Filtering)

**Purpose:** Apply industry-standard QC filters beyond Boughter's CDR-only checks

**Input:** `train_datasets/boughter/VH_only_boughter_training.csv` (914 sequences)

**Filters to Apply:**
1. ✅ **Filter X in full VH sequence** (not just CDRs)
2. ✅ **Filter other non-standard amino acids** (B, Z, J, U, O) if present
3. ⚠️ **Optional:** Filter extreme length outliers (>2.5 SD from mean) - REVIEW FIRST

**Output:** `train_datasets/boughter/VH_only_boughter_training_strict_qc.csv` (~852 sequences)

---

## File Naming Convention

### Philosophy

- **Base files** (Boughter QC only): `*_boughter.csv` or `*_boughter_training.csv`
- **Additional QC files**: `*_boughter_training_strict_qc.csv` or `*_boughter_training_clean.csv`
- **Fragments with additional QC**: `*_boughter_training_strict_qc.csv` (e.g., `H-CDR3_boughter_training_strict_qc.csv`)

### Naming Pattern

```
{fragment}_{dataset}_{split}_{qc_level}.csv

Where:
- fragment: VH_only, H-CDR3, All-CDRs, etc.
- dataset: boughter
- split: training, test (optional)
- qc_level: (none) = Boughter QC only
            strict_qc = Boughter QC + industry standard
            clean = same as strict_qc (alternative name)
```

### Examples

**Current files (keep as-is):**
```
train_datasets/boughter/VH_only_boughter.csv                    # All 1,065 sequences
train_datasets/boughter/VH_only_boughter_training.csv           # 914 sequences (0 and 4+ flags)
train_datasets/boughter/H-CDR3_boughter_training.csv            # 914 CDR-H3 sequences
train_datasets/boughter/All-CDRs_boughter_training.csv          # 914 concatenated CDRs
```

**NEW files (to be created):**
```
train_datasets/boughter/VH_only_boughter_training_strict_qc.csv        # ~852 sequences (X filtered)
train_datasets/boughter/H-CDR3_boughter_training_strict_qc.csv         # ~852 CDR-H3 sequences
train_datasets/boughter/All-CDRs_boughter_training_strict_qc.csv       # ~852 concatenated CDRs
train_datasets/boughter/Full_boughter_training_strict_qc.csv           # ~852 full VH+VL
```

---

## Directory Structure

### Current Structure (DO NOT MODIFY)

```
train_datasets/
├── boughter_raw/                              # Intermediate processing files
│   ├── stage1_translated.csv                  # After DNA translation (1,171 seqs)
│   └── stage2_stage3_annotated_qc.csv         # After ANARCI + Boughter QC (1,065 seqs)
│
├── boughter/                                  # Final training datasets
│   ├── README.md                              # Dataset documentation
│   ├── VH_only_boughter.csv                   # All sequences (1,065)
│   ├── VH_only_boughter_training.csv          # 0 and 4+ flags (914)
│   ├── H-CDR3_boughter_training.csv           # CDR-H3 fragments (914)
│   ├── All-CDRs_boughter_training.csv         # All CDRs concatenated (914)
│   └── ... (other fragments)
│
└── BOUGHTER_DATA_PROVENANCE.md                # Dataset source documentation
```

### NEW Structure (to be added)

```
train_datasets/
├── boughter/
│   ├── README.md                              # UPDATE: document strict_qc variants
│   │
│   ├── VH_only_boughter.csv                   # ✅ Existing (1,065 seqs)
│   ├── VH_only_boughter_training.csv          # ✅ Existing (914 seqs, Boughter QC)
│   ├── VH_only_boughter_training_strict_qc.csv    # 🆕 NEW (~852 seqs, + industry QC)
│   │
│   ├── H-CDR3_boughter_training.csv           # ✅ Existing (914 seqs)
│   ├── H-CDR3_boughter_training_strict_qc.csv     # 🆕 NEW (~852 seqs)
│   │
│   ├── All-CDRs_boughter_training.csv         # ✅ Existing (914 seqs)
│   ├── All-CDRs_boughter_training_strict_qc.csv   # 🆕 NEW (~852 seqs)
│   │
│   └── ... (other fragment variants)
│
└── BOUGHTER_DATA_PROVENANCE.md                # UPDATE: document additional QC stage
```

---

## Implementation Steps

### Step 1: Create Stage 4 QC Script

**File:** `preprocessing/boughter/stage4_additional_qc.py`

**Purpose:** Apply industry-standard full-sequence QC filters

**Input:** `train_datasets/boughter/VH_only_boughter_training.csv` (914 sequences)

**Filters:**
1. Remove X anywhere in full VH sequence (not just CDRs)
2. Remove non-standard amino acids (B, Z, J, U, O) if present
3. Optional: Review and potentially remove extreme length outliers

**Output:** `train_datasets/boughter/VH_only_boughter_training_strict_qc.csv` (~852 sequences)

**Pseudocode:**
```python
#!/usr/bin/env python3
"""
Stage 4: Additional QC Filtering for Boughter Training Set
===========================================================

Purpose: Apply industry-standard full-sequence QC beyond Boughter's CDR-only checks

Input: train_datasets/boughter/VH_only_boughter_training.csv (914 sequences)
Output: train_datasets/boughter/VH_only_boughter_training_strict_qc.csv (~852 sequences)

Filters:
1. Remove X anywhere in full VH sequence
2. Remove non-standard amino acids (B, Z, J, U, O)
3. Optional: Remove extreme length outliers (after manual review)

Date: 2025-11-04
"""

import pandas as pd
from pathlib import Path

# Paths
INPUT_FILE = Path("train_datasets/boughter/VH_only_boughter_training.csv")
OUTPUT_FILE = Path("train_datasets/boughter/VH_only_boughter_training_strict_qc.csv")

# Standard amino acids
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

def filter_x_in_sequence(df, seq_col='sequence'):
    """Remove sequences with X anywhere (not just CDRs)"""
    before = len(df)
    df_filtered = df[~df[seq_col].str.contains('X', na=False)]
    removed = before - len(df_filtered)
    print(f"Removed {removed} sequences with X in full sequence")
    return df_filtered

def filter_non_standard_aa(df, seq_col='sequence'):
    """Remove sequences with non-standard amino acids (B, Z, J, U, O)"""
    def has_non_standard(seq):
        if pd.isna(seq):
            return False
        return any(aa not in STANDARD_AA for aa in str(seq))

    before = len(df)
    df_filtered = df[~df[seq_col].apply(has_non_standard)]
    removed = before - len(df_filtered)
    print(f"Removed {removed} sequences with non-standard amino acids")
    return df_filtered

def main():
    print("Stage 4: Additional QC Filtering")
    print("=" * 80)

    # Load data
    df = pd.read_csv(INPUT_FILE, comment='#')
    print(f"Input: {len(df)} sequences from {INPUT_FILE}")

    # Determine sequence column
    seq_col = 'sequence' if 'sequence' in df.columns else 'heavy_seq'

    # Apply filters
    df = filter_x_in_sequence(df, seq_col)
    df = filter_non_standard_aa(df, seq_col)

    # Save output
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nOutput: {len(df)} sequences saved to {OUTPUT_FILE}")
    print(f"Removed: {914 - len(df)} sequences total")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Original (Boughter QC): 914 sequences")
    print(f"After strict QC: {len(df)} sequences")
    print(f"Reduction: {914 - len(df)} sequences ({(914 - len(df)) / 914 * 100:.1f}%)")

if __name__ == "__main__":
    main()
```

---

### Step 2: Generate All Fragment Variants with Strict QC

**File:** `preprocessing/boughter/generate_strict_qc_fragments.py`

**Purpose:** Generate all 16 fragment types from the strict QC filtered dataset

**Input:** `train_datasets/boughter/VH_only_boughter_training_strict_qc.csv`

**Outputs:** All fragment variants with `_strict_qc` suffix
- `H-CDR3_boughter_training_strict_qc.csv`
- `All-CDRs_boughter_training_strict_qc.csv`
- `Full_boughter_training_strict_qc.csv`
- etc. (all 16 fragments from Novo Table 4)

**Note:** This can likely be a simple wrapper around existing fragment generation logic, just reading from the strict QC file instead of the base training file.

---

### Step 3: Update Documentation

**Files to update:**

1. **train_datasets/boughter/README.md**
   - Document the two QC levels (Boughter vs strict)
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

### Step 4: Validation

**File:** `preprocessing/boughter/validate_stage4.py`

**Purpose:** Verify strict QC filtering worked correctly

**Checks:**
1. Confirm no X in any full VH sequence
2. Confirm no non-standard amino acids
3. Verify sequence count (~852 expected)
4. Check label distribution (should remain balanced)
5. Validate fragment integrity (all fragments match base sequences)

---

## QC Level Comparison

### Boughter QC (Current - 914 sequences)

**What it includes:**
- ✅ X filtering in CDRs only (L1, L2, L3, H1, H2, H3)
- ✅ Empty CDR filtering
- ✅ Flagging (0 and 4+ flags only, exclude 1-3)

**What it doesn't include:**
- ❌ X filtering in full sequence (Framework 1-4)
- ❌ Non-standard AA filtering (B, Z, J, U, O)
- ❌ Length outlier filtering

**Source:** Boughter et al. 2020, seq_loader.py lines 10-33

**Use case:** Exact replication of Boughter's published methodology

---

### Strict QC (NEW - ~852 sequences)

**What it includes:**
- ✅ All Boughter QC filters (above)
- ✅ **PLUS:** X filtering in FULL VH sequence
- ✅ **PLUS:** Non-standard AA filtering (B, Z, J, U, O)
- ⚠️ **Optional:** Length outlier filtering (needs review)

**Source:** Industry standard practice + Novo Nordisk likely methodology

**Use case:** Matching Novo's likely QC + modern ML best practices

---

## Expected Outcomes

### Sequence Counts

```
Pipeline Stage                        Sequences    Cumulative Reduction
─────────────────────────────────────────────────────────────────────
Raw DNA (6 subsets)                   1,171        -
Stage 1: DNA translation              1,171        0
Stage 2+3: ANARCI + Boughter QC       1,065        -106 (9.0%)
Flagging (0 and 4+ only)                914        -151 (12.9%)
Stage 4: Additional QC                 ~852        -62 (6.8% of 914)
─────────────────────────────────────────────────────────────────────
Total reduction (raw → strict QC)                  -319 (27.2%)
```

### Model Performance Hypothesis

**Current (Boughter QC, 914 sequences):**
- Accuracy: 67.5% ± 8.9% (10-fold CV)

**Expected (Strict QC, ~852 sequences):**
- Accuracy: ~71% (hypothesis: match Novo's reported performance)
- Reasoning: Remove noisy sequences with X at position 0

**Statistical note:**
- Current gap: 3.5 percentage points (0.4 standard deviations)
- Not statistically significant, but consistent trend

---

## File Overview

### New Files to Create

**Scripts:**
1. `preprocessing/boughter/stage4_additional_qc.py` - Main filtering script
2. `preprocessing/boughter/generate_strict_qc_fragments.py` - Fragment generation
3. `preprocessing/boughter/validate_stage4.py` - Validation checks

**Data files:**
1. `train_datasets/boughter/VH_only_boughter_training_strict_qc.csv` - Main filtered dataset
2. `train_datasets/boughter/H-CDR3_boughter_training_strict_qc.csv` - CDR-H3 fragment
3. `train_datasets/boughter/All-CDRs_boughter_training_strict_qc.csv` - All CDRs concatenated
4. ... (up to 16 fragment variants)

**Documentation updates:**
1. `train_datasets/boughter/README.md` - Dataset documentation
2. `train_datasets/BOUGHTER_DATA_PROVENANCE.md` - Provenance tracking
3. `preprocessing/boughter/README.md` - Pipeline documentation

---

## Testing Strategy

### Unit Tests

1. **test_stage4_qc.py** - Test filtering logic
   - Verify X detection in full sequence
   - Verify non-standard AA detection
   - Test edge cases (empty sequences, all-X sequences, etc.)

2. **test_fragment_generation.py** - Test fragment consistency
   - Verify fragments match base sequences
   - Check all 16 fragment types
   - Validate sequence counts

### Integration Tests

1. **End-to-end pipeline test:**
   - Run full pipeline from raw DNA → strict QC
   - Verify all intermediate files
   - Check final sequence counts

2. **Comparison test:**
   - Compare Boughter QC (914) vs Strict QC (~852)
   - Verify only X/non-standard sequences removed
   - Check label distribution remains balanced

---

## Next Steps (Implementation Order)

### Phase 1: Core Filtering (Priority 1)
1. ✅ Create this plan document (DONE)
2. 🔧 Create `preprocessing/boughter/stage4_additional_qc.py`
3. 🔧 Run Stage 4 QC on 914 training sequences
4. 🔧 Validate output (~852 sequences expected)

### Phase 2: Fragment Generation (Priority 2)
5. 🔧 Create `preprocessing/boughter/generate_strict_qc_fragments.py`
6. 🔧 Generate all 16 fragment variants with `_strict_qc` suffix
7. 🔧 Validate fragment integrity

### Phase 3: Documentation (Priority 3)
8. 🔧 Update `train_datasets/boughter/README.md`
9. 🔧 Update `train_datasets/BOUGHTER_DATA_PROVENANCE.md`
10. 🔧 Update `preprocessing/boughter/README.md`

### Phase 4: Testing (Priority 4)
11. 🔧 Create `preprocessing/boughter/validate_stage4.py`
12. 🔧 Write unit tests for filtering logic
13. 🔧 Run end-to-end pipeline validation

### Phase 5: Model Training (Priority 5)
14. 🔧 Train model on strict QC dataset (~852 sequences)
15. 🔧 Compare performance: Boughter QC (914) vs Strict QC (~852)
16. 🔧 Analyze results and update documentation

---

## Key Principles

1. ✅ **DO NOT modify existing files/scripts** - only extend the pipeline
2. ✅ **Use clear naming conventions** - `_strict_qc` suffix for additional QC
3. ✅ **Maintain parallel structure** - all fragments have both variants
4. ✅ **Document everything** - update all relevant README/provenance docs
5. ✅ **Validate thoroughly** - confirm filtering worked correctly
6. ✅ **Preserve reproducibility** - keep both QC levels available

---

## References

**Discovery:**
- QC audit script: `scripts/audit_boughter_training_qc.py`
- Audit results: 62/914 sequences with X at position 0

**Methodology clarification:**
- Master doc: `docs/boughter/BOUGHTER_NOVO_METHODOLOGY_CLARIFICATION.md`
- Boughter code: https://github.com/ctboughter/AIMS_manuscripts/blob/main/seq_loader.py

**Related documentation:**
- `docs/boughter/BOUGHTER_NOVO_REPLICATION_ANALYSIS.md`
- `docs/boughter/cdr_boundary_first_principles_audit.md`
- `docs/boughter/boughter_data_sources.md`

---

**Document Status:**
- **Version:** 1.0
- **Date:** 2025-11-04
- **Status:** 📋 Ready for implementation
- **Next Action:** Create `preprocessing/boughter/stage4_additional_qc.py`
