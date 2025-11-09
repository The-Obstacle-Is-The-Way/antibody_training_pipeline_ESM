# Boughter Dataset - P0 Blocker Fix Report

**Date:** 2025-11-02
**Branch:** ray/learning
**Issue:** Boughter dataset preprocessing P0 blocker
**Status:** ✅ **P0 BLOCKER RESOLVED**

---

## Executive Summary

The Boughter dataset processing script had the **EXACT SAME P0 BLOCKER** as Harvey/Shehata PLUS an additional constant region contamination issue:

1. **Gap Character Bug**: Using `annotation.sequence_alignment_aa` (IMGT-aligned WITH gap characters) instead of `annotation.sequence_aa` (raw sequence WITHOUT gaps)
2. **Constant Region Contamination**: Using `annotation.sequence_aa` (full translated DNA including constant region with stop codons) instead of reconstructing V-domain from fragments

**Impact:**
- 13 sequences (1.2%) in fragment CSV files contained gap characters `-`, causing ESM-1v embedding validation to crash
- 241 sequences (22.6%) contained stop codons `*` from constant region, also causing ESM validation failure

**Fix:** V-domain reconstruction from ANARCI fragments (FWR1+CDR1+FWR2+CDR2+FWR3+CDR3+FWR4)

**Status:** ✅ All 1,065 sequences now gap-free and ESM-1v compatible

**Tests:** ✅ 5/5 comprehensive tests passing

---

## P0 Blocker Details

### The Double Bug

**File:** `preprocessing/boughter/stage2_stage3_annotation_qc.py`

**Bug 1 (Line 89 - Original Code):** Using wrong ANARCI attribute

```python
# WRONG (original code)
fragments = {
    f"full_seq_{chain}": safe_str(annotation.sequence_alignment_aa),  # IMGT-aligned with gaps
    ...
}
```

**Bug 2 (First Fix Attempt):** Still using full translated DNA with constant region

```python
# STILL WRONG (first fix attempt)
fragments = {
    f"full_seq_{chain}": safe_str(annotation.sequence_aa),  # Gap-free but includes constant region
    ...
}
```

**Solution (Final Fix):** Reconstruct V-domain from annotated fragments

```python
# CORRECT (final fix)
# Extract individual fragments
fragments = {
    f"fwr1_aa_{chain}": safe_str(annotation.fwr1_aa),
    f"cdr1_aa_{chain}": safe_str(annotation.cdr1_aa),
    f"fwr2_aa_{chain}": safe_str(annotation.fwr2_aa),
    f"cdr2_aa_{chain}": safe_str(annotation.cdr2_aa),
    f"fwr3_aa_{chain}": safe_str(annotation.fwr3_aa),
    f"cdr3_aa_{chain}": safe_str(annotation.cdr3_aa),
    f"fwr4_aa_{chain}": safe_str(annotation.fwr4_aa),
}

# Reconstruct full V-domain from fragments (avoids constant region garbage)
# This is gap-free and clean (P0 fix + constant region removal)
fragments[f"full_seq_{chain}"] = "".join([
    fragments[f"fwr1_aa_{chain}"],
    fragments[f"cdr1_aa_{chain}"],
    fragments[f"fwr2_aa_{chain}"],
    fragments[f"cdr2_aa_{chain}"],
    fragments[f"fwr3_aa_{chain}"],
    fragments[f"cdr3_aa_{chain}"],
    fragments[f"fwr4_aa_{chain}"],
])
```

### Root Cause

The `riot_na` library (ANARCI wrapper) provides multiple sequence attributes:

- `annotation.sequence_alignment_aa`: IMGT-numbered alignment WITH gaps (`-` characters) - **WRONG for ESM**
- `annotation.sequence_aa`: Raw amino acid sequence WITHOUT gaps - **WRONG for Boughter (includes constant region)**
- `annotation.fwr*_aa`, `annotation.cdr*_aa`: Individual fragments (gap-free, V-domain only) - **CORRECT**

**Why Boughter is Different:**
- Harvey/Shehata: Input sequences were already clean V-domains → `sequence_aa` works
- Boughter: Input sequences are full DNA translations (signal peptide + V-domain + constant region) → `sequence_aa` includes garbage

**ESM-1v requirement:** Only accepts valid amino acids `"ACDEFGHIKLMNPQRSTVWYX"` (no `-` gap character, no `*` stop codon)

**Result:** The IMGT-aligned sequence with gaps OR the full DNA translation with stop codons both cause ESM-1v to reject input during validation (model.py:86-90).

---

## Impact Assessment

### Before Fix (Original Processing)

**Generated:** 2025-11-01 (original run)
**Source:** `preprocessing/boughter/stage2_stage3_annotation_qc.py` with `sequence_alignment_aa`

**Gap Character Contamination:**

| File | Sequences | Gaps | Gap % |
|------|-----------|------|-------|
| VH_only_boughter.csv | 1,065 | **11** | **1.0%** |
| VL_only_boughter.csv | 1,065 | **2** | **0.2%** |
| Full_boughter.csv | 1,065 | **13** | **1.2%** |
| All CDR/FWR files | 1,065 | 0 | 0% |

**Stop Codon Contamination (after first fix attempt):**

| File | Sequences | Stop Codons | Stop % |
|------|-----------|-------------|--------|
| VH_only_boughter.csv | 1,065 | **157** | **14.7%** |
| VL_only_boughter.csv | 1,065 | **226** | **21.2%** |
| Full_boughter.csv | 1,065 | **241** | **22.6%** |

**Example of contaminated sequence:**
```
ID: flu_000138
Sequence: RLQLQESGPGLVKPSETLSLTCTASGGSVNSGGYYWGWIR...WSSAQDWLNAXYSAXSTXSXHR
                                                     ^
                                                     Stop codon at position 275
                                                     Followed by X's (unknown amino acids)
```

**Critical:** Only VH/VL/Full were affected. CDR/FWR fragments were clean because they use `.cdr*_aa` and `.fwr*_aa` attributes, which are V-domain-only by design.

### After Fix (Regenerated with V-Domain Reconstruction)

**Generated:** 2025-11-02
**Source:** `preprocessing/boughter/stage2_stage3_annotation_qc.py` with fragment concatenation

| File | Sequences | Gaps | Stop Codons | Status |
|------|-----------|------|-------------|--------|
| VH_only_boughter.csv | 1,065 | **0** | **0** | ✅ |
| VL_only_boughter.csv | 1,065 | **0** | **0** | ✅ |
| Full_boughter.csv | 1,065 | **0** | **0** | ✅ |
| All CDR files | 1,065 | 0 | 0 | ✅ |
| All FWR files | 1,065 | 0 | 0 | ✅ |

**Result:** ✅ **All 1,065 sequences are now gap-free, stop-free, and ESM-1v compatible**

---

## Data Source Clarification

**Source:** Official Boughter repository + AIMS_manuscripts
**Repo:** `ctboughter/AIMS_manuscripts`
**Location:** `app_data/full_sequences/`

**Files:**
- 6 subsets: flu, mouse_iga, hiv_nat, hiv_cntrl, hiv_plos, gut_hiv
- DNA FASTA files (paired heavy/light chains)
- Polyreactivity flag files (NumReact 0-7 or Y/N format)

**Total:** 1,171 raw DNA sequences → 1,117 after translation → 1,110 after ANARCI → 1,065 after QC (90.9% overall retention)

**Processing Pipeline:**
- **Stage 1:** DNA translation (hybrid: direct-first, ATG fallback)
- **Stage 2:** ANARCI annotation (IMGT numbering)
- **Stage 3:** Post-annotation QC (filter X in CDRs, empty CDRs)

---

## Fix Implementation

### Step 1: Apply P0 Fix

**File:** `preprocessing/boughter/stage2_stage3_annotation_qc.py:93-147`

Changed from:
```python
f"full_seq_{chain}": safe_str(annotation.sequence_aa),
```

To:
```python
# Extract individual fragments
fragments = {
    f"fwr1_aa_{chain}": safe_str(annotation.fwr1_aa),
    ... # all individual fragments
}

# Reconstruct full V-domain from fragments
fragments[f"full_seq_{chain}"] = "".join([...])
```

### Step 2: Regenerate All Boughter Fragments

```bash
python3 preprocessing/boughter/stage2_stage3_annotation_qc.py
```

**Runtime:** ~2-3 minutes
**Output:** 16 fragment CSV files (1,065 sequences each)

### Step 3: Validate Gap and Stop Codon Removal

```bash
python3 -c "
import pandas as pd

# Check for gaps
vh = pd.read_csv('train_datasets/boughter/annotated/VH_only_boughter.csv', comment='#')
print(f'Gaps: {vh[\"sequence\"].str.contains(\"-\", na=False).sum()}')

# Check for stop codons
print(f'Stop codons: {vh[\"sequence\"].str.contains(\"*\", regex=False, na=False).sum()}')
"
```

**Result:**
```
Gaps: 0
Stop codons: 0
```

### Step 4: Run Comprehensive Test Suite

```bash
python3 tests/test_boughter_embedding_compatibility.py
```

**Result:** 5/5 tests passed ✅

---

## Test Suite Results

### Test 1: Gap Character Detection
✅ **PASS** - All 16 fragment files gap-free (1,065 sequences each)

### Test 2: Amino Acid Validation
✅ **PASS** - All sequences contain only valid amino acids (5,325 sequences validated across 5 files)

### Test 3: Previously Affected Sequences
✅ **PASS** - Spot-checked 10 sequences (5 VH, 5 VL), all gap-free
- Before fix: VH had 11 gaps (1.0%), VL had 2 gaps (0.2%)
- After first fix: VH had 157 stop codons (14.7%), VL had 226 stop codons (21.2%)
- After final fix: 0 gaps, 0 stop codons (0%)

### Test 4: ESM Model Validation Simulation
✅ **PASS** - All 3,195 sequences passed model.py:86-90 validation logic
- VH_only: 1,065/1,065 ✅
- VL_only: 1,065/1,065 ✅
- Full: 1,065/1,065 ✅

### Test 5: Data Integrity
✅ **PASS** - All 16 files present with 1,065 rows
- Label distribution: 48.5% specific, 51.5% non-specific (balanced ✓)
- Training set: 914 sequences (443 specific + 471 non-specific)
- Excluded (1-3 flags): 151 sequences

---

## Comparison with Harvey/Shehata Fixes

| Aspect | Shehata | Harvey | **Boughter** |
|--------|---------|--------|--------------|
| **Bug Location** | preprocessing/shehata/step2_extract_fragments.py:63 | preprocessing/harvey/process_harvey.py:48 | preprocessing/boughter/stage2_stage3_annotation_qc.py:93-147 |
| **Bug Type** | sequence_alignment_aa (gaps) | sequence_alignment_aa (gaps) | sequence_alignment_aa (gaps) + sequence_aa (constant region) |
| **Fix** | `→ sequence_aa` (gap-free) | `→ sequence_aa` (gap-free) | `→ fragment concatenation` (V-domain only) |
| **Affected Files** | VH_only, VL_only | VHH_only | VH_only, VL_only, Full |
| **Gap Impact** | 100% (398/398) | 8.6% (12,116/141,021) | **1.2% (13/1,065)** |
| **Stop Codon Impact** | N/A (protein input) | N/A (protein input) | **22.6% (241/1,065)** |
| **Test Suite** | ✅ test_shehata_embedding_compatibility.py | ✅ test_harvey_embedding_compatibility.py | ✅ test_boughter_embedding_compatibility.py |
| **Test Results** | 5/5 passed ✅ | 5/5 passed ✅ | **5/5 passed ✅** |

**Key Difference:** Boughter required a more sophisticated fix because the input data is DNA (not protein), so `sequence_aa` returns the full translation including constant region garbage (stop codons, X's). The solution was to reconstruct the V-domain by concatenating ANARCI's annotated fragments.

---

## Files Modified

### Code Changes
1. ✅ `preprocessing/boughter/stage2_stage3_annotation_qc.py:93-147` - P0 fix applied (V-domain reconstruction)
2. ✅ `tests/integration/test_boughter_embedding_compatibility.py` - New test suite created

### Data Regenerated
3. ✅ `train_datasets/boughter/annotated/VH_only_boughter.csv` - 11 gaps + 157 stop codons → 0
4. ✅ `train_datasets/boughter/annotated/VL_only_boughter.csv` - 2 gaps + 226 stop codons → 0
5. ✅ `train_datasets/boughter/annotated/Full_boughter.csv` - 13 gaps + 241 stop codons → 0
6. ✅ All 13 other fragment files regenerated (already clean)

### Documentation
7. ✅ `docs/boughter/p0_fix_report.md` - This report
8. ✅ `docs/boughter/archive/boughter_processing_status.md` - Archived (pre-P0 fix snapshot)
9. ✅ `docs/boughter/archive/accuracy_verification_report.md` - Archived (pre-P0 fix snapshot)

---

## Lessons Learned

### Why This Happened

1. **Input Data Difference**: Boughter uses DNA sequences (requiring translation), while Harvey/Shehata use protein sequences
2. **ANARCI Attribute Confusion**: `sequence_aa` means different things depending on input:
   - For clean V-domain protein input: Returns same sequence without gaps ✓
   - For full DNA translation: Returns everything including constant region ✗
3. **Insufficient Testing**: Original validation scripts didn't check for stop codons or gap characters
4. **Silent Failure**: ANARCI doesn't warn about constant region contamination; validation only fails at ESM-1v embedding time

### Prevention

1. **First-principles validation:** Always check for gap characters AND stop codons after ANARCI annotation
2. **Input-aware processing:** DNA inputs require V-domain reconstruction from fragments, not direct sequence extraction
3. **Comprehensive test suites:** Include gap detection, stop codon detection, and ESM validation simulation for ALL fragment types
4. **Code review:** Verify fragment concatenation logic matches expected V-domain boundaries
5. **Documentation:** Clearly document the distinction between:
   - `sequence_alignment_aa` (with gaps, V-domain only)
   - `sequence_aa` (no gaps, but may include non-V-domain regions)
   - Fragment concatenation (no gaps, V-domain only - BEST for DNA inputs)

---

## References

- **Boughter Paper:** Boughter et al. 2020, eLife 9:e61393
- **Official Repo:** https://github.com/ctboughter/AIMS_manuscripts
- **Novo Nordisk Paper:** Sakhnini et al. 2025, bioRxiv (pending)
- **ANARCI:** Dunbar & Deane 2016, Bioinformatics
- **riot_na Library:** v4.0.5 (ANARCI Python wrapper)
- **ESM-1v Model:** facebook/esm1v_t33_650M_UR90S_1

---

**Report Generated:** 2025-11-02
**Branch:** ray/learning
**Commit:** [pending]
**Status:** ✅ Ready for final commit and PR

