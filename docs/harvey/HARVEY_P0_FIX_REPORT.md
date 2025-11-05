# Harvey Dataset - P0 Blocker Fix Report

**Date:** 2025-11-02
**Branch:** ray/learning → feat/harvey-preprocessing
**Issue:** #4 – Harvey dataset preprocessing
**Status:** ✅ **P0 BLOCKER RESOLVED**

---

## Executive Summary

The Harvey dataset processing script had the **EXACT SAME P0 BLOCKER** as Shehata: using `annotation.sequence_alignment_aa` (IMGT-aligned WITH gap characters) instead of `annotation.sequence_aa` (raw sequence WITHOUT gaps) for the full VHH sequence.

**Impact:** 12,116 sequences (8.6%) in `VHH_only_harvey.csv` contained gap characters `-`, causing ESM-1v embedding validation to crash.

**Fix:** One-line change in `preprocessing/process_harvey.py:48`
**Status:** ✅ All 141,021 sequences now gap-free
**Tests:** ✅ 5/5 comprehensive tests passing

---

## P0 Blocker Details

### The Bug

**File:** `preprocessing/process_harvey.py`
**Line:** 48
**Issue:** Using wrong attribute from riot_na annotation

```python
# BEFORE (with gaps - WRONG)
fragments = {
    "full_seq_H": annotation.sequence_alignment_aa,  # IMGT-aligned with gaps
    ...
}
```

```python
# AFTER (gap-free - CORRECT)
fragments = {
    "full_seq_H": annotation.sequence_aa,  # Raw sequence, no gaps (P0 fix)
    ...
}
```

### Root Cause

The `riot_na` library (ANARCI wrapper) provides two sequence attributes:
- `annotation.sequence_alignment_aa`: IMGT-numbered alignment WITH gaps (`-` characters)
- `annotation.sequence_aa`: Raw amino acid sequence WITHOUT gaps

**ESM-1v requirement:** Only accepts valid amino acids `"ACDEFGHIKLMNPQRSTVWYX"` (no `-` gap character)

**Result:** The IMGT-aligned sequence with gaps causes ESM-1v to reject the input during validation (model.py:86-90).

---

## Impact Assessment

### Before Fix (Original Processing)

**Generated:** 2025-11-01 (original run)
**Source:** `preprocessing/process_harvey.py` with `sequence_alignment_aa`

| File | Sequences | Gaps | Gap % |
|------|-----------|------|-------|
| VHH_only_harvey.csv | 141,021 | **12,116** | **8.6%** |
| H-CDR1_harvey.csv | 141,021 | 0 | 0% |
| H-CDR2_harvey.csv | 141,021 | 0 | 0% |
| H-CDR3_harvey.csv | 141,021 | 0 | 0% |
| H-CDRs_harvey.csv | 141,021 | 0 | 0% |
| H-FWRs_harvey.csv | 141,021 | 0 | 0% |

**Critical:** Only the full VHH sequence was affected because CDR/FWR fragments use `.cdr*_aa` and `.fwr*_aa` attributes, which are gap-free by design.

### After Fix (Regenerated with P0 Fix)

**Generated:** 2025-11-02
**Source:** `preprocessing/process_harvey.py` with `sequence_aa` (gap-free)

| File | Sequences | Gaps | Gap % |
|------|-----------|------|-------|
| VHH_only_harvey.csv | 141,021 | **0** | **0%** ✅ |
| H-CDR1_harvey.csv | 141,021 | 0 | 0% |
| H-CDR2_harvey.csv | 141,021 | 0 | 0% |
| H-CDR3_harvey.csv | 141,021 | 0 | 0% |
| H-CDRs_harvey.csv | 141,021 | 0 | 0% |
| H-FWRs_harvey.csv | 141,021 | 0 | 0% |

**Result:** ✅ **All 141,021 sequences are now gap-free and ESM-1v compatible**

---

## Data Source Clarification

**IMPORTANT:** Previous documentation incorrectly stated that the Harvey dataset came from HuggingFace `ZYMScott/polyreaction` (a Harvey + GP-nano combined dataset). This is **INCORRECT**.

### Actual Data Source (Correct)

**Source:** Official Harvey Lab GitHub Repository
**Repo:** `debbiemarkslab/nanobody-polyreactivity`
**Location:** `backend/app/experiments/`
**Files:**
- `high_polyreactivity_high_throughput.csv` (71,772 sequences)
- `low_polyreactivity_high_throughput.csv` (69,702 sequences)

**Total:** 141,474 sequences → 141,021 after ANARCI annotation (99.68% success rate)

**Conversion Script:** `scripts/conversion/convert_harvey_csvs.py`
- Extracts full sequences from IMGT position columns (1-128)
- Combines high/low CSVs with binary labels (0=low, 1=high)
- Outputs: `test_datasets/harvey/processed/harvey.csv`

### What ZYMScott/polyreaction Is (NOT Used)

Per NbBench paper (Zhang et al. 2025, arxiv:2505.02022):
- Harvey [52] + GP-nano [53] **COMBINED** dataset
- Created by NbBench team as curated benchmark
- **NOT** the pure Harvey 2022 dataset

**We did NOT use this.** Our data comes directly from the official Harvey repository.

---

## Fix Implementation

### Step 1: Apply P0 Fix

**File:** `preprocessing/process_harvey.py:48`

```diff
- "full_seq_H": annotation.sequence_alignment_aa,
+ "full_seq_H": annotation.sequence_aa,  # Gap-free sequence (P0 fix)
```

### Step 2: Regenerate All Harvey Fragments

```bash
python3 preprocessing/process_harvey.py
```

**Runtime:** ~10 minutes
**Output:** 6 fragment CSV files (141,021 sequences each)

### Step 3: Validate Gap Removal

```bash
python3 -c "import pandas as pd; \
  vhh = pd.read_csv('test_datasets/harvey/fragments/VHH_only_harvey.csv'); \
  print(f'Gaps: {vhh[\"sequence\"].str.contains(\"-\", na=False).sum()}')"
```

**Result:** `Gaps: 0` ✅

### Step 4: Run Comprehensive Test Suite

```bash
python3 tests/test_harvey_embedding_compatibility.py
```

**Result:** 5/5 tests passed ✅

---

## Test Suite Results

### Test 1: Gap Character Detection
✅ **PASS** - All 6 fragment files gap-free (141,021 sequences each)

### Test 2: Amino Acid Validation
✅ **PASS** - All sequences contain only valid amino acids (423,063 sequences validated)

### Test 3: Previously Affected Sequences
✅ **PASS** - Spot-checked 5 sequences, all gap-free
- Before fix: 12,116 sequences with gaps (8.6%)
- After fix: 0 sequences with gaps (0%)

### Test 4: ESM Model Validation Simulation
✅ **PASS** - All 141,021 sequences passed model.py:86-90 validation logic

### Test 5: Data Integrity
✅ **PASS** - All 6 files present with 141,021 rows
- Label distribution: 49.1% low, 50.9% high (balanced ✓)

---

## Comparison with Shehata Fix

Both datasets had the **EXACT SAME P0 BLOCKER** - here's the parallel:

| Aspect | Shehata | Harvey |
|--------|---------|--------|
| **Bug Location** | `process_shehata.py:63` | `process_harvey.py:48` |
| **Bug Type** | `sequence_alignment_aa` (gaps) | `sequence_alignment_aa` (gaps) |
| **Fix** | `→ sequence_aa` (gap-free) | `→ sequence_aa` (gap-free) |
| **Affected File** | VH_only_shehata.csv | VHH_only_harvey.csv |
| **Impact** | 100% of 398 sequences | 8.6% of 141,021 sequences |
| **Test Suite** | test_shehata_embedding_compatibility.py | test_harvey_embedding_compatibility.py |
| **Test Results** | 5/5 passed ✅ | 5/5 passed ✅ |

**Key Difference:** Shehata's IMGT alignment had gaps in ALL sequences (VH and VL), while Harvey only had gaps in ~8.6% of VHH sequences due to ANARCI's specific insertion handling for nanobodies.

---

## Files Modified

### Code Changes
1. ✅ `preprocessing/process_harvey.py:48` - P0 fix applied
2. ✅ `tests/test_harvey_embedding_compatibility.py` - New test suite created

### Data Regenerated
3. ✅ `test_datasets/harvey/fragments/VHH_only_harvey.csv` - 12,116 gaps removed
4. ✅ `test_datasets/harvey/fragments/H-CDR1_harvey.csv` - Already gap-free
5. ✅ `test_datasets/harvey/fragments/H-CDR2_harvey.csv` - Already gap-free
6. ✅ `test_datasets/harvey/fragments/H-CDR3_harvey.csv` - Already gap-free
7. ✅ `test_datasets/harvey/fragments/H-CDRs_harvey.csv` - Already gap-free
8. ✅ `test_datasets/harvey/fragments/H-FWRs_harvey.csv` - Already gap-free

### Documentation
9. ✅ `docs/harvey/HARVEY_P0_FIX_REPORT.md` - This report
10. ⬜ `docs/harvey/harvey_data_sources.md` - Needs update to correct ZYMScott misinformation

---

## Next Steps

### On ray/learning Branch (Current)
- ✅ P0 fix applied
- ✅ All fragments regenerated (gap-free)
- ✅ Test suite created and passing
- ✅ Documentation created

### On feat/harvey-preprocessing Branch (Next)
1. ⬜ Cherry-pick P0 fix commit from ray/learning
2. ⬜ Regenerate Harvey fragments on that branch
3. ⬜ Run test suite to validate
4. ⬜ Update documentation to reflect correct data source
5. ⬜ Ready for PR (do not push until confirmed)

---

## Lessons Learned

### Why This Happened Twice

1. **Non-obvious API:** `riot_na` library provides both `sequence_aa` and `sequence_alignment_aa` without clear documentation about which to use
2. **Inconsistent behavior:** CDR/FWR fragments use gap-free attributes (`.cdr*_aa`, `.fwr*_aa`), but full sequence requires explicit `.sequence_aa` selection
3. **Silent failure:** ANARCI doesn't warn about gaps; validation only fails at ESM-1v embedding time

### Prevention

1. **First-principles validation:** Always check for gap characters after ANARCI annotation
2. **Comprehensive test suites:** Include gap detection tests for ALL fragment types
3. **Code review:** Double-check riot_na attribute selection in all preprocessing scripts
4. **Documentation:** Clearly document the `.sequence_aa` (gap-free) vs `.sequence_alignment_aa` (with gaps) distinction

---

## References

- **Harvey Paper:** Harvey et al. 2022, Nature Communications 13, 7554
- **Official Repo:** https://github.com/debbiemarkslab/nanobody-polyreactivity
- **Novo Nordisk Paper:** Sakhnini et al. 2025, bioRxiv 2025.04.28.650927
- **ANARCI:** Dunbar & Deane 2016, Bioinformatics
- **riot_na Library:** v4.0.5 (ANARCI Python wrapper)
- **ESM-1v Model:** facebook/esm1v_t33_650M_UR90S_1

---

**Report Generated:** 2025-11-02
**Branch:** ray/learning
**Commit:** [pending]
**Status:** ✅ Ready for cherry-pick to feat/harvey-preprocessing
