# Jain Dataset P0 Blocker Fix - Report

**Date**: 2025-11-02
**Issue**: Gap characters in VH/VL sequences causing ESM-1v embedding failures
**Status**: ✅ **RESOLVED**

---

## Executive Summary

**P0 Blocker Found**: Gap characters (`-`) present in 15 Jain antibody sequences (5 VH + 11 VL unique), preventing ESM-1v model inference.

**Root Cause**: `preprocessing/process_jain.py` used `annotation.sequence_alignment_aa` (aligned sequences with gaps) instead of reconstructing V-domains from clean fragments.

**Fix Applied**: V-domain reconstruction from ANARCI fragments (FWR1+CDR1+FWR2+CDR2+FWR3+CDR3+FWR4) - same fix as Harvey, Shehata, and Boughter datasets.

**Verification**: All 5 ESM compatibility tests passed (5/5). 137 sequences ready for inference.

---

## Problem Details

### P0 Blocker Detection

Created `tests/test_jain_embedding_compatibility.py` following the pattern from Harvey/Shehata/Boughter datasets. Initial test run revealed:

**Test Results (Before Fix)**:
- ✗ Gap Character Detection: **FAIL** - 15 sequences with gaps
- ✗ Amino Acid Validation: **FAIL** - Invalid character `-` detected
- ✓ Stop Codon Detection: PASS - No stop codons found
- ✗ ESM Model Validation: **FAIL** - 15 sequences incompatible
- ✓ Data Integrity: PASS - 137 sequences, correct label distribution

### Affected Sequences

**VH sequences with gaps (5 antibodies)**:
- dalotuzumab
- lebrikizumab
- panobacumab
- (+ 2 others)

**VL sequences with gaps (11 antibodies)**:
- daclizumab
- galiximab
- lenzilumab
- (+ 8 others)

**Total unique affected**: 15 antibodies (11.0% of dataset)

**Example gap pattern**:
```
dalotuzumab VH: QVQLQESGPGLVKPSETLSLTCTVSGYSI-TGGYLWNWIRQPPGKGLEWI...
                                          ^ gap character from alignment
```

---

## Root Cause Analysis

### Original Code (Broken)

**File**: `preprocessing/process_jain.py:63`

```python
# BROKEN: Used aligned sequence with gaps
fragments = {
    f"full_seq_{chain}": annotation.sequence_alignment_aa,  # ❌ Contains gaps
    f"fwr1_aa_{chain}": annotation.fwr1_aa,
    # ...
}
```

**Why it failed**:
- `annotation.sequence_alignment_aa` returns IMGT-aligned sequences
- IMGT alignment inserts gap characters (`-`) to maintain positional numbering
- ESM-1v tokenizer crashes on non-amino-acid characters
- Same root cause as Harvey (HARVEY_P0_FIX_REPORT.md), Shehata (SHEHATA_BLOCKER_ANALYSIS.md), and Boughter (BOUGHTER_P0_FIX_REPORT.md)

---

## Solution

### Fix Applied

**File**: `preprocessing/process_jain.py:73-84`

```python
# FIXED: Extract fragments first, then reconstruct
fragments = {
    f"fwr1_aa_{chain}": annotation.fwr1_aa,
    f"cdr1_aa_{chain}": annotation.cdr1_aa,
    f"fwr2_aa_{chain}": annotation.fwr2_aa,
    f"cdr2_aa_{chain}": annotation.cdr2_aa,
    f"fwr3_aa_{chain}": annotation.fwr3_aa,
    f"cdr3_aa_{chain}": annotation.cdr3_aa,
    f"fwr4_aa_{chain}": annotation.fwr4_aa,
}

# Reconstruct full V-domain from fragments (gap-free, P0 fix)
# This avoids gap characters from sequence_alignment_aa
# Same fix as applied to Harvey/Shehata/Boughter datasets
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

**Key Changes**:
1. Extract individual fragments from ANARCI annotation
2. Concatenate fragments to reconstruct V-domain (gap-free)
3. Avoid using `sequence_alignment_aa` for full sequences
4. CDR/FWR fragments were already clean (no gaps in individual regions)

---

## Verification

### Fragment Regeneration

```bash
python3 preprocessing/process_jain.py
```

**Output**:
- ✓ 137/137 antibodies successfully annotated
- ✓ 16 fragment CSV files regenerated
- ✓ Label distribution: 67 specific, 3 non-specific, 67 mild

### ESM Compatibility Test Suite

```bash
python3 tests/test_jain_embedding_compatibility.py
```

**Test Results (After Fix)**:
```
======================================================================
Test Summary
======================================================================
  ✓ PASS: Gap Character Detection       (16 files, 137 sequences each)
  ✓ PASS: Amino Acid Validation         (685 sequences validated)
  ✓ PASS: Stop Codon Detection          (137 sequences x 3 files)
  ✓ PASS: ESM Model Validation          (411 sequences validated)
  ✓ PASS: Data Integrity                (137 rows per file, correct labels)

  Total: 5/5 tests passed

  ✓ ALL TESTS PASSED - Jain dataset is ESM-1v compatible!
  ✓ Ready for model inference and confusion matrix generation
```

---

## Impact Analysis

### Before Fix
- ❌ VH_only: 5 sequences with gaps (3.6%)
- ❌ VL_only: 11 sequences with gaps (8.0%)
- ❌ Full/VH+VL: 15 sequences with gaps (11.0%)
- ❌ ESM-1v inference would crash on 15 antibodies
- ❌ Cannot generate confusion matrix for comparison to Novo/Hybri

### After Fix
- ✅ VH_only: 0 gaps (100% clean)
- ✅ VL_only: 0 gaps (100% clean)
- ✅ Full/VH+VL: 0 gaps (100% clean)
- ✅ All 137 sequences compatible with ESM-1v
- ✅ Ready for model inference to generate confusion matrix

### Test Set Status

**Test Set (excluding mild 1-3 flags)**:
- 70 sequences total (67 specific + 3 non-specific)
- 100% compatible with ESM-1v
- Ready for comparison to Novo (69% accuracy) and Hybri (65.1% accuracy)

**Confusion Matrix Targets**:
- Novo: [[40 19][10 17]] = 69% accuracy
- Hybri: [[39 19][11 17]] = 65.1% accuracy
- Our target: ~65-69% (same ballpark)

---

## Files Modified

### Code Changes

| File | Line | Change |
|------|------|--------|
| `preprocessing/process_jain.py` | 63-84 | Replaced `sequence_alignment_aa` with V-domain reconstruction |

### Files Regenerated

All 16 fragment CSVs in `test_datasets/jain/`:
- VH_only_jain.csv ✓
- VL_only_jain.csv ✓
- H-CDR1_jain.csv ✓
- H-CDR2_jain.csv ✓
- H-CDR3_jain.csv ✓
- L-CDR1_jain.csv ✓
- L-CDR2_jain.csv ✓
- L-CDR3_jain.csv ✓
- H-CDRs_jain.csv ✓
- L-CDRs_jain.csv ✓
- H-FWRs_jain.csv ✓
- L-FWRs_jain.csv ✓
- VH+VL_jain.csv ✓
- All-CDRs_jain.csv ✓
- All-FWRs_jain.csv ✓
- Full_jain.csv ✓

### New Test Files

| File | Purpose |
|------|---------|
| `tests/test_jain_embedding_compatibility.py` | ESM-1v compatibility validation (5 tests) |

---

## Comparison to Other Datasets

| Dataset | P0 Issue | Fix Location | Affected Sequences | Fix Date |
|---------|----------|--------------|-------------------|----------|
| **Harvey** | Gap characters | `process_harvey.py` | Unknown count | Prior |
| **Shehata** | Gap characters | `process_shehata.py` | Unknown count | Prior |
| **Boughter** | Gaps + stop codons | `process_boughter.py:101` | 11 VH + 2 VL | Prior |
| **Jain** | Gap characters | `process_jain.py:76` | 5 VH + 11 VL | 2025-11-02 |

**Consistent Solution**: All four datasets use V-domain reconstruction from ANARCI fragments instead of aligned sequences.

---

## Next Steps

1. ✅ **DONE**: Jain P0 blocker fixed and verified
2. **TODO**: Test trained Boughter model on Jain dataset
3. **TODO**: Generate confusion matrix for comparison
4. **TODO**: Compare to Novo [[40 19][10 17]] and Hybri [[39 19][11 17]]
5. **OPTIONAL**: Re-evaluate with >=3 flag threshold (86 sequences vs 70)

---

## References

- **Similar fixes**:
  - `docs/harvey/HARVEY_P0_FIX_REPORT.md`
  - `docs/shehata/SHEHATA_BLOCKER_ANALYSIS.md`
  - `docs/boughter/BOUGHTER_P0_FIX_REPORT.md`
- **Test pattern**: `tests/test_boughter_embedding_compatibility.py`
- **Jain source data**: `docs/jain/jain_data_sources.md`
- **Jain conversion**: `docs/jain/jain_conversion_verification_report.md`

---

## Conclusion

✅ **P0 Blocker RESOLVED** - Jain dataset is now fully compatible with ESM-1v embedding pipeline.

✅ **All Tests Passing** - 5/5 ESM compatibility tests passed, 137 sequences validated.

✅ **Ready for Inference** - Model trained on Boughter (67.5% CV accuracy) can now be tested on Jain to generate confusion matrix for comparison with Novo (69%) and Hybri (65.1%).

**Assessment**: Same P0 issue (gap characters) and same solution (V-domain reconstruction) as other datasets. Fix is consistent and verified.
