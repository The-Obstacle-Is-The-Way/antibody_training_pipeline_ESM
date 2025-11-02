# Shehata Dataset P0-P4 Blocker Analysis

**Date:** 2025-11-02 (Updated with accurate counts)
**Analyst:** Claude Code
**Context:** Deep review before potential Discord criticism of "vibe coding"
**Validation:** First principles validation documented in `P0_BLOCKER_FIRST_PRINCIPLES_VALIDATION.md`

---

## Executive Summary

**CRITICAL P0 BLOCKER CONFIRMED:** Gap characters in fragment sequences will cause ESM embedding failures.

**Status:**
- ✅ Phase 1 conversion: CORRECT (gaps removed from base CSV)
- ❌ Phase 2 fragments: **BROKEN** (gaps present in VH/VL/Full fragment files)

**Validation Status:** ✅ Bug confirmed real (not hallucinated) via:
- Code inspection (model.py validation logic)
- Runtime behavior testing
- Actual file verification
- Paper methodology cross-reference
- Domain knowledge verification

---

## P0 BLOCKERS (CRITICAL - BREAKS FUNCTIONALITY)

### P0-1: Gap Characters in Fragment Sequences

**Issue:** VH, VL, and Full fragment CSVs contain IMGT gap characters ("-") that will break ESM-1v embedding.

**Accurate Evidence (Verified 2025-11-02):**
```bash
# Gap counts in fragment files:
VH_only_shehata.csv: 13 sequences with "-" gaps (NOT 7)
VL_only_shehata.csv: 4 sequences with "-" gaps (NOT 1)
Full_shehata.csv: 17 sequences with "-" gaps (NOT 8)
VH+VL_shehata.csv: 17 sequences with "-" gaps

# Affected antibody IDs:
VH gaps (13): ADI-47173, ADI-47060, ADI-45440, ADI-47105, ADI-47224,
              ADI-47140, ADI-45389, ADI-47169, ADI-47071, ADI-47107,
              ADI-47267, ADI-47246, ADI-47141

VL gaps (4): ADI-47223, ADI-47114, ADI-47163, ADI-47211

Full gaps (17): All VH gaps + all VL gaps (union)

# Example (ADI-47173 VH):
EVQLVESGGGVVQPGRSLRLSCAASGFTFDRYGMHWIRQAPGKGLECVALISFDGSHK-YADSVKGRFTISRDNSRNTLY...
                                                           ^
                                                        GAP CHAR
```

**Root Cause (Verified):**
- **File:** `preprocessing/process_shehata.py:63`
- **Issue:** Uses `annotation.sequence_alignment_aa` (IMGT-aligned with gaps)
- **Should use:** `annotation.sequence_aa` (raw sequence without gaps)

**Impact:**
- **Runtime:** ESM-1v validation at `model.py:36` raises `ValueError: "Invalid amino acid characters in sequence"`
- **Behavior:** Logs warning, replaces with placeholder "M", returns garbage embeddings
- **Result:** Silent failure → 17 incorrect embeddings → wrong model results
- **Paper reproduction:** IMPOSSIBLE with current implementation

**Severity:** P0 (breaks core functionality, produces incorrect results)

**Why Paper Must Use Gap-Free Sequences:**
- Paper methodology (Sakhnini et al. 2025): "sequences were annotated in the CDRs using ANARCI following the IMGT numbering scheme. Following this, 16 different antibody fragment sequences were assembled and embedded by... ESM 1v"
- ESM-1v tokenizer requires valid amino acids only: `"ACDEFGHIKLMNPQRSTVWYX"`
- IMGT gaps ("-") are NOT valid amino acids
- Paper achieved successful embeddings → must have used gap-free field

**First Principles Validation:**
✅ Validated in `docs/shehata/P0_BLOCKER_FIRST_PRINCIPLES_VALIDATION.md`:
- Code logic confirms `-` character fails validation (model.py:33-37)
- Actual files verified to have gaps (13 VH, 4 VL, 17 Full)
- Runtime behavior tested and confirmed
- Not a hallucination - reproducible bug with clear evidence

**Fix Required:**
```python
# File: preprocessing/process_shehata.py
# Line: 63

# Current (BROKEN):
f"full_seq_{chain}": annotation.sequence_alignment_aa  # IMGT-aligned WITH gaps

# Change to:
f"full_seq_{chain}": annotation.sequence_aa  # Raw sequence WITHOUT gaps
```

**Verification Status:**
✅ CDR/FWR fragments have 0 gaps (correct - individual regions don't have internal gaps)
❌ Full-length sequences (VH, VL, Full, VH+VL) have gaps (broken)

---

## P1 BLOCKERS (HIGH SEVERITY - INCORRECT RESULTS)

### P1-1: PSR Threshold Verification

**Issue:** Binary labels use 98.24th percentile threshold (0.31002), but paper doesn't explicitly state this value.

**Evidence (Verified 2025-11-02):**
```
Paper states: "7 out of 398 antibodies characterised as non-specific"
Our threshold: 0.31002 PSR (98.24th percentile)
Result: 7 labeled as non-specific ✓

Top 7 PSR scores (all labeled non-specific):
  ADI-45498: PSR=0.710695, label=1
  ADI-45499: PSR=0.710238, label=1
  ADI-47276: PSR=0.567043, label=1
  ADI-47265: PSR=0.487817, label=1
  ADI-45435: PSR=0.457177, label=1
  ADI-47275: PSR=0.418042, label=1
  ADI-45433: PSR=0.364564, label=1

Next highest (labeled specific):
  8th: PSR=0.310024 (just below threshold)
```

**Current Status:**
✅ Threshold produces correct count (7/398)
✅ Top 7 PSR scores match non-specific labels
✅ Clean separation at 98.24th percentile
⚠️ Threshold value not explicitly confirmed from paper

**Risk:**
- If paper used different threshold with same count, labels might differ
- Need to verify these exact 7 antibodies match paper's non-specific set

**Severity:** P1 (labels likely correct, but need confirmation)

**Mitigation:**
- Threshold documented: `scripts/convert_shehata_excel_to_csv.py:163`
- Check paper supplement for non-specific antibody IDs
- If IDs match, threshold is validated
- If IDs differ, adjust threshold to match paper's exact cutoff

---

## P2 BLOCKERS (MODERATE - METHODOLOGY CONCERNS)

### P2-1: Re-annotation vs Provided CDR Annotations

**Issue:** mmc2.xlsx contains pre-annotated CDR regions, but we re-annotate with ANARCI.

**Justification:**
- Ensures consistency with other datasets (Jain, Harvey, Boughter)
- Uses same riot_na version (4.0.5) across all datasets
- IMGT scheme consistent

**Risk:**
- Annotations might differ slightly from paper's
- If paper used their Excel CDRs directly, results could vary

**Severity:** P2 (methodology consistency)

**Mitigation Done:**
- Documented choice in code
- 100% annotation success rate (398/398)
- Fragment lengths match expected ranges

---

## P3 BLOCKERS (LOW - MINOR ISSUES)

### P3-1: Dataset Location (test_datasets vs train_datasets)

**Issue:** Shehata is in `test_datasets/` (correct per paper - external test set only).

**Evidence from paper:**
> "the most balanced dataset (i.e. Boughter one) was selected for training of ML models, while the remaining three (i.e. Jain, Shehata and Harvey) were used for testing."

**Status:** ✅ CORRECT location

**Severity:** P3 (documentation clarity)

---

## P4 BLOCKERS (INFORMATIONAL - NICE TO HAVE)

### P4-1: Missing Validation Against Paper Statistics

**Issue:** No verification that our extracted CDR lengths match paper's reported statistics.

**What We Have:**
- Fragment length ranges documented
- All 398 antibodies processed
- 16 fragment types created

**What's Missing:**
- Cross-check CDR3 length distribution with paper Figure/Table
- Verify mean fragment lengths
- Compare PSR score distribution shape

**Severity:** P4 (nice to have for confidence)

---

## Non-Issues (Investigated, Confirmed Correct)

### ✅ Sequence Sanitization in Phase 1
- `convert_shehata_excel_to_csv.py` correctly removes gaps
- shehata.csv has clean sequences
- No gaps in base CSV ✓

### ✅ Label Distribution
- 391 specific (98.2%)
- 7 non-specific (1.8%)
- Matches paper ✓

### ✅ B Cell Subset Distribution
- All subsets present
- Correct column preservation ✓

### ✅ Fragment Count
- 16 fragment types created
- All match paper methodology ✓

### ✅ Integration Compatibility
- CSV format correct
- Column names standardized
- Compatible with data.load_local_data() ✓

---

## Comparison with Novo Paper Requirements

**From Sakhnini et al. 2025 Methods 4.3:**

| Requirement | Our Implementation | Status |
|-------------|-------------------|---------|
| ANARCI annotation | ✅ riot_na 4.0.5 | PASS |
| IMGT numbering | ✅ IMGT scheme | PASS |
| 16 fragment types | ✅ All created | PASS |
| ESM-1v embedding | ❌ Will fail (gaps) | **FAIL** |
| Mean pooling | N/A (in data.py) | N/A |
| External test set | ✅ test_datasets/ | PASS |

---

## Documentation Contradictions

### ⚠️ Phase 2 Completion Report Contradiction

**File:** `docs/shehata/shehata_phase2_completion_report.md:90-122`

**Claim:**
> "✅ **All fragment lengths match expected antibody structure**"
> "✅ **All fragments preserve original label distribution**"
> "✅ **All fragment files validated**"
> "- **No missing sequences:** ✓"

**Reality:**
❌ Fragment files contain gap characters (13 VH, 4 VL, 17 Full)
❌ Sequences will fail ESM validation
❌ Phase 2 NOT fully validated

**Resolution:**
- This blocker analysis supersedes completion report
- Completion report must be updated post-fix
- Add gap detection to validation checklist

---

## Action Items

### URGENT (P0):
1. **Fix gap characters in fragment sequences**
   - **File:** `preprocessing/process_shehata.py`
   - **Change line 63:**
     ```python
     # FROM: f"full_seq_{chain}": annotation.sequence_alignment_aa
     # TO:   f"full_seq_{chain}": annotation.sequence_aa
     ```
   - Re-run fragment extraction: `python3 preprocessing/process_shehata.py`
   - Validate all gaps removed: `grep -c '\-' test_datasets/shehata/*.csv` (should be 0)
   - Test ESM embedding on all 17 previously-affected sequences

2. **Add gap detection to validation**
   - **File:** `scripts/validate_shehata_conversion.py`
   - Add fragment-level validation function
   - Assert no `-` characters in any fragment CSV
   - Prevent regression

3. **Update completion report**
   - **File:** `docs/shehata/shehata_phase2_completion_report.md`
   - Remove "✅ validated" claims until post-fix
   - Add note about P0 blocker discovered and fixed
   - Update validation checklist to include gap detection

### HIGH PRIORITY (P1):
4. **Verify PSR threshold against paper**
   - Check paper supplement for non-specific antibody IDs
   - Compare: ADI-45498, ADI-45499, ADI-47276, ADI-47265, ADI-45435, ADI-47275, ADI-45433
   - If mismatch, adjust threshold to match paper's exact antibodies
   - Document verification in data sources doc

### MEDIUM PRIORITY (P2):
5. **Validate re-annotation choice**
   - Spot-check CDR boundaries vs Excel annotations
   - Document any differences
   - Confirm consistency justification

### LOW PRIORITY (P3-P4):
6. **Statistical validation**
   - Compare fragment length distributions with paper
   - Verify PSR score distribution shape
   - Cross-reference metadata

---

## Testing Checklist Before PR

- [ ] No gap characters in ANY fragment CSV
- [ ] ESM embedding works on all fragments
- [ ] Label distribution: 391/7 split maintained
- [ ] All 398 sequences in all 16 files
- [ ] Spot-check 10 sequences against mmc2.xlsx
- [ ] Compare with paper Figure 3C-D statistics
- [ ] Test load with data.load_local_data()
- [ ] Run through existing model (if available)

---

## Conclusion

**PRIMARY BLOCKER:** Gap characters in fragment sequences (P0)
**CONFIDENCE:** High - clear technical error with definitive fix
**RISK TO DISCORD CRITICISM:** **ELIMINATED after fix**

After fixing P0, implementation will be technically sound and defensible.
