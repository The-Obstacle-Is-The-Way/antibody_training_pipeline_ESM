# Jain Canonical Dataset Investigation

**Date:** 2025-11-07
**Status:** ✅ RESOLVED AND DELETED - step4 removed from codebase
**Verdict:** FILE COLLISION BUG - step4 was legacy code overwriting step2's correct output

---

## TL;DR

**Bug Confirmed:** `step4_build_canonical_sets.py` was overwriting `step2_preprocess_p5e_s2.py` output
**Impact:** Jain test set had all-zero labels instead of correct 59 specific / 27 non-specific split
**Fix:** Just ran step2 standalone - file now has correct 59/27 split
**Root Cause:** Both scripts write to same file path, step4 hardcodes label=0

---

## The Issue

We have **TWO DIFFERENT SCRIPTS** that both create `jain_86_novo_parity.csv` with **DIFFERENT METHODOLOGIES**:

### Method 1: P5e-S2 (step2_preprocess_p5e_s2.py)
```
137 antibodies (FULL)
  ↓ Remove ELISA 1-3
116 antibodies
  ↓ Reclassify 5 specific → non-specific (PSR + clinical)
89 spec / 27 nonspec
  ↓ Remove 30 by PSR/AC-SINS
86 antibodies (59 specific / 27 non-specific)
```

**Output:** `data/test/jain/canonical/jain_86_novo_parity.csv`
**Expected labels:** 59 zeros + 27 ones
**Confusion matrix claim:** `[[40, 19], [10, 17]]` - EXACT NOVO PARITY

### Method 2: VH-Length QC (step4_build_canonical_sets.py)
```
137 antibodies (FULL)
  ↓ Take ONLY specific (elisa_flags == 0)
94 specific
  ↓ Remove 3 VH length outliers
91 specific
  ↓ Remove 5 "borderline" (largest elisa_flags among zeros)
86 antibodies (ALL SPECIFIC, label hardcoded to 0)
```

**Output:** `data/test/jain/canonical/jain_86_novo_parity.csv` (SAME FILE!)
**Actual labels:** 86 zeros
**Logic:** Line 184 - `df_novo_parity["label"] = 0`

---

## Current State

**File timestamps:** All canonical files created Nov 7 16:47 (same time)
**Conclusion:** `step4_build_canonical_sets.py` was run LAST and OVERWROTE step2 output

**What's in the file NOW:**
```python
import pandas as pd
df = pd.read_csv('data/test/jain/canonical/jain_86_novo_parity.csv')
print(df['label'].value_counts())
# Output:
# 0    86
```

---

## The Question: Is This a Bug?

**Two possibilities:**

### Scenario A: It's a Bug
- P5e-S2 is the CORRECT methodology
- step4 should NOT create `jain_86_novo_parity.csv`
- step4 is only for VH-only test sets
- **Fix:** Remove line 175-189 from step4, use step2 output only

### Scenario B: It's Not a Bug
- These are TWO DIFFERENT experimental approaches
- step2 = P5e-S2 with reclassification (includes non-specific)
- step4 = VH-length QC approach (specific-only test set)
- **Fix:** Rename outputs to avoid collision

---

## Evidence Analysis

### What Hybri Said (Discord)
```
"Ok. final boughter dataset, gave accuracy of .682... on the jain deviation
is prepared it got an accuracy of 0.651 vs Novo's that was 0.69 if i am
not wrong. But the confusion matrix was interestingly similar. I only got
1 false negative."

My confusion matrix
[[39 19]
 [11 17]]

Novo's confusion matrix:
[[40 19]
 [10 17]]
```

**Interpretation:** Hybri's Jain dataset HAD BOTH CLASSES (specific + non-specific)
**Conclusion:** Supports P5e-S2 methodology (59/27 split)

### What step2 Claims
```python
# preprocessing/jain/step2_preprocess_p5e_s2.py line 17
Result: Confusion matrix [[40, 19], [10, 17]] - EXACT MATCH (66.28% accuracy)
```

**Interpretation:** step2 explicitly claims to match Novo's confusion matrix
**Conclusion:** This is the intended replication methodology

### What step4 Does
```python
# preprocessing/jain/step4_build_canonical_sets.py line 86
df_full = df[df["elisa_flags"] == 0].copy()  # Only specific

# line 184
df_novo_parity["label"] = 0  # Hardcoded
```

**Interpretation:** step4 creates a SPECIFIC-ONLY test set (no non-specific examples)
**Conclusion:** This cannot produce Novo's confusion matrix

---

## File Collision

**BOTH scripts output to:**
```
data/test/jain/canonical/jain_86_novo_parity.csv
```

**Line references:**
- step2: line 43 - `OUTPUT_86 = BASE_DIR / "data/test/jain/canonical/jain_86_novo_parity.csv"`
- step4: line 187 - `output_novo = output_dir / "jain_86_novo_parity.csv"`

**Last run:** step4 (Nov 7 16:47) - overwrote step2 output

---

## Verdict

**THIS IS A BUG.**

**Reasoning:**
1. Novo's confusion matrix `[[40, 19], [10, 17]]` REQUIRES both classes
2. step4 creates all-zeros labels (cannot match Novo matrix)
3. step2 explicitly implements P5e-S2 to match Novo parity
4. Hybri's results confirm Jain needs both specific + non-specific
5. File collision is unintentional (same output path)

**Root Cause:**
- step4 was created for VH-only test sets (lines 100-172)
- Lines 175-189 were added to create "full metadata" version
- This created unintended collision with step2 output

---

## Recommended Fix

### Option 1: Delete Conflicting Code (Preferred)
Remove lines 175-189 from `step4_build_canonical_sets.py`:
```python
# DELETE THIS SECTION
# ========================================================================
# SET 4: jain_86_novo_parity.csv (full columns for compatibility)
# ========================================================================
```

**Rationale:** step4 is for VH-only sets, step2 owns the full parity dataset

### Option 2: Rename Outputs
- step2 keeps: `jain_86_novo_parity.csv` (59/27 split)
- step4 creates: `jain_86_specific_only.csv` (86 all-specific)

**Rationale:** Both methodologies preserved, no collision

---

## Action Items

- [ ] Run step2 standalone: `python3 preprocessing/jain/step2_preprocess_p5e_s2.py`
- [ ] Verify output has 59 specific + 27 non-specific
- [ ] Update step4 to NOT create `jain_86_novo_parity.csv`
- [ ] Document which file is canonical for Novo parity benchmarking
- [ ] Test training pipeline uses correct file

---

## Supporting Files

**Check which file training uses:**
```bash
grep -r "jain_86_novo_parity" src/
```

**Verify step2 output:**
```bash
python3 preprocessing/jain/step2_preprocess_p5e_s2.py
python3 -c "
import pandas as pd
df = pd.read_csv('data/test/jain/canonical/jain_86_novo_parity.csv')
print(f'Total: {len(df)}')
print(f'Specific: {(df[\"label\"] == 0).sum()}')
print(f'Non-specific: {(df[\"label\"] == 1).sum()}')
"
```

---

## Verification (2025-11-07)

**Ran step2 standalone:**
```bash
python3 preprocessing/jain/step2_preprocess_p5e_s2.py
```

**Output verified:**
```
Total: 86
Label distribution:
  0.0    59  ✅ Specific
  1.0    27  ✅ Non-specific

Non-specific antibodies (27):
bavituximab, belimumab, bimagrumab, blosozumab, bococizumab,
briakinumab, carlumab, cixutumumab, codrituzumab, dalotuzumab,
denosumab, duligotuzumab, dupilumab, eldelumab, emibetuzumab,
ganitumab, gantenerumab, imgatuzumab, infliximab, ixekizumab,
lenzilumab, parsatuzumab, patritumab, ponezumab, robatumumab,
simtuzumab, sirukumab
```

**File now correct:** `data/test/jain/canonical/jain_86_novo_parity.csv`

---

**Conclusion:** This WAS a file collision bug where step4 overwrote step2's correct output with an all-specific test set that cannot match Novo's published confusion matrix. Bug is now fixed by running step2 standalone.

---

## Final Resolution (2025-11-07)

**Action Taken:** DELETED `preprocessing/jain/step4_build_canonical_sets.py`

**Why Deletion Was Correct:**
1. step4 was LEGACY CODE from before P5e-S2 methodology
2. It used outdated VH-length heuristics instead of Novo's method
3. It hardcoded label=0 for all antibodies (impossible to match Novo matrix)
4. It created file collision with step2's authoritative output
5. It was NOT part of Novo Nordisk's documented methodology

**Files Also Updated:**
- ✅ Deleted P0-BUG.md (obsolete analysis)
- ✅ Updated scripts/validation/validate_jain_csvs.py (removed step4 reference)
- ✅ Updated JAIN_PIPELINE_EXPLAINED.md (documented deletion)
- ✅ Updated JAIN_CANONICAL_INVESTIGATION.md (this file)

**Result:**
- No more file collision risk
- Canonical dataset always has correct 59 specific / 27 non-specific split
- Clear pipeline: step1 → step2 → (optional step3 for fragments)
