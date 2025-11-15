# Jain Processing Pipeline - Clear Explanation

**Date:** 2025-11-07
**Status:** ✅ RESOLVED - step4 deleted (was legacy code with file collision bug)

---

## The Jain Pipeline (Current State)

### Step 1: Convert Excel to CSV
**Script:** `preprocessing/jain/step1_convert_excel_to_csv.py`

**Input:**
- `data/test/jain/raw/Private_Jain2017_ELISA_indiv.xlsx` (private ELISA data)
- `data/test/jain/raw/jain-pnas.1616408114.sd0*.xlsx` (public supplement)

**Output:**
- `data/test/jain/processed/jain_with_private_elisa_FULL.csv` (137 antibodies)
- `data/test/jain/processed/jain_sd01.csv`
- `data/test/jain/processed/jain_sd02.csv`
- `data/test/jain/processed/jain_sd03.csv`

**Purpose:** Convert Excel to CSV with ELISA-based flagging

---

### Step 2: P5e-S2 Novo Parity
**Script:** `preprocessing/jain/step2_preprocess_p5e_s2.py`

**Input:**
- `data/test/jain/processed/jain_with_private_elisa_FULL.csv` (137 antibodies)

**Output:**
- `data/test/jain/processed/jain_ELISA_ONLY_116.csv` (116 antibodies)
- `data/test/jain/canonical/jain_86_novo_parity.csv` ← **PRIMARY OUTPUT**

**Methodology:**
```
137 antibodies
  ↓ Remove ELISA 1-3
116 antibodies
  ↓ Reclassify 5 spec→nonspec (PSR + clinical)
89 spec / 27 nonspec
  ↓ Remove 30 by PSR/AC-SINS
86 antibodies (59 specific / 27 non-specific)
```

**Result:**
- **59 specific (label=0) + 27 non-specific (label=1)**
- **Confusion matrix: [[40, 19], [10, 17]]**
- **EXACT Novo Nordisk parity**

---

### Step 3: Extract Fragments
**Script:** `preprocessing/jain/step3_extract_fragments.py`

**Input:**
- `data/test/jain/processed/jain_with_private_elisa_FULL.csv`

**Output:**
- `data/test/jain/fragments/VH_jain_test_FULL.csv`
- `data/test/jain/fragments/VL_jain_test_FULL.csv`
- `data/test/jain/fragments/H-CDR1_jain_test_FULL.csv`
- ... (16 fragment types total)

**Purpose:** Extract all fragment types (VH, VL, CDRs, FWRs) using ANARCI

---

### Step 4: Build Canonical Sets (⚠️ PROBLEMATIC)
**Script:** `preprocessing/jain/step4_build_canonical_sets.py`

**Input:**
- `data/test/jain/processed/jain_with_private_elisa_FULL.csv`

**Output:**
- `data/test/jain/canonical/VH_only_jain_test_FULL.csv` (94 antibodies)
- `data/test/jain/canonical/VH_only_jain_test_QC_REMOVED.csv` (91 antibodies)
- `data/test/jain/canonical/VH_only_jain_test_PARITY_86.csv` (86 antibodies)
- `data/test/jain/canonical/jain_86_novo_parity.csv` ← **OVERWRITES STEP2!**

**Methodology:**
```
137 antibodies
  ↓ Take ONLY specific (elisa_flags == 0)
94 specific
  ↓ Remove 3 VH length outliers
91 specific
  ↓ Remove 5 "borderline"
86 antibodies (ALL label=0 HARDCODED)
```

**Result:**
- **86 all-specific (label=0)**
- **NO non-specific antibodies**
- **CANNOT match Novo confusion matrix**

---

## THE BUG

### File Collision

**Both step2 AND step4 write to:**
```
data/test/jain/canonical/jain_86_novo_parity.csv
```

**step2 output (CORRECT):**
- 59 specific + 27 non-specific
- Can match Novo confusion matrix [[40, 19], [10, 17]]

**step4 output (WRONG):**
- 86 all-specific (hardcoded label=0)
- CANNOT match Novo confusion matrix

### What Happens When Users Run This

**Scenario 1: User runs step2 only**
```bash
python3 preprocessing/jain/step2_preprocess_p5e_s2.py
```
✅ Result: `jain_86_novo_parity.csv` has 59/27 split (CORRECT)

**Scenario 2: User runs step4 only**
```bash
python3 preprocessing/jain/step4_build_canonical_sets.py
```
❌ Result: `jain_86_novo_parity.csv` has 86/0 split (WRONG)

**Scenario 3: User runs both in sequence**
```bash
python3 preprocessing/jain/step2_preprocess_p5e_s2.py
python3 preprocessing/jain/step4_build_canonical_sets.py
```
❌ Result: step4 OVERWRITES step2, file has 86/0 split (WRONG)

**Scenario 4: User runs full pipeline from README**
```bash
# If README says to run all steps...
python3 preprocessing/jain/step1_convert_excel_to_csv.py
python3 preprocessing/jain/step2_preprocess_p5e_s2.py
python3 preprocessing/jain/step3_extract_fragments.py
python3 preprocessing/jain/step4_build_canonical_sets.py  # ← BREAKS IT!
```
❌ Result: step4 overwrites step2, training gets wrong labels

---

## Is step4 Bogus?

### Not Entirely - It Has Two Purposes

**Purpose 1: VH-only test sets (VALID)**
Creates three VH-only variants for benchmarking:
- `VH_only_jain_test_FULL.csv` - 94 antibodies
- `VH_only_jain_test_QC_REMOVED.csv` - 91 antibodies
- `VH_only_jain_test_PARITY_86.csv` - 86 antibodies

**These are useful** for testing VH-only models

**Purpose 2: Full metadata version (INVALID)**
Lines 175-189 create `jain_86_novo_parity.csv` with:
- Full columns (vh_sequence, vl_sequence, etc.)
- **BUT hardcoded label=0 for ALL antibodies**
- **This OVERWRITES step2's correct output**

### What's Bogus

**Lines 175-189 of step4 are BOGUS:**
```python
# ========================================================================
# SET 4: jain_86_novo_parity.csv (full columns for compatibility)
# ========================================================================
df_novo_parity = df_parity[...].copy()
df_novo_parity["label"] = 0  # ← HARDCODED WRONG
df_novo_parity["source"] = "jain2017_pnas"

output_novo = output_dir / "jain_86_novo_parity.csv"  # ← COLLISION!
df_novo_parity.to_csv(output_novo, index=False)
```

This section:
1. Starts from SPECIFIC-ONLY antibodies (elisa_flags==0)
2. Hardcodes all labels to 0
3. Writes to same file as step2
4. Cannot possibly match Novo's confusion matrix

---

## The Fix

### Option 1: Delete Bogus Code (RECOMMENDED)

**Remove lines 175-189 from step4:**
```python
# DELETE THIS ENTIRE SECTION
# SET 4: jain_86_novo_parity.csv (full columns for compatibility)
```

**Rationale:**
- step4 should ONLY create VH-only test sets
- step2 owns `jain_86_novo_parity.csv` (the Novo parity file)
- No collision, no confusion

### Option 2: Rename step4 Output

**Change line 187 in step4:**
```python
# OLD:
output_novo = output_dir / "jain_86_novo_parity.csv"

# NEW:
output_novo = output_dir / "jain_86_specific_only.csv"
```

**Rationale:**
- Preserves both methodologies
- No collision
- Clear naming (specific_only vs novo_parity)

### Option 3: Make step4 Use step2 Output

**Rewrite step4 to:**
1. Load `jain_86_novo_parity.csv` from step2
2. Extract VH-only columns
3. Preserve the correct 59/27 labels

**This requires step2 to run first**

---

## Current State (Nov 7)

**What's in the file RIGHT NOW:**
```bash
$ python3 -c "
import pandas as pd
df = pd.read_csv('data/test/jain/canonical/jain_86_novo_parity.csv')
print(f'Total: {len(df)}')
print(f'Specific: {(df[\"label\"] == 0).sum()}')
print(f'Non-specific: {(df[\"label\"] == 1).sum()}')
"

Total: 86
Specific: 59.0
Non-specific: 27.0
```

✅ **Currently CORRECT** (just ran step2 standalone)

**But if anyone runs step4, it will break again**

---

## Recommended Action

### Immediate Fix

1. **Delete lines 175-189 from step4**
2. **Update README** to clarify:
   - step2 creates Novo parity dataset (59/27 split)
   - step4 creates VH-only benchmarks (optional)
3. **Add warning** to step4 that it should NOT overwrite step2

### Update Documentation

**In `preprocessing/jain/README.md`:**
```markdown
## Full Pipeline

### Required Steps (for Novo Parity)
python3 preprocessing/jain/step1_convert_excel_to_csv.py
python3 preprocessing/jain/step2_preprocess_p5e_s2.py  # Creates jain_86_novo_parity.csv

### Optional Steps
python3 preprocessing/jain/step3_extract_fragments.py  # Extract all fragment types
python3 preprocessing/jain/step4_build_canonical_sets.py  # VH-only benchmarks

IMPORTANT: step4 is OPTIONAL and creates VH-only test sets.
           Do NOT run step4 if you need the full Novo parity dataset with both classes.
```

---

## Summary

**Is step4 bogus?**
- **Partially.** VH-only sets are useful, but the `jain_86_novo_parity.csv` section is wrong.

**Will users break things?**
- **Yes, if they run step4 after step2**, they'll overwrite correct labels with all-zeros.

**What's the fix?**
- **Delete lines 175-189 from step4** so it only creates VH-only files and stops overwriting step2.

**What does step2 do?**
- **Creates the CORRECT Novo parity dataset** with 59 specific + 27 non-specific antibodies that can match the confusion matrix [[40, 19], [10, 17]].

---

**Bottom line:** step4 had a FILE COLLISION BUG and has been DELETED (2025-11-07).

## Resolution (2025-11-07)

**Action Taken:** Deleted `preprocessing/jain/step4_build_canonical_sets.py`

**Rationale:**
- step4 was LEGACY CODE from before P5e-S2 implementation
- It created VH-only files using outdated heuristics (VH length filtering)
- It hardcoded all labels to 0, making it impossible to match Novo's confusion matrix
- It overwrote step2's correct output whenever run

**Official Jain Pipeline:**
```bash
# Step 1: Convert Excel to CSV
python3 preprocessing/jain/step1_convert_excel_to_csv.py

# Step 2: Create Novo parity dataset (59 specific + 27 non-specific)
python3 preprocessing/jain/step2_preprocess_p5e_s2.py

# Step 3 (Optional): Extract fragment files for VH/VL/CDR slices
python3 preprocessing/jain/step3_extract_fragments.py
```

**Result:** No more file collision, canonical dataset always has correct 59/27 split.
