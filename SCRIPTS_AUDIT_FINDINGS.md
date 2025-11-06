# Scripts & Preprocessing Audit - Full Findings Report

**Date:** 2025-11-05
**Branch:** leroy-jenkins/full-send
**Status:** 🔍 RESEARCH COMPLETE - Awaiting Senior Approval
**Safety:** ✅ SAFE (main branch backup at commit a868338)

---

## Executive Summary

**Audit Scope:** All scripts in `preprocessing/` and `scripts/` directories

**Key Findings:**
- 🚨 **1 P0 Critical Issue:** Duplicate preprocessing script with wrong paths
- ⚠️  **1 P1 Path Issue:** Missing subdirectory in demo script
- ✅ **0 P2 Issues:** Threshold work (0.5 vs 0.5495) is all SAFE and correct
- ✅ **Legacy folder pattern:** Working well (scripts/conversion/legacy/)

**Bottom Line:**
- **DELETE:** 1 bad preprocessing script
- **FIX:** 1 path in demo script
- **KEEP:** ALL threshold analysis work (rethreshold_harvey.py, etc.)

---

## 🚨 P0 Critical: Duplicate Preprocessing Scripts

### Issue: preprocessing/process_jain.py (BAD - DELETE THIS)

**Problem:** Uses OLD flat directory paths that NO LONGER EXIST

**Bad Paths in script:**
```python
INPUT_137 = "test_datasets/jain_with_private_elisa_FULL.csv"  # ❌ File doesn't exist
INPUT_SD03 = "test_datasets/jain_sd03.csv"                     # ❌ File doesn't exist
OUTPUT_116 = "test_datasets/jain_ELISA_ONLY_116.csv"           # ❌ File doesn't exist
OUTPUT_86 = "test_datasets/jain/jain_86_novo_parity.csv"       # ❌ File doesn't exist
```

**Verification:**
```bash
$ ls test_datasets/jain_with_private_elisa_FULL.csv
ls: test_datasets/jain_with_private_elisa_FULL.csv: No such file or directory ❌

$ ls test_datasets/jain/jain_86_novo_parity.csv
ls: test_datasets/jain/jain_86_novo_parity.csv: No such file or directory ❌
```

**Risk Level:** 🔴 CRITICAL
- If run accidentally, script will FAIL immediately
- Could confuse users about which script to use
- Referenced in OLD docs that haven't been updated

**Correct Script:** `preprocessing/preprocess_jain_p5e_s2.py`

**Correct Paths:**
```python
INPUT_137 = "test_datasets/jain/processed/jain_with_private_elisa_FULL.csv"  # ✅ EXISTS
INPUT_SD03 = "test_datasets/jain/processed/jain_sd03.csv"                     # ✅ EXISTS
OUTPUT_116 = "test_datasets/jain/processed/jain_ELISA_ONLY_116.csv"           # ✅ EXISTS
OUTPUT_86 = "test_datasets/jain/canonical/jain_86_novo_parity.csv"            # ✅ EXISTS
```

**Documentation Reference:**
- `docs/jain/JAIN_REORGANIZATION_COMPLETE.md` line 52 confirms `preprocess_jain_p5e_s2.py` is current
- `test_datasets/jain/README.md` shows 4-tier structure (raw/, processed/, canonical/, fragments/)

**Recommendation:**
```bash
# DELETE THIS FILE
git rm preprocessing/process_jain.py
git rm -r preprocessing/__pycache__  # Clean up Python cache
```

**Git History:** Preserved via git log (can always recover if needed)

---

## ⚠️ P1 Path Issues

### Issue: scripts/testing/demo_assay_specific_thresholds.py

**Problem:** Missing `canonical/` subdirectory in Jain path

**Location:** Line 84

**Current (WRONG):**
```python
jain_cm, jain_acc = test_with_assay_type(
    model,
    "test_datasets/jain/VH_only_jain_test_QC_REMOVED.csv",  # ❌ Missing canonical/
    "Jain",
    "ELISA",
    novo_jain,
)
```

**Correct:**
```python
jain_cm, jain_acc = test_with_assay_type(
    model,
    "test_datasets/jain/canonical/VH_only_jain_test_QC_REMOVED.csv",  # ✅ Add canonical/
    "Jain",
    "ELISA",
    novo_jain,
)
```

**Verification:**
```bash
$ ls test_datasets/jain/VH_only_jain_test_QC_REMOVED.csv
ls: test_datasets/jain/VH_only_jain_test_QC_REMOVED.csv: No such file or directory ❌

$ ls test_datasets/jain/canonical/VH_only_jain_test_QC_REMOVED.csv
-rw-r--r--@ 1 ray  staff  15K Nov  5 19:03 test_datasets/jain/canonical/VH_only_jain_test_QC_REMOVED.csv ✅
```

**Risk Level:** 🟡 MEDIUM
- Script will fail when run
- Easy fix (add one subdirectory)
- Doesn't corrupt data

**Recommendation:**
```python
# Fix line 84 in demo_assay_specific_thresholds.py
"test_datasets/jain/canonical/VH_only_jain_test_QC_REMOVED.csv"
```

---

## ✅ Verified SAFE Scripts

### Threshold Analysis Work (User Requested to Keep)

**All threshold work is SAFE and uses correct paths:**

#### 1. scripts/rethreshold_harvey.py ✅
- **Path:** `test_datasets/harvey/fragments/VHH_only_harvey.csv` (line 48)
- **Status:** ✅ Correct 4-tier path
- **File exists:** YES
- **Purpose:** Experiments with 0.5 vs 0.5495 thresholds for Harvey
- **Action:** KEEP (user explicitly requested)

#### 2. scripts/analysis/analyze_threshold_optimization.py ✅
- **Status:** ✅ Generic analysis script, no hardcoded paths
- **Purpose:** Threshold optimization analysis
- **Action:** KEEP

#### 3. scripts/testing/demo_assay_specific_thresholds.py ⚠️
- **Status:** ⚠️ Has path issue (see P1 above) but script logic is sound
- **Purpose:** Demo of assay-specific thresholds (PSR vs ELISA)
- **Paths:**
  - Jain: Missing `canonical/` subdirectory (FIXABLE)
  - Shehata: `test_datasets/shehata/fragments/VH_only_shehata.csv` ✅ CORRECT
- **Action:** KEEP + FIX path

---

### Core Preprocessing Scripts (All Correct)

#### 1. preprocessing/preprocess_jain_p5e_s2.py ✅
- **Status:** ✅ CURRENT, correct 4-tier paths
- **Purpose:** P5e-S2 method for Jain 86-antibody benchmark
- **Paths:** All use `jain/processed/` and `jain/canonical/`
- **Action:** KEEP

#### 2. preprocessing/process_harvey.py ✅
- **Status:** ✅ Correct 4-tier paths
- **Paths:**
  - Input: `harvey/raw/*.csv` ✅
  - Output: `harvey/processed/harvey.csv` ✅
- **Action:** KEEP

#### 3. preprocessing/process_shehata.py ✅
- **Status:** ✅ Correct 4-tier paths
- **Action:** KEEP

#### 4. preprocessing/boughter/ (7 scripts) ✅
- **Status:** ✅ All correct, well-organized subdirectory
- **Scripts:**
  - stage1_dna_translation.py ✅
  - stage2_stage3_annotation_qc.py ✅
  - stage4_additional_qc.py ✅
  - validate_stage1.py ✅
  - validate_stages2_3.py ✅
  - validate_stage4.py ✅
  - README.md ✅ (excellent documentation)
- **Action:** KEEP ALL

---

### Conversion Scripts (All Correct)

#### 1. scripts/conversion/convert_harvey_csvs.py ✅
- **Paths:**
  - Input: `harvey/raw/high_polyreactivity_high_throughput.csv` ✅
  - Input: `harvey/raw/low_polyreactivity_high_throughput.csv` ✅
  - Output: `harvey/processed/harvey.csv` ✅
- **Action:** KEEP

#### 2. scripts/conversion/convert_jain_excel_to_csv.py ✅
- **Paths:**
  - Input: `jain/raw/*.xlsx` ✅
  - Output: `jain/processed/*.csv` ✅
- **Action:** KEEP

#### 3. scripts/conversion/convert_shehata_excel_to_csv.py ✅
- **Paths:**
  - Input: `shehata/raw/shehata-mmc2.xlsx` ✅
  - Output: `shehata/processed/shehata.csv` ✅
- **Action:** KEEP

#### 4. scripts/conversion/legacy/ (2 scripts) ✅
- **convert_jain_excel_to_csv_OLD_BACKUP.py** - Properly archived
- **convert_jain_excel_to_csv_TOTAL_FLAGS_WRONG.py** - Properly archived
- **README.md** - Documents why these are wrong
- **Action:** KEEP (good practice to preserve history)

---

### Testing Scripts (All Correct)

#### 1. scripts/testing/test_harvey_psr_threshold.py ✅
- **Status:** ✅ Correct paths
- **Purpose:** Harvey test (141K nanobodies, 61.5% accuracy)
- **Action:** KEEP

#### 2. scripts/testing/test_jain_novo_parity.py ✅
- **Status:** ✅ Correct paths (uses `jain/canonical/`)
- **Purpose:** Jain test (86 antibodies, 66.28% accuracy)
- **Action:** KEEP

---

### Validation Scripts (All Correct)

#### 1. scripts/validation/validate_fragments.py ✅
- **Status:** ✅ Generic validation, no hardcoded paths
- **Action:** KEEP

#### 2. scripts/validation/validate_jain_conversion.py ✅
- **Status:** ✅ Updated to use `jain/raw/` and `jain/processed/`
- **Action:** KEEP

#### 3. scripts/validation/validate_shehata_conversion.py ✅
- **Status:** ✅ Correct paths
- **Action:** KEEP

---

### Analysis Scripts (All Correct)

#### 1. scripts/analysis/compare_jain_methodologies.py ✅
- **Status:** ✅ Correct paths
- **Purpose:** Compare different Jain filtering methods
- **Action:** KEEP

#### 2. scripts/analysis/zscore_jain_116_outliers.py ✅
- **Status:** ✅ Correct paths
- **Purpose:** Z-score analysis of Jain 116-antibody set
- **Action:** KEEP

---

### Root-Level Scripts

#### 1. scripts/audit_boughter_training_qc.py ✅
- **Status:** ✅ Correct paths (train_datasets/boughter/)
- **Purpose:** Audit Boughter training QC
- **Recommendation:** Could move to `scripts/validation/` for organization
- **Action:** KEEP (optional: move to validation/)

#### 2. scripts/verify_novo_parity.py ✅
- **Status:** ✅ Correct paths
- **Purpose:** Verify Novo Nordisk parity across all benchmarks
- **Recommendation:** Could move to `scripts/validation/` for organization
- **Action:** KEEP (optional: move to validation/)

#### 3. scripts/train_hyperparameter_sweep.py ✅
- **Status:** ✅ Correct paths
- **Purpose:** Hyperparameter tuning
- **Recommendation:** Could create `scripts/training/` for organization
- **Action:** KEEP (optional: move to new training/ dir)

---

## Summary of Issues

### P0 Critical (1 issue)

| File | Issue | Risk | Action |
|------|-------|------|--------|
| `preprocessing/process_jain.py` | Wrong paths, duplicate | 🔴 HIGH | DELETE |

### P1 Path Issues (1 issue)

| File | Line | Issue | Risk | Action |
|------|------|-------|------|--------|
| `scripts/testing/demo_assay_specific_thresholds.py` | 84 | Missing `canonical/` | 🟡 MED | FIX PATH |

### P2 Organizational (3 files)

| File | Issue | Risk | Action |
|------|-------|------|--------|
| `scripts/audit_boughter_training_qc.py` | Uncategorized | 🟢 LOW | Optional: move to validation/ |
| `scripts/verify_novo_parity.py` | Uncategorized | 🟢 LOW | Optional: move to validation/ |
| `scripts/train_hyperparameter_sweep.py` | Uncategorized | 🟢 LOW | Optional: move to training/ |

---

## Documentation Issues Found

### Old Documentation References

These docs reference the BAD script (`process_jain.py`) and need updating:

1. **scripts/conversion/legacy/README.md** (line 45)
   - Says: "Use `preprocessing/process_jain.py`"
   - Should say: "Use `preprocessing/preprocess_jain_p5e_s2.py`"

2. **docs/harvey/harvey_data_cleaning_log.md** (line 3)
   - Says: "adapt from process_jain.py"
   - Not critical (Harvey already done), but could update for clarity

3. **docs/harvey/harvey_preprocessing_implementation_plan.md**
   - Says: "Adapt from `process_jain.py`"
   - Not critical (Harvey already done)

4. **docs/jain/jain_data_sources.md**
   - Says: "python3 preprocessing/process_jain.py"
   - Should say: "python3 preprocessing/preprocess_jain_p5e_s2.py"

5. **docs/jain/jain_conversion_verification_report.md** (line 2)
   - Says: "via `python3 preprocessing/process_jain.py`"
   - Should say: "via `python3 preprocessing/preprocess_jain_p5e_s2.py`"

**Note:** `docs/jain/JAIN_REORGANIZATION_COMPLETE.md` is CORRECT (uses preprocess_jain_p5e_s2.py)

---

## Recommended Actions

### Phase 1: Delete Bad Script (P0)

```bash
# Delete duplicate script with wrong paths
git rm preprocessing/process_jain.py

# Clean up Python cache
git rm -r preprocessing/__pycache__

# Commit
git commit -m "fix: Delete preprocessing/process_jain.py (wrong paths, superseded by preprocess_jain_p5e_s2.py)"
```

**Justification:**
- File uses paths that don't exist
- Superseded by `preprocess_jain_p5e_s2.py`
- Git history preserved (can recover if needed)
- Eliminates confusion about which script is current

---

### Phase 2: Fix Path Issue (P1)

**File:** `scripts/testing/demo_assay_specific_thresholds.py`

**Change line 84:**
```python
# OLD (wrong)
"test_datasets/jain/VH_only_jain_test_QC_REMOVED.csv"

# NEW (correct)
"test_datasets/jain/canonical/VH_only_jain_test_QC_REMOVED.csv"
```

**Commit:**
```bash
git add scripts/testing/demo_assay_specific_thresholds.py
git commit -m "fix: Correct Jain path in demo_assay_specific_thresholds.py (add canonical/ subdirectory)"
```

---

### Phase 3: Update Documentation (Optional but Recommended)

**Files to update:**
1. `scripts/conversion/legacy/README.md` line 45
2. `docs/jain/jain_data_sources.md`
3. `docs/jain/jain_conversion_verification_report.md`

**Change all references from:**
```
preprocessing/process_jain.py
```

**To:**
```
preprocessing/preprocess_jain_p5e_s2.py
```

**Commit:**
```bash
git add docs/jain/*.md scripts/conversion/legacy/README.md
git commit -m "docs: Update Jain preprocessing script references (process_jain.py → preprocess_jain_p5e_s2.py)"
```

---

### Phase 4: Optional Reorganization (P2)

**If desired, create better organization:**

```bash
# Option 1: Create scripts/training/ for training scripts
mkdir scripts/training
git mv scripts/train_hyperparameter_sweep.py scripts/training/
echo "# Training Scripts" > scripts/training/README.md

# Option 2: Move validation scripts to scripts/validation/
git mv scripts/audit_boughter_training_qc.py scripts/validation/
git mv scripts/verify_novo_parity.py scripts/validation/

# Option 3: Move analysis script to scripts/analysis/
git mv scripts/rethreshold_harvey.py scripts/analysis/

# Update READMEs to list new scripts
# (manual editing required)

# Commit
git add -A
git commit -m "refactor: Organize scripts into proper subdirectories"
```

**Note:** This is OPTIONAL for cleaner organization. Not required for functionality.

---

## Verification Checklist

After making changes, verify:

### Preprocessing Scripts
- [ ] `preprocessing/preprocess_jain_p5e_s2.py` exists and works ✅
- [ ] `preprocessing/process_harvey.py` exists and works ✅
- [ ] `preprocessing/process_shehata.py` exists and works ✅
- [ ] `preprocessing/boughter/` directory intact (7 scripts) ✅
- [ ] `preprocessing/process_jain.py` deleted ✅

### Scripts Paths
- [ ] `scripts/testing/demo_assay_specific_thresholds.py` path fixed ✅
- [ ] All conversion scripts use 4-tier paths ✅
- [ ] All testing scripts use 4-tier paths ✅
- [ ] All validation scripts use 4-tier paths ✅

### Documentation
- [ ] Jain docs reference correct script ✅
- [ ] Legacy folder READMEs updated ✅
- [ ] No broken links to deleted script ✅

### Functionality
- [ ] Run Jain preprocessing to verify:
  ```bash
  python3 preprocessing/preprocess_jain_p5e_s2.py
  ```
  Expected: Generates `jain_86_novo_parity.csv` in `test_datasets/jain/canonical/`

- [ ] Run demo script to verify fix:
  ```bash
  python3 scripts/testing/demo_assay_specific_thresholds.py
  ```
  Expected: No file not found errors

- [ ] Run Novo parity verification:
  ```bash
  python3 scripts/verify_novo_parity.py
  ```
  Expected: All benchmarks pass

---

## Safety Notes

**Current Status:**
- Branch: `leroy-jenkins/full-send`
- Backup: `main` (synced to same commit a868338)
- Upstream: Removed (complete independence)

**Rollback Plan:**
If anything breaks:
```bash
# Check what we changed
git log --oneline | head -5

# Revert to backup
git reset --hard a868338

# Or revert specific commits
git revert <commit-hash>
```

**Git History Preservation:**
- All deletions via `git rm` (preserves history)
- Can recover deleted files: `git checkout <commit> -- <file>`
- Example: `git checkout 349318e -- preprocessing/process_jain.py`

---

## What We're NOT Touching

**User explicitly wants to KEEP:**
- ✅ `scripts/rethreshold_harvey.py` - Threshold work (0.5 vs 0.5495)
- ✅ `scripts/analysis/analyze_threshold_optimization.py` - Threshold analysis
- ✅ `scripts/testing/demo_assay_specific_thresholds.py` - Assay-specific thresholds (just fixing path)

**All threshold work is SAFE and stays as-is (except minor path fix in demo script)**

---

## Execution Plan Summary

**Required Actions (Phases 1-2):**
1. Delete `preprocessing/process_jain.py` (wrong paths)
2. Fix path in `demo_assay_specific_thresholds.py` (add `canonical/`)
3. Update 3-5 docs to reference correct script

**Optional Actions (Phases 3-4):**
4. Move 3 root-level scripts to subdirectories for organization
5. Create `scripts/training/` directory
6. Update subdirectory READMEs

**Time Estimate:**
- Phase 1-2: 5 minutes (required fixes)
- Phase 3-4: 15 minutes (optional organization)

**Risk Level:** 🟢 LOW (have backup, can rollback anytime)

---

## References

- **Dataset reorganization:** `docs/TEST_DATASETS_REORGANIZATION_PLAN.md`
- **Jain cleanup:** `docs/jain/JAIN_REORGANIZATION_COMPLETE.md`
- **Harvey cleanup:** `docs/harvey/HARVEY_CLEANUP_VERIFICATION.md`
- **Boughter pipeline:** `preprocessing/boughter/README.md`
- **4-tier structure:** `test_datasets/*/README.md`

---

## Questions for Senior Approval

1. **Delete vs Archive:** Delete `preprocessing/process_jain.py` entirely, or move to `preprocessing/legacy/`?
   - **Recommendation:** DELETE (git history preserves it anyway)

2. **Documentation updates:** Update 3-5 docs now, or later?
   - **Recommendation:** Update NOW (prevents confusion)

3. **Optional reorganization:** Move root-level scripts to subdirectories?
   - **Recommendation:** YES (cleaner structure, but not critical)

4. **Commit strategy:** Single commit or multiple commits per phase?
   - **Recommendation:** Multiple commits (clearer git history)

---

**Audit Completed:** 2025-11-05
**Auditor:** Claude Code
**Status:** 🟡 AWAITING SENIOR APPROVAL
**Ready to Execute:** ✅ YES (after approval)
