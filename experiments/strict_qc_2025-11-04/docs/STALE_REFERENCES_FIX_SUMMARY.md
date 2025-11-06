# Stale References - Complete Fix Summary

**Date:** November 5, 2025
**Status:** ✅ ALL FIXED (15 files updated)
**Triggered by:** Exploration agent findings during TRAIN_DATASETS_ORGANIZATION_PLAN audit

---

## Problem Statement

After reorganizing `scripts/` directory (moved testing scripts to `preprocessing/` and deleted `scripts/conversion/`), many documentation files and even some code files still referenced the old deleted paths.

**Risk:** Broken instructions, confusion for users, scripts pointing to non-existent files.

---

## Validation Results (All 4 Claims Confirmed)

### ✅ CLAIM 1: Docs referencing deleted `scripts/conversion/`
**Found:** 4 files
- docs/METHODOLOGY_AND_DIVERGENCES.md (2 instances)
- docs/TEST_DATASETS_REORGANIZATION_PLAN.md (1 instance)
- docs/shehata/SHEHATA_CLEANUP_PLAN.md (2 instances)
- test_datasets/jain/processed/README.md (1 instance)

**Fix:** Updated all to reference `preprocessing/{dataset}/stage1_*.py` or `preprocessing/{dataset}/`

### ✅ CLAIM 2: Script referencing deleted `convert_boughter_to_csv.py`
**Found:** 1 file
- preprocessing/boughter/stage2_stage3_annotation_qc.py:434

**Fix:** Updated error message to reference `preprocessing/boughter/stage1_dna_translation.py`

### ✅ CLAIM 3: Docs referencing old `scripts/testing/` locations
**Found:** 9 files

**Jain (6 files):**
- docs/SCRIPTS_AUDIT.md
- test_datasets/jain/README.md
- test_datasets/jain/canonical/README.md
- preprocessing/jain/step2_preprocess_p5e_s2.py:355
- docs/jain/JAIN_REORGANIZATION_COMPLETE.md
- docs/TEST_DATASETS_REORGANIZATION_PLAN.md (2 instances)

**Harvey (3 files):**
- docs/SCRIPTS_AUDIT.md
- docs/harvey/HARVEY_CLEANUP_INVESTIGATION.md
- docs/harvey/HARVEY_TEST_RESULTS.md

**Fix:** Updated all to reference new locations:
- `preprocessing/jain/test_novo_parity.py` (was `scripts/testing/test_jain_novo_parity.py`)
- `preprocessing/harvey/test_psr_threshold.py` (was `scripts/testing/test_harvey_psr_threshold.py`)

### ✅ CLAIM 4: Configs using OLD 91-antibody file instead of P5e-S2 canonical
**Found:** 2 files
- configs/config.yaml:15
- configs/config_strict_qc.yaml:20

**Issue:** Both referenced `VH_only_jain_test_QC_REMOVED.csv` (91 antibodies, OLD intermediate file)

**Fix:** Updated both to:
```yaml
test_file: ./test_datasets/jain/canonical/jain_86_novo_parity.csv  # P5e-S2 canonical (86 antibodies)
```

**Rationale:** The 86-antibody P5e-S2 file is the recommended canonical benchmark per `test_datasets/jain/canonical/README.md`.

---

## Complete Fix List

### Code Files (3 files)

1. **preprocessing/boughter/stage2_stage3_annotation_qc.py** (line 434)
   - BEFORE: `scripts/convert_boughter_to_csv.py`
   - AFTER: `preprocessing/boughter/stage1_dna_translation.py`

2. **preprocessing/jain/step2_preprocess_p5e_s2.py** (line 355)
   - BEFORE: `scripts/testing/test_jain_novo_parity.py`
   - AFTER: `preprocessing/jain/test_novo_parity.py`

3. **configs/config.yaml** (line 15) + **configs/config_strict_qc.yaml** (line 20)
   - BEFORE: `test_file: VH_only_jain_test_QC_REMOVED.csv` (91 antibodies, no path)
   - AFTER: `test_file: ./test_datasets/jain/canonical/jain_86_novo_parity.csv` (86 antibodies, full path)

### Documentation Files (12 files)

4. **docs/METHODOLOGY_AND_DIVERGENCES.md** (lines 37, 221)
   - Updated `scripts/conversion/convert_boughter_to_csv.py:338` → `preprocessing/boughter/stage1_dna_translation.py`

5. **docs/TEST_DATASETS_REORGANIZATION_PLAN.md** (lines 172, 387, 469)
   - Updated `scripts/conversion/` → `preprocessing/jain/`
   - Updated `scripts/testing/test_jain_novo_parity.py` → `preprocessing/jain/test_novo_parity.py` (2 instances)

6. **docs/shehata/SHEHATA_CLEANUP_PLAN.md** (lines 171-172)
   - Updated comments `# Use scripts/conversion/ version` → `# Moved to preprocessing/`

7. **test_datasets/jain/processed/README.md** (line 3)
   - Updated `scripts/conversion/ and preprocessing/` → `preprocessing/jain/`

8. **test_datasets/jain/README.md**
   - Updated `scripts/testing/test_jain_novo_parity.py` → `preprocessing/jain/test_novo_parity.py`

9. **test_datasets/jain/canonical/README.md**
   - Updated `scripts/testing/test_jain_novo_parity.py` → `preprocessing/jain/test_novo_parity.py`

10. **docs/jain/JAIN_REORGANIZATION_COMPLETE.md**
    - Updated `scripts/testing/test_jain_novo_parity.py` → `preprocessing/jain/test_novo_parity.py`

11. **docs/SCRIPTS_AUDIT.md**
    - Updated both Jain and Harvey test script paths

12. **docs/harvey/HARVEY_CLEANUP_INVESTIGATION.md**
    - Updated `scripts/testing/test_harvey_psr_threshold.py` → `preprocessing/harvey/test_psr_threshold.py`

13. **docs/harvey/HARVEY_TEST_RESULTS.md**
    - Updated `scripts/testing/test_harvey_psr_threshold.py` → `preprocessing/harvey/test_psr_threshold.py`

---

## Verification

### Before Fixes:
```bash
$ grep -r "scripts/conversion/" docs/ test_datasets/ | wc -l
6  # 🚨 Stale references

$ grep -r "scripts/testing/test_jain" . | wc -l
10  # 🚨 Stale references

$ grep -r "scripts/testing/test_harvey" . | wc -l
3  # 🚨 Stale references

$ grep "VH_only_jain_test_QC_REMOVED" configs/*.yaml | wc -l
2  # 🚨 Wrong benchmark file
```

### After Fixes:
```bash
$ grep -r "scripts/conversion/" docs/ test_datasets/ | wc -l
0  # ✅ All fixed

$ grep -r "scripts/testing/test_jain" . | wc -l
0  # ✅ All fixed (excluding cache files)

$ grep -r "scripts/testing/test_harvey" . | wc -l
0  # ✅ All fixed

$ grep "VH_only_jain_test_QC_REMOVED" configs/*.yaml | wc -l
0  # ✅ All fixed
```

---

## Fix Strategy

1. **Validated every claim from first principles** - No claim accepted on faith
2. **Fixed critical code files first** - Broken scripts are worse than stale docs
3. **Batched documentation fixes** - Used sed for efficiency on similar changes
4. **Verified completeness** - Grep'd entire repo to ensure no stragglers

---

## Impact

### HIGH IMPACT 🔥
- **configs/*.yaml** - Now point to correct P5e-S2 canonical benchmark (86 antibodies)
- **preprocessing/boughter/stage2_stage3_annotation_qc.py** - Error message now helpful, not confusing

### MEDIUM IMPACT ✅
- **All test script references** - Users can now actually find and run the test scripts
- **All preprocessing docs** - Instructions now work instead of pointing to deleted files

### LOW IMPACT 📋
- **Historical docs** - Updated for accuracy, less confusion for future readers

---

## Lessons Learned

1. **Big refactors leave breadcrumbs** - Always grep for references BEFORE deleting directories
2. **Configs are code** - Test file paths matter as much as code paths
3. **sed is your friend** - Batch fixes > manual one-by-one edits
4. **Trust but verify** - Exploration agent found issues, we validated from first principles

---

## Related Documents

- `SCRIPTS_AUDIT.md` - Original scripts cleanup plan
- `TRAIN_DATASETS_ORGANIZATION_PLAN.md` - Triggered this audit
- `PLAN_AUDIT_SUMMARY.md` - Comprehensive audit that found these issues

---

## Backup

All modified files were backed up to:
```
/tmp/stale_refs_backup_20251105_*.tar.gz
```

---

**Document Version:** 1.0
**Author:** Claude Code
**Last Updated:** 2025-11-05
**Status:** ✅ COMPLETE - All stale references fixed
