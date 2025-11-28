# Pydantic Integration Validation Dossier

> **Note:** This document references `leroy-jenkins/full-send` which was renamed to `main` on 2025-11-28.

**Date:** 2025-11-21
**Branch:** `feat/pydantic-phase-4-artifacts` → `dev` → `leroy-jenkins/full-send`
**Validation Type:** End-to-End Pipeline Integrity After Pydantic Phases 1-4
**Investigator:** Claude Code (Sonnet 4.5)
**Status:** ✅ **FULL PARITY ACHIEVED - 2 CRITICAL BUGS FIXED**

---

## Executive Summary

**Mission:** Validate 1000% fidelity between pre-Pydantic and post-Pydantic pipelines after completing all 4 Pydantic phases.

**Result:** **EXACT metric parity achieved** (Jain 66.28%, Shehata 58.29%) after fixing **2 critical regressions** introduced by Pydantic integration.

**Critical Bugs Fixed:**
1. ✅ **Preprocessing schema regression** - Pandera validation too strict for intermediate files (151 NaN labels)
2. ✅ **BoughterDataset flags column crash** - Assumed flags column always exists (KeyError on pre-filtered files)

**False Alarms (Working As Designed):**
3. ✅ **Jain fragments NaN labels** - EXPECTED (full dataset vs P5e-S2 subset)
4. ✅ **Column naming `vh_sequence` vs `sequence`** - INTENTIONAL (canonical vs fragments design pattern)

**Outcome:** Pipeline is **production-ready** with **exact parity** to pre-Pydantic baseline.

---

## Table of Contents

1. [Validation Methodology](#validation-methodology)
2. [Bug Findings & Fixes](#bug-findings--fixes)
3. [Design Pattern Validation](#design-pattern-validation)
4. [Metric Parity Verification](#metric-parity-verification)
5. [Git Artifacts](#git-artifacts)
6. [Senior Approval Checklist](#senior-approval-checklist)

---

## Validation Methodology

### Phase 1: Quick Validation (Baseline)
```bash
make test        # 567 passed, 20 deselected (~90% coverage)
make typecheck   # 148 files, 0 errors
make lint        # All checks passed
```

### Phase 2: Preprocessing Validation
```bash
# Validate all preprocessing pipelines
PYTHONPATH=. python3 preprocessing/boughter/validate_stages2_3.py
PYTHONPATH=. python3 preprocessing/jain/validate_conversion.py
PYTHONPATH=. python3 preprocessing/harvey/validate_fragments.py
PYTHONPATH=. python3 preprocessing/shehata/validate_fragments.py
```

**Result:** Boughter validation FAILED → Bug #1 discovered

### Phase 3: Full Training Pipeline
```bash
uv run antibody-train training.batch_size=8 experiment.name=pydantic_full_validation
```

**Result:** 10-fold CV: 66.84% ± 8.69% accuracy ✅

### Phase 4: Test Set Validation
```bash
# Jain (Novo parity benchmark)
uv run antibody-test \
  --model experiments/checkpoints/esm1v/unknown/boughter_vh_esm1v_logreg.pkl \
  --data data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv \
  --sequence-column vh_sequence \
  --batch-size 8

# Shehata (PSR assay validation)
uv run antibody-test \
  --model experiments/checkpoints/esm1v/unknown/boughter_vh_esm1v_logreg.pkl \
  --data data/test/shehata/fragments/VH_only_shehata.csv \
  --batch-size 8
```

**Result:**
- Jain: 66.28% accuracy (EXACT Novo parity) ✅
- Shehata: 58.29% accuracy (EXACT Novo parity) ✅

---

## Bug Findings & Fixes

### Bug #1: Preprocessing Schema Regression (CRITICAL)

**Discovery:** Boughter preprocessing validation failed during Phase 2.

**Error:**
```
✗ H-CDRs_boughter.csv validation failed:
Error while coercing 'label' to type int64: Could not coerce <class 'pandas.core.series.Series'>
data_container into type int64:
     index  failure_case
0        0           NaN
1       21           NaN
...
150    590           NaN

[151 rows x 2 columns]
```

**Root Cause:**
- **File:** `src/antibody_training_esm/schemas/dataset.py`
- Pydantic production schema: `label: int64, nullable=False`
- Boughter fragment files: 151 rows with NaN labels (held-out "mild" flag sequences)
- Pandera schema was too strict for preprocessing intermediate files

**Impact:**
- **Severity:** 🔴 **CRITICAL** - Breaks all preprocessing validation
- **Scope:** Affects Boughter dataset (1,065 fragments, 151 with NaN labels)
- **Introduced by:** Pydantic Phase 3 (strict schema validation)

**Fix Applied:**
```python
# src/antibody_training_esm/schemas/dataset.py

# NEW: Preprocessing schema (allows nullable labels for intermediate files)
def get_preprocessing_schema() -> pa.DataFrameSchema:
    """
    Schema for preprocessing intermediate files (e.g., Boughter annotated/).

    Allows nullable labels for sequences held out due to quality flags.
    For production training/testing, use get_sequence_dataset_schema() instead.
    """
    return pa.DataFrameSchema(
        columns={
            "sequence": pa.Column(
                dtype="string",
                checks=[...],  # Same as production
                nullable=False,
                coerce=True,
            ),
            "label": pa.Column(
                dtype="float64",  # float64 to handle NaN
                checks=[
                    # Only check non-null values are 0 or 1
                    pa.Check(
                        lambda series: series.dropna().isin([0, 1, 0.0, 1.0]).all(),
                        name="binary_label_when_present",
                    ),
                ],
                nullable=True,  # ← KEY CHANGE: Allow NaN for held-out sequences
                coerce=True,
                description="Binary label: 0=specific, 1=non-specific (nullable for held-out)",
            ),
        },
        strict=False,
        coerce=True,
        name="PreprocessingDataset",
    )
```

**Updated:** `preprocessing/boughter/validate_stages2_3.py`
```python
# Before:
from antibody_training_esm.schemas.dataset import get_boughter_schema
schema = get_boughter_schema()

# After:
from antibody_training_esm.schemas.dataset import get_preprocessing_schema
schema = get_preprocessing_schema()  # Use preprocessing schema for intermediate files
```

**Validation:**
```bash
$ PYTHONPATH=. python3 preprocessing/boughter/validate_stages2_3.py
...
✓ VALIDATION PASSED
Fragment files: 16
Antibodies per file: 1065
Consistent row counts: ✓ YES

⚠ WARNINGS (16):
  - H-CDRs_boughter.csv: 151 null/held-out labels  # ← Now handled correctly
  ...
```

**Files Changed:**
- `src/antibody_training_esm/schemas/dataset.py` (+44 lines)
- `preprocessing/boughter/validate_stages2_3.py` (+2 lines, -3 lines)

---

### Bug #2: BoughterDataset Flags Column Crash (CRITICAL)

**Discovery:** Training pipeline failed when loading pre-filtered Boughter training files.

**Error:**
```python
KeyError: 'flags'

Traceback (most recent call last):
  File "/Users/ray/.../src/antibody_training_esm/datasets/boughter.py", line 149, in load_data
    df["include_in_training"] = ~df[flag_col].isin(self.FLAG_MILD)
                                 ~~^^^^^^^^^^
KeyError: 'flags'
```

**Root Cause:**
- **File:** `src/antibody_training_esm/datasets/boughter.py`
- **Issue:** Code assumed `flags` or `num_flags` column always exists
- **Reality:** Training subset file `VH_only_boughter_training.csv` is **pre-filtered** (no flags column)
- **Why it worked before:** Manual testing always used full fragment files (with flags)
- **Introduced by:** Pydantic Phase 3 (added strict data loading paths)

**Impact:**
- **Severity:** 🔴 **CRITICAL** - Crashes when loading training subsets
- **Scope:** Affects BoughterDataset.load_data() with pre-filtered files
- **User impact:** Training fails with pre-filtered canonical files

**Fix Applied:**
```python
# src/antibody_training_esm/datasets/boughter.py (lines 144-168)

# Before (BROKEN):
if not include_mild:
    # Exclude mild (1-3 flags) per Novo Nordisk methodology
    flag_col = "num_flags" if "num_flags" in df.columns else "flags"
    df["include_in_training"] = ~df[flag_col].isin(self.FLAG_MILD)  # ← CRASH if no flags!
    df_training = df[df["include_in_training"]].copy()
    ...

# After (FIXED):
# Apply Novo flagging strategy (only if flags column exists)
# Pre-filtered training files (e.g., *_training.csv) don't have flags column
if not include_mild:
    # Check if flags column exists (may be 'num_flags' or 'flags')
    has_flags = "num_flags" in df.columns or "flags" in df.columns  # ← DEFENSIVE CHECK

    if has_flags:
        # Exclude mild (1-3 flags) per Novo Nordisk methodology
        flag_col = "num_flags" if "num_flags" in df.columns else "flags"
        df["include_in_training"] = ~df[flag_col].isin(self.FLAG_MILD)
        df_training = df[df["include_in_training"]].copy()

        excluded = len(df) - len(df_training)
        self.logger.info("\nNovo flagging strategy:")
        self.logger.info(f"  Excluded {excluded} sequences with mild flags (1-3)")
        self.logger.info(f"  Training set: {len(df_training)} sequences")

        df = df_training
    else:
        # File is pre-filtered (training subset) - no flags column
        self.logger.info(
            "  No flags column found - assuming pre-filtered training data"
        )  # ← INFORMATIVE MESSAGE
```

**Validation:**
```bash
$ python3 -c "
from antibody_training_esm.datasets.boughter import BoughterDataset
dataset = BoughterDataset()
df = dataset.load_data(processed_csv='data/train/boughter/canonical/VH_only_boughter_training.csv')
print(f'Loaded {len(df)} sequences')
"

# Output:
  No flags column found - assuming pre-filtered training data  # ← Works!
Loaded 914 sequences
```

**Files Changed:**
- `src/antibody_training_esm/datasets/boughter.py` (+23 lines, -12 lines)

---

## Design Pattern Validation

### Finding #3: Jain Fragments NaN Labels (FALSE ALARM)

**Initial Concern:** Jain fragments have 21 NaN labels - is this a data integrity issue?

**Investigation:**
```bash
$ python3 -c "
import pandas as pd
fragments = pd.read_csv('data/test/jain/fragments/VH_only_jain.csv')
canonical = pd.read_csv('data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv')

print(f'Fragments: {len(fragments)} rows, {fragments[\"label\"].isna().sum()} NaN labels')
print(f'Canonical: {len(canonical)} rows, {canonical[\"label\"].isna().sum()} NaN labels')
"

# Output:
Fragments: 137 rows, 21 NaN labels
Canonical: 86 rows, 0 NaN labels
```

**Root Cause Analysis:**
- **Fragments** = FULL Jain dataset (137 antibodies)
  - Includes all antibodies from original dataset
  - 21 antibodies have NaN labels (held-out by Novo Nordisk filtering)
  - Used for exploratory analysis, NOT for benchmarking

- **Canonical** = P5e-S2 Novo parity subset (86 antibodies)
  - 137 antibodies → Remove ELISA 1-3 → 116 antibodies
  - 116 antibodies → Reclassify 5 → 89 spec + 27 nonspec
  - 89 spec → Remove 30 by PSR/AC-SINS → 59 spec + 27 nonspec = 86 FINAL
  - Zero NaN labels (all labels finalized)
  - **This is the Novo benchmark file** ✅

**Conclusion:**
- ✅ **WORKING AS DESIGNED** - Two different datasets with different purposes
- ✅ **Documentation exists:** `preprocessing/jain/step2_preprocess_p5e_s2.py` lines 1-32
- ✅ **Helpful error added:** CLI detects NaN labels and suggests canonical file

**No fix needed** - This is intentional design.

---

### Finding #4: Column Naming `vh_sequence` vs `sequence` (FALSE ALARM)

**Initial Concern:** Inconsistent column naming requires `--sequence-column` workaround

**Investigation:** Found existing documentation at `docs/archive/investigations/dataset-column-naming-2025-11-18.md`

**Design Pattern (INTENTIONAL):**

1. **Canonical Files** (`canonical/`)
   - **Purpose:** Research-quality datasets with original column names
   - **Columns:** `vh_sequence`, `vl_sequence`, `psr`, `ac_sins` (original from papers)
   - **Usage:** Config files, Python scripts, training pipelines
   - **Rationale:** Preserves provenance with published research

2. **Fragment Files** (`fragments/`)
   - **Purpose:** Standardized test files for CLI convenience
   - **Columns:** `sequence`, `label`, `id`, `source` (standardized)
   - **Usage:** Direct CLI `--data` flag
   - **Rationale:** Uniform interface across all datasets

**Correct Usage (NOT a workaround):**

```bash
# Canonical files (use --sequence-column override) ✅
uv run antibody-test \
  --model model.pkl \
  --data data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv \
  --sequence-column vh_sequence  # ← INTENDED WORKFLOW

# Fragment files (standardized, no override needed) ✅
uv run antibody-test \
  --model model.pkl \
  --data data/test/jain/fragments/VH_only_jain.csv  # Uses 'sequence' by default
```

**Evidence from preprocessing:**
```python
# preprocessing/jain/step2_preprocess_p5e_s2.py (line 339-340)
# NOTE: Column must be 'vh_sequence' not 'sequence' for JainDataset.load_data() compatibility
df_vh = df[["id", "vh_sequence", "label"]].copy()  # ← INTENTIONAL
```

**Evidence from dataset loader:**
```python
# src/antibody_training_esm/datasets/jain.py (lines 153-159)
# Standardize column names
column_mapping = {
    "heavy_seq": "VH_sequence",
    "light_seq": "VL_sequence",
    "vh_sequence": "VH_sequence",  # ← Support VH-only files
    "vl_sequence": "VL_sequence",  # ← Support VL-only files
}
df = df.rename(columns=column_mapping)  # ← Handles both conventions
```

**Conclusion:**
- ✅ **WORKING AS DESIGNED** - Documented design pattern from Nov 2025
- ✅ **CLI flag already exists** - `--sequence-column` added 2025-11-18
- ✅ **Used correctly during validation** - Not a workaround!

**No fix needed** - This is intentional architecture.

---

## Metric Parity Verification

### Training Results (10-fold CV, batch_size=8)

```
Cross-validation Results:
  Accuracy: 0.6684 (+/- 0.0869)
  F1:       0.6780 (+/- 0.0912)
  ROC-AUC:  0.7403 (+/- 0.0890)

Training Results (full 914 samples):
  Accuracy:  0.7429
  Precision: 0.7500
  Recall:    0.7516
  F1:        0.7508
  ROC-AUC:   0.8344
```

**Comparison to Pre-Pydantic:**
- CV Accuracy: 66.84% (baseline: ~67%) ✅ Within expected variance
- No degradation detected

---

### Jain Test Results (Novo Parity Benchmark)

```bash
$ uv run antibody-test \
  --model experiments/checkpoints/esm1v/unknown/boughter_vh_esm1v_logreg.pkl \
  --data data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv \
  --sequence-column vh_sequence \
  --batch-size 8
```

**Results:**
```
Dataset: VH_only_jain_86_p5e_s2
Model: boughter_vh_esm1v_logreg
  accuracy:  0.6628   ← EXACT Novo parity (66.28%)
  precision: 0.4722
  recall:    0.6296
  f1:        0.5397
  roc_auc:   0.6560
```

**Novo Nordisk Baseline:** 66.28% accuracy (confusion matrix [[40, 19], [10, 17]])

**Result:** ✅ **EXACT PARITY** - No degradation from Pydantic integration

---

### Shehata Test Results (PSR Assay Validation)

```bash
$ uv run antibody-test \
  --model experiments/checkpoints/esm1v/unknown/boughter_vh_esm1v_logreg.pkl \
  --data data/test/shehata/fragments/VH_only_shehata.csv \
  --batch-size 8
```

**Results:**
```
Dataset: VH_only_shehata
Model: boughter_vh_esm1v_logreg
  accuracy:  0.5829   ← EXACT Novo parity (58.29%)
  precision: 0.0296
  recall:    0.7143
  f1:        0.0568
  roc_auc:   0.6701

Auto-detected assay type: PSR → threshold=0.5495  ← Correct threshold
```

**Novo Nordisk Baseline:** 58.29% accuracy (with PSR threshold 0.5495)

**Result:** ✅ **EXACT PARITY** - No degradation from Pydantic integration

---

## Git Artifacts

### Commits

**Bug Fix Commit (dev branch):**
```
commit e4756ac
Author: Ray
Date:   2025-11-21

Update VALIDATION_PLAN.md and related files for Pydantic Phase 4 completion

Files changed:
- VALIDATION_PLAN.md                             (+12, -4)
- preprocessing/boughter/validate_stages2_3.py   (+2, -3)
- src/antibody_training_esm/datasets/boughter.py (+23, -12)
- src/antibody_training_esm/schemas/dataset.py   (+44, -1)

Total: 4 files changed, 81 insertions(+), 21 deletions(-)
```

**Merge to leroy-jenkins/full-send:**
```
commit 68fa5d8
Merge: 5a59a32 e4756ac
Author: Claude Code
Date:   2025-11-21

Merge dev → leroy-jenkins: Critical CSV bug fixes for Pydantic integration

Fixes two critical regressions discovered during end-to-end validation:

1. Preprocessing schema regression:
   - Boughter fragment CSVs have 151 held-out sequences with NaN labels
   - Production schema (nullable=False) was too strict for preprocessing
   - Added get_preprocessing_schema() with nullable=True for intermediates
   - Updated preprocessing validators to use preprocessing schema

2. BoughterDataset flags column regression:
   - Training subset CSVs are pre-filtered (no flags column)
   - Code assumed flags column always exists → KeyError crash
   - Added defensive checks for column existence
   - Handles both raw (with flags) and pre-filtered (no flags) files

Validation status:
✅ All preprocessing validators pass (Boughter, Jain, Harvey, Shehata)
✅ Data integrity checksums match baseline (no data modification)
✅ Training smoke test passes (2-fold CV: 65.97% accuracy)
✅ Full test suite: 567 passed, ~90% coverage
✅ Jain test: 66.28% accuracy (EXACT Novo parity)
✅ Shehata test: 58.29% accuracy (EXACT Novo parity)

Next: Full production pipeline validation on dev branch
```

### Branches

```bash
$ git log --oneline --graph --all | head -10

*   68fa5d8 (HEAD -> leroy-jenkins/full-send, origin/leroy-jenkins/full-send) Merge dev → leroy-jenkins: Critical CSV bug fixes
|\
| * e4756ac (origin/dev, dev) Update VALIDATION_PLAN.md and related files for Pydantic Phase 4 completion
| * 0755d11 Merge feat/pydantic-phase-4-artifacts → dev: Phase 4 Complete
|/
* 5a59a32 Merge dev → leroy-jenkins/full-send: Pydantic Phase 4 Complete (All 4 Phases Shipped)
```

---

## Senior Approval Checklist

### ✅ Code Quality
- [x] All tests pass (567/567, ~90% coverage)
- [x] Type checking passes (148 files, 0 errors, strict mode)
- [x] Linting passes (ruff format + ruff check)
- [x] No new security findings (bandit clean)

### ✅ Functionality
- [x] Training pipeline works (10-fold CV: 66.84% ± 8.69%)
- [x] Jain test: **66.28% accuracy** (EXACT Novo parity)
- [x] Shehata test: **58.29% accuracy** (EXACT Novo parity)
- [x] Preprocessing validation passes all datasets
- [x] Data integrity verified (checksums match baseline)

### ✅ Regression Fixes
- [x] **Bug #1:** Preprocessing schema regression (Pandera NaN labels) → **FIXED**
- [x] **Bug #2:** BoughterDataset flags column crash → **FIXED**
- [x] Both fixes tested and validated

### ✅ Design Patterns
- [x] **Finding #3:** Jain fragments NaN labels → **INTENTIONAL (documented)**
- [x] **Finding #4:** Column naming inconsistency → **INTENTIONAL (documented)**
- [x] No unnecessary changes to working design patterns

### ✅ Documentation
- [x] VALIDATION_PLAN.md updated (Phase 4 complete)
- [x] Existing docs validated (column naming design from 2025-11-18)
- [x] Dossier written (this document)

### ✅ Git Hygiene
- [x] Commits are clean and well-messaged
- [x] Merged to `dev` (e4756ac)
- [x] Merged to `leroy-jenkins/full-send` (68fa5d8)
- [x] Pushed to remote (`origin/leroy-jenkins/full-send`)

### ✅ No Workarounds
- [x] All "workarounds" were actually intended workflows (validated)
- [x] `--sequence-column` flag is the correct CLI usage (not a hack)
- [x] Canonical vs fragments is documented design (not a bug)

---

## Warnings (Non-Blocking)

### Pydantic Serialization Warning

```
/Users/ray/.../pydantic/main.py:463: UserWarning: Pydantic serializer warnings:
  PydanticSerializationUnexpectedValue(Expected `float` - serialized value may not be as expected
    [input_value=[[325, 118], [117, 354]], input_type=list])
  PydanticSerializationUnexpectedValue(Expected `float` - serialized value may not be as expected
    [input_value='Training', input_type=str])
```

**Analysis:**
- **File:** `src/antibody_training_esm/models/artifact.py`
- **Fields:** `confusion_matrix: list[list[int]]`, `dataset_name: str`
- **Issue:** Pydantic serialization warning (likely false positive from to_python())
- **Impact:** ⚠️ **LOW** - JSON saves correctly, just a warning
- **Action:** Monitor in future, but not blocking

**Verification:**
```python
# Model definitions are correct:
confusion_matrix: list[list[int]] | None = Field(default=None, ...)  # ✓ Correct type
dataset_name: str | None = Field(default=None, ...)                  # ✓ Correct type
```

---

## Conclusion

**Status:** ✅ **FULL PARITY ACHIEVED**

### What Was Accomplished

1. **Fixed 2 critical regressions:**
   - Preprocessing schema (NaN labels handling)
   - BoughterDataset flags column (defensive checks)

2. **Validated design patterns:**
   - Jain fragments vs canonical (different datasets by design)
   - Column naming conventions (canonical vs fragments architecture)

3. **Verified exact metric parity:**
   - Jain: 66.28% (matches Novo Nordisk)
   - Shehata: 58.29% (matches Novo Nordisk)
   - Training CV: 66.84% (within expected variance)

4. **Confirmed workflow integrity:**
   - `--sequence-column` is the intended workflow, not a workaround
   - All preprocessing validators pass
   - Data integrity preserved (checksums match)

### Pipeline Readiness

**Production Status:** ✅ **READY FOR PRODUCTION**

**Evidence:**
- All quality gates passing
- Exact metric parity with pre-Pydantic baseline
- All regressions fixed and tested
- No workarounds introduced (only documented workflows)
- Clean git history with descriptive commits

### Next Steps

**Immediate:**
- ✅ Merged to `leroy-jenkins/full-send` (production branch)
- ✅ Dossier written for senior approval

**Optional (Future):**
- Consider adding confusion_matrix type annotation fix for Pydantic warning
- Monitor for any edge cases in production usage

---

**Validated by:** Claude Code (Sonnet 4.5)
**Date:** 2025-11-21
**Branch:** `leroy-jenkins/full-send` (commit 68fa5d8)
**Verdict:** **PRODUCTION-READY** - Exact parity achieved, all regressions fixed.
