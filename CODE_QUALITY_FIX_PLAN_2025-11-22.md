# Code Quality Fix Plan - Iron-Clad Analysis
**Date:** 2025-11-22
**Status:** VERIFIED AND PRIORITIZED
**Based on:** CODE_QUALITY_AUDIT_2025-11-22.md

---

## 🎯 Executive Summary

**Audit Baseline (corrected):** 8 issues (1 P1, 5 P2, 2 P3)
**Verification Status:** ✅ **Issues confirmed; counts corrected (sequence preview appears in 3 places, not 4)**
**Critical Finding:** The sys.path hack and type gap are real; remaining items are quality polish/tech debt

**Action Required:** Fix 3 high-impact issues immediately, defer the low-impact backlog items

---

## ✅ Verification Results

### P1 Issue - VERIFIED ✓
**sys.path manipulation:** CONFIRMED unnecessary
- ✅ `validation/validate_experiment_artifacts.py` inserts repo `/src` on sys.path
- ✅ With the project installed (uv/editable), imports resolve without the hack
- ✅ This is a **legitimate portability risk**

### P2 Issues - All VERIFIED ✓
**Magic numbers:** CONFIRMED scattered
- ✅ `50` for sequence preview (3 locations: embeddings x2, CLI output)
- ✅ `59, 27, 86, 57` Novo parity constants (multiple files)
- ✅ `[-0.5, 0.5, 3.5, 6.5]` ELISA flag bins (preprocessing)
- ✅ `60` for log separator width

**Type safety gap:** CONFIRMED real
- ✅ `self._classifier: Any = None` in prediction.py:53
- ✅ Defeats type checking in core component

**Deprecated module:** CONFIRMED still imported
- ✅ `default_paths.py` marked DEPRECATED but still used by 4 dataset loaders
- ✅ Creates indirection through settings.py

---

## 🔥 Iron-Clad Fix Plan

### **TIER 1: FIX IMMEDIATELY (High Impact, Low Effort)**

These are **real bugs** or **major code smells** that should be fixed ASAP.

#### 1. Remove sys.path Manipulation ⚡ CRITICAL
**File:** `validation/validate_experiment_artifacts.py:21`
**Issue:** Breaks portability - won't work when package installed normally
**Priority:** P0 (disguised as P1)
**Effort:** 2 minutes
**Fix:**

```python
# DELETE THIS LINE (line 21):
# sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Package is already installed - just import normally:
from antibody_training_esm.models.artifact import (
    CVResults,
    ModelArtifactMetadata,
)
```

**Verification:**
```bash
# Test the script works without sys.path hack
uv run python validation/validate_experiment_artifacts.py --help
```

**Impact:** Fixes portability bug that breaks in production

---

#### 2. Fix Type Safety Gap in Predictor ⚡ HIGH
**File:** `src/antibody_training_esm/core/prediction.py:53,56`
**Issue:** `Any` type defeats type checking in core prediction component
**Priority:** P1
**Effort:** 15 minutes
**Fix:**

```python
# BEFORE (line 53, 56):
self._classifier: Any = None

@property
def classifier(self) -> Any:

# AFTER:
from antibody_training_esm.core.classifier import BinaryClassifier
from sklearn.linear_model import LogisticRegression

self._classifier: BinaryClassifier | LogisticRegression | None = None

@property
def classifier(self) -> BinaryClassifier | LogisticRegression:
    """
    Lazy loads the classifier.

    Returns:
        Trained classifier (BinaryClassifier or LogisticRegression)

    Raises:
        RuntimeError: If classifier fails to load
    """
    if self._classifier is None:
        # ... existing load logic ...
    return self._classifier  # Type checker now knows it's not None
```

**Impact:** Restores type safety in core prediction path

---

#### 3. Extract Sequence Preview Constant 📐 MEDIUM
**Files:** `src/antibody_training_esm/core/embeddings.py:126,137` + `cli/predict.py:41`
**Issue:** Magic number `50` scattered across codebase
**Priority:** P2
**Effort:** 10 minutes
**Fix:**

**Step 1:** Add to `src/antibody_training_esm/core/config.py`:
```python
# Error message formatting
SEQUENCE_PREVIEW_LENGTH = 50  # Max chars for sequence previews in logs/errors
```

**Step 2:** Update all usages:
```python
# In embeddings.py and predict.py:
from antibody_training_esm.core.config import SEQUENCE_PREVIEW_LENGTH

# Replace all instances of:
sequence[:50]
# With:
sequence[:SEQUENCE_PREVIEW_LENGTH]
```

**Locations to update:**
- `src/antibody_training_esm/core/embeddings.py:126` (error log)
- `src/antibody_training_esm/core/embeddings.py:137` (seq_preview variable)
- `src/antibody_training_esm/cli/predict.py:41` (result output)

**Impact:** Single source of truth for formatting constant

---

### **TIER 2: FIX NEXT SPRINT (Medium Impact, Medium Effort)**

These improve code quality significantly but aren't urgent.

#### 4. Centralize Novo Parity Constants 📊 MEDIUM
**Files:** `src/antibody_training_esm/datasets/jain.py:321,327,332,333` + `preprocessing/jain/`
**Issue:** Scientific constants (59, 27, 86, 57) scattered without names
**Priority:** P2
**Effort:** 20 minutes
**Fix:**

Add to `src/antibody_training_esm/datasets/jain.py` (top of file, after imports):
```python
# Novo Nordisk Parity Constants (from Sakhnini et al. 2025)
# Paper benchmark: 86 antibodies with [[40, 19], [10, 17]] confusion matrix
NOVO_PARITY_SPECIFIC_COUNT = 59      # Specific antibodies in parity set
NOVO_PARITY_NONSPECIFIC_COUNT = 27   # Non-specific antibodies in parity set
NOVO_PARITY_TOTAL = 86               # Total parity set size (59 + 27)
NOVO_PARITY_EXPECTED_CORRECT = 57    # Expected correct predictions (40 + 17)
NOVO_PARITY_ACCURACY = 66.28         # Expected accuracy (57/86 = 0.6628)

# Sanity checks
assert NOVO_PARITY_SPECIFIC_COUNT + NOVO_PARITY_NONSPECIFIC_COUNT == NOVO_PARITY_TOTAL
assert NOVO_PARITY_EXPECTED_CORRECT / NOVO_PARITY_TOTAL == NOVO_PARITY_ACCURACY
```

Update all usages in `jain.py`:
```python
# Line 321:
specific_keep = specific_sorted.tail(NOVO_PARITY_SPECIFIC_COUNT)

# Line 327:
df_86 = pd.concat([specific_keep, nonspecific], ignore_index=True)
assert len(df_86) == NOVO_PARITY_TOTAL

# Lines 332-333:
self.logger.info(f"  Specific: {spec_count} (expected {NOVO_PARITY_SPECIFIC_COUNT})")
self.logger.info(f"  Non-specific: {nonspec_count} (expected {NOVO_PARITY_NONSPECIFIC_COUNT})")
```

Update `preprocessing/jain/test_novo_parity.py:165`:
```python
from antibody_training_esm.datasets.jain import NOVO_PARITY_ACCURACY

novo_accuracy = NOVO_PARITY_ACCURACY  # 66.28% (was: 57/86)
```

**Impact:** Self-documenting scientific constants with provenance

---

#### 5. Extract ELISA Flag Bins 🏷️ MEDIUM
**File:** `preprocessing/jain/step1_convert_excel_to_csv.py:161-165`
**Issue:** Flagging strategy encoded in magic numbers
**Priority:** P2
**Effort:** 15 minutes
**Fix:**

Add at top of `step1_convert_excel_to_csv.py`:
```python
# Novo Nordisk ELISA Flagging Strategy
# Paper methodology: 0 flags = specific, 1-3 = mild, 4-6 = non-specific
FLAG_BINS = [-0.5, 0.5, 3.5, 6.5]  # Bin edges for pd.cut
FLAG_CATEGORIES = ["specific", "mild", "non_specific"]

# Explanation:
# - [-0.5, 0.5): 0 flags → specific
# - [0.5, 3.5): 1-3 flags → mild polyreactivity
# - [3.5, 6.5]: 4-6 flags → non-specific
```

Update usage (line 161):
```python
df["flag_category"] = pd.cut(
    df["elisa_flags"],
    bins=FLAG_BINS,
    labels=FLAG_CATEGORIES,
)
```

**Impact:** Documents scientific methodology in code

---

#### 6. Migrate from default_paths.py to settings.py 🗑️ MEDIUM
**Files:** 4 dataset loaders + `default_paths.py`
**Issue:** Deprecated module still imported, creates unnecessary indirection
**Priority:** P2 (tech debt)
**Effort:** 30 minutes
**Fix:**

**Step 1:** Update all 4 dataset loaders:
```python
# OLD (in boughter.py, jain.py, harvey.py, shehata.py):
from antibody_training_esm.datasets.default_paths import BOUGHTER_ANNOTATED_DIR

# NEW:
from antibody_training_esm.settings import settings
BOUGHTER_ANNOTATED_DIR = settings.BOUGHTER_ANNOTATED_DIR
```

**Affected files:**
- `src/antibody_training_esm/datasets/boughter.py:37`
- `src/antibody_training_esm/datasets/jain.py:40`
- `src/antibody_training_esm/datasets/harvey.py:31`
- `src/antibody_training_esm/datasets/shehata.py:31`

**Step 2:** Delete `src/antibody_training_esm/datasets/default_paths.py`

**Step 3:** Verify no imports remain:
```bash
rg "from.*default_paths import" src/
# Should return: no matches
```

**Impact:** Removes deprecated code, simplifies architecture

---

### **TIER 3: BACKLOG (Low Impact, Cosmetic)**

These are nice-to-haves that don't significantly impact quality.

#### 7. Extract Log Separator Width (Cosmetic)
**Priority:** P3
**Effort:** 5 minutes
**Rationale:** Purely cosmetic - separator width doesn't affect functionality

#### 8. Add Jain Stage Filtering Tests (Low Priority)
**Priority:** P3
**Effort:** 2 hours
**Rationale:** Existing tests cover the critical paths; this is belt-and-suspenders

---

## 📋 Implementation Order

### Sprint 1 (Immediate - ~30 minutes)
1. Remove sys.path hack (2 min) → **FIXES PORTABILITY BUG**
2. Fix type safety gap (15 min) → **RESTORES TYPE SAFETY IN PREDICTOR**
3. Extract sequence preview constant (10 min) → **SINGLE SOURCE OF TRUTH**

### Sprint 2 (Next - ~1 hour)
4. Centralize Novo constants (20 min) → **SELF-DOCUMENTING SCIENCE**
5. Extract ELISA flag bins (15 min) → **DOCUMENTS METHODOLOGY**
6. Migrate from default_paths.py (30 min) → **REMOVES TECH DEBT**

### Backlog (Defer)
7. Log separator width (cosmetic)
8. Jain stage filtering tests (low ROI)

---

## 🎯 Expected Outcomes

### After Tier 1 (30 minutes):
- ✅ sys.path hack removed
- ✅ Predictor classifier types explicit instead of `Any`
- ✅ Sequence preview length centralized across embeddings and CLI

### After Tier 2 (1.5 hours total):
- ✅ Novo parity and ELISA flag constants named with provenance
- ✅ Deprecated `default_paths` removed in favor of `settings`
- ✅ Preprocessing logic for ELISA flags documented

### Final State:
- ✅ Identified magic numbers centralized
- ✅ No remaining `Any` in the predictor path
- ✅ Deprecated module removed; remaining backlog is cosmetic (log width) + test coverage

---

## ✅ Verification Plan

### Pre-Fix Baseline
```bash
# Verify current issues exist
rg "sys.path.insert" validation/
rg ": Any" src/antibody_training_esm/core/prediction.py
rg "sequence\[:50\]" src/
rg "tail\(59\)" src/
rg "from.*default_paths import" src/
```

### Post-Fix Verification
```bash
# Ensure all issues resolved
make all  # All tests pass + type check + lint
rg "sys.path.insert" validation/  # Should be empty
rg ": Any" src/antibody_training_esm/core/prediction.py  # Should be empty
rg "SEQUENCE_PREVIEW_LENGTH" src/  # Should find constant + usages
rg "NOVO_PARITY" src/  # Should find constants + usages
rg "from.*default_paths import" src/  # Should be empty
```

### Regression Testing
```bash
# Ensure no functionality broken
uv run pytest tests/ -v
uv run python validation/validate_experiment_artifacts.py experiments/runs/
```

---

## 💯 Quality Assessment

### Current State (Post-Pydantic v0.7.0)
- Strengths: structured pipeline with enforced typing/linting, Hydra configs, and CI checks
- Gaps: sys.path hack, predictor `Any`, sequence preview literal, Novo parity + ELISA flag constants unnamed, deprecated `default_paths` imports, placeholder Jain stage test, cosmetic log separator

### After Tier 1 Fixes
- Portability gap closed; predictor typing tightened; sequence preview length centralized

### After Tier 2 Fixes
- Scientific constants named/documented; deprecated `default_paths` removed; ELISA flag bins documented

### Backlog
- Cosmetic log separator constant; add fixtures/tests for Jain stage filtering

---

## 🏆 Conclusion

- Plan now reflects **8 confirmed issues** (1 P1, 5 P2, 2 P3) based on code review.
- Tier 1 + Tier 2 changes are scoped to ~1.5–2 hours of engineering time; Jain stage test fixtures will add additional effort.
- Address Tier 1 first to remove the portability gap and tighten typing; execute Tier 2 next to document scientific constants and retire deprecated imports.

**Audit Verified By:** Deep review of current codebase (2025-11-22)
**Status:** ✅ **READY FOR EXECUTION**
