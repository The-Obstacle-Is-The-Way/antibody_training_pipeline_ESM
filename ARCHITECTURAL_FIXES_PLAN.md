# Remaining Technical Debt

**Last Updated:** 2025-11-20
**Status:** Phase 2 (P1) - In Progress
**Goal:** Complete remaining high-priority architectural improvements

**What's Been Completed:**
- ✅ Phase 1 (P0): All critical fixes (sys.path hacks, pytest markers, documentation)
- ✅ Phase 1.5: Validation gap closure (Novo parity, GitIgnore, multi-backbone testing)
- ✅ Fix #5: Print → logging migration (799 statements converted)
- ✅ Fix #6 (partial): Split cli/test.py into modular structure

**This Document:** Remaining tech debt organized by priority

---

## Priority Overview

### ✅ Completed (Phase 1 & 1.5)
- Phase 1 (P0): sys.path hacks, pytest markers, documentation (25 min)
- Phase 1.5: Validation gap closure, multi-backbone testing (~1 hour)
- Fix #5: Print → logging migration (799 statements, ~3 hours)
- Fix #6 (partial): cli/test.py refactor (1/4 files)

### 📋 Remaining Work

**Phase 2 (P1) - High Priority:** ~8-10 hours
- Fix #6: Split remaining 3 overly long files (trainer.py, 2 preprocessing scripts)
- Fix #7: Centralize hardcoded paths
- Fix #8: Standardize file permissions
- Fix #9: Fix bare except blocks
- Fix #10: Address type ignores
- Fix #11: Clean up utils directory
- Fix #12: Merge config directories

**Phase 3 (P2) - Medium Priority:** ~5-7 hours
- Fix #14: Document PSR threshold differences
- Fix #15-16: Extract duplicated code (validation, fragments)
- Fix #17-18: Standardize shebangs & pytest config
- Fix #19-22: Minor cleanup (constants, TODOs, bug refs)

**Phase 4 (P3) - Low Priority:** ~1-2 hours
- Fix #24: Standardize docstring style
- Fix #25: Add missing docstrings

---

## Phase 2: High Priority Fixes (P1)

### Fix #6: Split Overly Long Files 📝

**Priority:** P1 (HIGH)
**Effort:** 2-3 hours (3 files remaining)
**Status:** ⚠️ PARTIAL (1/4 complete)

**Problem:** 4 files exceed 500 lines, violating Single Responsibility Principle.

**Progress:**
- ✅ `src/antibody_training_esm/cli/test.py` → Split into `cli/testing/` subdirectory
- ⏳ `src/antibody_training_esm/core/trainer.py` (934 lines) - PENDING
- ⏳ `preprocessing/boughter/stage1_dna_translation.py` (590 lines) - PENDING
- ⏳ `preprocessing/boughter/stage2_stage3_annotation_qc.py` (514 lines) - PENDING

**Recommended Refactoring:**

**trainer.py → core/training/ directory:**
```
core/
├── trainer.py (main train_model function, ~300 lines)
└── training/
    ├── __init__.py
    ├── cache.py (CacheManager for embedding cache ops)
    ├── metrics.py (MetricsLogger for evaluation)
    └── serialization.py (ModelSerializer for .pkl handling)
```

**stage1_dna_translation.py → boughter/translation/ directory:**
```
preprocessing/boughter/
├── stage1_dna_translation.py (main orchestration, ~200 lines)
└── translation/
    ├── __init__.py
    ├── dna_translator.py (DNATranslator class)
    └── validation_utils.py (Translation validation)
```

**stage2_stage3_annotation_qc.py → boughter/annotation/ directory:**
```
preprocessing/boughter/
├── stage2_stage3_annotation_qc.py (main orchestration, ~200 lines)
└── annotation/
    ├── __init__.py
    ├── anarci_annotator.py (ANARCIAnnotator class)
    └── qc_filter.py (QCFilter class)
```

**Verification:**
```bash
# Check file sizes reduced
wc -l src/antibody_training_esm/core/trainer.py  # Should be <400
wc -l preprocessing/boughter/stage*.py  # Should be <300 each

# All tests still pass
uv run pytest

# Quality gates
make all
```

---

### Fix #7: Centralize Hardcoded Paths 📂

**Priority:** P1 (HIGH)
**Effort:** 2 hours
**Risk:** MEDIUM

**Problem:** 50+ hardcoded paths scattered across 17 preprocessing scripts.

**Examples:**
```python
# preprocessing/jain/step1_convert_excel_to_csv.py:45
RAW_DIR = Path("data/test/jain/raw")

# preprocessing/harvey/test_psr_threshold.py:86
CSV_PATH = "data/test/harvey/fragments/VHH_only_harvey.csv"
```

**Fix Plan:**

**Step 1: Create `preprocessing/paths.py`**

```python
"""
Centralized path configuration for preprocessing scripts.

All data paths for preprocessing pipelines defined here for easy modification.
"""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Base data directories
DATA_DIR = PROJECT_ROOT / "data"
DATA_TRAIN_DIR = DATA_DIR / "train"
DATA_TEST_DIR = DATA_DIR / "test"

# Boughter (training set)
BOUGHTER_DIR = DATA_TRAIN_DIR / "boughter"
BOUGHTER_RAW_DIR = BOUGHTER_DIR / "raw"
BOUGHTER_PROCESSED_DIR = BOUGHTER_DIR / "processed"
BOUGHTER_CANONICAL_DIR = BOUGHTER_DIR / "canonical"

# Jain (test set)
JAIN_DIR = DATA_TEST_DIR / "jain"
JAIN_RAW_DIR = JAIN_DIR / "raw"
JAIN_PROCESSED_DIR = JAIN_DIR / "processed"
JAIN_FRAGMENTS_DIR = JAIN_DIR / "fragments"
JAIN_CANONICAL_DIR = JAIN_DIR / "canonical"

# Harvey (test set - nanobodies)
HARVEY_DIR = DATA_TEST_DIR / "harvey"
HARVEY_RAW_DIR = HARVEY_DIR / "raw"
HARVEY_PROCESSED_DIR = HARVEY_DIR / "processed"
HARVEY_FRAGMENTS_DIR = HARVEY_DIR / "fragments"

# Shehata (test set - PSR)
SHEHATA_DIR = DATA_TEST_DIR / "shehata"
SHEHATA_RAW_DIR = SHEHATA_DIR / "raw"
SHEHATA_PROCESSED_DIR = SHEHATA_DIR / "processed"
SHEHATA_FRAGMENTS_DIR = SHEHATA_DIR / "fragments"
SHEHATA_CANONICAL_DIR = SHEHATA_DIR / "canonical"

# Experiments
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
CHECKPOINTS_DIR = EXPERIMENTS_DIR / "checkpoints"
CACHE_DIR = EXPERIMENTS_DIR / "cache"
BENCHMARKS_DIR = EXPERIMENTS_DIR / "benchmarks"
```

**Step 2: Migrate all preprocessing scripts (1.5 hours)**

**Pattern:**
```python
# BEFORE:
from pathlib import Path
RAW_DIR = Path("data/test/jain/raw")
OUTPUT_DIR = Path("data/test/jain/processed")

# AFTER:
from preprocessing.paths import JAIN_RAW_DIR, JAIN_PROCESSED_DIR
RAW_DIR = JAIN_RAW_DIR
OUTPUT_DIR = JAIN_PROCESSED_DIR
```

**Verification:**
```bash
# Check no hardcoded "data/" paths remain (except in paths.py)
grep -r "data/train\|data/test" preprocessing/*.py | grep -v "paths.py" | grep -v ".pyc"

# All scripts still work
uv run python preprocessing/jain/step1_convert_excel_to_csv.py
```

---

### Fix #8: Standardize File Permissions 🔐

**Priority:** P1 (HIGH)
**Effort:** 10 minutes
**Risk:** ZERO

**Problem:** Random subset of scripts are executable with no clear pattern.

**Currently Executable (6 files):**
- `preprocessing/boughter/train_hyperparameter_sweep.py`
- `preprocessing/boughter/validate_stages2_3.py`
- `preprocessing/shehata/step2_extract_fragments.py`
- `preprocessing/jain/test_novo_parity.py`
- `preprocessing/jain/step2_preprocess_p5e_s2.py`
- `scripts/validation/validate_fragments.py`

**Recommendation: Make ALL preprocessing scripts executable**

```bash
# Make all .py scripts in preprocessing/ executable
find preprocessing -name "*.py" ! -name "__init__.py" -exec chmod +x {} \;

# Verify all have execute permission
find preprocessing -name "*.py" ! -name "__init__.py" -exec ls -l {} \; | awk '{print $1}' | sort -u
# Should show: -rwxr-xr-x
```

**Verification:**
```bash
# Check all .py files have same permissions
find preprocessing -name "*.py" ! -name "__init__.py" -exec stat -c "%a %n" {} \; | awk '{print $1}' | sort -u
# Should show single value: 755
```

---

### Fix #9: Fix Bare except Exception 🐛

**Priority:** P1 (HIGH)
**Effort:** 10 minutes
**Risk:** LOW

**Problem:** Four bare `except Exception:` blocks in trainer.py catch too much.

**Locations:**
- `src/antibody_training_esm/core/trainer.py:176`
- `src/antibody_training_esm/core/trainer.py:831`
- `src/antibody_training_esm/core/trainer.py:875`
- `src/antibody_training_esm/core/trainer.py:927`

**Current Code (Line 176):**
```python
try:
    cache_path.unlink()
    logger.info(f"Deleted cache file: {cache_path}")
except Exception:  # ← TOO BROAD
    logger.warning(f"Could not delete cache file {cache_path}: {e}")
```

**Fix:**
```python
try:
    cache_path.unlink()
    logger.info(f"Deleted cache file: {cache_path}")
except (OSError, PermissionError) as e:  # ← SPECIFIC
    logger.warning(f"Could not delete cache file {cache_path}: {e}")
except Exception as e:  # Catch unexpected errors
    logger.error(f"Unexpected error deleting cache {cache_path}: {e}")
    raise  # Re-raise unexpected errors
```

**Verification:**
```bash
# Check no bare excepts remain
grep -n "except Exception:" src/antibody_training_esm/core/trainer.py
# Should return nothing

# Tests still pass
uv run pytest tests/unit/core/test_trainer.py
```

---

### Fix #10: Address type: ignore Comments 🔍

**Priority:** P1 (HIGH)
**Effort:** 30 minutes
**Risk:** LOW

**Problem:** 2 `type: ignore` comments indicate incomplete type coverage.

**Locations:**
1. `src/antibody_training_esm/core/embeddings.py:60`: `# type: ignore[no-untyped-call]`
2. `tests/unit/datasets/test_base.py:265`: `# type: ignore`

**Fix #1 - embeddings.py:60:**

**Current:**
```python
self.tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
    model_name, revision=revision
)
```

**Fix (add explanatory comment):**
```python
# Type ignore needed: transformers.AutoTokenizer lacks type stubs
# See: https://github.com/huggingface/transformers/issues/
self.tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
    model_name, revision=revision
)
```

**Fix #2 - test_base.py:265:**

**Current:**
```python
mock_dataset = MockDataset()  # type: ignore
```

**Fix (add proper type annotation):**
```python
mock_dataset: AntibodyDataset = MockDataset()  # Explicit type for mypy
```

**Verification:**
```bash
# Run mypy strict mode
uv run mypy src/antibody_training_esm tests --strict

# Count remaining type: ignore
grep -r "type: ignore" src/ tests/ | wc -l
# Should be 1 (only the HuggingFace one with explanation)
```

---

### Fix #11: Delete Empty utils/ Directory 🗑️

**Priority:** P1 (HIGH)
**Effort:** 5 minutes
**Risk:** ZERO

**Problem:** `src/antibody_training_esm/utils/` contains only `__init__.py`, serves no purpose.

**Recommended Action: Delete it**

```bash
# Check nothing imports from utils
grep -r "from antibody_training_esm.utils import" src/ tests/

# If nothing imports it, delete
rm -rf src/antibody_training_esm/utils/
```

**Verification:**
```bash
# Verify deletion
ls src/antibody_training_esm/utils  # Should error

# All tests pass
uv run pytest
```

---

### Fix #12: Merge Duplicate Config Directories 📁

**Priority:** P1 (HIGH)
**Effort:** 15 minutes
**Risk:** LOW

**Problem:** Two config directories: `configs/` (root) and `src/antibody_training_esm/conf/` (package).

**Current State:**
```
configs/testing/jain_p5e_s2.yaml  # Root location
src/antibody_training_esm/conf/  # Canonical Hydra configs
```

**Fix Plan:**

```bash
# Move testing configs into package
mkdir -p src/antibody_training_esm/conf/testing/
mv configs/testing/jain_p5e_s2.yaml src/antibody_training_esm/conf/testing/

# Delete empty configs/
rmdir configs/testing/
rmdir configs/
```

**Verification:**
```bash
# Configs directory deleted
ls configs/  # Should error

# New location exists
ls src/antibody_training_esm/conf/testing/jain_p5e_s2.yaml

# Hydra can still find configs
uv run antibody-train --help
```

---

## Phase 3: Medium Priority Fixes (P2)

### Fix #14: Document PSR Threshold Differences 📊

**Priority:** P2 (MEDIUM)
**Effort:** 20 minutes

**Problem:** Two different PSR thresholds used (0.4 and 0.5495), confusing which is "correct".

**Locations:**
- `src/antibody_training_esm/core/classifier.py:30`: `"PSR": 0.5495`
- `preprocessing/jain/step2_preprocess_p5e_s2.py:51`: `PSR_THRESHOLD = 0.4`

**Fix: Add documentation comments**

```python
# src/antibody_training_esm/core/classifier.py:30
ASSAY_THRESHOLDS = {
    "ELISA": 0.5,
    # PSR threshold for prediction (Novo Nordisk exact parity)
    # NOTE: Different from Jain preprocessing threshold (0.4) which is used
    # for reclassifying Tier A antibodies during data preparation
    "PSR": 0.5495,
}
```

```python
# preprocessing/jain/step2_preprocess_p5e_s2.py:51
# PSR threshold for Jain dataset reclassification (Tier A only)
# This is used during preprocessing to reclassify certain antibodies,
# NOT for prediction. Prediction uses 0.5495 (see classifier.py)
PSR_THRESHOLD = 0.4
```

---

### Fix #15-16: Extract Duplicated Code 🔄

**Priority:** P2 (MEDIUM)
**Effort:** 5 hours combined
**Risk:** MEDIUM

**Problem:**
- Validation logic duplicated across 4 scripts (~60-80% overlap)
- Fragment extraction logic duplicated across 3 scripts (~200 lines each)

**Fix #15: Create `preprocessing/validation_utils.py`**

**Fix #16: Create `preprocessing/fragment_utils.py`**

(See detailed implementation specs in original plan if needed)

---

### Fix #17-18: Standardize Shebangs & Pytest Config

**Fix #17: Standardize Shebangs** (10 min)
- Add `#!/usr/bin/env python3` to ALL preprocessing .py files
- Or specify `#!/usr/bin/env python3.12` for version clarity

**Fix #18: Delete pytest.ini** (10 min)
- Move all pytest config to `pyproject.toml`
- Delete `pytest.ini`

---

### Fix #19-22: Minor Cleanup

**Fix #19: Centralize Logging Setup** - Already done via Fix #5 ✅

**Fix #20: Wrap Global Constants** (1 hour)
- Create config classes for magic values
- Move to dedicated config files

**Fix #21: Remove Stale TODOs** (5 min)
- Search for TODO comments
- Either implement or delete

**Fix #22: Update Bug Reference Comments** (10 min)
- Replace bug doc references with explanations

---

## Phase 4: Low Priority Fixes (P3)

### Fix #24: Standardize Docstring Style

**Effort:** 2 hours
**Goal:** Apply Google-style docstrings consistently across preprocessing/

### Fix #25: Add Missing Docstrings

**Effort:** 1 hour
**Goal:** Add docstrings to all public functions

---

## Verification Checklist

### Phase 2 (P1) Verification
- [ ] No files >500 lines: `find src preprocessing -name "*.py" -exec wc -l {} \; | awk '$1 > 500'`
- [ ] Centralized paths: `grep -r "data/train\|data/test" preprocessing/*.py | grep -v paths.py`
- [ ] Consistent permissions: `find preprocessing -name "*.py" ! -name "__init__.py" -exec stat -c "%a" {} \; | sort -u`
- [ ] No bare except: `grep "except Exception:" src/antibody_training_esm/core/trainer.py`
- [ ] Minimal type: ignore: `grep -r "type: ignore" src/ | wc -l` (should be ≤1)
- [ ] No empty utils/: `ls src/antibody_training_esm/utils/*.py 2>/dev/null | wc -l` (should error)
- [ ] Single config location: `ls configs/ 2>/dev/null` (should error)

### Overall Code Quality
- [ ] All tests pass: `uv run pytest`
- [ ] Type checking passes: `uv run mypy src/ --strict`
- [ ] Linting passes: `uv run ruff check src/ preprocessing/`
- [ ] Formatting consistent: `uv run ruff format --check src/ preprocessing/`
- [ ] Coverage ≥70%: `uv run pytest --cov=. --cov-fail-under=70`
- [ ] Security scan clean: `uv run bandit -r src/ preprocessing/`

---

## Quick Reference Commands

### Run All Quality Gates
```bash
make all  # format → lint → typecheck → test
```

### Check File Sizes
```bash
find src preprocessing -name "*.py" -exec wc -l {} \; | awk '$1 > 500 {print $2": "$1" lines"}'
```

### Find Hardcoded Paths
```bash
grep -rn "data/train\|data/test" preprocessing/ --include="*.py" | grep -v paths.py
```

### Check Permission Consistency
```bash
find preprocessing -name "*.py" ! -name "__init__.py" -exec stat -f "%Sp %N" {} \; | awk '{print $1}' | sort | uniq -c
```

---

**END OF REMAINING TECH DEBT SPEC**

**Next Actions:**
1. Review this cleaned-up spec
2. Decide which fixes to tackle next
3. Consider creating a refactor branch for Phase 2 work
