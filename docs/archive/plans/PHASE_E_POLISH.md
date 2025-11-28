# Phase E: Polish & Documentation

> **Note:** This document references `leroy-jenkins/full-send` which was renamed to `main` on 2025-11-28.

**Effort:** 2-3 hours
**Risk:** LOW
**Dependencies:** Phases A-D complete
**Branch:** `claude/refactor-phase-e`

---

## Overview

Final touches to achieve pristine codebase: documentation improvements, standardization, and cleanup.

**Goal:** Polish the codebase with cosmetic improvements, consistent conventions, and complete documentation.

**Why this is LOW risk:**
- No logic changes
- Mostly comments and docstrings
- Easy to review and verify

---

## Fixes Included

| Fix # | Description | Effort | Risk |
|-------|-------------|--------|------|
| #14 | Document PSR threshold differences | 20 min | ZERO |
| #17 (verify) | Pytest config single source (pytest.ini already removed) | 10 min | ZERO |
| #5 follow-up | Replace/justify remaining `print()` diagnostics | 20-30 min | LOW |
| #21 | Remove stale TODOs (currently 1) | 5 min | ZERO |
| #22 | Update stale bug references (CLI override doc) | 10 min | ZERO |
| #24-25 | Docstring polish after Phase C/D splits | 1-2 hours | LOW |

**Total Time:** 2-3 hours

---

## Task E1: Document PSR Threshold Differences (20 min)

### Problem
Two different PSR thresholds used (0.4 vs 0.5495) with no explanation.

**Locations:**
- `src/antibody_training_esm/core/classifier.py:30` (0.5495)
- `preprocessing/jain/step2_preprocess_p5e_s2.py:51` (0.4)

### Solution
Add explanatory comments to both locations.

### Implementation

**File 1: classifier.py**

```python
# BEFORE:
ASSAY_THRESHOLDS = {
    "ELISA": 0.5,
    "PSR": 0.5495,
}

# AFTER:
ASSAY_THRESHOLDS = {
    "ELISA": 0.5,
    # PSR threshold for prediction (Novo Nordisk exact parity)
    # This is the threshold used at INFERENCE TIME for Harvey/Shehata datasets
    # to classify antibodies based on PSR assay predictions.
    #
    # NOTE: Different from Jain preprocessing threshold (0.4) which is used
    # for RECLASSIFYING Tier A antibodies during data preparation. These are
    # different use cases with different optimal thresholds.
    #
    # See: docs/research/assay-thresholds.md for full explanation
    "PSR": 0.5495,
}
```

**File 2: step2_preprocess_p5e_s2.py**

```python
# BEFORE:
PSR_THRESHOLD = 0.4

# AFTER:
# PSR threshold for Jain dataset reclassification (Tier A only)
# This is used during PREPROCESSING to reclassify certain antibodies
# based on PSR assay results, NOT for prediction.
#
# Different from prediction threshold (0.5495) in classifier.py which is
# used at inference time. The 0.4 threshold is specific to the clinical
# decision boundary for Tier A antibodies in the Jain dataset.
#
# See: docs/research/assay-thresholds.md for full explanation
PSR_THRESHOLD = 0.4
```

### Verification
```bash
# Check comments added
grep -A5 "PSR.*0.5495\|PSR_THRESHOLD.*0.4" src/antibody_training_esm/core/classifier.py preprocessing/jain/step2_preprocess_p5e_s2.py
```

### Success Criteria
- [ ] Both files have explanatory comments
- [ ] Comments explain WHY two thresholds exist

---

## Task E2: Pytest Config Single Source (10 min)

### Problem
Pytest configuration already lives in `pyproject.toml`; ensure no drift or stray `pytest.ini` reappears.

### Solution
- Verify `pyproject.toml` contains required pytest options (testpaths/markers/etc.).
- Confirm `pytest.ini` is absent.

### Verification
```bash
rg "\\[tool.pytest" -n -C 3 pyproject.toml
ls pytest.ini  # Should error/no output
uv run pytest -q --disable-warnings --maxfail=1
```

### Success Criteria
- [ ] Pytest settings live only in `pyproject.toml`
- [ ] No `pytest.ini` present
- [ ] Tests still collect/run

---

## Task E3: Replace/Justify Remaining `print()` Diagnostics (20-30 min)

### Problem
`print()` calls remain in preprocessing and library code (e.g., 22 matches in `preprocessing/`, 36 in `src/`). Some are user-facing summaries; others duplicate logging.

### Solution
- Convert non-user-facing diagnostics to `logger.info`/`logger.warning`.
- If a `print()` is intentionally user-facing (CLI summary), add a brief comment documenting why it remains.

### Verification
```bash
rg "print\\(" src preprocessing | grep -v README | wc -l  # Track count reduction
rg "print\\(" preprocessing | grep -v README
```

### Success Criteria
- [ ] All diagnostic output goes through logging
- [ ] Remaining `print()` calls are documented as intentional UX output

---

## Task E4: Remove Stale TODOs (5 min)

### Problem
Only one TODO remains: `tests/integration/test_dataset_pipeline.py` ("Create distinct mock CSVs..."). Decide whether to implement now or replace with a linked issue.

### Implementation

**Find all TODOs:**

```bash
# Search for TODO comments
grep -rn "TODO\|FIXME\|XXX" src/ preprocessing/ tests/ --include="*.py"
```

**For each TODO found:**
- If trivial → implement it now
- If tracked elsewhere → delete the comment
- If still valid → keep it (rare)

**Example:**

```python
# BEFORE:
# TODO: Add validation for sequence length

# AFTER (if trivial):
if len(sequence) < 10:
    raise ValueError("Sequence too short")

# OR (if not needed):
# (delete the comment)
```

### Verification
```bash
# Check for remaining TODOs
grep -rn "TODO" src/ preprocessing/ tests/ --include="*.py"

# Should return ZERO or very few
```

### Success Criteria
- [ ] All stale TODOs removed or implemented
- [ ] Only valid TODOs remain (if any)

---

## Task E5: Update Bug References (10 min)

### Problem
Two files reference `CLI_OVERRIDE_BUG` docs that are not present (`config_schema.py`, `tests/unit/core/test_structured_configs.py`). Clarify or link to an existing write-up.

### Implementation

**Find bug references:**

```bash
# Search for bug references
grep -rn "CLI_OVERRIDE_BUG\|BUG:\|HACK:\|WORKAROUND:" src/ preprocessing/ tests/ --include="*.py"
```

**For each reference:**
- If bug is fixed → replace with explanation of WHY the code exists
- If bug still exists → update comment with issue tracker link

**Example:**

```python
# BEFORE:
# WORKAROUND for CLI_OVERRIDE_BUG: Force reload config
reload_config()

# AFTER:
# Config must be reloaded after Hydra initialization to pick up CLI overrides.
# This ensures --sequence-column and --label-column flags take precedence.
reload_config()
```

### Verification
```bash
# Check no old bug references remain
grep -rn "CLI_OVERRIDE_BUG" src/ preprocessing/ tests/

# Should return NOTHING
```

### Success Criteria
- [ ] No references to resolved bugs
- [ ] All workarounds have clear explanations

---

## Task E6: Docstring Polish (1-2 hours)

### Problem
Inconsistent docstring style and missing docstrings in newly created modules.

### Solution
Apply Google-style docstrings consistently, focusing on new modules from Phases C & D.

### Target Files

**Phase C modules (expected after split):**
- `src/antibody_training_esm/core/training/cache.py`
- `src/antibody_training_esm/core/training/metrics.py`
- `src/antibody_training_esm/core/training/serialization.py`
- `preprocessing/boughter/translation/dna_translator.py`
- `preprocessing/boughter/translation/validation.py`
- `preprocessing/boughter/annotation/anarci.py`
- `preprocessing/boughter/annotation/qc.py`
- `src/antibody_training_esm/datasets/base_components/validation.py`
- `src/antibody_training_esm/datasets/base_components/annotation.py`
- `src/antibody_training_esm/datasets/base_components/fragments.py`

**Phase D modules (2 files):**
- `preprocessing/validation_utils.py`
- `preprocessing/fragment_utils.py`

### Google-Style Docstring Template

```python
def function_name(arg1: str, arg2: int = 10) -> bool:
    """
    Short one-line summary (imperative mood).

    Longer description if needed. Explain what the function does,
    not how it does it. Describe edge cases and important behavior.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2 (default: 10)

    Returns:
        Description of return value

    Raises:
        ValueError: If arg1 is empty
        OSError: If file operation fails

    Example:
        >>> result = function_name("foo", 20)
        >>> print(result)
        True
    """
    ...
```

### Implementation

**Step 1: Review each new module (1 hour)**

For each file:
1. Check all public functions have docstrings
2. Verify docstrings follow Google style
3. Add missing docstrings
4. Fix inconsistent formatting

**Step 2: Spot check preprocessing scripts (30 min)**

Review main preprocessing scripts for missing docstrings:
- `preprocessing/boughter/stage1_dna_translation.py`
- `preprocessing/jain/step2_preprocess_p5e_s2.py`
- Add module-level docstrings if missing

### Verification
```bash
# Check all public functions have docstrings (manual review)
# Focus on new modules from Phases C & D

# Type checking should pass
uv run mypy src/ preprocessing/ --strict
```

### Success Criteria
- [ ] All new modules (Phase C & D) have complete docstrings
- [ ] Docstrings follow Google style consistently
- [ ] All public functions documented
- [ ] Type checking passes

---

## Phase Completion Checklist

### All Tasks Complete
- [ ] Task E1: PSR threshold comments added
- [ ] Task E2: Pytest config verified (single source)
- [ ] Task E3: `print()` diagnostics replaced/justified
- [ ] Task E4: Stale TODOs removed
- [ ] Task E5: Bug references updated
- [ ] Task E6: Docstrings polished

### Quality Gates
- [ ] All tests pass: `uv run pytest`
- [ ] Type checking: `uv run mypy src/ preprocessing/ --strict`
- [ ] Linting: `uv run ruff check src/ preprocessing/`
- [ ] Formatting: `uv run ruff format --check src/ preprocessing/`
- [ ] Security scan: `uv run bandit -r src/ preprocessing/`
- [ ] `make all` passes

### Git Workflow
```bash
# Create branch
git checkout dev
git pull origin dev
git checkout -b claude/refactor-phase-e

# Make changes (complete all 6 tasks above)

# Commit
git add -A
git commit -m "$(cat <<'EOF'
refactor: Phase E - Polish & documentation cleanup

Final polish for pristine codebase: documentation improvements,
standardization, and cleanup.

**Task E1: Document PSR Threshold Differences (20 min)**
Added explanatory comments to both PSR threshold locations:
- classifier.py: 0.5495 (prediction/inference threshold)
- step2_preprocess_p5e_s2.py: 0.4 (preprocessing threshold)
Clarified WHY two different thresholds exist (different use cases)

**Task E2: Pytest Config Single Source (10 min)**
Verified pytest config lives in pyproject.toml; ensured no stray pytest.ini

**Task E3: Replace/justify print() diagnostics (20-30 min)**
Converted remaining diagnostics to logging; documented intentional CLI prints

**Task E4: Remove Stale TODOs (5 min)**
Cleared remaining TODOs or linked to tracking issues

**Task E5: Update Bug References (10 min)**
Updated CLI_OVERRIDE_BUG references with live links/explanations

**Task E6: Docstring Polish (1-2 hours)**
Applied Google-style docstrings to all new modules:
- Phase C modules (cache, metrics, serialization, translation, annotation, datasets/base_components)
- Phase D modules (validation_utils, fragment_utils)
Added missing docstrings to public functions
Standardized formatting across all modules

**Quality Gates: ✅ ALL PASSED**
- pytest (full suite): PASSED
- mypy strict: PASSED
- ruff check: PASSED
- ruff format: PASSED
- bandit security scan: PASSED
- make all: PASSED

**Impact:**
- Better documentation: All public APIs documented
- Consistent conventions: Shebangs, pytest config, docstrings
- Cleaner code: No stale TODOs or bug references
- Clearer intent: PSR thresholds explained

**Files Changed:**
- MODIFIED: ~30 files (docstrings, comments, logging cleanups)

**Next:** PRISTINE CODEBASE ACHIEVED! 🎉
All 5 phases complete. Ready to merge to dev → leroy-jenkins/full-send.
EOF
)"

# Push and create PR
git push -u origin claude/refactor-phase-e
gh pr create --title "Phase E: Polish & Documentation - Final Cleanup" \
  --body "Completes Phase E (FINAL) of technical debt cleanup. See commit message for details." \
  --base dev
```

---

## Success Metrics

**Before Phase E (validated 2025-11-20):**
- PSR thresholds: No inline explanation of 0.4 vs 0.5495
- Pytest config: Already consolidated in `pyproject.toml` (no `pytest.ini`)
- `print()` diagnostics: 20+ in preprocessing, 30+ in src
- TODOs: 1 (tests/integration/test_dataset_pipeline.py)
- Bug references: `CLI_OVERRIDE_BUG` mentioned but supporting doc missing
- Docstrings: Will be missing in new modules after Phases C/D

**After Phase E (target):**
- PSR thresholds: Fully documented ✅
- Pytest config: Verified single source ✅
- `print()` diagnostics: Converted to logging or documented ✅
- TODOs: Removed or linked to issues ✅
- Bug references: Updated with explanations/links ✅
- Docstrings: Complete Google-style for new modules ✅

---

## Final Verification: All 5 Phases Complete

### Checklist

**Phase A: Quick Wins**
- [ ] File permissions standardized/documented
- [ ] Bare except blocks fixed
- [ ] Type ignores reduced to ≤2 with justification
- [ ] Empty utils/ deleted
- [ ] Config directories merged

**Phase B: Path Centralization**
- [ ] `preprocessing/paths.py` created
- [ ] Zero hardcoded paths in scripts/tests
- [ ] All scripts/tests use centralized paths

**Phase C: File Splitting**
- [ ] No files >500 lines
- [ ] trainer.py split into training modules
- [ ] datasets/base.py split into base_components
- [ ] boughter stage1/stage2 scripts split

**Phase D: Code Deduplication**
- [ ] validation_utils.py created
- [ ] fragment_utils.py created
- [ ] Duplicate validation/fragment logic removed

**Phase E: Polish**
- [ ] PSR thresholds documented
- [ ] `print()` diagnostics cleaned up
- [ ] TODOs cleaned up or linked
- [ ] Bug references updated
- [ ] Docstrings complete

### Final Quality Gates

```bash
# 1. All tests pass
uv run pytest

# 2. Type checking
uv run mypy src/ preprocessing/ --strict

# 3. Linting
uv run ruff check src/ preprocessing/

# 4. Formatting
uv run ruff format --check src/ preprocessing/

# 5. Security
uv run bandit -r src/ preprocessing/

# 6. Full quality suite
make all
```

### Codebase Quality Score

**Before Refactoring (validated 2025-11-20):**
- Hardcoded paths: 100+ across preprocessing + tests
- Files >500 lines: 4 (`trainer.py`, `datasets/base.py`, `stage1_dna_translation.py`, `stage2_stage3_annotation_qc.py`)
- `type: ignore`: 5 occurrences
- Duplicate preprocessing logic: ~1.6k LOC overlap (validation + fragments)
- Config sources: packages + root `configs/`

**After All 5 Phases (target):**
- Hardcoded paths: centralized (0 inline)
- Files >500 lines: 0
- `type: ignore`: ≤2 with justification
- Duplicate preprocessing logic: 0 (shared utils with byte-for-byte parity)
- Config sources: single package location
- Type coverage: 100% (1 documented ignore) ✅

---

**🎉 PHASE E COMPLETE! PRISTINE CODEBASE ACHIEVED! 🎉**

**Total Effort Across All Phases:** 14-18 hours
**Total Files Changed:** ~92 files
**Total Lines Reduced:** ~490 lines (through deduplication)
**Quality Improvement:** B+ → A+

**Ready to merge all phases:**
```bash
# Merge dev → leroy-jenkins/full-send
git checkout leroy-jenkins/full-send
git pull origin leroy-jenkins/full-send
git merge dev
git push origin leroy-jenkins/full-send
```

**🚀 Ship it! 🚀**
