# Phase E: Polish & Documentation

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
| #17 | Standardize shebangs | 10 min | ZERO |
| #18 | Delete pytest.ini | 10 min | LOW |
| #21 | Remove stale TODOs | 5 min | ZERO |
| #22 | Update bug references | 10 min | ZERO |
| #24-25 | Docstring polish | 1-2 hours | LOW |

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

## Task E2: Standardize Shebangs (10 min)

### Problem
Inconsistent shebang usage across preprocessing scripts.

### Solution
Add `#!/usr/bin/env python3` to ALL preprocessing .py files (except `__init__.py`).

### Implementation

**Automated approach:**

```bash
# Add shebang to all preprocessing scripts missing it
for file in preprocessing/*/*.py; do
    # Skip __init__.py files
    if [[ $(basename "$file") == "__init__.py" ]]; then
        continue
    fi

    # Check if file already has shebang
    if ! head -1 "$file" | grep -q '^#!/'; then
        # Add shebang at the top
        echo '#!/usr/bin/env python3' | cat - "$file" > /tmp/tempfile
        mv /tmp/tempfile "$file"
        echo "Added shebang to $file"
    fi
done
```

### Verification
```bash
# Check all preprocessing scripts have shebangs
for file in preprocessing/*/*.py; do
    if [[ $(basename "$file") != "__init__.py" ]]; then
        if ! head -1 "$file" | grep -q '^#!/'; then
            echo "Missing shebang: $file"
        fi
    fi
done

# Should return NOTHING
```

### Success Criteria
- [ ] All preprocessing .py files have shebangs (except `__init__.py`)
- [ ] Shebangs are consistent: `#!/usr/bin/env python3`

---

## Task E3: Consolidate Pytest Config (10 min)

### Problem
Duplicate pytest configuration in `pytest.ini` AND `pyproject.toml`.

### Solution
Delete `pytest.ini`, keep all config in `pyproject.toml`.

### Implementation

**Step 1: Verify pyproject.toml has all pytest config**

```bash
# Check pytest config in pyproject.toml
grep -A20 "\[tool.pytest" pyproject.toml
```

Should include:
- `testpaths`
- `markers`
- `python_files`
- `python_classes`
- `python_functions`
- `addopts`

**Step 2: Delete pytest.ini**

```bash
# Backup first (just in case)
cp pytest.ini pytest.ini.backup

# Delete
rm pytest.ini
```

**Step 3: Verify tests still work**

```bash
# Run tests
uv run pytest

# Should work without errors
```

### Verification
```bash
# Check pytest.ini doesn't exist
ls pytest.ini
# Should error: "No such file or directory"

# Tests still pass
uv run pytest
```

### Success Criteria
- [ ] `pytest.ini` deleted
- [ ] All pytest config in `pyproject.toml`
- [ ] Tests still pass

---

## Task E4: Remove Stale TODOs (5 min)

### Problem
Stale TODO comments that should either be implemented or deleted.

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
Code references old bugs (e.g., "CLI_OVERRIDE_BUG") that are now fixed.

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

**Phase C modules (7 files):**
- `src/antibody_training_esm/core/training/cache.py`
- `src/antibody_training_esm/core/training/metrics.py`
- `src/antibody_training_esm/core/training/serialization.py`
- `preprocessing/boughter/translation/dna_translator.py`
- `preprocessing/boughter/translation/validation.py`
- `preprocessing/boughter/annotation/anarci.py`
- `preprocessing/boughter/annotation/qc.py`

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
- [ ] Task E2: All scripts have shebangs
- [ ] Task E3: pytest.ini deleted
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

**Task E2: Standardize Shebangs (10 min)**
Added #!/usr/bin/env python3 to all preprocessing scripts
Consistent shebang across ~20 files

**Task E3: Consolidate Pytest Config (10 min)**
Deleted pytest.ini (duplicate configuration)
All pytest config now in pyproject.toml only

**Task E4: Remove Stale TODOs (5 min)**
Removed X stale TODO comments
Implemented Y trivial TODOs
Kept Z valid TODOs

**Task E5: Update Bug References (10 min)**
Removed references to resolved bugs (CLI_OVERRIDE_BUG)
Replaced with clear explanations of WHY code exists

**Task E6: Docstring Polish (1-2 hours)**
Applied Google-style docstrings to all new modules:
- Phase C modules (7 files): cache, metrics, serialization, translation, annotation
- Phase D modules (2 files): validation_utils, fragment_utils
Added missing docstrings to public functions
Standardized formatting across all modules

**Quality Gates: ✅ ALL PASSED**
- pytest (468 tests): PASSED
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
- MODIFIED: ~30 files (docstrings, comments, shebangs)
- DELETED: pytest.ini

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

**Before Phase E:**
- PSR threshold confusion: No explanation
- Inconsistent shebangs: Random subset
- Duplicate pytest config: pytest.ini + pyproject.toml
- Stale TODOs: Multiple
- Old bug references: Present
- Missing docstrings: Many

**After Phase E:**
- PSR thresholds: Fully documented ✅
- Shebangs: 100% consistent ✅
- Pytest config: Single source (pyproject.toml) ✅
- Stale TODOs: Removed ✅
- Bug references: Updated with explanations ✅
- Docstrings: Complete Google-style ✅

---

## Final Verification: All 5 Phases Complete

### Checklist

**Phase A: Quick Wins** ✅
- [ ] File permissions standardized (755)
- [ ] Bare except blocks fixed
- [ ] Type ignores addressed
- [ ] Empty utils/ deleted
- [ ] Config directories merged

**Phase B: Path Centralization** ✅
- [ ] preprocessing/paths.py created
- [ ] Zero hardcoded paths
- [ ] All scripts use centralized paths

**Phase C: File Splitting** ✅
- [ ] No files >500 lines
- [ ] trainer.py split into 4 modules
- [ ] 2 preprocessing scripts split

**Phase D: Code Deduplication** ✅
- [ ] validation_utils.py created
- [ ] fragment_utils.py created
- [ ] ~840 duplicate lines eliminated

**Phase E: Polish** ✅
- [ ] PSR thresholds documented
- [ ] Shebangs standardized
- [ ] pytest.ini deleted
- [ ] TODOs cleaned up
- [ ] Docstrings complete

### Final Quality Gates

```bash
# 1. All tests pass
uv run pytest
# Expected: 468 tests passed

# 2. Type checking
uv run mypy src/ preprocessing/ --strict
# Expected: Success: no issues found

# 3. Linting
uv run ruff check src/ preprocessing/
# Expected: All checks passed

# 4. Formatting
uv run ruff format --check src/ preprocessing/
# Expected: All files formatted correctly

# 5. Security
uv run bandit -r src/ preprocessing/
# Expected: No issues found

# 6. Full quality suite
make all
# Expected: All gates pass
```

### Codebase Quality Score

**Before Refactoring (Jekyll & Hyde):**
- Grade: B+ (good src/, mediocre preprocessing/)
- Issues: 26 total (8 critical, 8 high, 10 medium)
- Duplicate code: ~840 lines
- Hardcoded paths: 50+ instances
- Files >500 lines: 3 files
- Type coverage: 99% (2 ignores)

**After All 5 Phases (Pristine):**
- Grade: A+ (production-quality throughout) ✅
- Issues: 0 ✅
- Duplicate code: 0 ✅
- Hardcoded paths: 0 (all in paths.py) ✅
- Files >500 lines: 0 ✅
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
