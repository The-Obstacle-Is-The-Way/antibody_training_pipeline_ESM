# Phase A: Quick Wins (Low-Hanging Fruit)

**Effort:** 1-1.5 hours
**Risk:** LOW
**Dependencies:** None
**Branch:** `claude/refactor-phase-a`

---

## Overview

Knock out 5 trivial fixes in ~1 hour for immediate visible progress. All changes are mechanical with minimal risk.

**Goal:** Get easy wins on the board to build momentum for larger refactoring phases.

---

## Fixes Included

| Fix # | Description | Effort | Risk |
|-------|-------------|--------|------|
| #8 | Standardize file permissions | 10 min | ZERO |
| #9 | Fix bare except blocks | 10 min | LOW |
| #10 | Address type: ignore comments | 30 min | LOW |
| #11 | Delete empty utils/ directory | 5 min | ZERO |
| #12 | Merge config directories | 15 min | LOW |

**Total Time:** 1-1.5 hours

---

## Task A1: Standardize File Permissions (10 min)

### Problem
Random subset of 6 scripts are executable with no clear pattern.

### Solution
Make ALL preprocessing scripts executable (755 permissions).

### Commands
```bash
# Make all .py scripts in preprocessing/ executable
find preprocessing -name "*.py" ! -name "__init__.py" -exec chmod +x {} \;

# Verify all have execute permission
find preprocessing -name "*.py" ! -name "__init__.py" -exec ls -l {} \; | awk '{print $1}' | sort -u
# Expected output: -rwxr-xr-x
```

### Verification
```bash
# Check all .py files have same permissions (755)
find preprocessing -name "*.py" ! -name "__init__.py" -exec stat -f "%Sp %N" {} \; | awk '{print $1}' | sort | uniq -c
# Should show single value: -rwxr-xr-x

# Verify scripts still run
uv run python preprocessing/jain/validate_conversion.py
```

### Success Criteria
- [ ] All preprocessing .py files have 755 permissions
- [ ] All scripts remain executable

---

## Task A2: Fix Bare Except Blocks (10 min)

### Problem
Four bare `except Exception:` blocks in trainer.py catch too much.

**Locations:**
- `src/antibody_training_esm/core/trainer.py:176`
- `src/antibody_training_esm/core/trainer.py:831`
- `src/antibody_training_esm/core/trainer.py:875`
- `src/antibody_training_esm/core/trainer.py:927`

### Solution
Replace with specific exception types.

### Changes

**Location 1 (Line 176) - Cache deletion:**
```python
# BEFORE:
try:
    cache_path.unlink()
    logger.info(f"Deleted cache file: {cache_path}")
except Exception:  # ← TOO BROAD
    logger.warning(f"Could not delete cache file {cache_path}: {e}")

# AFTER:
try:
    cache_path.unlink()
    logger.info(f"Deleted cache file: {cache_path}")
except (OSError, PermissionError) as e:
    logger.warning(f"Could not delete cache file {cache_path}: {e}")
```

**Locations 2-4 (Lines 831, 875, 927):**
Follow same pattern - identify specific exceptions that can be raised and catch those explicitly.

### Verification
```bash
# Check no bare except Exception: remain
grep -n "except Exception:" src/antibody_training_esm/core/trainer.py
# Should return nothing

# Tests still pass
uv run pytest tests/unit/core/test_trainer.py -v
```

### Success Criteria
- [ ] Zero bare `except Exception:` in trainer.py
- [ ] All trainer tests pass
- [ ] Mypy passes

---

## Task A3: Address type: ignore Comments (30 min)

### Problem
2 `type: ignore` comments indicate incomplete type coverage.

**Locations:**
1. `src/antibody_training_esm/core/embeddings.py:60`
2. `tests/unit/datasets/test_base.py:265`

### Solution

**Fix #1 - embeddings.py:60 (Add explanatory comment):**

```python
# BEFORE:
self.tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
    model_name, revision=revision
)

# AFTER:
# Type ignore needed: transformers.AutoTokenizer lacks type stubs
# This is a known limitation of the HuggingFace transformers library
# See: https://github.com/huggingface/transformers/issues/
self.tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
    model_name, revision=revision
)
```

**Fix #2 - test_base.py:265 (Add proper type annotation):**

```python
# BEFORE:
mock_dataset = MockDataset()  # type: ignore

# AFTER:
mock_dataset: AntibodyDataset = MockDataset()  # Explicit type for mypy
```

### Verification
```bash
# Run mypy strict mode
uv run mypy src/antibody_training_esm tests --strict

# Count remaining type: ignore
grep -r "type: ignore" src/ tests/ | wc -l
# Should be 1 (only the HuggingFace one with explanation)
```

### Success Criteria
- [ ] ≤1 `type: ignore` comments remain
- [ ] Remaining ignore has explanatory comment
- [ ] Mypy strict mode passes

---

## Task A4: Delete Empty utils/ Directory (5 min)

### Problem
`src/antibody_training_esm/utils/` contains only `__init__.py`, serves no purpose.

### Solution
Delete the entire directory.

### Commands
```bash
# Check nothing imports from utils
grep -r "from antibody_training_esm.utils import" src/ tests/
# Should return nothing

# If nothing imports it, delete
rm -rf src/antibody_training_esm/utils/
```

### Verification
```bash
# Verify deletion
ls src/antibody_training_esm/utils
# Should error: "No such file or directory"

# All tests pass
uv run pytest
```

### Success Criteria
- [ ] `src/antibody_training_esm/utils/` doesn't exist
- [ ] All tests pass
- [ ] No import errors

---

## Task A5: Merge Config Directories (15 min)

### Problem
Two config directories exist:
- `configs/` (root)
- `src/antibody_training_esm/conf/` (package)

### Solution
Move everything to package location, delete root configs/.

### Commands
```bash
# Move testing configs into package
mkdir -p src/antibody_training_esm/conf/testing/
mv configs/testing/jain_p5e_s2.yaml src/antibody_training_esm/conf/testing/

# Delete empty configs/
rmdir configs/testing/
rmdir configs/
```

### Verification
```bash
# Configs directory deleted
ls configs/
# Should error: "No such file or directory"

# New location exists
ls src/antibody_training_esm/conf/testing/jain_p5e_s2.yaml
# Should succeed

# Hydra can still find configs
uv run antibody-train --help
# Should work without errors
```

### Success Criteria
- [ ] `configs/` directory doesn't exist
- [ ] All configs in `src/antibody_training_esm/conf/`
- [ ] `antibody-train --help` works
- [ ] All tests pass

---

## Phase Completion Checklist

### Quality Gates
- [ ] All 5 tasks complete
- [ ] Run `make all` (format → lint → typecheck → test)
- [ ] Run full test suite: `uv run pytest` (all 468 tests pass)
- [ ] Run security scan: `uv run bandit -r src/ preprocessing/`
- [ ] Verify no regressions in preprocessing scripts

### Git Workflow
```bash
# Create branch
git checkout dev
git pull origin dev
git checkout -b claude/refactor-phase-a

# Make changes (complete all 5 tasks above)

# Commit
git add -A
git commit -m "$(cat <<'EOF'
refactor: Phase A - Quick wins (5 trivial fixes)

Completed 5 low-risk improvements for immediate progress:

**Task A1: Standardize file permissions**
- Made all preprocessing scripts executable (755)
- Consistent permissions across ~20 files

**Task A2: Fix bare except blocks**
- Replaced 4 bare `except Exception:` with specific types
- File: src/antibody_training_esm/core/trainer.py

**Task A3: Address type: ignore comments**
- Added explanatory comment to embeddings.py (HuggingFace lacks stubs)
- Fixed test_base.py with proper type annotation
- Reduced type ignores from 2 to 1

**Task A4: Delete empty utils/ directory**
- Removed src/antibody_training_esm/utils/ (only contained __init__.py)

**Task A5: Merge config directories**
- Moved configs/testing/ → src/antibody_training_esm/conf/testing/
- Deleted empty configs/ directory

**Quality Gates: ✅ ALL PASSED**
- make all: PASSED
- pytest (468 tests): PASSED
- bandit security scan: PASSED
- mypy strict: PASSED

**Impact:**
- Improved code clarity (specific exception handling)
- Better type safety (fewer ignores)
- Cleaner directory structure (no empty dirs)
- Consistent file permissions

**Next:** Phase B - Path Centralization
EOF
)"

# Push and create PR
git push -u origin claude/refactor-phase-a

# Create PR (if using gh CLI)
gh pr create --title "Phase A: Quick Wins - 5 Trivial Fixes" \
  --body "Completes Phase A of technical debt cleanup. See commit message for details." \
  --base dev
```

### Final Checks
- [ ] All tasks verified individually
- [ ] Quality gates passed
- [ ] Committed with detailed message
- [ ] PR created (if applicable)
- [ ] Ready for senior review

---

## Success Metrics

**Before Phase A:**
- Bare except blocks: 4
- type: ignore comments: 2
- Empty directories: 1
- Config locations: 2
- File permission chaos: Yes

**After Phase A:**
- Bare except blocks: 0 ✅
- type: ignore comments: 1 (with explanation) ✅
- Empty directories: 0 ✅
- Config locations: 1 ✅
- File permissions: Consistent (755) ✅

---

**Phase A Complete! Ready for Phase B (Path Centralization)**
