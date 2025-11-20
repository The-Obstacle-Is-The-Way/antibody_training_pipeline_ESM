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
Random subset of scripts are executable with no clear pattern.

**Currently executable (755):**
- `preprocessing/boughter/train_hyperparameter_sweep.py`
- `preprocessing/boughter/validate_stages2_3.py`
- `preprocessing/jain/step2_preprocess_p5e_s2.py`
- `preprocessing/jain/test_novo_parity.py`
- `preprocessing/shehata/step2_extract_fragments.py`
- `scripts/validation/validate_fragments.py`

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
Two bare `except Exception:` blocks in `src/antibody_training_esm/core/trainer.py` swallow unexpected errors:
- `setup_logging` Hydra fallback (~line 176)
- `train_pipeline` Hydra/legacy output-dir fallback (~line 858)

### Solution
Replace with specific exception types.

### Changes

**Location 1 (setup_logging fallback ~176):**
```python
try:
    hydra_cfg = HydraConfig.get()
    output_dir = Path(hydra_cfg.runtime.output_dir)
    log_file = output_dir / log_file_str
    log_file.parent.mkdir(parents=True, exist_ok=True)
except (ValueError, AttributeError, OSError) as e:
    logger.warning("Hydra output dir not available, falling back to config log path: %s", e)
    log_file = Path(log_file_str)
    if not log_file.is_absolute():
        log_file = Path.cwd() / log_file_str
    log_file.parent.mkdir(parents=True, exist_ok=True)
except Exception as e:
    logger.exception("Unexpected error determining log file path")
    raise
```

**Location 2 (Hydra/legacy output-dir fallback ~858):**
```python
try:
    from hydra.core.hydra_config import HydraConfig

    hydra_cfg = HydraConfig.get()
    cv_output_dir = Path(hydra_cfg.runtime.output_dir)
    experiment_name = cfg.experiment.name
    logger.info(f"Saving CV results to Hydra output dir: {cv_output_dir}")
except (ImportError, AttributeError, OSError, ValueError) as e:
    model_save_dir = config.get("training", {}).get("model_save_dir", "./outputs")
    cv_output_dir = Path(model_save_dir)
    experiment_name = config.get("experiment", {}).get("name", "training")
    logger.info("Running without Hydra, saving CV results to %s (reason: %s)", cv_output_dir, e)
except Exception as e:
    logger.exception("Unexpected error determining CV output directory")
    raise
```

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
Five `type: ignore` comments remain (goal: ≤2 with explicit justification).

**Locations:**
1. `src/antibody_training_esm/core/embeddings.py:60` (`AutoTokenizer.from_pretrained` lacks stubs)
2. `src/antibody_training_esm/core/classifier_factory.py:138` (factory Protocol init)
3. `src/antibody_training_esm/data/loaders.py:16` (`datasets` missing typing for `load_dataset`)
4. `tests/unit/datasets/test_base.py:265` (passing `None` to `sanitize_sequence`)
5. `tests/unit/core/strategies/test_logistic_regression.py:344` (`np.savez` kwargs typing)

### Solution (per-file)

1) **embeddings.py:** Keep ignore but add explanatory comment linking to missing HF stubs.  
2) **classifier_factory.py:** Replace ignore by tightening typing (e.g., `Protocol`/`TypeAlias` for strategy factories with `__init__(config: dict[str, Any])`).  
3) **data/loaders.py:** Keep or wrap import with `TYPE_CHECKING` + stub alias; add comment that `datasets.load_dataset` lacks stubs (attr-defined).  
4) **tests/unit/datasets/test_base.py:** Remove ignore by updating `sanitize_sequence` to accept `str | None` (and keep raising ValueError) or cast to `Any` with explanatory comment.  
5) **tests/unit/core/strategies/test_logistic_regression.py:** Annotate `arrays_dict` as `Mapping[str, np.ndarray]` so `np.savez` call is typed without an ignore.

### Verification
```bash
# Run mypy strict mode
uv run mypy src/antibody_training_esm tests --strict

# Count remaining type: ignore
grep -r "type: ignore" src/ tests/ | wc -l
# Target: ≤2 (HF tokenizer + datasets import) with inline comments
```

### Success Criteria
- [ ] ≤2 `type: ignore` comments remain, both justified by external stubs
- [ ] Remaining ignores have explanatory comments + links
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
- [ ] Run full test suite: `uv run pytest` (currently ~500+ tests collected)
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
- Applied a consistent execute policy across preprocessing scripts
- Documented expected permission (`755` for runnable scripts)

**Task A2: Fix bare except blocks**
- Replaced 2 bare `except Exception:` with specific handling + re-raise
- File: src/antibody_training_esm/core/trainer.py

**Task A3: Address type: ignore comments**
- Added explanatory comment to embeddings.py (HuggingFace lacks stubs)
- Tightened typing around classifier_factory/data.loaders to drop ignores
- Fixed failing tests by removing test-time ignores
- Reduced ignores from 5 to ≤2 and documented remaining ones

**Task A4: Delete empty utils/ directory**
- Removed src/antibody_training_esm/utils/ (only contained __init__.py)

**Task A5: Merge config directories**
- Moved configs/testing/ → src/antibody_training_esm/conf/testing/
- Deleted empty configs/ directory

**Quality Gates: ✅ ALL PASSED**
- make all: PASSED
- pytest (full suite): PASSED
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

**Before Phase A (validated 2025-11-20):**
- Bare except blocks: 2 (`core/trainer.py`)
- `type: ignore` comments: 5
- Empty directories: 1 (`src/antibody_training_esm/utils/`)
- Config locations: 2 (`configs/` + `src/antibody_training_esm/conf/`)
- File permissions: mixed (`755` on 6 scripts, `644` elsewhere)

**After Phase A (target):**
- Bare except blocks: 0 ✅
- `type: ignore` comments: ≤2 with justification ✅
- Empty directories: 0 ✅
- Config locations: 1 ✅
- File permissions: Consistent + documented ✅

---

**Phase A Complete! Ready for Phase B (Path Centralization)**
