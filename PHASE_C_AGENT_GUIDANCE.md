# Phase C: File Splitting - EXPLICIT AGENT GUIDANCE

**Status:** Ready to start
**Branch:** Create `claude/refactor-phase-c` from `dev`
**Estimated Time:** 3-4 hours
**Risk Level:** HIGH (structural changes, many imports)

---

## ⚠️ CRITICAL LESSONS FROM PHASE A & B

### What Went Wrong Before:
1. **Phase B broke e2e tests** by removing mock fixtures without checking test logic
2. **Tests hung for 4+ minutes** downloading 650MB ESM models in CI
3. **Accuracy dropped from 66% to 48%** due to incorrect data loading changes
4. **Multiple reactive fix attempts** instead of thinking first principles

### What We Learned:
✅ **ALWAYS run `make test` before AND after refactoring**
✅ **NEVER remove fixtures without understanding what uses them**
✅ **Check test output carefully** - look for hangs, skips, or failures
✅ **Preserve existing logic** - if a test uses mocks, keep using mocks
✅ **Think before acting** - when stuck, stop and analyze from first principles

---

## 📋 PRE-FLIGHT CHECKLIST (DO THIS FIRST!)

Before touching ANY code:

```bash
# 1. Verify clean starting state
git checkout dev
git pull origin dev
git status  # Should be clean

# 2. Run baseline tests (record results)
make test 2>&1 | tail -10
# Expected: 513 passed, 20 deselected in ~95s

# 3. Record current line counts
cloc src/antibody_training_esm/core/trainer.py
cloc src/antibody_training_esm/datasets/base.py
cloc preprocessing/boughter/stage1_dna_translation.py
cloc preprocessing/boughter/stage2_stage3_annotation_qc.py

# 4. Create Phase C branch
git checkout -b claude/refactor-phase-c
```

**DO NOT PROCEED until all 4 steps pass.**

---

## 🎯 PHASE C OBJECTIVES

Split 4 massive files into maintainable modules:

| File | Current | Target | New Modules |
|------|---------|--------|-------------|
| `core/trainer.py` | 961 lines | ~350 | 3 modules (cache, metrics, serialization) |
| `datasets/base.py` | 627 lines | ~350 | 2-3 modules/mixins |
| `boughter/stage1_dna_translation.py` | 598 lines | ~250 | 2 modules |
| `boughter/stage2_stage3_annotation_qc.py` | 519 lines | ~250 | 2 modules |

**Success Criteria:**
- ✅ All 4 files split correctly
- ✅ All imports updated (no broken imports)
- ✅ All tests pass (`make test`)
- ✅ No regressions in behavior
- ✅ Type checking passes (`mypy`)
- ✅ Documentation updated

---

## 🚨 MANDATORY WORKFLOW FOR EACH FILE SPLIT

For EACH file you split, follow this exact sequence:

### 1. READ & UNDERSTAND (15 min per file)
```bash
# Read the entire file first
cat src/antibody_training_esm/core/trainer.py

# Identify logical sections
grep -n "^def \|^class " src/antibody_training_esm/core/trainer.py

# Check what imports it
grep -r "from.*trainer import\|import.*trainer" src/ tests/

# Check what tests use it
grep -r "trainer" tests/
```

**DO NOT PROCEED until you understand:**
- What functions/classes exist
- What imports this file
- What tests depend on it
- What the logical sections are

### 2. PLAN THE SPLIT (10 min per file)
Write down EXACTLY:
- What goes in each new module
- What stays in the original file
- What imports need updating
- What tests might break

**Example plan:**
```
trainer.py SPLIT PLAN:
- cache.py: get_cache_key(), save_embeddings_cache(), load_embeddings_cache()
- metrics.py: calculate_metrics(), log_metrics(), plot_confusion_matrix()
- serialization.py: save_model(), load_model()
- trainer.py: train_model() (orchestration only)

IMPORTS TO UPDATE:
- tests/unit/core/test_trainer.py
- tests/e2e/test_train_pipeline.py
- src/antibody_training_esm/cli/train.py

TESTS TO RUN AFTER:
- make test
- pytest tests/unit/core/test_trainer.py -v
```

### 3. EXECUTE THE SPLIT (30-45 min per file)
```bash
# Create new module directory
mkdir -p src/antibody_training_esm/core/training
touch src/antibody_training_esm/core/training/__init__.py

# Extract functions to new modules
# (Use Edit tool to move code, preserve type hints, docstrings)

# Update imports in __init__.py to re-export public APIs

# Update imports in original file

# Update imports in dependent files
```

**RULES:**
- ✅ Preserve ALL type hints
- ✅ Preserve ALL docstrings
- ✅ Preserve ALL functionality
- ✅ NO logic changes
- ✅ NO optimization
- ✅ NO "improvements"
- ❌ DO NOT change behavior
- ❌ DO NOT remove tests
- ❌ DO NOT remove fixtures

### 4. VERIFY INCREMENTALLY (15 min per file)
After EACH file split, run:

```bash
# 1. Check imports work
python3 -c "from antibody_training_esm.core.trainer import train_model; print('OK')"

# 2. Run type checking
mypy src/antibody_training_esm/core/

# 3. Run unit tests for this module
pytest tests/unit/core/test_trainer.py -v

# 4. Run fast test suite
make test

# 5. Check for regressions
# Compare with baseline - should be same number passed/skipped
```

**IF ANY TEST FAILS:**
1. **STOP** immediately
2. Read the error message carefully
3. Check what you changed in the last step
4. Revert if needed: `git diff` → `git checkout <file>`
5. Fix the specific issue
6. Re-run tests before continuing

### 5. COMMIT INCREMENTALLY (5 min per file)
Commit AFTER each successful file split:

```bash
git add src/antibody_training_esm/core/trainer.py \
        src/antibody_training_esm/core/training/ \
        tests/unit/core/test_trainer.py

git commit -m "$(cat <<'EOF'
refactor(core): Split trainer.py into modular components

Extracted trainer.py (961 lines) into:
- core/training/cache.py (200 lines) - Embedding cache management
- core/training/metrics.py (250 lines) - Evaluation metrics
- core/training/serialization.py (150 lines) - Model save/load
- core/trainer.py (350 lines) - Main orchestration

All tests pass (513 passed, 20 deselected).
No regressions. Type checking passes.
EOF
)"
```

---

## 📝 TASK-BY-TASK EXECUTION PLAN

### Task C1: Split trainer.py (1.5 hours)

**Target Structure:**
```
src/antibody_training_esm/core/
├── trainer.py                    # Main orchestration (~350 lines)
└── training/
    ├── __init__.py               # Re-export public APIs
    ├── cache.py                  # Embedding cache (~200 lines)
    ├── metrics.py                # Metrics calculation (~250 lines)
    └── serialization.py          # Model save/load (~150 lines)
```

**Checklist:**
- [ ] Read PHASE_C_FILE_SPLITTING.md task C1 section
- [ ] Read current trainer.py in full
- [ ] Create training/ directory
- [ ] Extract cache.py (functions: get_cache_key, save_embeddings_cache, load_embeddings_cache)
- [ ] Extract metrics.py (functions: calculate_metrics, log_metrics, plot_confusion_matrix)
- [ ] Extract serialization.py (functions: save_model, load_model)
- [ ] Update __init__.py to re-export
- [ ] Update imports in trainer.py
- [ ] Update imports in cli/train.py, cli/test.py
- [ ] Update imports in tests/unit/core/test_trainer.py
- [ ] Run `make test` - verify 513 passed
- [ ] Run `mypy src/`
- [ ] Commit

### Task C2: Split base.py (1 hour)

**Target Structure:**
```
src/antibody_training_esm/datasets/
├── base.py                       # Core AntibodyDataset (~350 lines)
└── mixins/
    ├── __init__.py
    ├── annotation_mixin.py       # ANARCI annotation (~200 lines)
    └── fragment_mixin.py         # Fragment extraction (~150 lines)
```

**Checklist:**
- [ ] Read PHASE_C_FILE_SPLITTING.md task C2 section
- [ ] Read current base.py in full
- [ ] Create mixins/ directory
- [ ] Extract annotation_mixin.py (ANARCI annotation methods)
- [ ] Extract fragment_mixin.py (fragment extraction methods)
- [ ] Update base.py to inherit mixins
- [ ] Update imports in dataset loaders (boughter.py, jain.py, harvey.py, shehata.py)
- [ ] Update imports in tests/unit/datasets/test_base.py
- [ ] Run `make test` - verify 513 passed
- [ ] Run `mypy src/`
- [ ] Commit

### Task C3: Split stage1_dna_translation.py (45 min)

**Target Structure:**
```
preprocessing/boughter/
├── stage1_dna_translation.py     # Main pipeline (~250 lines)
└── translation/
    ├── __init__.py
    ├── translator.py             # Translation logic (~200 lines)
    └── readers.py                # FASTA parsing (~150 lines)
```

**Checklist:**
- [ ] Read PHASE_C_FILE_SPLITTING.md task C3 section
- [ ] Read current stage1_dna_translation.py
- [ ] Create translation/ directory
- [ ] Extract translator.py (DNA → protein translation)
- [ ] Extract readers.py (FASTA file parsing)
- [ ] Update stage1_dna_translation.py imports
- [ ] Run stage1_dna_translation.py to verify it works
- [ ] Run `make test` - verify 513 passed
- [ ] Commit

### Task C4: Split stage2_stage3_annotation_qc.py (45 min)

**Target Structure:**
```
preprocessing/boughter/
├── stage2_stage3_annotation_qc.py  # Main pipeline (~250 lines)
└── annotation/
    ├── __init__.py
    ├── annotator.py                # ANARCI annotation (~150 lines)
    └── qc.py                        # Quality control (~150 lines)
```

**Checklist:**
- [ ] Read PHASE_C_FILE_SPLITTING.md task C4 section
- [ ] Read current stage2_stage3_annotation_qc.py
- [ ] Create annotation/ directory
- [ ] Extract annotator.py (ANARCI annotation logic)
- [ ] Extract qc.py (quality control filters)
- [ ] Update stage2_stage3_annotation_qc.py imports
- [ ] Run stage2_stage3_annotation_qc.py to verify it works
- [ ] Run `make test` - verify 513 passed
- [ ] Commit

---

## 🧪 CONTINUOUS VERIFICATION

Run these commands AFTER EVERY TASK:

```bash
# 1. Type checking
mypy src/

# 2. Fast test suite
make test

# 3. Check for import errors
python3 -c "
from antibody_training_esm.core.trainer import train_model
from antibody_training_esm.datasets.base import AntibodyDataset
print('✅ Imports OK')
"

# 4. Git status
git status
```

**Expected output:**
- mypy: `Success: no issues found in 115 source files`
- make test: `513 passed, 20 deselected in ~95s`
- imports: `✅ Imports OK`
- git status: Only intended changes

**IF ANYTHING FAILS:**
1. DO NOT CONTINUE to next task
2. Fix the current issue first
3. Re-run verification
4. Only proceed when all checks pass

---

## 🚫 COMMON MISTAKES TO AVOID

### ❌ DON'T DO THIS:
- Removing mock fixtures from tests
- Changing test logic during refactoring
- "Improving" code while splitting files
- Skipping type hints or docstrings
- Forgetting to update imports in tests
- Continuing after test failures
- Making multiple changes before testing

### ✅ DO THIS INSTEAD:
- Preserve all existing logic exactly
- Move code without changing it
- Update imports everywhere
- Test after each file split
- Commit incrementally
- Stop and debug if tests fail
- Keep refactoring separate from features

---

## 📊 FINAL VERIFICATION (Before Merge)

Before creating PR, run full validation:

```bash
# 1. Clean build
make clean
uv sync --all-extras

# 2. Full quality gate
make all

# 3. E2E tests (optional, but recommended)
make test-e2e

# 4. Line count verification
echo "=== BEFORE (from baseline) ==="
echo "trainer.py: 961 lines"
echo "base.py: 627 lines"
echo "stage1_dna_translation.py: 598 lines"
echo "stage2_stage3_annotation_qc.py: 519 lines"
echo ""
echo "=== AFTER (current) ==="
cloc src/antibody_training_esm/core/trainer.py
cloc src/antibody_training_esm/core/training/
cloc src/antibody_training_esm/datasets/base.py
cloc src/antibody_training_esm/datasets/mixins/
cloc preprocessing/boughter/stage1_dna_translation.py
cloc preprocessing/boughter/translation/
cloc preprocessing/boughter/stage2_stage3_annotation_qc.py
cloc preprocessing/boughter/annotation/

# 5. Git diff stats
git diff dev --stat
```

**Expected Results:**
- ✅ make all passes (format, lint, typecheck, test)
- ✅ 513 tests passed, 20 deselected
- ✅ Line counts match targets (±10%)
- ✅ No unintended changes in git diff

---

## 🎯 COMPLETION CRITERIA

Phase C is COMPLETE when:

1. **All 4 files split successfully**
   - trainer.py → 4 modules
   - base.py → 3 modules
   - stage1_dna_translation.py → 3 modules
   - stage2_stage3_annotation_qc.py → 3 modules

2. **All tests passing**
   - `make test`: 513 passed, 20 deselected
   - `make test-e2e`: All e2e tests pass (optional)

3. **All quality gates passing**
   - `make format`: No changes
   - `make lint`: All checks passed
   - `make typecheck`: Success: no issues found

4. **Git history clean**
   - 4 incremental commits (1 per file split)
   - Clear commit messages
   - No merge conflicts

5. **Documentation updated**
   - PHASE_C_FILE_SPLITTING.md marked complete
   - REFACTOR_PHASES_OVERVIEW.md updated

---

## 🚀 FINAL STEPS

When Phase C is complete:

```bash
# 1. Update status docs
# Edit PHASE_C_FILE_SPLITTING.md - mark all tasks ✅
# Edit REFACTOR_PHASES_OVERVIEW.md - update Phase C status

# 2. Final commit
git add PHASE_C_FILE_SPLITTING.md REFACTOR_PHASES_OVERVIEW.md
git commit -m "docs: Mark Phase C (File Splitting) as complete"

# 3. Push branch
git push origin claude/refactor-phase-c

# 4. Verify GitHub Actions pass (if enabled)

# 5. Create PR to dev
gh pr create --base dev --head claude/refactor-phase-c \
  --title "Phase C: File Splitting (Core Refactoring)" \
  --body "$(cat <<'EOF'
## Summary
Split 4 massive files into modular components following SRP.

## Changes
- trainer.py (961 → 350 lines) + 3 modules
- base.py (627 → 350 lines) + 2 mixins
- stage1_dna_translation.py (598 → 250 lines) + 2 modules
- stage2_stage3_annotation_qc.py (519 → 250 lines) + 2 modules

## Testing
- make all passes ✅
- 513 tests passed, 20 deselected ✅
- No regressions ✅

## Verification
\`\`\`bash
make test    # 513 passed in ~95s
make all     # All quality gates pass
\`\`\`
EOF
)"
```

---

## 📞 WHEN TO ASK FOR HELP

Ask the user if:
- ❓ Tests fail and you can't figure out why after 2 attempts
- ❓ Import errors that aren't obvious
- ❓ Uncertain about where to split a function
- ❓ Circular import dependencies
- ❓ Type checking errors that seem wrong

**DO NOT:**
- Continue with failing tests
- Make guesses about test fixtures
- Remove code without understanding it
- Change test logic during refactoring

---

## ✅ READY TO START?

**Pre-flight checklist complete?**
- [ ] On `dev` branch with latest changes
- [ ] Baseline tests pass (513 passed)
- [ ] Line counts recorded
- [ ] Created `claude/refactor-phase-c` branch

**If YES:** Proceed with Task C1 (Split trainer.py)
**If NO:** Complete pre-flight checklist first

---

**Remember:** SLOW IS SMOOTH, SMOOTH IS FAST. Think first, act second. Test always. 🚀
