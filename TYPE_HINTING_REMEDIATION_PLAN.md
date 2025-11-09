# Type-Hint Remediation Strategy

_Last updated: 2025-11-08 (Phase C – CLI/integration next)_

## Executive Summary

**TL;DR:** Another AI enforced strict mypy (`disallow_untyped_defs = true`) without adding type annotations to test/preprocessing code. Result: **546 errors** across **46 files**. Production code (`src/`) is clean. This is **mechanical but time-consuming** (5-7 hours). Plan below provides step-by-step instructions to fix systematically.

**Status:** Phases A & B complete; Phase C in progress – fixtures, `tests/unit/data`, `tests/unit/datasets`, `tests/conftest.py`, and the entire `tests/unit/core/` tree are now typed; remaining Phase C clusters (CLI, integration, e2e) are pending.

**Impact:** Blocks `make all` and CI until fixed.

**Risk:** Low - all errors are in test/support code, not production.

## 1. Background

- Commit `5003e6c` flipped `[tool.mypy].disallow_untyped_defs = true` for the entire repo.
- Unit/integration tests, fixtures, and preprocessing scripts still contain large blocks of legacy `def foo(...)` signatures without annotations.
- `make all` (which runs `uv run mypy .`) now fails with hundreds of `no-untyped-def` errors even though runtime behaviour is unchanged.

## 2. Problem Statement

Type enforcement landed before the codebase was ready. This blocks CI / `make all`, hides real typing issues behind noise, and risks regressions when contributors start adding `# type: ignore` everywhere just to get builds green.

## 3. Goals

1. Restore a passing `make all` without weakening the stricter mypy settings.
2. Eliminate “missing type annotation” errors systematically (no mass `Any`, no broad config relaxations).
3. Keep runtime code paths untouched; only add hints or light refactors that clarify intent.

## Progress Snapshot (2025-11-08)

- ✅ Phase A – Inventory completed via `.mypy_failures.txt` (546-error baseline)
- ✅ Phase B – Fixtures typed (`tests/fixtures/mock_*`) – commit `3352ad7`
- ✅ Phase C – `tests/unit/data/test_loaders.py` typed and API tightened to accept `Path | str` / `Sequence[...]` (commit `0b274a3` for loaders, `0dcb166` for tests)
- ✅ Phase C – `tests/unit/datasets/*` typed (base, annotation, boughter, harvey, jain, shehata) – commit `7d79238`
- ✅ Shared pytest glue – `tests/conftest.py` fully annotated (commit `fc68eb8`)
- ✅ `tests/unit/core/test_classifier.py` typed with passing slice tests (commit `d82dcda`)
- ✅ `tests/unit/core/test_embeddings.py` + `tests/unit/core/test_trainer.py` typed (mypy + pytest slices passing; commit pending in this change set)
- ⏳ Remaining **313** mypy errors (updated `.mypy_failures.txt`, 2025-11-08) concentrated in the yet-to-fix Phase C/D/E directories (breakdown below)

## 4. Scope

**Verified as of 2025-11-08:**

- **546 mypy errors** across **46 files**
- **0 errors in `src/`** (production code is clean ✅)
- **All errors are in test/support code** (safe to fix without runtime risk)

### Error Breakdown by Directory (post-Phase C updates)

```
Priority 1 (Shared fixtures/utilities):
   0 errors - tests/fixtures/ ✅
   0 errors - tests/conftest.py ✅

Priority 2 (Test Files - Mechanical Fixes):
   0 errors - tests/unit/core/         ✅ (classifier, embeddings, trainer)
 100 errors - tests/unit/cli/          (test_preprocess.py, test_test.py, test_train.py, test_model_tester.py)
 108 errors - tests/integration/       (model tester/persistence/dataset pipeline + dataset compat suites)
  23 errors - tests/e2e/               (train pipeline + reproduce novo flows)
   0 errors - tests/unit/data/ ✅
   0 errors - tests/unit/datasets/ ✅

Priority 3 (Preprocessing Scripts):
  34 errors - preprocessing/boughter/
  18 errors - preprocessing/jain/
   9 errors - preprocessing/shehata/
   9 errors - preprocessing/harvey/

Priority 4 (Utility Scripts):
   9 errors - scripts/validation/
   3 errors - scripts/testing/
```

**All remaining failures are `no-untyped-def`; no behavioural regressions observed so far.**

## 5. Strategy

### Phase A – Inventory & Ordering ✅ COMPLETE

1. ✅ Generated `.mypy_failures.txt` with 546 errors
2. ✅ Grouped failures by path cluster
3. ✅ Prioritized shared utilities first (fixtures, helpers)

### Phase B – Fix Shared Utilities (Priority 1) ✅ COMPLETE

**Time Estimate: 30 minutes**

**Status:** Completed 2025-11-08 (commit `3352ad7`). Fixtures now provide typed mocks/models that downstream tests rely on.

**Files to Fix:**
1. `tests/fixtures/mock_models.py` (11 errors)
   - Add `-> None` to all `__init__` methods
   - Add return types to all mock methods
   - Add parameter types to all functions

**Commands:**
```bash
# Fix fixtures
# (Apply type annotations to all functions)

# Verify
uv run mypy tests/fixtures/ --no-error-summary
# Expected: 0 errors

# Commit
git add tests/fixtures/
git commit -m "fix: Add type annotations to test fixtures"
```

### Phase C – Fix Test Files (Priority 2)

**Time Estimate: 3-4 hours**

**Files to Fix (in order):**
1. ✅ `tests/unit/data/test_loaders.py` (22 errors) – done 2025-11-08 (commit `0dcb166`)
2. ✅ `tests/unit/datasets/test_base.py` – included in commit `7d79238`
3. ✅ `tests/unit/datasets/test_base_annotation.py` – included in commit `7d79238`
4. ✅ `tests/unit/datasets/test_*.py` (boughter, harvey, jain, shehata) – included in commit `7d79238`
5. ✅ `tests/unit/core/test_*.py` – classifier, embeddings, and trainer suites fully typed (commits `d82dcda` + current change)
6. `tests/unit/cli/test_*.py` (**~100 errors**) – ~45 minutes (three CLI entrypoint suites + model tester)
7. `tests/integration/test_*.py` (**~108 errors**) – ~45 minutes (model tester/persistence/dataset compatibility suites)
8. `tests/e2e/test_*.py` (**23 errors**) – ~20 minutes (train pipeline + reproduce novo)

**Pattern to Follow:**
```python
# Before:
def test_something(mock_data):
    result = function_under_test()
    assert result is not None

# After:
def test_something(mock_data: pd.DataFrame) -> None:
    result = function_under_test()
    assert result is not None
```

**Common Type Annotations:**
- Test functions: `-> None`
- Pytest fixtures: `-> Generator[Type, None, None]` or `-> Type`
- Mock data: `-> pd.DataFrame`, `-> dict[str, Any]`, `-> list[str]`

**Commands:**
```bash
# Fix each test file
# (Apply type annotations)

# Verify after each file/group
uv run mypy tests/unit/data/ --no-error-summary
uv run mypy tests/unit/datasets/ --no-error-summary
# etc.

# Commit in logical chunks (by subdirectory)
git add tests/unit/data/
git commit -m "fix: Add type annotations to data loader tests"

git add tests/unit/datasets/
git commit -m "fix: Add type annotations to dataset tests"
# etc.
```

### Phase D – Fix Preprocessing Scripts (Priority 3)

**Time Estimate: 1-2 hours**

**Files to Fix:**
1. `preprocessing/boughter/*.py` (34 errors) - 30 min
2. `preprocessing/jain/*.py` (18 errors) - 20 min
3. `preprocessing/shehata/*.py` (9 errors) - 15 min
4. `preprocessing/harvey/*.py` (9 errors) - 15 min

**Pattern to Follow:**
```python
# Before:
def process_data(input_file):
    df = pd.read_csv(input_file)
    return df

# After:
def process_data(input_file: str | Path) -> pd.DataFrame:
    df = pd.read_csv(input_file)
    return df
```

**Commands:**
```bash
# Fix each preprocessing directory
# (Apply type annotations)

# Verify
uv run mypy preprocessing/boughter/ --no-error-summary
# etc.

# Commit by dataset
git add preprocessing/boughter/
git commit -m "fix: Add type annotations to Boughter preprocessing"
# etc.
```

### Phase E – Fix Utility Scripts (Priority 4)

**Time Estimate: 15-30 minutes**

**Files to Fix:**
1. `scripts/validation/*.py` (9 errors)
2. `scripts/testing/*.py` (3 errors)

**Commands:**
```bash
# Fix scripts
# (Apply type annotations)

# Verify
uv run mypy scripts/ --no-error-summary

# Commit
git add scripts/
git commit -m "fix: Add type annotations to utility scripts"
```

### Phase F – Final Verification

**Time Estimate: 10 minutes**

**Commands:**
```bash
# Run full mypy check
uv run mypy .
# Expected: Found 0 errors

# Run full test suite
uv run pytest -v
# Expected: All tests pass

# Run all code quality checks
make all
# Expected: All checks pass

# Remove temporary file
rm .mypy_failures.txt

# Final commit
git add .
git commit -m "chore: Complete type annotation remediation - mypy clean"
```

## 6. Guardrails

- **No blanket `# type: ignore`** unless there is a library stub gap; justify each with a comment.
- **No config weakening** (do not flip `disallow_untyped_defs` back off).
- **Minimal churn**: only add annotations or micro-refactors needed to express types.

## 7. Deliverables

- [ ] Clean `mypy` run (zero errors)
- [ ] All 400+ tests still passing
- [ ] `make all` passes (ruff + mypy + pytest + coverage)
- [ ] `.mypy_failures.txt` removed
- [x] Logical git commits (one per phase/directory) – ongoing per-cluster commits
- [x] Updated documentation as progress advances (this file)

## 8. Total Time Estimate

**Total: 5-7 hours of focused work**

- Phase A (Inventory): ✅ DONE
- Phase B (Fixtures): 30 min
- Phase C (Tests): 3-4 hours
- Phase D (Preprocessing): 1-2 hours
- Phase E (Scripts): 15-30 min
- Phase F (Verification): 10 min

**Can be parallelized:** Multiple AI agents or developers can work on different phases simultaneously (e.g., one on tests, one on preprocessing).

## 9. Execution Checklist

- [x] Phase B: Fix `tests/fixtures/` (11 errors)
- [x] Phase C: Fix `tests/unit/data/` (22 errors)
- [x] Phase C: Fix `tests/unit/datasets/` (140 errors)
- [x] Phase C: Fix `tests/unit/core/` (classifier + embeddings + trainer)
- [ ] Phase C: Fix `tests/unit/cli/` (~100 errors)
- [ ] Phase C: Fix `tests/integration/` (~108 errors)
- [ ] Phase C: Fix `tests/e2e/` (23 errors)
- [ ] Phase D: Fix `preprocessing/boughter/` (34 errors)
- [ ] Phase D: Fix `preprocessing/jain/` (18 errors)
- [ ] Phase D: Fix `preprocessing/shehata/` (9 errors)
- [ ] Phase D: Fix `preprocessing/harvey/` (9 errors)
- [ ] Phase E: Fix `scripts/` (12 errors total – validation 9, testing 3)
- [ ] Phase F: Run `uv run mypy .` → 0 errors
- [ ] Phase F: Run `uv run pytest -v` → all pass
- [ ] Phase F: Run `make all` → all pass

## 10. How Another AI Agent Should Execute This

**Step-by-step instructions:**

1. Read this entire document first
2. Start with Phase B (fixtures) - **DO NOT SKIP THIS**
3. For each file:
   - Read the file
   - Find all functions missing type annotations
   - Add `-> None` for functions that don't return anything
   - Add `-> Type` for functions that return values
   - Add parameter types (use mypy error messages as hints)
4. After each file/group, verify with `uv run mypy <path>`
5. Commit when a phase is complete
6. Move to next phase
7. Final verification with `make all`

**DO NOT:**
- Skip fixtures (causes cascade failures)
- Use `# type: ignore` without justification
- Use `Any` as a cop-out (be specific with types)
- Batch all changes into one commit (commit per phase)

Let's execute this plan carefully so we keep strict typing **and** a green CI. Homie, we've got this. 💪
