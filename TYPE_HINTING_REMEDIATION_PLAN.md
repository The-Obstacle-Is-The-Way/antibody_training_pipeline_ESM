# Type-Hint Remediation Strategy

_Last updated: 2025-11-08_

## Executive Summary

**TL;DR:** Another AI enforced strict mypy (`disallow_untyped_defs = true`) without adding type annotations to test/preprocessing code. Result: **546 errors** across **46 files**. Production code (`src/`) is clean. This is **mechanical but time-consuming** (5-7 hours). Plan below provides step-by-step instructions to fix systematically.

**Status:** Phases A & B complete; Phase C underway (tests/unit/data ✅, tests/unit/datasets ✅, remaining clusters pending).

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
- ✅ Phase C (kickoff) – `tests/unit/data/test_loaders.py` typed – commit `0dcb166`
- ✅ Phase C – `tests/unit/datasets/*` typed (base, annotation, boughter, harvey, jain, shehata) – commit `fix: Add type annotations to dataset tests`
- ⏳ Remaining ≈373 errors (546 − 173 fixed) across the rest of Phases C–E

## 4. Scope

**Verified as of 2025-11-08:**

- **546 mypy errors** across **46 files**
- **0 errors in `src/`** (production code is clean ✅)
- **All errors are in test/support code** (safe to fix without runtime risk)

### Error Breakdown by Directory

```
Priority 1 (Fix First - Cascades to Tests):
  11 errors - tests/fixtures/          (mock models, sequences)

Priority 2 (Test Files - Mechanical Fixes):
 140 errors - tests/unit/datasets/     (dataset loader tests)
  99 errors - tests/unit/core/         (core module tests)
  88 errors - tests/integration/       (integration tests)
  76 errors - tests/unit/cli/          (CLI tests)
  22 errors - tests/unit/data/         (data loader tests)
  22 errors - tests/e2e/               (end-to-end tests)

Priority 3 (Preprocessing Scripts):
  29 errors - preprocessing/boughter/  (Boughter preprocessing)
  15 errors - preprocessing/jain/      (Jain preprocessing)
   6 errors - preprocessing/shehata/   (Shehata preprocessing)
   6 errors - preprocessing/harvey/    (Harvey preprocessing)

Priority 4 (Utility Scripts):
   8 errors - scripts/validation/      (validation scripts)
   2 errors - scripts/testing/         (testing scripts)
```

**All errors are `Function is missing a type annotation [no-untyped-def]` - very mechanical to fix.**

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
2. ✅ `tests/unit/datasets/test_base.py` (many errors) – included in commit `fix: Add type annotations to dataset tests`
3. ✅ `tests/unit/datasets/test_base_annotation.py` (many errors) – included in commit `fix: Add type annotations to dataset tests`
4. ✅ `tests/unit/datasets/test_*.py` (boughter, harvey, jain, shehata) – included in commit `fix: Add type annotations to dataset tests`
5. `tests/unit/core/test_*.py` (99 errors) - 1 hour
6. `tests/unit/cli/test_*.py` (76 errors) - 45 min
7. `tests/integration/test_*.py` (88 errors) - 45 min
8. `tests/e2e/test_*.py` (22 errors) - 20 min

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
1. `preprocessing/boughter/*.py` (29 errors) - 30 min
2. `preprocessing/jain/*.py` (15 errors) - 20 min
3. `preprocessing/shehata/*.py` (6 errors) - 15 min
4. `preprocessing/harvey/*.py` (6 errors) - 15 min

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
1. `scripts/validation/*.py` (8 errors)
2. `scripts/testing/*.py` (2 errors)

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

1. ✅ Clean `mypy` run (zero errors)
2. ✅ All 400+ tests still passing
3. ✅ `make all` passes (ruff + mypy + pytest + coverage)
4. ✅ `.mypy_failures.txt` removed
5. ✅ Logical git commits (one per phase/directory)
6. ✅ Updated documentation if needed

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
- [ ] Phase C: Fix `tests/unit/core/` (99 errors)
- [ ] Phase C: Fix `tests/unit/cli/` (76 errors)
- [ ] Phase C: Fix `tests/integration/` (88 errors)
- [ ] Phase C: Fix `tests/e2e/` (22 errors)
- [ ] Phase D: Fix `preprocessing/boughter/` (29 errors)
- [ ] Phase D: Fix `preprocessing/jain/` (15 errors)
- [ ] Phase D: Fix `preprocessing/shehata/` (6 errors)
- [ ] Phase D: Fix `preprocessing/harvey/` (6 errors)
- [ ] Phase E: Fix `scripts/` (10 errors)
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
