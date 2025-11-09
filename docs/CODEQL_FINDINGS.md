# CodeQL Security Findings

**Date**: 2025-11-08
**Branch**: `leroy-jenkins/full-send`
**Scanner**: GitHub CodeQL (security-extended + security-and-quality queries)

## Summary

| # | Severity | Rule | File | Line | Status |
|---|----------|------|------|------|--------|
| 3 | **ERROR** | Potentially uninitialized local variable | `experiments/novo_parity/scripts/batch_permutation_test.py` | 309 | 🔴 MUST FIX |
| 2 | NOTE | Unused global variable | `src/antibody_training_esm/core/trainer.py` | 386 | 🟡 CLEANUP |
| 1 | NOTE | Unused local variable | `experiments/novo_parity/scripts/run_exp05_inference.py` | 73 | 🟡 CLEANUP |

---

## 🔴 CRITICAL: Issue #3 - Potentially Uninitialized Local Variable

### Location
`experiments/novo_parity/scripts/batch_permutation_test.py:309`

### Finding
```python
# Line 301-309
reclass_func = RECLASSIFICATION_STRATEGIES[reclass_strategy][1]
if callable(reclass_func):
    if reclass_strategy in ["R1", "R2"]:
        reclass_ids = reclass_func()
    else:
        reclass_ids = reclass_func(df)

print(f"\n✅ Reclassification: {len(reclass_ids)} antibodies")  # ← ERROR: reclass_ids may not exist
```

### Root Cause
If `reclass_func` is **not callable**, the `if callable(reclass_func)` block is skipped, and `reclass_ids` is never initialized. Then line 309 tries to use `reclass_ids`, causing `NameError`.

### Impact
- **Runtime crash**: `NameError: name 'reclass_ids' is not defined`
- **Production risk**: Batch permutation tests would fail silently or crash
- **Data integrity**: Permutation experiments would be incomplete

### Fix Plan

**Option A: Initialize with empty list (defensive)**
```python
# Line 301-309 (FIXED)
reclass_func = RECLASSIFICATION_STRATEGIES[reclass_strategy][1]
reclass_ids = []  # ← Initialize to empty list

if callable(reclass_func):
    if reclass_strategy in ["R1", "R2"]:
        reclass_ids = reclass_func()
    else:
        reclass_ids = reclass_func(df)

print(f"\n✅ Reclassification: {len(reclass_ids)} antibodies")  # ← Safe now
```

**Option B: Fail fast with assertion (strict)**
```python
# Line 301-309 (ALTERNATIVE)
reclass_func = RECLASSIFICATION_STRATEGIES[reclass_strategy][1]
assert callable(reclass_func), f"Invalid reclass_func for strategy {reclass_strategy}"

if reclass_strategy in ["R1", "R2"]:
    reclass_ids = reclass_func()
else:
    reclass_ids = reclass_func(df)

print(f"\n✅ Reclassification: {len(reclass_ids)} antibodies")
```

**Recommendation**: **Option A** (defensive). This file is in `experiments/` (research code), so graceful degradation (empty list = no reclassification) is better than crashing the entire batch run.

---

## 🟡 CLEANUP: Issue #2 - Unused Global Variable

### Location
`src/antibody_training_esm/core/trainer.py:386`

### Finding
```python
# Line 386
results = {}  # ← Global variable never used
```

### Impact
- **Code smell**: Dead code clutters namespace
- **Maintenance**: Confusing to future developers
- **No runtime impact**: Variable exists but is never read

### Fix Plan
**Simply delete the line**:
```python
# Delete line 386 entirely
```

**Verification**: Search codebase for `results` variable to ensure no hidden dependencies:
```bash
grep -rn "results" src/antibody_training_esm/core/trainer.py
```

---

## 🟡 CLEANUP: Issue #1 - Unused Local Variable

### Location
`experiments/novo_parity/scripts/run_exp05_inference.py:73`

### Finding
```python
# Line 73
y_proba = ...  # ← Local variable assigned but never used
```

### Impact
- **Code smell**: Dead code
- **Memory waste**: Holds data unnecessarily
- **No runtime impact**: Variable exists but is never read

### Fix Plan

**Option A: Delete the assignment** (if truly unused)
```python
# Delete line 73 or comment out with explanation
```

**Option B: Use it** (if it was meant to be logged/returned)
```python
# Line 73+
y_proba = classifier.predict_proba(X_test)
print(f"Prediction probabilities: {y_proba[:5]}")  # ← Add usage
```

**Action Required**: Read context around line 73 to determine if `y_proba` was intended for logging, debugging, or should be deleted entirely.

---

## Fix Priority & Timeline

### Immediate (This Sprint)
- [ ] **Issue #3** - Fix uninitialized variable bug (CRITICAL, runtime crash risk)

### Next Sprint (Code Quality)
- [ ] **Issue #2** - Remove unused global variable in trainer.py
- [ ] **Issue #1** - Remove/use unused local variable in run_exp05_inference.py

---

## Testing Plan

### Issue #3 Fix Validation
```bash
# Run batch permutation test with all reclassification strategies
python experiments/novo_parity/scripts/batch_permutation_test.py --reclass-strategy R0
python experiments/novo_parity/scripts/batch_permutation_test.py --reclass-strategy R1
python experiments/novo_parity/scripts/batch_permutation_test.py --reclass-strategy R2
python experiments/novo_parity/scripts/batch_permutation_test.py --reclass-strategy R3

# Ensure no NameError crashes
```

### Issue #2 Fix Validation
```bash
# Run trainer tests
uv run pytest tests/unit/core/test_trainer.py -v

# Run full test suite
uv run pytest tests/unit/ tests/integration/ -v
```

### Issue #1 Fix Validation
```bash
# Run exp05 inference script
python experiments/novo_parity/scripts/run_exp05_inference.py

# Ensure no regressions
```

---

## Approval Required

**Before implementing fixes, require approval from:**
- [ ] Senior engineer review of fix strategies
- [ ] Confirmation that Issue #1 variable (`y_proba`) is truly unused (not debugging artifact)
- [ ] Confirmation that Issue #3 Option A (empty list) is preferred over Option B (assertion)

**Review Questions:**
1. Should we add type hints to prevent future uninitialized variable bugs?
2. Should we add pre-commit hook to catch unused variables automatically?
3. Are there other similar patterns in `experiments/` that need hardening?

---

## References

- **CodeQL Rule Docs**: https://codeql.github.com/codeql-query-help/python/
- **GitHub Security Tab**: https://github.com/The-Obstacle-Is-The-Way/antibody_training_pipeline_ESM/security/code-scanning
- **Related Issues**: None (first CodeQL scan)
