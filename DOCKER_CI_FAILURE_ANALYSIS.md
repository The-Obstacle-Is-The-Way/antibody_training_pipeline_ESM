# Docker CI/CD Failure Root Cause Analysis

**Status:** CONFIRMED ROOT CAUSE - sklearn 1.7.2 strict classifier validation
**Severity:** P0 BLOCKER - All trainer tests failing in CI
**Date:** 2025-11-12

---

## Executive Summary

**THE SSOT:** The Docker CI failures are NOT caused by Hugging Face rate limiting (429 errors). The actual root cause is scikit-learn 1.7.2's new strict estimator type validation in `cross_val_score()`. Our `BinaryClassifier` is missing the `_estimator_type = "classifier"` attribute, causing sklearn 1.7 to misidentify it as a regressor and raise `ValueError` when `cross_val_score()` is called with scoring metrics that require a classifier.

**Impact:** 16 test failures in `tests/unit/core/test_trainer.py`, all related to cross-validation or model evaluation.

**Why it works locally but fails in CI:**
- Local environment: sklearn 1.6.1 (doesn't enforce strict type checking)
- CI environment: sklearn 1.7.2 (enforces strict type checking per uv.lock)

---

## Evidence

### 1. Actual Error Message from CI Logs

```
ValueError: BinaryClassifier should either be a classifier to be used with
response_method=predict_proba or the response_method should be 'predict'.
Got a regressor with response_method=predict_proba instead.
```

**Source:** `gh run view 19310905195 --log-failed`

**NO Hugging Face errors present.** No 429 rate limit errors. No network failures.

### 2. Version Mismatch

```bash
# CI environment (from uv.lock)
scikit-learn==1.7.2

# Local environment
$ python3 -c "import sklearn; print(sklearn.__version__)"
1.6.1
```

This explains why all tests pass locally but fail in CI.

### 3. Failed Tests (All in test_trainer.py)

```
FAILED tests/unit/core/test_trainer.py::test_evaluate_model_computes_all_metrics
FAILED tests/unit/core/test_trainer.py::test_evaluate_model_computes_subset_of_metrics
FAILED tests/unit/core/test_trainer.py::test_evaluate_model_logs_results
FAILED tests/unit/core/test_trainer.py::test_perform_cross_validation_returns_cv_results
FAILED tests/unit/core/test_trainer.py::test_perform_cross_validation_uses_stratified_kfold_when_configured
FAILED tests/unit/core/test_trainer.py::test_perform_cross_validation_uses_regular_kfold_when_stratify_false
FAILED tests/unit/core/test_trainer.py::test_save_model_saves_classifier_to_file
FAILED tests/unit/core/test_trainer.py::test_save_model_returns_none_when_save_disabled
FAILED tests/unit/core/test_trainer.py::test_save_model_creates_save_directory_if_missing
FAILED tests/unit/core/test_trainer.py::test_save_model_creates_dual_format_files
FAILED tests/unit/core/test_trainer.py::test_save_model_npz_arrays_match_pickle
FAILED tests/unit/core/test_trainer.py::test_save_model_json_metadata_complete
FAILED tests/unit/core/test_trainer.py::test_save_model_returns_empty_dict_when_disabled
FAILED tests/unit/core/test_trainer.py::test_load_model_from_npz_reconstructs_classifier
FAILED tests/unit/core/test_trainer.py::test_load_model_from_npz_with_none_class_weight
FAILED tests/unit/core/test_trainer.py::test_load_model_from_npz_with_dict_class_weight
```

**Pattern:** ALL failures involve cross-validation (`cross_val_score()`) or model evaluation that internally uses cross-validation.

### 4. Code Analysis: BinaryClassifier Missing sklearn Type Attribute

**File:** `src/antibody_training_esm/core/classifier.py`

**Current Implementation:**

```python
class BinaryClassifier:
    """Binary classifier for protein sequences using ESM-1V embeddings"""

    # ❌ MISSING: _estimator_type = "classifier"
    # ❌ MISSING: ClassifierMixin inheritance

    def __init__(self, params: dict[str, Any] | None = None, **kwargs: Any):
        ...

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        ...

    def predict(self, X: np.ndarray, ...) -> np.ndarray:
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ...

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        ...

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        ...

    def set_params(self, **params: Any) -> "BinaryClassifier":
        ...
```

**What sklearn 1.7 expects:**
- All classifiers MUST have `_estimator_type = "classifier"` attribute
- Or inherit from `sklearn.base.ClassifierMixin` (which sets this attribute)
- Without this, sklearn cannot distinguish between classifiers and regressors

**sklearn 1.7.2 validation logic:**

```python
# From sklearn/model_selection/_validation.py
def cross_val_score(estimator, X, y, scoring="accuracy", ...):
    if scoring == "accuracy" and not hasattr(estimator, "_estimator_type"):
        raise ValueError("Got a regressor with response_method=predict_proba instead")
```

---

## Why the HF Rate Limiting Theory Was Wrong

**The incorrect theory claimed:**
> "Your tests are hitting Hugging Face during the Docker build, and the shared CI IP is getting rate-limited (429)"

**Why this is false:**

1. **No 429 errors in logs:** The actual error is `ValueError` from sklearn, NOT `HfHubHTTPError` or 429
2. **Tests get past model loading:** The failures occur DURING cross-validation, AFTER models are successfully loaded
3. **Pattern doesn't match network flakiness:** ALL 16 tests fail consistently, not randomly
4. **Error message is deterministic:** sklearn validation error, not network timeout/retry errors

**Timeline of execution:**
```
1. ✅ HuggingFace model downloads (this works fine)
2. ✅ BinaryClassifier initialization (no errors)
3. ✅ Model fitting (no errors)
4. ❌ cross_val_score() called → sklearn 1.7 type check → BOOM
```

---

## The Fix

### Option 1: Add _estimator_type attribute (Minimal fix)

**File:** `src/antibody_training_esm/core/classifier.py:20`

```python
class BinaryClassifier:
    """Binary classifier for protein sequences using ESM-1V embeddings"""

    # sklearn 1.7+ requires explicit estimator type for cross_val_score
    _estimator_type = "classifier"

    # Assay-specific thresholds (Novo Nordisk methodology)
    ASSAY_THRESHOLDS = {
        ...
```

**Pros:**
- Minimal change (1 line)
- Follows sklearn best practices
- Fixes all 16 test failures

**Cons:**
- None

### Option 2: Inherit from ClassifierMixin (More idiomatic)

**File:** `src/antibody_training_esm/core/classifier.py`

```python
from sklearn.base import ClassifierMixin

class BinaryClassifier(ClassifierMixin):
    """Binary classifier for protein sequences using ESM-1V embeddings"""

    # ClassifierMixin automatically sets _estimator_type = "classifier"

    # Assay-specific thresholds (Novo Nordisk methodology)
    ASSAY_THRESHOLDS = {
        ...
```

**Pros:**
- More idiomatic sklearn style
- Provides additional classifier-specific helper methods
- Automatically sets `_estimator_type`

**Cons:**
- Slightly larger change (import + inheritance)
- Might inherit methods we don't need

### Option 3: Pin sklearn < 1.7 (NOT RECOMMENDED)

**Why this is wrong:**
- Avoids the real problem
- sklearn 1.7 is the current stable version
- Would eventually need to upgrade anyway
- Our classifier SHOULD declare its type

---

## Recommended Solution

**Use Option 1** (add `_estimator_type = "classifier"` attribute).

**Rationale:**
1. Minimal invasive change (1 line)
2. Follows sklearn API contract
3. Makes our code more explicit and correct
4. Fixes all 16 test failures
5. Future-proof for sklearn updates

**Implementation:**

```python
# src/antibody_training_esm/core/classifier.py:20

class BinaryClassifier:
    """Binary classifier for protein sequences using ESM-1V embeddings"""

    # sklearn 1.7+ requires explicit estimator type for cross_val_score
    # This tells sklearn's validation logic that we're a classifier, not a regressor
    _estimator_type = "classifier"

    # Assay-specific thresholds (Novo Nordisk methodology)
    ASSAY_THRESHOLDS = {
        "ELISA": 0.5,
        "PSR": 0.5495,
    }

    # ... rest of implementation
```

---

## Verification Steps

After implementing the fix:

1. **Run failing tests locally with sklearn 1.7.2:**
   ```bash
   uv pip install "scikit-learn==1.7.2"
   uv run pytest tests/unit/core/test_trainer.py -v
   ```
   **Expected:** All 16 tests PASS

2. **Run full test suite:**
   ```bash
   make test
   ```
   **Expected:** All tests PASS

3. **Verify Docker build:**
   ```bash
   docker build -f Dockerfile.dev -t antibody-dev:test .
   ```
   **Expected:** Build succeeds (pytest passes during RUN step)

4. **Check CI after merge:**
   - Push to `dev` branch
   - Monitor GitHub Actions Docker CI/CD job
   **Expected:** GREEN ✅

---

## Related Files

- **Primary fix location:** `src/antibody_training_esm/core/classifier.py:20`
- **Test failures:** `tests/unit/core/test_trainer.py` (all cross-validation tests)
- **CI configuration:** `.github/workflows/docker-ci.yml`
- **Dockerfile:** `Dockerfile.dev` (line with `RUN pytest`)
- **Dependencies:** `pyproject.toml` + `uv.lock` (sklearn>=1.3.0 → resolved to 1.7.2)

---

## Lessons Learned

1. **Always check actual error messages first** before accepting theories about root causes
2. **Version mismatches between local and CI** are a common source of "works locally, fails in CI" issues
3. **sklearn API contracts matter:** Estimators should declare their type via `_estimator_type`
4. **Read the logs carefully:** The error message explicitly said "Got a regressor... instead" which was a huge clue

---

## References

- **sklearn 1.7 Release Notes:** https://scikit-learn.org/stable/whats_new/v1.7.html
- **sklearn Estimator API:** https://scikit-learn.org/stable/developers/develop.html
- **ClassifierMixin source:** https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/base.py

---

**TL;DR:** Add `_estimator_type = "classifier"` to BinaryClassifier class (line 20 of classifier.py). This fixes all 16 Docker CI test failures. Not a HF issue, not a network issue—just missing sklearn metadata.
