# XGBoost Classifier Integration - Technical Specification

**Status:** Draft
**Version:** 1.0
**Date:** 2025-11-15
**Priority:** Phase 1, Week 1-2 (Roadmap)
**Expected Impact:** +1-3% accuracy improvement over Logistic Regression

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Motivation & Business Value](#motivation--business-value)
3. [Current Architecture Analysis](#current-architecture-analysis)
4. [Design Decisions](#design-decisions)
5. [Implementation Strategy](#implementation-strategy)
6. [Technical Requirements](#technical-requirements)
7. [Risk Assessment](#risk-assessment)
8. [Success Metrics](#success-metrics)
9. [Timeline & Milestones](#timeline--milestones)

---

## Executive Summary

This specification outlines the integration of **XGBoost** as an alternative classifier backend for the antibody polyreactivity prediction pipeline. XGBoost will complement the existing Logistic Regression classifier, enabling:

- **Performance gains:** Expected +1-3% accuracy improvement on benchmark datasets
- **Model diversity:** Completes the 3×3 model grid (3 backbones × 3 classifiers)
- **Foundation for future work:** Enables comprehensive model comparison and HuggingFace Hub publishing

### Key Design Principles

1. **Non-Breaking:** Zero changes to existing `BinaryClassifier` API for LogReg users
2. **SOLID:** Strategy pattern for classifier backend swapping
3. **DRY:** Shared code for embedding extraction, validation, serialization
4. **Testable:** 100% test coverage with TDD approach
5. **Type-Safe:** Full mypy strict mode compliance

---

## Motivation & Business Value

### Why XGBoost?

**Empirical Evidence from Literature:**
- XGBoost typically outperforms Logistic Regression on tabulated embeddings (Chen & Guestrin, 2016)
- Non-linear decision boundaries better capture protein representation patterns
- Handles feature interactions that linear models miss

**Expected Results:**

| Backbone  | Classifier | Jain Acc (Expected) | Improvement |
|-----------|------------|---------------------|-------------|
| ESM-1v    | LogReg     | 66.28% (baseline)   | —           |
| ESM-1v    | XGBoost    | **67-68%**          | +1-2%       |
| ESM2-650M | LogReg     | 62.79% (baseline)   | —           |
| ESM2-650M | XGBoost    | **64-66%**          | +2-3%       |

### Roadmap Alignment

This is **Phase 1, Week 1-2** of the roadmap:
- Week 1-2: XGBoost Classifier ← **WE ARE HERE**
- Week 3-4: AntiBERTa Backbone
- Week 5-6: MLP Classifier + Benchmark Analysis

Completing XGBoost unblocks:
1. Comprehensive 9-model benchmark (3 backbones × 3 classifiers)
2. HuggingFace Hub publishing (Phase 2)
3. Inference API library (Phase 3)
4. Web demo (Phase 4)

---

## Current Architecture Analysis

### Existing Classifier Implementation

**File:** `src/antibody_training_esm/core/classifier.py`

```python
class BinaryClassifier:
    """Binary classifier for protein sequences using ESM-1V embeddings"""

    def __init__(self, params: dict[str, Any] | None = None, **kwargs: Any):
        # Initialize ESM embedding extractor
        self.embedding_extractor = ESMEmbeddingExtractor(...)

        # Hardcoded LogisticRegression (PROBLEM!)
        self.classifier = LogisticRegression(
            C=C, penalty=penalty, solver=solver, ...
        )
```

**Current Limitations:**
1. ❌ Hardcoded `LogisticRegression` - no abstraction for other classifiers
2. ❌ No strategy pattern for swapping classifier backends
3. ❌ Hyperparameters mixed between embedding extractor and classifier
4. ❌ Serialization logic assumes LogReg attributes (`coef_`, `intercept_`)

**What Works Well:**
1. ✅ sklearn API compatibility (`fit()`, `predict()`, `get_params()`, `set_params()`)
2. ✅ Embedding extraction separated from classification
3. ✅ Dual serialization (Pickle + NPZ+JSON)
4. ✅ Assay-specific thresholds (ELISA vs PSR)

### Hydra Configuration Structure

**File:** `src/antibody_training_esm/conf/config.yaml`

```yaml
defaults:
  - model: esm1v          # Backbone: esm1v, esm2_650m
  - classifier: logreg    # ← Need to add: xgboost
  - data: boughter_jain
```

**Classifier Config:** `conf/classifier/logreg.yaml`

```yaml
type: logistic_regression  # Used for factory pattern
C: 1.0
penalty: l2
solver: lbfgs
max_iter: 1000
random_state: ${training.random_state}
class_weight: null
cv_folds: 10
stratify: true
```

**Observation:** The `type` field is perfect for factory pattern implementation!

---

## Design Decisions

### Decision 1: Refactor `BinaryClassifier` to Support Multiple Backends

**Problem:** Current implementation hardcodes `LogisticRegression`.

**Solution:** Use **Strategy Pattern** with classifier factory.

```python
# NEW: Abstract base class for classifier strategies
from abc import ABC, abstractmethod

class ClassifierStrategy(ABC):
    """Abstract base class for classifier backends"""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for NPZ+JSON format"""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, config: dict[str, Any]) -> "ClassifierStrategy":
        """Deserialize from dict"""
        pass

# Concrete implementations
class LogisticRegressionStrategy(ClassifierStrategy):
    """Wrapper for sklearn LogisticRegression"""
    ...

class XGBoostStrategy(ClassifierStrategy):
    """Wrapper for XGBoost XGBClassifier"""
    ...

# Factory function
def create_classifier(config: dict[str, Any]) -> ClassifierStrategy:
    """Factory for creating classifier strategies"""
    classifier_type = config.get("type", "logistic_regression")

    if classifier_type == "logistic_regression":
        return LogisticRegressionStrategy(config)
    elif classifier_type == "xgboost":
        return XGBoostStrategy(config)
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")
```

**Updated `BinaryClassifier`:**

```python
class BinaryClassifier:
    def __init__(self, params: dict[str, Any] | None = None, **kwargs: Any):
        # Initialize ESM embedding extractor (unchanged)
        self.embedding_extractor = ESMEmbeddingExtractor(...)

        # NEW: Use factory to create classifier strategy
        self.classifier = create_classifier(params)
```

**Benefits:**
- ✅ Open/Closed Principle: Open for extension (new classifiers), closed for modification
- ✅ Non-breaking: Existing LogReg configs work unchanged
- ✅ Testable: Easy to mock classifier strategies
- ✅ Type-safe: Abstract base class enforces interface contracts

### Decision 2: XGBoost Hyperparameter Defaults

**Problem:** XGBoost has 100+ hyperparameters. Which ones should we expose?

**Solution:** Start with **minimal viable config**, expand later based on experimentation.

**Initial Config** (`conf/classifier/xgboost.yaml`):

```yaml
type: xgboost

# Core hyperparameters (sklearn-compatible XGBClassifier)
n_estimators: 100         # Number of boosting rounds (LogReg has no equivalent)
max_depth: 6              # Maximum tree depth (controls overfitting)
learning_rate: 0.3        # Shrinkage (eta)
subsample: 1.0            # Row subsampling (no bagging by default)
colsample_bytree: 1.0     # Column subsampling
reg_alpha: 0.0            # L1 regularization (similar to LogReg penalty=l1)
reg_lambda: 1.0           # L2 regularization (similar to LogReg penalty=l2)

# sklearn compatibility
random_state: ${training.random_state}
objective: binary:logistic  # Binary classification
eval_metric: logloss        # Loss function (matches LogReg)
use_label_encoder: false    # Deprecated in xgboost>=1.6.0
enable_categorical: false   # We use embeddings (continuous), not categorical

# Cross-validation (inherited from parent config)
cv_folds: 10
stratify: true
```

**Rationale:**
- `n_estimators=100`: Standard default, balance speed vs performance
- `max_depth=6`: XGBoost default, prevents overfitting on small datasets (Boughter has 914 samples)
- `learning_rate=0.3`: XGBoost default
- `reg_lambda=1.0`: Equivalent to LogReg's L2 penalty (C=1.0)

**Future Hyperparameter Tuning:**
- Phase 2: Grid search over `n_estimators`, `max_depth`, `learning_rate`
- Phase 3: Advanced parameters (`gamma`, `min_child_weight`, `scale_pos_weight`)

### Decision 3: Serialization Strategy

**Problem:** XGBoost models have different internal state than LogisticRegression.

**Current LogReg Serialization:**
```python
# NPZ format
np.savez(
    path,
    coef=classifier.coef_,           # (n_features,) array
    intercept=classifier.intercept_,  # scalar
    classes=classifier.classes_,      # [0, 1]
    ...
)
```

**XGBoost Internal State:**
- Trees (JSON format via `get_booster().save_raw()`)
- Feature importances
- Best iteration (if early stopping)

**Solution:** Use XGBoost's native serialization + metadata JSON.

```python
# XGBoost serialization (dual format)

# 1. Pickle (research/debugging) - unchanged
pickle.dump(classifier, file)

# 2. Production format (NPZ + JSON)
# - *.xgb: XGBoost binary format (ubj or json)
# - *_config.json: Metadata (hyperparameters, sklearn version, etc.)

# Save XGBoost model
classifier.save_model(f"{base_path}.xgb")  # Native XGBoost format

# Save metadata JSON
metadata = {
    "model_type": "xgboost",
    "xgboost_version": xgboost.__version__,
    "n_estimators": classifier.n_estimators,
    "max_depth": classifier.max_depth,
    ...
}
json.dump(metadata, f)
```

**Benefits:**
- ✅ Production-ready: No pickle dependency for deployment
- ✅ Cross-language: `.xgb` format can be loaded in R, Java, etc.
- ✅ HuggingFace compatible: Standard format for model hub
- ✅ Efficient: Binary format is compact and fast to load

### Decision 4: Backward Compatibility

**Requirement:** Existing code using `BinaryClassifier` with LogReg must continue to work.

**Strategy:** Default to LogReg if `type` is not specified.

```python
def create_classifier(config: dict[str, Any]) -> ClassifierStrategy:
    # Default to logistic_regression for backward compatibility
    classifier_type = config.get("type", "logistic_regression")
    ...
```

**Test Coverage:**
```python
@pytest.mark.unit
def test_backward_compatibility_no_type_field():
    """Verify BinaryClassifier defaults to LogReg if type not specified"""
    params = {
        "model_name": "facebook/esm1v_t33_650M_UR90S_1",
        "device": "cpu",
        "random_state": 42,
        "max_iter": 1000,
        # No "type" field - should default to logistic_regression
    }

    classifier = BinaryClassifier(params)

    # Verify it's using LogisticRegression
    assert isinstance(classifier.classifier, LogisticRegressionStrategy)
```

---

## Implementation Strategy

### Phase 1: Refactoring (Non-Breaking)

**Goal:** Extract classifier abstraction without changing behavior.

**Tasks:**
1. Create `ClassifierStrategy` abstract base class
2. Implement `LogisticRegressionStrategy` wrapper (existing behavior)
3. Update `BinaryClassifier` to use strategy pattern
4. Add factory function `create_classifier()`
5. **Verify all existing tests pass** (no behavior changes)

**Success Criteria:**
- ✅ All existing tests pass unchanged
- ✅ `BinaryClassifier` API unchanged
- ✅ Type safety maintained (mypy strict mode)
- ✅ No performance regression

### Phase 2: XGBoost Implementation (New Feature)

**Goal:** Add XGBoost as alternative classifier backend.

**Tasks:**
1. Add `xgboost>=2.0.0` to `pyproject.toml`
2. Implement `XGBoostStrategy` class
3. Create `conf/classifier/xgboost.yaml`
4. Write unit tests for `XGBoostStrategy`
5. Write integration tests (embedding → XGBoost → predictions)
6. Update serialization logic for XGBoost models

**Success Criteria:**
- ✅ Unit tests pass (100% coverage for XGBoostStrategy)
- ✅ Integration tests pass
- ✅ Can train model: `antibody-train classifier=xgboost`
- ✅ Model serialization works (both pickle and .xgb+JSON)

### Phase 3: Benchmarking & Validation

**Goal:** Train models and compare performance vs LogReg.

**Tasks:**
1. Train ESM-1v + XGBoost on Boughter
2. Train ESM2-650M + XGBoost on Boughter
3. Test on Jain/Harvey/Shehata datasets
4. Generate comparison tables (XGBoost vs LogReg)
5. Update `docs/research/benchmark-results.md`

**Success Criteria:**
- ✅ XGBoost achieves ≥65% accuracy on Jain (target: 67-68%)
- ✅ XGBoost outperforms LogReg on at least 2/3 test datasets
- ✅ Models saved in hierarchical directory structure
- ✅ Results documented with statistical significance tests

---

## Technical Requirements

### Dependencies

**Add to `pyproject.toml`:**

```toml
[project]
dependencies = [
    # ... existing dependencies ...
    "xgboost>=2.0.0",  # NEW: XGBoost for gradient boosting classifier
]
```

**Version Justification:**
- `xgboost>=2.0.0`: Latest stable, sklearn API improvements, better GPU support
- Tested compatibility with Python 3.12, sklearn 1.3+, numpy 1.24+

### Configuration Schema

**New File:** `src/antibody_training_esm/conf/classifier/xgboost.yaml`

```yaml
# XGBoost Classifier Configuration
# See: https://xgboost.readthedocs.io/en/stable/parameter.html

type: xgboost

# Core hyperparameters
n_estimators: 100
max_depth: 6
learning_rate: 0.3
subsample: 1.0
colsample_bytree: 1.0
reg_alpha: 0.0
reg_lambda: 1.0

# sklearn compatibility
random_state: ${training.random_state}
objective: binary:logistic
eval_metric: logloss
use_label_encoder: false
enable_categorical: false

# Cross-validation (inherited)
cv_folds: 10
stratify: true
```

### File Structure Changes

**New Files:**
```
src/antibody_training_esm/core/
├── classifier.py                      # Modified: Use strategy pattern
├── classifier_strategy.py             # NEW: Abstract base class
├── strategies/                        # NEW: Strategy implementations
│   ├── __init__.py
│   ├── logistic_regression.py        # NEW: LogReg strategy
│   └── xgboost_strategy.py            # NEW: XGBoost strategy

src/antibody_training_esm/conf/
└── classifier/
    ├── logreg.yaml                    # Existing
    └── xgboost.yaml                   # NEW

tests/unit/core/strategies/            # NEW: Strategy tests
├── __init__.py
├── test_logistic_regression.py
└── test_xgboost_strategy.py

tests/integration/
└── test_xgboost_training.py           # NEW

docs/developer-guide/
├── xgboost-integration-spec.md        # This file
├── xgboost-api-design.md              # Next doc
└── xgboost-test-plan.md               # Next doc
```

### Type Annotations

**Strict mypy compliance required:**

```python
from abc import ABC, abstractmethod
from typing import Any, Protocol

class ClassifierStrategy(Protocol):
    """Protocol for classifier strategies (duck typing for sklearn compat)"""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ...
```

---

## Risk Assessment

### Technical Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| **Breaking existing LogReg users** | High | Low | Comprehensive backward compatibility tests |
| **XGBoost overfitting on small datasets** | Medium | Medium | Start with conservative hyperparameters (max_depth=6) |
| **Serialization format incompatibility** | Medium | Low | Use XGBoost native format (.xgb), well-tested |
| **Type safety violations** | Low | Low | Run mypy in strict mode throughout |
| **Performance regression** | Low | Low | Benchmark embeddings extraction time (unchanged) |

### Dependency Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| **XGBoost version conflicts** | Medium | Low | Pin to `xgboost>=2.0.0,<3.0.0` |
| **GPU driver compatibility** | Low | Medium | CPU fallback, document GPU requirements |
| **Increased package size** | Low | High | Accept trade-off (XGBoost adds ~50MB) |

### Schedule Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| **Refactoring takes longer than expected** | Medium | Medium | Phase 1 is non-breaking, can ship incrementally |
| **Hyperparameter tuning delays** | Low | High | Ship with defaults first, tune later |
| **Integration test failures** | Medium | Low | TDD approach catches issues early |

---

## Success Metrics

### Phase 1 (Refactoring)

- ✅ All 150+ existing tests pass unchanged
- ✅ Zero mypy errors (strict mode)
- ✅ Zero ruff errors
- ✅ No performance regression (benchmark embedding extraction)

### Phase 2 (XGBoost Implementation)

- ✅ 100% test coverage for new code
- ✅ Can train model via CLI: `antibody-train classifier=xgboost`
- ✅ Model loads from both pickle and .xgb+JSON formats
- ✅ Type annotations complete (mypy strict mode)

### Phase 3 (Benchmarking)

**Target Metrics:**

| Backbone  | Classifier | Jain Acc | Jain F1 | Jain ROC-AUC | Hypothesis |
|-----------|------------|----------|---------|--------------|------------|
| ESM-1v    | LogReg     | 66.28%   | 0.63    | 0.71         | Baseline   |
| ESM-1v    | XGBoost    | **≥67%** | **≥0.65** | **≥0.72** | +1-2% gain |
| ESM2-650M | LogReg     | 62.79%   | 0.60    | 0.67         | Baseline   |
| ESM2-650M | XGBoost    | **≥64%** | **≥0.62** | **≥0.69** | +2-3% gain |

**Success Criteria:**
- ✅ XGBoost outperforms LogReg on ≥2 of 3 test datasets (Jain, Harvey, Shehata)
- ✅ Improvement is statistically significant (p < 0.05, McNemar's test)
- ✅ Training time ≤2× LogReg training time
- ✅ Model size ≤100MB per model

---

## Timeline & Milestones

### Week 1: Refactoring & XGBoost Implementation

**Days 1-2: Refactoring (Non-Breaking)**
- Create `ClassifierStrategy` abstraction
- Implement `LogisticRegressionStrategy`
- Update `BinaryClassifier` to use strategy pattern
- Verify all existing tests pass
- **Deliverable:** PR #1 - "Refactor: Extract classifier strategy pattern"

**Days 3-4: XGBoost Implementation**
- Add `xgboost` dependency
- Implement `XGBoostStrategy`
- Create `conf/classifier/xgboost.yaml`
- Write unit tests (TDD approach)
- **Deliverable:** PR #2 - "feat: Add XGBoost classifier support"

**Days 5-7: Integration & Testing**
- Write integration tests
- Update serialization logic
- Test CLI: `antibody-train classifier=xgboost`
- Fix any issues found
- **Deliverable:** PR #3 - "test: Add XGBoost integration tests"

### Week 2: Benchmarking & Documentation

**Days 8-10: Training & Benchmarking**
- Train ESM-1v + XGBoost on Boughter
- Train ESM2-650M + XGBoost on Boughter
- Test on Jain/Harvey/Shehata
- Generate performance comparison tables
- **Deliverable:** Trained models in `models/esm1v/xgboost/`, `models/esm2_650m/xgboost/`

**Days 11-12: Analysis & Documentation**
- Statistical significance testing (McNemar's test)
- Update `docs/research/benchmark-results.md`
- Write blog post draft: "XGBoost vs Logistic Regression for Antibody Polyreactivity"
- **Deliverable:** PR #4 - "docs: Add XGBoost benchmark results"

**Days 13-14: Buffer & Final Review**
- Address PR feedback
- Final testing (e2e tests)
- Merge to `dev` branch
- **Deliverable:** XGBoost integration complete ✅

---

## Next Steps

After reading this spec, proceed to:

1. **API Design Document** (`xgboost-api-design.md`)
   - Detailed class diagrams
   - API contracts for `ClassifierStrategy`
   - Method signatures with type annotations

2. **Test Plan** (`xgboost-test-plan.md`)
   - Unit test specifications
   - Integration test scenarios
   - Performance benchmarks
   - Coverage targets

3. **Implementation** (TDD approach)
   - Phase 1: Refactoring
   - Phase 2: XGBoost implementation
   - Phase 3: Benchmarking

---

## References

- XGBoost Documentation: https://xgboost.readthedocs.io/
- sklearn API Reference: https://scikit-learn.org/stable/developers/develop.html
- Hydra Documentation: https://hydra.cc/docs/intro/
- Project Roadmap: `/home/user/antibody_training_pipeline_ESM/ROADMAP.md`
- Current Architecture: `docs/developer-guide/architecture.md`

---

**Document Status:** Draft → Ready for Review
**Next Action:** Write API Design Document
**Assigned To:** Claude (Autonomous Implementation)
**Review Required:** Yes (before implementation begins)
