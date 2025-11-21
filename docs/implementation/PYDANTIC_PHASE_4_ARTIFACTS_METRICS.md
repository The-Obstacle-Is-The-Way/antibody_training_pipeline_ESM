# Pydantic Phase 4: Artifacts & Metrics

**Status:** Completed
**Priority:** MEDIUM (Robustness)
**Risk:** LOW (Isolated to serialization)
**Dependencies:** Phase 1 (Pydantic installed)

---

## Overview

Replace manual dictionary construction/parsing in `serialization.py` and `metrics.py` with Pydantic models. This ensures model artifacts are self-describing, version-compatible, and load reliably across environments.

**Key Benefits:**
- **Self-Describing Models:** JSON sidecar contains full metadata
- **Type Safety:** Serialization/deserialization is type-checked
- **Version Compatibility:** Detect incompatible models at load time
- **No Manual Type Casting:** Pydantic handles complex types (e.g., `class_weight` with int keys)

---

## Dependencies

**Already installed from Phase 1:**
```toml
[project.optional-dependencies]
validation = [
    "pydantic>=2.10.0",
]
```

---

## Implementation Scope

### Files to Modify

1. **Create:** `src/antibody_training_esm/models/artifact.py`
   - `ModelArtifactMetadata` (model metadata)
   - `EvaluationMetrics` (training/test metrics)
   - `CVResults` (cross-validation results)

2. **Modify:** `src/antibody_training_esm/core/training/serialization.py`
   - Use `ModelArtifactMetadata` in `save_model()`
   - Use `ModelArtifactMetadata` in `load_model_from_npz()`
   - Remove manual JSON type casting

3. **Modify:** `src/antibody_training_esm/core/training/metrics.py`
   - Use `EvaluationMetrics` in `evaluate_model()`
   - Use `CVResults` in `save_cv_results()`
   - Replace dict construction with `.model_dump()`

---

## Model Specifications

### 1. `ModelArtifactMetadata` (Saved Model Config)

**Location:** `src/antibody_training_esm/models/artifact.py`

```python
from pydantic import BaseModel, Field
from typing import Literal, Any

class ModelArtifactMetadata(BaseModel):
    """
    Metadata for saved model artifacts.

    This model structures the JSON sidecar file that accompanies
    NPZ/XGB model files. It enables version checking and parameter
    reconstruction.
    """
    # Model architecture
    model_name: str = Field(
        ...,
        description="HuggingFace ESM model ID",
        examples=["facebook/esm1v_t33_650M_UR90S_1"],
    )

    model_type: Literal["logistic_regression", "xgboost", "random_forest"] = Field(
        ...,
        description="Classifier type",
    )

    sklearn_version: str = Field(
        ...,
        description="scikit-learn version used for training",
        examples=["1.3.0"],
    )

    # Classifier configuration (strategy-specific)
    classifier: dict[str, Any] = Field(
        ...,
        description="Full classifier config from to_dict() method",
    )

    # ESM embedding extractor params
    esm_model: str = Field(
        ...,
        description="ESM model name (redundant with model_name, kept for compat)",
    )

    esm_revision: str = Field(
        default="main",
        description="HuggingFace model revision (commit hash)",
    )

    batch_size: int = Field(
        default=16,
        ge=1,
        description="Batch size for embedding extraction",
    )

    device: str = Field(
        default="cpu",
        description="Device used during training",
    )

    # Legacy flat fields (LogReg only, for backward compatibility)
    C: float | None = Field(
        default=None,
        description="LogReg: Inverse regularization strength",
    )

    penalty: Literal["l1", "l2"] | None = Field(
        default=None,
        description="LogReg: Regularization type",
    )

    solver: str | None = Field(
        default=None,
        description="LogReg: Optimization algorithm",
    )

    class_weight: Literal["balanced"] | dict[int, float] | None = Field(
        default=None,
        description="Class weighting strategy",
    )

    max_iter: int | None = Field(
        default=None,
        description="LogReg: Maximum iterations",
    )

    random_state: int | None = Field(
        default=None,
        description="Random seed",
    )

    # Optional metrics from training
    training_metrics: dict[str, float] | None = Field(
        default=None,
        description="Metrics from final training run",
    )

    @classmethod
    def from_classifier(cls, classifier: Any) -> "ModelArtifactMetadata":
        """
        Construct metadata from BinaryClassifier instance.

        Args:
            classifier: Trained BinaryClassifier

        Returns:
            ModelArtifactMetadata
        """
        import sklearn

        strategy_config = classifier.classifier.to_dict()
        classifier_type = strategy_config.get("type", "logistic_regression")

        metadata_dict = {
            # Model architecture
            "model_name": classifier.model_name,
            "model_type": classifier_type,
            "sklearn_version": sklearn.__version__,
            # Classifier config (strategy-specific)
            "classifier": strategy_config,
            # ESM params
            "esm_model": classifier.model_name,
            "esm_revision": classifier.revision,
            "batch_size": classifier.batch_size,
            "device": classifier.device,
        }

        # Add legacy flat fields for LogReg (backward compat)
        if classifier_type == "logistic_regression":
            metadata_dict.update({
                "C": classifier.C,
                "penalty": classifier.penalty,
                "solver": classifier.solver,
                "class_weight": classifier.class_weight,
                "max_iter": classifier.max_iter,
                "random_state": classifier.random_state,
            })

        return cls.model_validate(metadata_dict)

    def to_classifier_params(self) -> dict[str, Any]:
        """
        Extract parameters for BinaryClassifier reconstruction.

        Returns:
            Dict of parameters for BinaryClassifier(...) init
        """
        return {
            # ESM params
            "model_name": self.esm_model,
            "device": self.device,
            "batch_size": self.batch_size,
            "revision": self.esm_revision,
            # Classifier params (merge legacy + modern)
            **self.classifier,
        }
```

### 2. `EvaluationMetrics` (Training/Test Metrics)

```python
class EvaluationMetrics(BaseModel):
    """
    Evaluation metrics for a single dataset.

    Used for training set, test set, and cross-validation fold results.
    """
    accuracy: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Classification accuracy (0-1)",
    )

    precision: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Precision (positive predictive value)",
    )

    recall: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Recall (sensitivity, true positive rate)",
    )

    f1: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="F1 score (harmonic mean of precision and recall)",
    )

    roc_auc: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Area under ROC curve",
    )

    # Optional confusion matrix
    confusion_matrix: list[list[int]] | None = Field(
        default=None,
        description="Confusion matrix [[TN, FP], [FN, TP]]",
    )

    # Dataset metadata
    dataset_name: str | None = Field(
        default=None,
        description="Name of evaluated dataset (e.g., 'Jain', 'Training')",
    )

    n_samples: int | None = Field(
        default=None,
        ge=0,
        description="Number of samples in dataset",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "accuracy": 0.6628,
                    "precision": 0.47,
                    "recall": 0.63,
                    "f1": 0.54,
                    "roc_auc": 0.68,
                    "confusion_matrix": [[40, 19], [10, 17]],
                    "dataset_name": "Jain",
                    "n_samples": 86,
                }
            ]
        }
    }

    @classmethod
    def from_sklearn_metrics(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None = None,
        dataset_name: str | None = None,
    ) -> "EvaluationMetrics":
        """
        Construct metrics from sklearn predictions.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (for ROC-AUC)
            dataset_name: Name of dataset

        Returns:
            EvaluationMetrics
        """
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
            confusion_matrix,
        )

        metrics_dict = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "dataset_name": dataset_name,
            "n_samples": len(y_true),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }

        # ROC-AUC requires probabilities
        if y_proba is not None:
            metrics_dict["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))

        return cls.model_validate(metrics_dict)
```

### 3. `CVResults` (Cross-Validation Results)

```python
class CVResults(BaseModel):
    """
    Cross-validation results with mean and std for each metric.

    Aggregates metrics across all CV folds.
    """
    cv_accuracy: dict[Literal["mean", "std"], float] = Field(
        ...,
        description="Mean and std of accuracy across folds",
    )

    cv_precision: dict[Literal["mean", "std"], float] | None = Field(
        default=None,
        description="Mean and std of precision",
    )

    cv_recall: dict[Literal["mean", "std"], float] | None = Field(
        default=None,
        description="Mean and std of recall",
    )

    cv_f1: dict[Literal["mean", "std"], float] | None = Field(
        default=None,
        description="Mean and std of F1 score",
    )

    cv_roc_auc: dict[Literal["mean", "std"], float] | None = Field(
        default=None,
        description="Mean and std of ROC-AUC",
    )

    n_splits: int = Field(
        ...,
        ge=2,
        description="Number of cross-validation folds",
    )

    # Optional: per-fold results
    fold_results: list[EvaluationMetrics] | None = Field(
        default=None,
        description="Metrics for each individual fold",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "cv_accuracy": {"mean": 0.82, "std": 0.05},
                    "cv_precision": {"mean": 0.78, "std": 0.06},
                    "cv_recall": {"mean": 0.85, "std": 0.04},
                    "cv_f1": {"mean": 0.81, "std": 0.05},
                    "cv_roc_auc": {"mean": 0.87, "std": 0.03},
                    "n_splits": 10,
                }
            ]
        }
    }

    @classmethod
    def from_sklearn_cv_results(
        cls,
        cv_scores: dict[str, list[float]],
        n_splits: int,
    ) -> "CVResults":
        """
        Construct CVResults from sklearn cross_validate output.

        Args:
            cv_scores: Dict like {"test_accuracy": [...], "test_f1": [...]}
            n_splits: Number of folds

        Returns:
            CVResults
        """
        import numpy as np

        results_dict = {"n_splits": n_splits}

        # Map sklearn metric names to our field names
        metric_map = {
            "test_accuracy": "cv_accuracy",
            "test_precision": "cv_precision",
            "test_recall": "cv_recall",
            "test_f1": "cv_f1",
            "test_roc_auc": "cv_roc_auc",
        }

        for sklearn_name, pydantic_name in metric_map.items():
            if sklearn_name in cv_scores:
                scores = cv_scores[sklearn_name]
                results_dict[pydantic_name] = {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                }

        return cls.model_validate(results_dict)
```

---

## Integration Steps (TDD)

### Step 1: Write Tests FIRST

**Create:** `tests/unit/models/test_artifact.py`

```python
"""Unit tests for artifact models."""

import pytest
import numpy as np
from pydantic import ValidationError

from antibody_training_esm.models.artifact import (
    ModelArtifactMetadata,
    EvaluationMetrics,
    CVResults,
)


class TestModelArtifactMetadata:
    """Test ModelArtifactMetadata validation."""

    def test_valid_metadata(self):
        """Valid metadata constructs correctly."""
        metadata = ModelArtifactMetadata(
            model_name="facebook/esm1v_t33_650M_UR90S_1",
            model_type="logistic_regression",
            sklearn_version="1.3.0",
            classifier={"type": "logistic_regression", "C": 1.0},
            esm_model="facebook/esm1v_t33_650M_UR90S_1",
        )
        assert metadata.model_type == "logistic_regression"

    def test_class_weight_with_int_keys(self):
        """class_weight dict with int keys is preserved."""
        metadata = ModelArtifactMetadata(
            model_name="facebook/esm1v_t33_650M_UR90S_1",
            model_type="logistic_regression",
            sklearn_version="1.3.0",
            classifier={"type": "logistic_regression"},
            esm_model="facebook/esm1v_t33_650M_UR90S_1",
            class_weight={0: 1.0, 1: 2.0},  # Int keys
        )
        assert metadata.class_weight == {0: 1.0, 1: 2.0}

    def test_to_classifier_params(self):
        """to_classifier_params() extracts init params."""
        metadata = ModelArtifactMetadata(
            model_name="facebook/esm1v_t33_650M_UR90S_1",
            model_type="logistic_regression",
            sklearn_version="1.3.0",
            classifier={"type": "logistic_regression", "C": 1.0},
            esm_model="facebook/esm1v_t33_650M_UR90S_1",
            esm_revision="abc123",
            batch_size=32,
            device="cuda",
        )

        params = metadata.to_classifier_params()
        assert params["model_name"] == "facebook/esm1v_t33_650M_UR90S_1"
        assert params["device"] == "cuda"
        assert params["batch_size"] == 32


class TestEvaluationMetrics:
    """Test EvaluationMetrics validation."""

    def test_valid_metrics(self):
        """Valid metrics construct correctly."""
        metrics = EvaluationMetrics(
            accuracy=0.85,
            precision=0.80,
            recall=0.90,
            f1=0.85,
            roc_auc=0.88,
        )
        assert metrics.accuracy == 0.85

    def test_metrics_out_of_range_rejected(self):
        """Metrics must be 0-1."""
        with pytest.raises(ValidationError):
            EvaluationMetrics(
                accuracy=1.5,  # Out of range
            )

    def test_from_sklearn_metrics(self):
        """from_sklearn_metrics() constructs from arrays."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        y_proba = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.1, 0.9]])

        metrics = EvaluationMetrics.from_sklearn_metrics(
            y_true, y_pred, y_proba, dataset_name="Test"
        )

        assert metrics.accuracy == 0.75
        assert metrics.dataset_name == "Test"
        assert metrics.n_samples == 4
        assert metrics.confusion_matrix is not None


class TestCVResults:
    """Test CVResults validation."""

    def test_valid_cv_results(self):
        """Valid CV results construct correctly."""
        cv = CVResults(
            cv_accuracy={"mean": 0.82, "std": 0.05},
            n_splits=10,
        )
        assert cv.cv_accuracy["mean"] == 0.82

    def test_from_sklearn_cv_results(self):
        """from_sklearn_cv_results() constructs from sklearn output."""
        cv_scores = {
            "test_accuracy": [0.8, 0.82, 0.85, 0.81],
            "test_f1": [0.75, 0.78, 0.80, 0.77],
        }

        cv = CVResults.from_sklearn_cv_results(cv_scores, n_splits=4)

        assert cv.n_splits == 4
        assert cv.cv_accuracy is not None
        assert 0.81 < cv.cv_accuracy["mean"] < 0.83
```

**Run tests (should FAIL initially):**
```bash
uv run pytest tests/unit/models/test_artifact.py -xvs
```

### Step 2: Implement Models

Create `src/antibody_training_esm/models/artifact.py` with specifications above.

**Update:** `src/antibody_training_esm/models/__init__.py`

```python
# ... existing imports ...

from antibody_training_esm.models.artifact import (
    ModelArtifactMetadata,
    EvaluationMetrics,
    CVResults,
)

__all__ = [
    # ... existing exports ...
    "ModelArtifactMetadata",
    "EvaluationMetrics",
    "CVResults",
]
```

**Run tests (should PASS):**
```bash
uv run pytest tests/unit/models/test_artifact.py -v
```

### Step 3: Integrate into Serialization

**Modify:** `src/antibody_training_esm/core/training/serialization.py`

```python
from antibody_training_esm.models.artifact import ModelArtifactMetadata


def save_model(
    classifier: BinaryClassifier, config: dict[str, Any], logger: logging.Logger
) -> dict[str, str]:
    """Save model with Pydantic metadata."""
    # ... existing NPZ/XGB saving logic ...

    # Format 3: JSON metadata (Pydantic)
    json_path = f"{base_path}_config.json"

    # Construct metadata from classifier (Pydantic handles serialization)
    metadata = ModelArtifactMetadata.from_classifier(classifier)

    # Add training metrics if available
    if "train_metrics" in config:
        metadata.training_metrics = config["train_metrics"]

    # Save as JSON (Pydantic handles type conversion)
    with open(json_path, "w") as f:
        json.dump(metadata.model_dump(), f, indent=2)

    logger.info(f"Saved JSON config: {json_path}")
    saved_paths["config"] = str(json_path)

    return saved_paths


def load_model_from_npz(npz_path: str, json_path: str) -> BinaryClassifier:
    """Load model with Pydantic metadata validation."""
    # Load arrays (same as before)
    arrays = np.load(npz_path)
    # ... existing array loading ...

    # Load metadata (Pydantic validates)
    with open(json_path) as f:
        metadata_dict = json.load(f)

    metadata = ModelArtifactMetadata.model_validate(metadata_dict)

    # Construct BinaryClassifier from metadata
    params = metadata.to_classifier_params()
    classifier = BinaryClassifier(params)

    # Restore fitted state (same as before)
    # ... existing state restoration ...

    return classifier
```

### Step 4: Integrate into Metrics

**Modify:** `src/antibody_training_esm/core/training/metrics.py`

```python
from antibody_training_esm.models.artifact import (
    EvaluationMetrics,
    CVResults,
)


def evaluate_model(
    classifier: Any,
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    metrics: list[str],
    logger: logging.Logger,
) -> EvaluationMetrics:
    """
    Evaluate model and return Pydantic metrics.

    Returns:
        EvaluationMetrics (not dict)
    """
    y_pred = classifier.predict(X)
    y_proba = classifier.predict_proba(X)

    # Use Pydantic model constructor
    eval_metrics = EvaluationMetrics.from_sklearn_metrics(
        y, y_pred, y_proba, dataset_name=dataset_name
    )

    # Log results
    logger.info(f"\n{dataset_name} Metrics:")
    logger.info(f"  Accuracy:  {eval_metrics.accuracy:.4f}")
    if eval_metrics.precision is not None:
        logger.info(f"  Precision: {eval_metrics.precision:.4f}")
    if eval_metrics.recall is not None:
        logger.info(f"  Recall:    {eval_metrics.recall:.4f}")
    if eval_metrics.f1 is not None:
        logger.info(f"  F1:        {eval_metrics.f1:.4f}")
    if eval_metrics.roc_auc is not None:
        logger.info(f"  ROC-AUC:   {eval_metrics.roc_auc:.4f}")

    return eval_metrics


def perform_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    config: dict[str, Any],
    logger: logging.Logger,
) -> CVResults:
    """
    Perform cross-validation and return Pydantic CVResults.

    Returns:
        CVResults (not dict)
    """
    from sklearn.model_selection import cross_validate

    # ... existing CV logic ...

    cv_scores = cross_validate(
        classifier, X, y, cv=n_splits, scoring=scoring, return_train_score=False
    )

    # Convert to Pydantic model
    cv_results = CVResults.from_sklearn_cv_results(cv_scores, n_splits=n_splits)

    # Log results
    logger.info(f"\nCross-Validation Results ({n_splits} folds):")
    logger.info(
        f"  Accuracy: {cv_results.cv_accuracy['mean']:.4f} "
        f"(+/- {cv_results.cv_accuracy['std'] * 2:.4f})"
    )

    return cv_results


def save_cv_results(
    cv_results: CVResults,
    output_dir: Path,
    experiment_name: str,
    logger: logging.Logger,
) -> None:
    """
    Save CV results using Pydantic serialization.

    Args:
        cv_results: CVResults model (not dict)
        output_dir: Directory to save results
        experiment_name: Experiment name
        logger: Logger
    """
    output_path = output_dir / f"{experiment_name}_cv_results.yaml"

    # Pydantic handles type conversion automatically
    with open(output_path, "w") as f:
        yaml.dump(cv_results.model_dump(), f)

    logger.info(f"Saved CV results to {output_path}")
```

---

## Success Criteria

### Functional Requirements

- [x] `ModelArtifactMetadata` serializes/deserializes correctly
- [x] `class_weight` with int keys preserved (no string conversion)
- [x] `EvaluationMetrics` constructs from sklearn predictions
- [x] `CVResults` constructs from sklearn cross_validate output
- [x] `save_model()` writes Pydantic metadata to JSON
- [x] `load_model_from_npz()` validates metadata with Pydantic
- [x] No manual type casting in serialization code

### Quality Gates

- [x] All unit tests pass (artifact unit tests + legacy suites)
- [x] `make test` passes (≈556 tests, ~89% coverage)
- [x] `make lint` passes
- [x] `make typecheck` passes
- [x] Code coverage ≥70%
- [x] Saved models load correctly with Pydantic validation

---

## Rollout Plan

1. **PR 1: Models Only**
   - Add `models/artifact.py`
   - Add tests

2. **PR 2: Serialization Integration**
   - Update `save_model()` and `load_model_from_npz()`

3. **PR 3: Metrics Integration**
   - Update `evaluate_model()` and `perform_cross_validation()`

---

## Backward Compatibility

**JSON format unchanged:**
- Pydantic produces same JSON structure as before
- Legacy models can still load (Pydantic is lenient)

**Function signature changes:**
- `evaluate_model()` returns `EvaluationMetrics` (not `dict`)
- `perform_cross_validation()` returns `CVResults` (not `dict`)
- Callers need to update: `metrics.accuracy` instead of `metrics["accuracy"]`

---

## Non-Goals (Out of Scope)

- ❌ Prediction validation (Phase 1)
- ❌ Config validation (Phase 2)
- ❌ DataFrame schemas (Phase 3)
- ❌ Migration of existing model artifacts (manual if needed)

---

**Last Updated:** 2025-11-21
**Completion:** All 4 phases documented
