#!/usr/bin/env python3
"""
Unit Tests for BinaryClassifier

Tests the BinaryClassifier class for antibody non-specificity prediction.
Focus: behavior testing, not implementation details.

Philosophy:
- Test behaviors (WHAT code does), not implementation (HOW it does it)
- Minimal mocking (only ESM model loading)
- Test edge cases and error handling

Date: 2025-11-07
Coverage Target: 90%+
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from antibody_training_esm.core.classifier import BinaryClassifier
from tests.conftest import assert_valid_predictions

# ============================================================================
# Initialization Tests
# ============================================================================


@pytest.mark.unit
def test_classifier_initializes_with_default_params(
    mock_transformers_model: Any,
) -> None:
    """Verify classifier initializes with default hyperparameters"""
    # Arrange
    params = {
        "model_name": "facebook/esm1v_t33_650M_UR90S_1",
        "device": "cpu",
        "random_state": 42,
        "max_iter": 1000,
    }

    # Act
    classifier = BinaryClassifier(params=params)

    # Assert
    assert classifier.C == 1.0  # Default C
    assert classifier.penalty == "l2"  # Default penalty
    assert classifier.solver == "lbfgs"  # Default solver
    assert classifier.random_state == 42
    assert classifier.max_iter == 1000
    assert classifier.is_fitted is False


@pytest.mark.unit
def test_classifier_initializes_with_custom_params(
    mock_transformers_model: Any,
) -> None:
    """Verify classifier accepts custom hyperparameters"""
    # Arrange
    params = {
        "model_name": "facebook/esm1v_t33_650M_UR90S_1",
        "device": "cpu",
        "random_state": 42,
        "max_iter": 500,
        "C": 0.1,
        "penalty": "l1",
        "solver": "liblinear",
        "class_weight": "balanced",
    }

    # Act
    classifier = BinaryClassifier(params=params)

    # Assert
    assert classifier.C == 0.1
    assert classifier.penalty == "l1"
    assert classifier.solver == "liblinear"
    assert classifier.class_weight == "balanced"
    assert classifier.max_iter == 500


@pytest.mark.unit
def test_classifier_initializes_with_kwargs(mock_transformers_model: Any) -> None:
    """Verify classifier supports sklearn-style kwargs initialization"""
    # Arrange / Act
    classifier = BinaryClassifier(
        model_name="facebook/esm1v_t33_650M_UR90S_1",
        device="cpu",
        random_state=42,
        max_iter=1000,
        C=0.5,
        penalty="l2",
    )

    # Assert
    assert classifier.C == 0.5
    assert classifier.penalty == "l2"
    assert classifier.random_state == 42


@pytest.mark.unit
def test_classifier_stores_embedding_extractor(mock_transformers_model: Any) -> None:
    """Verify classifier creates embedding extractor during initialization"""
    # Arrange
    params = {
        "model_name": "facebook/esm1v_t33_650M_UR90S_1",
        "device": "cpu",
        "random_state": 42,
        "max_iter": 1000,
    }

    # Act
    classifier = BinaryClassifier(params=params)

    # Assert
    assert classifier.embedding_extractor is not None
    assert classifier.model_name == "facebook/esm1v_t33_650M_UR90S_1"
    assert classifier.device == "cpu"


# ============================================================================
# sklearn API Compatibility Tests
# ============================================================================


@pytest.mark.unit
def test_classifier_implements_get_params(mock_transformers_model: Any) -> None:
    """Verify classifier implements sklearn get_params() interface"""
    # Arrange
    params = {
        "model_name": "facebook/esm1v_t33_650M_UR90S_1",
        "device": "cpu",
        "random_state": 42,
        "max_iter": 1000,
        "C": 0.5,
        "penalty": "l2",
    }
    classifier = BinaryClassifier(params=params)

    # Act
    retrieved_params = classifier.get_params()

    # Assert
    assert "C" in retrieved_params
    assert "penalty" in retrieved_params
    assert "random_state" in retrieved_params
    assert "model_name" in retrieved_params
    assert retrieved_params["C"] == 0.5
    assert retrieved_params["penalty"] == "l2"
    assert retrieved_params["random_state"] == 42


@pytest.mark.unit
def test_classifier_implements_set_params(mock_transformers_model: Any) -> None:
    """Verify classifier implements sklearn set_params() interface"""
    # Arrange
    params = {
        "model_name": "facebook/esm1v_t33_650M_UR90S_1",
        "device": "cpu",
        "random_state": 42,
        "max_iter": 1000,
        "C": 1.0,
    }
    classifier = BinaryClassifier(params=params)

    # Act
    classifier.set_params(C=0.1, max_iter=500)

    # Assert
    assert classifier.C == 0.1
    assert classifier.max_iter == 500


@pytest.mark.unit
def test_get_params_returns_only_valid_constructor_params(
    mock_transformers_model: Any,
) -> None:
    """Verify get_params() excludes non-constructor params like 'cv_folds'"""
    # Arrange
    params = {
        "model_name": "facebook/esm1v_t33_650M_UR90S_1",
        "device": "cpu",
        "random_state": 42,
        "max_iter": 1000,
        "cv_folds": 5,  # Not a constructor param
        "stratify": True,  # Not a constructor param
    }
    classifier = BinaryClassifier(params=params)

    # Act
    retrieved_params = classifier.get_params()

    # Assert - should NOT include non-constructor params
    assert "cv_folds" not in retrieved_params
    assert "stratify" not in retrieved_params
    assert "C" in retrieved_params  # Valid param


# ============================================================================
# Fitting Tests
# ============================================================================


@pytest.mark.unit
def test_classifier_fits_on_embeddings(
    mock_embeddings: dict[str, Any],
    default_classifier_params: dict[str, Any],
    mock_transformers_model: Any,
) -> None:
    """Verify classifier can be fitted on embedding arrays"""
    # Arrange
    classifier = BinaryClassifier(params=default_classifier_params)
    X = mock_embeddings["X_train"]
    y = mock_embeddings["y_train"]

    # Act
    classifier.fit(X, y)

    # Assert
    assert classifier.is_fitted is True


@pytest.mark.unit
def test_fit_accepts_numpy_arrays(
    mock_embeddings: dict[str, Any],
    default_classifier_params: dict[str, Any],
    mock_transformers_model: Any,
) -> None:
    """Verify fit() accepts np.ndarray for X and y"""
    # Arrange
    classifier = BinaryClassifier(params=default_classifier_params)
    X = mock_embeddings["X_train"]
    y = mock_embeddings["y_train"]

    # Act
    classifier.fit(X, y)

    # Assert
    assert classifier.is_fitted is True
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)


@pytest.mark.unit
def test_fit_handles_balanced_dataset(
    default_classifier_params: dict[str, Any],
    mock_transformers_model: Any,
) -> None:
    """Verify classifier handles balanced dataset (50/50 split)"""
    # Arrange
    classifier = BinaryClassifier(params=default_classifier_params)
    X = np.random.rand(100, 1280).astype(np.float32)
    y = np.array([0, 1] * 50)  # Balanced

    # Act
    classifier.fit(X, y)

    # Assert
    assert classifier.is_fitted is True


@pytest.mark.unit
def test_fit_handles_imbalanced_dataset(
    default_classifier_params: dict[str, Any],
    mock_transformers_model: Any,
) -> None:
    """Verify classifier handles imbalanced dataset (90/10 split)"""
    # Arrange
    classifier = BinaryClassifier(params=default_classifier_params)
    X = np.random.rand(100, 1280).astype(np.float32)
    y = np.array([0] * 90 + [1] * 10)  # Imbalanced

    # Act
    classifier.fit(X, y)

    # Assert
    assert classifier.is_fitted is True


@pytest.mark.unit
def test_fit_with_class_weight_balanced(mock_transformers_model: Any) -> None:
    """Verify classifier handles class_weight='balanced' for imbalanced data"""
    # Arrange
    params = {
        "model_name": "facebook/esm1v_t33_650M_UR90S_1",
        "device": "cpu",
        "random_state": 42,
        "max_iter": 1000,
        "class_weight": "balanced",
    }
    classifier = BinaryClassifier(params=params)
    X = np.random.rand(100, 1280).astype(np.float32)
    y = np.array([0] * 90 + [1] * 10)  # Imbalanced

    # Act
    classifier.fit(X, y)

    # Assert
    assert classifier.is_fitted is True
    assert classifier.class_weight == "balanced"


# ============================================================================
# Prediction Tests (Binary Classification)
# ============================================================================


@pytest.mark.unit
def test_predict_returns_binary_labels(
    fitted_classifier: Any, mock_embeddings: dict[str, Any]
) -> None:
    """Verify predictions are binary (0 or 1)"""
    # Arrange
    X_test = mock_embeddings["X_test"]

    # Act
    predictions = fitted_classifier.predict(X_test)

    # Assert
    assert_valid_predictions(predictions, len(X_test))


@pytest.mark.unit
def test_predict_returns_correct_shape(
    fitted_classifier: Any, mock_embeddings: dict[str, Any]
) -> None:
    """Verify predictions have correct shape (n_samples,)"""
    # Arrange
    X_test = mock_embeddings["X_test"]

    # Act
    predictions = fitted_classifier.predict(X_test)

    # Assert
    assert predictions.shape == (len(X_test),)


@pytest.mark.unit
def test_predict_handles_single_sample(fitted_classifier: Any) -> None:
    """Verify classifier handles single embedding (edge case)"""
    # Arrange
    X_single = np.random.rand(1, 1280).astype(np.float32)

    # Act
    predictions = fitted_classifier.predict(X_single)

    # Assert
    assert len(predictions) == 1
    assert predictions[0] in [0, 1]


@pytest.mark.unit
def test_predict_handles_large_batch(fitted_classifier: Any) -> None:
    """Verify classifier handles large batches efficiently"""
    # Arrange
    X_large = np.random.rand(1000, 1280).astype(np.float32)

    # Act
    predictions = fitted_classifier.predict(X_large)

    # Assert
    assert len(predictions) == 1000
    assert all(pred in [0, 1] for pred in predictions)


@pytest.mark.unit
def test_predict_before_fit_raises_error(
    default_classifier_params: dict[str, Any],
    mock_transformers_model: Any,
) -> None:
    """Verify classifier raises error when predicting before fit"""
    # Arrange
    classifier = BinaryClassifier(params=default_classifier_params)
    X = np.random.rand(10, 1280).astype(np.float32)

    # Act & Assert
    with pytest.raises(ValueError, match="Classifier must be fitted"):
        classifier.predict(X)


# ============================================================================
# Threshold Calibration Tests (Assay-Specific)
# ============================================================================


@pytest.mark.unit
def test_predict_uses_default_threshold_0_5(fitted_classifier: Any) -> None:
    """Verify default threshold is 0.5 (ELISA)"""
    # Arrange
    X = np.random.rand(1, 1280).astype(np.float32)

    # Mock predict_proba to return known probability
    fitted_classifier.classifier.predict_proba = lambda X: np.array([[0.4, 0.6]])

    # Act
    prediction = fitted_classifier.predict(X)

    # Assert
    assert prediction[0] == 1  # 0.6 > 0.5 threshold


@pytest.mark.unit
def test_predict_applies_elisa_threshold(fitted_classifier: Any) -> None:
    """Verify assay_type='ELISA' uses 0.5 threshold"""
    # Arrange
    X = np.random.rand(1, 1280).astype(np.float32)

    # Mock predict_proba to return known probability
    fitted_classifier.classifier.predict_proba = lambda X: np.array([[0.45, 0.55]])

    # Act
    prediction = fitted_classifier.predict(X, assay_type="ELISA")

    # Assert
    assert prediction[0] == 1  # 0.55 > 0.5 threshold


@pytest.mark.unit
def test_predict_applies_psr_threshold(fitted_classifier: Any) -> None:
    """Verify assay_type='PSR' uses 0.5495 threshold (Novo parity)"""
    # Arrange
    X = np.random.rand(1, 1280).astype(np.float32)

    # Mock predict_proba to return known probability
    fitted_classifier.classifier.predict_proba = lambda X: np.array([[0.45, 0.55]])

    # Act
    prediction = fitted_classifier.predict(X, assay_type="PSR")

    # Assert
    assert prediction[0] == 1  # 0.55 > 0.5495 threshold


@pytest.mark.unit
def test_predict_psr_threshold_boundary_case(fitted_classifier: Any) -> None:
    """Verify PSR threshold boundary: 0.5495"""
    # Arrange
    X = np.random.rand(2, 1280).astype(np.float32)

    # Mock predict_proba: one just below, one just above threshold
    fitted_classifier.classifier.predict_proba = lambda X: np.array(
        [
            [0.5, 0.5494],  # Below threshold
            [0.5, 0.5496],  # Above threshold
        ]
    )

    # Act
    predictions = fitted_classifier.predict(X, assay_type="PSR")

    # Assert
    assert predictions[0] == 0  # 0.5494 < 0.5495
    assert predictions[1] == 1  # 0.5496 > 0.5495


@pytest.mark.unit
def test_predict_custom_threshold_overrides_default(fitted_classifier: Any) -> None:
    """Verify custom threshold overrides default 0.5"""
    # Arrange
    X = np.random.rand(1, 1280).astype(np.float32)

    # Mock predict_proba to return known probability
    fitted_classifier.classifier.predict_proba = lambda X: np.array([[0.4, 0.6]])

    # Act
    prediction = fitted_classifier.predict(X, threshold=0.7)

    # Assert
    assert prediction[0] == 0  # 0.6 < 0.7 custom threshold


@pytest.mark.unit
def test_predict_rejects_unknown_assay_type(fitted_classifier: Any) -> None:
    """Verify classifier raises error for unknown assay_type"""
    # Arrange
    X = np.random.rand(1, 1280).astype(np.float32)

    # Act & Assert
    with pytest.raises(ValueError, match="Unknown assay_type"):
        fitted_classifier.predict(X, assay_type="UNKNOWN")


# ============================================================================
# Probability Prediction Tests
# ============================================================================


@pytest.mark.unit
def test_predict_proba_returns_probabilities(
    fitted_classifier: Any, mock_embeddings: dict[str, Any]
) -> None:
    """Verify predict_proba returns class probabilities"""
    # Arrange
    X_test = mock_embeddings["X_test"]

    # Act
    probabilities = fitted_classifier.predict_proba(X_test)

    # Assert
    assert probabilities.shape == (len(X_test), 2)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1


@pytest.mark.unit
def test_predict_proba_before_fit_raises_error(
    default_classifier_params: dict[str, Any],
    mock_transformers_model: Any,
) -> None:
    """Verify predict_proba raises error before fit"""
    # Arrange
    classifier = BinaryClassifier(params=default_classifier_params)
    X = np.random.rand(10, 1280).astype(np.float32)

    # Act & Assert
    with pytest.raises(ValueError, match="Classifier must be fitted"):
        classifier.predict_proba(X)


# ============================================================================
# Score Tests (Accuracy)
# ============================================================================


@pytest.mark.unit
def test_score_returns_accuracy(
    fitted_classifier: Any, mock_embeddings: dict[str, Any]
) -> None:
    """Verify score() returns mean accuracy"""
    # Arrange
    X_test = mock_embeddings["X_test"]
    y_test = mock_embeddings["y_test"]

    # Act
    accuracy = fitted_classifier.score(X_test, y_test)

    # Assert
    assert 0.0 <= accuracy <= 1.0
    assert isinstance(accuracy, float)


@pytest.mark.unit
def test_score_before_fit_raises_error(
    default_classifier_params: dict[str, Any],
    mock_transformers_model: Any,
) -> None:
    """Verify score raises error before fit"""
    # Arrange
    classifier = BinaryClassifier(params=default_classifier_params)
    X = np.random.rand(10, 1280).astype(np.float32)
    y = np.array([0, 1] * 5)

    # Act & Assert
    with pytest.raises(ValueError, match="Classifier must be fitted before scoring"):
        classifier.score(X, y)


# ============================================================================
# Edge Case Tests
# ============================================================================


@pytest.mark.unit
def test_predict_handles_empty_embedding_array(fitted_classifier: Any) -> None:
    """Verify classifier behavior with empty embeddings array"""
    # Arrange
    empty_embeddings = np.array([]).reshape(0, 1280)

    # Act & Assert
    # sklearn LogisticRegression will raise on empty input
    with pytest.raises(ValueError):
        fitted_classifier.predict(empty_embeddings)


@pytest.mark.unit
def test_fit_handles_minimum_samples(
    default_classifier_params: dict[str, Any],
    mock_transformers_model: Any,
) -> None:
    """Verify classifier handles minimum training samples (2 per class)"""
    # Arrange
    classifier = BinaryClassifier(params=default_classifier_params)
    X = np.random.rand(4, 1280).astype(np.float32)
    y = np.array([0, 0, 1, 1])  # Minimum: 2 samples per class

    # Act
    classifier.fit(X, y)

    # Assert
    assert classifier.is_fitted is True


@pytest.mark.unit
def test_predict_handles_all_zero_embeddings(fitted_classifier: Any) -> None:
    """Verify classifier handles zero embeddings (edge case)"""
    # Arrange
    zero_embeddings = np.zeros((5, 1280), dtype=np.float32)

    # Act
    predictions = fitted_classifier.predict(zero_embeddings)

    # Assert - should not crash, predictions should be valid
    assert len(predictions) == 5
    assert all(pred in [0, 1] for pred in predictions)


@pytest.mark.unit
def test_predict_handles_nan_free_embeddings(fitted_classifier: Any) -> None:
    """Verify classifier handles NaN-free embeddings correctly"""
    # Arrange
    valid_embeddings = np.random.rand(10, 1280).astype(np.float32)

    # Act
    predictions = fitted_classifier.predict(valid_embeddings)

    # Assert
    assert len(predictions) == 10
    assert not np.isnan(predictions).any()


# ============================================================================
# Serialization Tests (Pickle Compatibility)
# ============================================================================


@pytest.mark.unit
def test_classifier_supports_getstate(fitted_classifier: Any) -> None:
    """Verify classifier implements __getstate__ for pickling"""
    # Act
    state = fitted_classifier.__getstate__()

    # Assert
    assert isinstance(state, dict)
    assert "embedding_extractor" not in state  # Should be excluded
    assert "classifier" in state  # LogisticRegression should be included
    assert "is_fitted" in state


@pytest.mark.unit
def test_classifier_supports_setstate(
    fitted_classifier: Any, mock_transformers_model: Any
) -> None:
    """Verify classifier implements __setstate__ for unpickling"""
    # Arrange
    state = fitted_classifier.__getstate__()

    # Act
    new_classifier = BinaryClassifier.__new__(BinaryClassifier)
    new_classifier.__setstate__(state)

    # Assert
    assert new_classifier.embedding_extractor is not None  # Recreated
    assert new_classifier.is_fitted is True
    assert new_classifier.model_name == fitted_classifier.model_name


# ============================================================================
# Integration-like Tests (Real Workflow)
# ============================================================================


@pytest.mark.unit
def test_full_fit_predict_workflow(
    default_classifier_params: dict[str, Any], mock_transformers_model: Any
) -> None:
    """Verify complete fit → predict workflow"""
    # Arrange
    classifier = BinaryClassifier(params=default_classifier_params)
    X_train = np.random.rand(50, 1280).astype(np.float32)
    y_train = np.array([0, 1] * 25)
    X_test = np.random.rand(10, 1280).astype(np.float32)

    # Act
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    # Assert
    assert classifier.is_fitted is True
    assert len(predictions) == 10
    assert all(pred in [0, 1] for pred in predictions)


@pytest.mark.unit
def test_refit_classifier_updates_model(
    default_classifier_params: dict[str, Any],
    mock_transformers_model: Any,
) -> None:
    """Verify classifier can be refitted with new data"""
    # Arrange
    classifier = BinaryClassifier(params=default_classifier_params)

    X_train1 = np.random.rand(50, 1280).astype(np.float32)
    y_train1 = np.array([0, 1] * 25)

    X_train2 = np.random.rand(100, 1280).astype(np.float32)
    y_train2 = np.array([0, 1] * 50)

    # Act: Fit, then refit
    classifier.fit(X_train1, y_train1)
    predictions1 = classifier.predict(X_train1[:5])

    classifier.fit(X_train2, y_train2)
    predictions2 = classifier.predict(X_train2[:5])

    # Assert - both should work
    assert len(predictions1) == 5
    assert len(predictions2) == 5
    assert classifier.is_fitted is True


# ============================================================================
# Docstring Examples Tests
# ============================================================================


@pytest.mark.unit
def test_readme_example_workflow(mock_transformers_model: Any) -> None:
    """Verify example from README/docstrings works correctly"""
    # This test ensures documentation examples stay accurate

    # Arrange
    params = {
        "model_name": "facebook/esm1v_t33_650M_UR90S_1",
        "device": "cpu",
        "random_state": 42,
        "max_iter": 1000,
    }
    classifier = BinaryClassifier(params=params)

    # Create sample data
    X_train = np.random.rand(100, 1280).astype(np.float32)
    y_train = np.array([0, 1] * 50)
    X_test = np.random.rand(10, 1280).astype(np.float32)

    # Act: Train and predict
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test, assay_type="PSR")

    # Assert
    assert len(predictions) == 10
    assert all(pred in [0, 1] for pred in predictions)
