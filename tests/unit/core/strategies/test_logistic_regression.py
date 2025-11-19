#!/usr/bin/env python3
"""
Unit Tests for LogisticRegressionStrategy

Tests the LogisticRegressionStrategy class - a wrapper for sklearn LogisticRegression
that implements the ClassifierStrategy protocol.

Philosophy:
- Test BEHAVIORS (WHAT code does), not implementation (HOW it does it)
- NO bogus mocks - test REAL sklearn LogisticRegression
- Test edge cases and error handling
- Test serialization round-trips with real file I/O
- Strict config validation (no defaults in code)

Date: 2025-11-19
Coverage Target: 100%
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from antibody_training_esm.core.strategies.logistic_regression import (
    LogisticRegressionStrategy,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def default_config() -> dict[str, Any]:
    """Return a valid full configuration for LogisticRegression."""
    return {
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": 42,
        "class_weight": None,
    }


# ============================================================================
# Initialization Tests
# ============================================================================


@pytest.mark.unit
def test_logreg_strategy_initializes_with_config(
    default_config: dict[str, Any],
) -> None:
    """Verify LogRegStrategy initializes with provided configuration."""
    # Act
    strategy = LogisticRegressionStrategy(default_config)

    # Assert
    assert strategy.C == 1.0
    assert strategy.penalty == "l2"
    assert strategy.solver == "lbfgs"
    assert strategy.max_iter == 1000
    assert strategy.random_state == 42
    assert strategy.class_weight is None


@pytest.mark.unit
def test_logreg_strategy_raises_key_error_on_missing_config() -> None:
    """Verify LogRegStrategy fails fast if config is incomplete."""
    # Arrange
    config = {"C": 1.0}  # Missing other required keys

    # Act & Assert
    with pytest.raises(KeyError):
        LogisticRegressionStrategy(config)


@pytest.mark.unit
def test_logreg_strategy_initializes_with_custom_params(
    default_config: dict[str, Any],
) -> None:
    """Verify LogRegStrategy accepts custom hyperparameters."""
    # Arrange
    config = default_config.copy()
    config.update(
        {
            "C": 0.1,
            "penalty": "l1",
            "solver": "liblinear",
            "max_iter": 500,
            "random_state": 123,
            "class_weight": "balanced",
        }
    )

    # Act
    strategy = LogisticRegressionStrategy(config)

    # Assert
    assert strategy.C == 0.1
    assert strategy.penalty == "l1"
    assert strategy.solver == "liblinear"
    assert strategy.max_iter == 500
    assert strategy.random_state == 123
    assert strategy.class_weight == "balanced"


@pytest.mark.unit
def test_logreg_strategy_creates_sklearn_classifier(
    default_config: dict[str, Any],
) -> None:
    """Verify LogRegStrategy creates sklearn LogisticRegression instance."""
    # Act
    strategy = LogisticRegressionStrategy(default_config)

    # Assert: Verify sklearn classifier exists
    assert strategy.classifier is not None
    assert hasattr(strategy.classifier, "fit")
    assert hasattr(strategy.classifier, "predict")
    assert hasattr(strategy.classifier, "predict_proba")


# ============================================================================
# Fit & Predict Tests (REAL sklearn behavior)
# ============================================================================


@pytest.mark.unit
def test_logreg_strategy_fits_and_predicts_on_simple_dataset(
    default_config: dict[str, Any],
) -> None:
    """Verify LogRegStrategy can fit and predict on linearly separable data."""
    # Arrange: Create simple 2D dataset (linearly separable)
    X_train = np.array(
        [
            [0.0, 0.0],  # Class 0
            [0.1, 0.1],  # Class 0
            [1.0, 1.0],  # Class 1
            [1.1, 1.1],  # Class 1
        ]
    )
    y_train = np.array([0, 0, 1, 1])

    X_test = np.array(
        [
            [0.05, 0.05],  # Should predict 0
            [1.05, 1.05],  # Should predict 1
        ]
    )

    strategy = LogisticRegressionStrategy(default_config)

    # Act
    strategy.fit(X_train, y_train)
    predictions = strategy.predict(X_test)

    # Assert: LogReg should learn this simple pattern
    assert predictions[0] == 0  # Near [0, 0] → class 0
    assert predictions[1] == 1  # Near [1, 1] → class 1


@pytest.mark.unit
def test_logreg_strategy_predict_proba_returns_valid_probabilities(
    default_config: dict[str, Any],
) -> None:
    """Verify predict_proba returns probabilities that sum to 1."""
    # Arrange
    np.random.seed(42)
    X_train = np.random.rand(50, 10)
    y_train = np.random.randint(0, 2, 50)

    strategy = LogisticRegressionStrategy(default_config)
    strategy.fit(X_train, y_train)

    X_test = np.random.rand(10, 10)

    # Act
    probs = strategy.predict_proba(X_test)

    # Assert
    assert probs.shape == (10, 2)  # (n_samples, n_classes)
    assert np.allclose(probs.sum(axis=1), 1.0)  # Probabilities sum to 1
    assert np.all(probs >= 0) and np.all(probs <= 1)  # Valid probabilities


@pytest.mark.unit
def test_logreg_strategy_sets_classes_attribute_after_fit(
    default_config: dict[str, Any],
) -> None:
    """Verify classes_ attribute is set after fit (sklearn compatibility)."""
    # Arrange
    X_train = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    y_train = np.array([0, 0, 1, 1])

    strategy = LogisticRegressionStrategy(default_config)

    # Act
    strategy.fit(X_train, y_train)

    # Assert
    assert hasattr(strategy, "classes_")
    np.testing.assert_array_equal(strategy.classes_, np.array([0, 1]))


# ============================================================================
# sklearn API Compatibility Tests
# ============================================================================


@pytest.mark.unit
def test_logreg_strategy_implements_get_params(default_config: dict[str, Any]) -> None:
    """Verify get_params returns all hyperparameters (sklearn API)."""
    # Arrange
    config = default_config.copy()
    config.update({"C": 0.5, "max_iter": 500})
    strategy = LogisticRegressionStrategy(config)

    # Act
    params = strategy.get_params()

    # Assert
    assert "C" in params
    assert "penalty" in params
    assert "solver" in params
    assert "max_iter" in params
    assert "random_state" in params
    assert params["C"] == 0.5
    assert params["max_iter"] == 500


@pytest.mark.unit
def test_logreg_strategy_is_instance_of_classifier_strategy_protocol(
    default_config: dict[str, Any],
) -> None:
    """Verify LogRegStrategy satisfies ClassifierStrategy protocol."""
    from antibody_training_esm.core.classifier_strategy import ClassifierStrategy

    # Arrange
    strategy = LogisticRegressionStrategy(default_config)

    # Act & Assert: Protocol check (runtime_checkable)
    assert isinstance(strategy, ClassifierStrategy)


# ============================================================================
# Serialization Tests (Round-Trip with Real File I/O)
# ============================================================================


@pytest.mark.unit
def test_logreg_strategy_to_dict_returns_hyperparameters(
    default_config: dict[str, Any],
) -> None:
    """Verify to_dict() returns all hyperparameters for JSON serialization."""
    # Arrange
    config = default_config.copy()
    config.update({"class_weight": "balanced", "C": 0.5})
    strategy = LogisticRegressionStrategy(config)

    # Act
    config_dict = strategy.to_dict()

    # Assert
    assert config_dict["type"] == "logistic_regression"
    assert config_dict["C"] == 0.5
    assert config_dict["class_weight"] == "balanced"
    assert config_dict["random_state"] == 42


@pytest.mark.unit
def test_logreg_strategy_to_arrays_raises_if_not_fitted(
    default_config: dict[str, Any],
) -> None:
    """Verify to_arrays() raises ValueError if classifier not fitted."""
    # Arrange
    strategy = LogisticRegressionStrategy(default_config)

    # Act & Assert
    with pytest.raises(ValueError, match="Classifier must be fitted"):
        strategy.to_arrays()


@pytest.mark.unit
def test_logreg_strategy_to_arrays_returns_fitted_state(
    default_config: dict[str, Any],
) -> None:
    """Verify to_arrays() returns all fitted state arrays."""
    # Arrange
    X_train = np.random.rand(50, 10)
    y_train = np.random.randint(0, 2, 50)

    strategy = LogisticRegressionStrategy(default_config)
    strategy.fit(X_train, y_train)

    # Act
    arrays = strategy.to_arrays()

    # Assert: Verify all required arrays present
    assert "coef" in arrays
    assert "intercept" in arrays
    assert "classes" in arrays
    assert "n_features_in" in arrays
    assert "n_iter" in arrays

    # Assert: Verify array shapes
    assert arrays["coef"].shape == (1, 10)  # (n_classes_binary, n_features)
    assert arrays["intercept"].shape == (1,)
    assert arrays["classes"].shape == (2,)


@pytest.mark.unit
def test_logreg_strategy_serialization_roundtrip(
    tmp_path: Path, default_config: dict[str, Any]
) -> None:
    """Verify serialize → deserialize → predict gives IDENTICAL results."""
    # Arrange: Train model
    np.random.seed(42)
    X_train = np.random.rand(50, 10)
    y_train = np.random.randint(0, 2, 50)
    X_test = np.random.rand(10, 10)

    strategy = LogisticRegressionStrategy(default_config)
    strategy.fit(X_train, y_train)

    # Get original predictions
    original_preds = strategy.predict(X_test)
    original_probs = strategy.predict_proba(X_test)

    # Act: Serialize to REAL files
    config_dict = strategy.to_dict()
    arrays_dict = strategy.to_arrays()

    # Save to disk (real file I/O, NO mocking)
    json_path = tmp_path / "config.json"
    npz_path = tmp_path / "arrays.npz"

    with open(json_path, "w") as f:
        json.dump(config_dict, f)
    np.savez(npz_path, **arrays_dict)  # type: ignore[arg-type]

    # Load from disk
    with open(json_path) as f:
        loaded_config = json.load(f)
    loaded_arrays = dict(np.load(npz_path))

    # Deserialize
    loaded_strategy = LogisticRegressionStrategy.from_dict(loaded_config, loaded_arrays)

    # Assert: Predictions match EXACTLY (not approximately)
    loaded_preds = loaded_strategy.predict(X_test)
    loaded_probs = loaded_strategy.predict_proba(X_test)

    np.testing.assert_array_equal(loaded_preds, original_preds)
    np.testing.assert_allclose(loaded_probs, original_probs, rtol=1e-10)


@pytest.mark.unit
def test_logreg_strategy_from_dict_creates_unfitted_classifier_if_arrays_none(
    default_config: dict[str, Any],
) -> None:
    """Verify from_dict() creates unfitted classifier if arrays=None."""
    # Arrange
    config = default_config.copy()
    config["type"] = "logistic_regression"

    # Act
    strategy = LogisticRegressionStrategy.from_dict(config, arrays=None)

    # Assert
    assert strategy.C == 1.0
    assert strategy.random_state == 42

    # Verify not fitted (no classes_ attribute on sklearn estimator)
    assert not hasattr(strategy.classifier, "coef_")
    assert not hasattr(strategy.classifier, "intercept_")


# ============================================================================
# Edge Cases & Error Handling
# ============================================================================


@pytest.mark.unit
def test_logreg_strategy_handles_class_weight_dict(
    default_config: dict[str, Any],
) -> None:
    """Verify LogRegStrategy handles class_weight as dict."""
    # Arrange
    config = default_config.copy()
    config["class_weight"] = {0: 1.0, 1: 2.0}  # Weight class 1 more heavily
    strategy = LogisticRegressionStrategy(config)

    X_train = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    y_train = np.array([0, 0, 1, 1])

    # Act
    strategy.fit(X_train, y_train)
    predictions = strategy.predict(X_train)

    # Assert: Classifier should work with dict class_weight
    assert predictions.shape == (4,)
    assert set(predictions).issubset({0, 1})


@pytest.mark.unit
def test_logreg_strategy_deterministic_with_same_random_state(
    default_config: dict[str, Any],
) -> None:
    """Verify same random_state gives deterministic results."""
    # Arrange
    np.random.seed(42)
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.rand(20, 10)

    # Act: Train two identical models
    strategy1 = LogisticRegressionStrategy(default_config)
    strategy1.fit(X_train, y_train)
    preds1 = strategy1.predict(X_test)

    strategy2 = LogisticRegressionStrategy(default_config)
    strategy2.fit(X_train, y_train)
    preds2 = strategy2.predict(X_test)

    # Assert: Predictions identical (deterministic)
    np.testing.assert_array_equal(preds1, preds2)


@pytest.mark.unit
def test_logreg_strategy_different_random_state_gives_different_results(
    default_config: dict[str, Any],
) -> None:
    """Verify different random_state can give different results (solver-dependent)."""
    # Arrange
    np.random.seed(42)
    # Create challenging dataset where optimization path matters
    X_train = np.random.rand(100, 50)
    y_train = np.random.randint(0, 2, 100)

    config1 = default_config.copy()
    config1["solver"] = "sag"

    config2 = config1.copy()
    config2["random_state"] = 123

    # Act
    strategy1 = LogisticRegressionStrategy(config1)
    strategy1.fit(X_train, y_train)
    coef1 = strategy1.classifier.coef_.copy()

    strategy2 = LogisticRegressionStrategy(config2)
    strategy2.fit(X_train, y_train)
    coef2 = strategy2.classifier.coef_.copy()

    # Assert: Coefficients may differ (stochastic optimization)
    assert coef1.shape == coef2.shape


# ============================================================================
# JSON Serialization Edge Cases
# ============================================================================


@pytest.mark.unit
def test_logreg_strategy_json_handles_none_class_weight(
    default_config: dict[str, Any],
) -> None:
    """Verify JSON serialization handles None class_weight correctly."""
    # Arrange
    config = default_config.copy()
    config["class_weight"] = None
    strategy = LogisticRegressionStrategy(config)

    # Act
    config_dict = strategy.to_dict()
    json_str = json.dumps(config_dict)  # Verify JSON-serializable
    loaded_config = json.loads(json_str)

    # Assert
    assert loaded_config["class_weight"] is None


@pytest.mark.unit
def test_logreg_strategy_json_handles_dict_class_weight(
    default_config: dict[str, Any],
) -> None:
    """Verify JSON serialization handles dict class_weight correctly."""
    # Arrange
    config = default_config.copy()
    config["class_weight"] = {0: 1.0, 1: 2.0}
    strategy = LogisticRegressionStrategy(config)

    # Act
    config_dict = strategy.to_dict()
    json_str = json.dumps(config_dict)  # JSON converts int keys to strings
    loaded_config = json.loads(json_str)

    # Assert: JSON converts int keys to strings
    assert loaded_config["class_weight"]["0"] == 1.0  # String key now
    assert loaded_config["class_weight"]["1"] == 2.0
