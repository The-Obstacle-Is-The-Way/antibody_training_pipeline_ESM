#!/usr/bin/env python3
"""
Unit Tests for XGBoostStrategy

Tests the XGBoostStrategy class - a wrapper for xgboost.XGBClassifier
that implements the ClassifierStrategy protocol.

Philosophy:
- Test BEHAVIORS (WHAT code does), not implementation (HOW it does it)
- NO bogus mocks - test REAL XGBoost
- Test edge cases and error handling
- Test serialization round-trips with real file I/O

Date: 2025-11-18
Coverage Target: 100%
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from antibody_training_esm.core.strategies.xgboost_strategy import XGBoostStrategy

# ============================================================================
# Initialization Tests
# ============================================================================


@pytest.mark.unit
def test_xgboost_strategy_initializes_with_defaults() -> None:
    """Verify XGBoostStrategy initializes with default hyperparameters."""
    # Arrange
    config: dict[str, Any] = {}  # Empty config - should use defaults

    # Act
    strategy = XGBoostStrategy(config)

    # Assert
    assert strategy.n_estimators == 100
    assert strategy.max_depth == 6
    assert strategy.learning_rate == 0.3
    assert strategy.subsample == 1.0
    assert strategy.colsample_bytree == 1.0
    assert strategy.reg_alpha == 0.0
    assert strategy.reg_lambda == 1.0
    assert strategy.random_state == 42
    assert strategy.objective == "binary:logistic"


@pytest.mark.unit
def test_xgboost_strategy_initializes_with_custom_params() -> None:
    """Verify XGBoostStrategy accepts custom hyperparameters."""
    # Arrange
    config = {
        "n_estimators": 50,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "random_state": 123,
    }

    # Act
    strategy = XGBoostStrategy(config)

    # Assert
    assert strategy.n_estimators == 50
    assert strategy.max_depth == 4
    assert strategy.learning_rate == 0.1
    assert strategy.subsample == 0.8
    assert strategy.colsample_bytree == 0.8
    assert strategy.reg_alpha == 0.1
    assert strategy.reg_lambda == 2.0
    assert strategy.random_state == 123


@pytest.mark.unit
def test_xgboost_strategy_creates_xgb_classifier() -> None:
    """Verify XGBoostStrategy creates XGBClassifier instance."""
    # Arrange
    config = {"n_estimators": 10}

    # Act
    strategy = XGBoostStrategy(config)

    # Assert: Verify XGBClassifier exists
    assert strategy.classifier is not None
    assert hasattr(strategy.classifier, "fit")
    assert hasattr(strategy.classifier, "predict")
    assert hasattr(strategy.classifier, "predict_proba")


# ============================================================================
# Fit & Predict Tests (REAL XGBoost behavior)
# ============================================================================


@pytest.mark.unit
def test_xgboost_strategy_fits_and_predicts_on_simple_dataset() -> None:
    """Verify XGBoost can fit and predict on linearly separable data."""
    # Arrange: Create simple 2D dataset (linearly separable)
    # Use 20 samples (10 per class) for reliable learning
    X_train = np.array(
        [
            [0.0 + i * 0.01, 0.0 + i * 0.01]
            for i in range(10)  # Class 0
        ]
        + [
            [1.0 + i * 0.01, 1.0 + i * 0.01]
            for i in range(10)  # Class 1
        ]
    )
    y_train = np.array([0] * 10 + [1] * 10)

    X_test = np.array(
        [
            [0.05, 0.05],  # Should predict 0
            [1.05, 1.05],  # Should predict 1
        ]
    )

    config = {"random_state": 42, "n_estimators": 20}
    strategy = XGBoostStrategy(config)

    # Act
    strategy.fit(X_train, y_train)
    predictions = strategy.predict(X_test)

    # Assert: XGBoost should learn this simple pattern
    assert predictions[0] == 0  # Near [0, 0] → class 0
    assert predictions[1] == 1  # Near [1, 1] → class 1


@pytest.mark.unit
def test_xgboost_strategy_handles_nonlinear_data() -> None:
    """Verify XGBoost learns non-linear decision boundary (XOR-like)."""
    # Arrange: XOR-like pattern (LogReg fails, XGBoost succeeds)
    np.random.seed(42)  # For reproducible noise
    # Create XOR pattern with slight noise for robustness
    X_train_list = []
    y_train_list = []
    for _ in range(50):
        for x1, x2, label in [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]:
            noise = np.random.randn(2) * 0.05  # Small noise
            X_train_list.append([x1 + noise[0], x2 + noise[1]])
            y_train_list.append(label)

    # Convert to numpy arrays
    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)

    X_test = np.array(
        [
            [0.1, 0.1],  # Should predict 0
            [0.1, 0.9],  # Should predict 1
            [0.9, 0.1],  # Should predict 1
            [0.9, 0.9],  # Should predict 0
        ]
    )

    config = {
        "random_state": 42,
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
    }
    strategy = XGBoostStrategy(config)

    # Act
    strategy.fit(X_train, y_train)
    predictions = strategy.predict(X_test)

    # Assert: XGBoost should learn XOR pattern (at least 75% accuracy)
    expected = np.array([0, 1, 1, 0])
    accuracy = (predictions == expected).mean()
    assert accuracy >= 0.75, f"XGBoost failed to learn XOR: accuracy={accuracy}"


@pytest.mark.unit
def test_xgboost_strategy_predict_proba_returns_valid_probabilities() -> None:
    """Verify predict_proba returns probabilities that sum to 1."""
    # Arrange
    np.random.seed(42)
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, 100)

    config = {"random_state": 42, "n_estimators": 10}
    strategy = XGBoostStrategy(config)
    strategy.fit(X_train, y_train)

    X_test = np.random.rand(20, 10)

    # Act
    probs = strategy.predict_proba(X_test)

    # Assert
    assert probs.shape == (20, 2)  # (n_samples, n_classes)
    assert np.allclose(probs.sum(axis=1), 1.0)  # Probabilities sum to 1
    assert np.all(probs >= 0) and np.all(probs <= 1)  # Valid probabilities


@pytest.mark.unit
def test_xgboost_strategy_sets_classes_attribute_after_fit() -> None:
    """Verify classes_ attribute is set after fit (sklearn compatibility)."""
    # Arrange
    X_train = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    y_train = np.array([0, 0, 1, 1])

    config = {"random_state": 42, "n_estimators": 10}
    strategy = XGBoostStrategy(config)

    # Act
    strategy.fit(X_train, y_train)

    # Assert
    assert hasattr(strategy, "classes_")
    np.testing.assert_array_equal(strategy.classes_, np.array([0, 1]))


# ============================================================================
# sklearn API Compatibility Tests
# ============================================================================


@pytest.mark.unit
def test_xgboost_strategy_implements_get_params() -> None:
    """Verify get_params returns all hyperparameters (sklearn API)."""
    # Arrange
    config = {
        "n_estimators": 50,
        "max_depth": 4,
        "learning_rate": 0.1,
        "random_state": 42,
    }
    strategy = XGBoostStrategy(config)

    # Act
    params = strategy.get_params()

    # Assert
    assert "n_estimators" in params
    assert "max_depth" in params
    assert "learning_rate" in params
    assert "random_state" in params
    assert params["n_estimators"] == 50
    assert params["max_depth"] == 4
    assert params["learning_rate"] == 0.1


@pytest.mark.unit
def test_xgboost_strategy_is_instance_of_classifier_strategy_protocol() -> None:
    """Verify XGBoostStrategy satisfies ClassifierStrategy protocol."""
    from antibody_training_esm.core.classifier_strategy import ClassifierStrategy

    # Arrange
    config = {"random_state": 42, "n_estimators": 10}
    strategy = XGBoostStrategy(config)

    # Act & Assert: Protocol check (runtime_checkable)
    assert isinstance(strategy, ClassifierStrategy)


@pytest.mark.unit
def test_xgboost_strategy_implements_score() -> None:
    """Verify score() returns mean accuracy."""
    # Arrange
    np.random.seed(42)
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.rand(20, 10)
    y_test = np.random.randint(0, 2, 20)

    config = {"random_state": 42, "n_estimators": 10}
    strategy = XGBoostStrategy(config)
    strategy.fit(X_train, y_train)

    # Act
    accuracy = strategy.score(X_test, y_test)

    # Assert
    assert 0.0 <= accuracy <= 1.0
    assert isinstance(accuracy, float)


# ============================================================================
# Serialization Tests (Native .xgb format)
# ============================================================================


@pytest.mark.unit
def test_xgboost_strategy_to_dict_returns_hyperparameters() -> None:
    """Verify to_dict() returns all hyperparameters for JSON serialization."""
    # Arrange
    config = {
        "n_estimators": 50,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "random_state": 42,
    }
    strategy = XGBoostStrategy(config)

    # Act
    config_dict = strategy.to_dict()

    # Assert
    assert config_dict["type"] == "xgboost"
    assert config_dict["n_estimators"] == 50
    assert config_dict["max_depth"] == 4
    assert config_dict["learning_rate"] == 0.1
    assert config_dict["subsample"] == 0.8
    assert config_dict["random_state"] == 42


@pytest.mark.unit
def test_xgboost_strategy_save_model_raises_if_not_fitted() -> None:
    """Verify save_model() raises ValueError if classifier not fitted."""
    # Arrange
    config = {"random_state": 42, "n_estimators": 10}
    strategy = XGBoostStrategy(config)

    # Act & Assert
    with pytest.raises(ValueError, match="Classifier must be fitted"):
        strategy.save_model("/tmp/model.xgb")


@pytest.mark.unit
def test_xgboost_strategy_save_and_load_model(tmp_path: Path) -> None:
    """Verify save_model() → load_model() gives IDENTICAL predictions."""
    # Arrange: Train model
    np.random.seed(42)
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.rand(20, 10)

    config = {"random_state": 42, "n_estimators": 20}
    strategy = XGBoostStrategy(config)
    strategy.fit(X_train, y_train)

    original_preds = strategy.predict(X_test)
    original_probs = strategy.predict_proba(X_test)

    # Act: Save model (REAL file I/O)
    xgb_path = tmp_path / "model.xgb"
    json_path = tmp_path / "config.json"

    strategy.save_model(str(xgb_path))
    config_dict = strategy.to_dict()
    with open(json_path, "w") as f:
        json.dump(config_dict, f)

    # Load model
    with open(json_path) as f:
        loaded_config = json.load(f)
    loaded_strategy = XGBoostStrategy.load_model(str(xgb_path), loaded_config)

    # Assert: Predictions match EXACTLY
    loaded_preds = loaded_strategy.predict(X_test)
    loaded_probs = loaded_strategy.predict_proba(X_test)

    np.testing.assert_array_equal(loaded_preds, original_preds)
    np.testing.assert_allclose(loaded_probs, original_probs, rtol=1e-10)


@pytest.mark.unit
def test_xgboost_strategy_from_dict_creates_unfitted_classifier() -> None:
    """Verify from_dict() creates unfitted classifier if no model file."""
    # Arrange
    config = {
        "type": "xgboost",
        "n_estimators": 50,
        "random_state": 42,
    }

    # Act
    strategy = XGBoostStrategy.from_dict(config)

    # Assert
    # Cast to concrete type for attribute checks (from_dict returns ClassifierStrategy protocol)
    assert isinstance(strategy, XGBoostStrategy)
    assert strategy.n_estimators == 50
    assert strategy.random_state == 42

    # Verify not fitted (no classes_ or get_booster)
    assert not hasattr(strategy.classifier, "_Booster")


# ============================================================================
# Edge Cases & Error Handling
# ============================================================================


@pytest.mark.unit
def test_xgboost_strategy_deterministic_with_same_random_state() -> None:
    """Verify same random_state gives deterministic results."""
    # Arrange
    np.random.seed(42)
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.rand(20, 10)

    config1 = {"random_state": 42, "n_estimators": 20}
    config2 = {"random_state": 42, "n_estimators": 20}

    # Act: Train two identical models
    strategy1 = XGBoostStrategy(config1)
    strategy1.fit(X_train, y_train)
    preds1 = strategy1.predict(X_test)

    strategy2 = XGBoostStrategy(config2)
    strategy2.fit(X_train, y_train)
    preds2 = strategy2.predict(X_test)

    # Assert: Predictions identical (deterministic)
    np.testing.assert_array_equal(preds1, preds2)


@pytest.mark.unit
def test_xgboost_strategy_handles_small_n_estimators() -> None:
    """Verify XGBoost works with very small n_estimators."""
    # Arrange
    X_train: np.ndarray[Any, Any] = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    y_train: np.ndarray[Any, Any] = np.array([0, 0, 1, 1])

    config = {"n_estimators": 1, "random_state": 42}  # Single tree
    strategy = XGBoostStrategy(config)

    # Act
    strategy.fit(X_train, y_train)
    predictions = strategy.predict(X_train)

    # Assert: Should still work (even with 1 tree)
    assert predictions.shape == (4,)
    assert set(predictions).issubset({0, 1})


@pytest.mark.unit
def test_xgboost_strategy_handles_large_n_estimators() -> None:
    """Verify XGBoost works with large n_estimators."""
    # Arrange
    np.random.seed(42)
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, 100)

    config = {"n_estimators": 500, "random_state": 42}
    strategy = XGBoostStrategy(config)

    # Act
    strategy.fit(X_train, y_train)
    predictions = strategy.predict(X_train)

    # Assert: Should work (just slower)
    assert predictions.shape == (100,)
    assert set(predictions).issubset({0, 1})


# ============================================================================
# JSON Serialization Tests
# ============================================================================


@pytest.mark.unit
def test_xgboost_strategy_json_serializable() -> None:
    """Verify to_dict() output is JSON-serializable."""
    # Arrange
    config = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.3,
        "random_state": 42,
    }
    strategy = XGBoostStrategy(config)

    # Act
    config_dict = strategy.to_dict()
    json_str = json.dumps(config_dict)  # Should not raise
    loaded_config = json.loads(json_str)

    # Assert
    assert loaded_config["type"] == "xgboost"
    assert loaded_config["n_estimators"] == 100
    assert loaded_config["max_depth"] == 6
