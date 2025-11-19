#!/usr/bin/env python3
"""
Integration Test: XGBoost Strategy via Factory

Verifies that XGBoostStrategy can be created via create_classifier factory
and works end-to-end with the BinaryClassifier.

Date: 2025-11-18
"""

from __future__ import annotations

import numpy as np
import pytest

from antibody_training_esm.core.classifier_factory import create_classifier
from antibody_training_esm.core.strategies.xgboost_strategy import XGBoostStrategy


@pytest.mark.integration
def test_factory_creates_xgboost_strategy() -> None:
    """Verify factory creates XGBoostStrategy when type=xgboost."""
    # Arrange
    config = {
        "type": "xgboost",
        "n_estimators": 50,
        "random_state": 42,
    }

    # Act
    strategy = create_classifier(config)

    # Assert
    assert isinstance(strategy, XGBoostStrategy)
    assert strategy.n_estimators == 50
    assert strategy.random_state == 42


@pytest.mark.integration
def test_xgboost_strategy_end_to_end() -> None:
    """Verify XGBoostStrategy works end-to-end: fit → predict → score."""
    # Arrange: Create synthetic dataset
    np.random.seed(42)
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.rand(20, 10)
    y_test = np.random.randint(0, 2, 20)

    config = {
        "type": "xgboost",
        "n_estimators": 50,
        "random_state": 42,
    }

    # Act
    strategy = create_classifier(config)
    strategy.fit(X_train, y_train)
    predictions = strategy.predict(X_test)
    probabilities = strategy.predict_proba(X_test)
    accuracy = strategy.score(X_test, y_test)

    # Assert
    assert predictions.shape == (20,)
    assert probabilities.shape == (20, 2)
    assert 0.0 <= accuracy <= 1.0
    assert set(predictions).issubset({0, 1})
    assert np.allclose(probabilities.sum(axis=1), 1.0)


@pytest.mark.integration
def test_xgboost_hydra_config_format() -> None:
    """Verify XGBoost config follows Hydra format (no magic numbers)."""
    # Arrange: Simulate Hydra config loading
    hydra_config = {
        "type": "xgboost",
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.3,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "random_state": 42,  # Would be ${training.random_state} in YAML
        "objective": "binary:logistic",
    }

    # Act
    strategy = create_classifier(hydra_config)

    # Assert: All hyperparameters correctly loaded
    assert isinstance(strategy, XGBoostStrategy)
    assert strategy.n_estimators == 100
    assert strategy.max_depth == 6
    assert strategy.learning_rate == 0.3
    assert strategy.subsample == 1.0
    assert strategy.colsample_bytree == 1.0
    assert strategy.reg_alpha == 0.0
    assert strategy.reg_lambda == 1.0
    assert strategy.random_state == 42
    assert strategy.objective == "binary:logistic"
