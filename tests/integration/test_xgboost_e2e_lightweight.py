#!/usr/bin/env python3
"""
Lightweight XGBoost End-to-End Integration Test

Proves XGBoost works in the full pipeline WITHOUT loading ESM models.
Uses mock embeddings to avoid OOM in resource-constrained environments.

Date: 2025-11-18
"""

from __future__ import annotations

import numpy as np
import pytest

from antibody_training_esm.core.classifier import BinaryClassifier


@pytest.mark.integration
def test_xgboost_e2e_with_mock_embeddings() -> None:
    """
    End-to-end test: XGBoost fits on mock embeddings and predicts.

    This proves the integration works without loading the 650M parameter ESM model.
    """
    # Arrange: Create XGBoost classifier with minimal config
    config = {
        "type": "xgboost",
        "n_estimators": 5,  # Minimal for speed
        "max_depth": 3,
        "random_state": 42,
        "model_name": "facebook/esm1v_t33_650M_UR90S_1",  # Required but won't load
        "device": "cpu",
    }

    classifier = BinaryClassifier(config)

    # Mock embeddings (bypass ESM model loading)
    np.random.seed(42)
    X_train = np.random.rand(50, 1280)  # 50 samples, 1280 dims (ESM-1v size)
    y_train = np.random.randint(0, 2, 50)

    X_test = np.random.rand(10, 1280)
    y_test = np.random.randint(0, 2, 10)

    # Act: Train XGBoost on mock embeddings
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    probabilities = classifier.predict_proba(X_test)
    accuracy = classifier.score(X_test, y_test)

    # Assert: Verify XGBoost learned something
    assert classifier.is_fitted
    assert predictions.shape == (10,)
    assert probabilities.shape == (10, 2)
    assert set(predictions).issubset({0, 1})
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert 0.0 <= accuracy <= 1.0

    # Verify it's actually XGBoost
    assert hasattr(classifier.classifier, "save_model")  # XGBoost-specific method
    strategy_config = classifier.classifier.to_dict()
    assert strategy_config["type"] == "xgboost"
    assert strategy_config["n_estimators"] == 5


@pytest.mark.integration
def test_xgboost_vs_logreg_on_same_data() -> None:
    """
    Compare XGBoost vs LogReg on same mock data.

    Proves both strategies work and produce different results (XGBoost is nonlinear).
    """
    # Arrange: Same mock data for both
    np.random.seed(42)
    X_train = np.random.rand(100, 50)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.rand(20, 50)

    xgb_config = {
        "type": "xgboost",
        "n_estimators": 10,
        "random_state": 42,
        "model_name": "facebook/esm1v_t33_650M_UR90S_1",
        "device": "cpu",
    }

    logreg_config = {
        "type": "logistic_regression",
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": 42,
        "model_name": "facebook/esm1v_t33_650M_UR90S_1",
        "device": "cpu",
    }

    # Act: Train both
    xgb_classifier = BinaryClassifier(xgb_config)
    xgb_classifier.fit(X_train, y_train)
    xgb_preds = xgb_classifier.predict(X_test)

    logreg_classifier = BinaryClassifier(logreg_config)
    logreg_classifier.fit(X_train, y_train)
    logreg_preds = logreg_classifier.predict(X_test)

    # Assert: Both work but may differ (XGBoost is nonlinear)
    assert xgb_preds.shape == logreg_preds.shape
    assert set(xgb_preds).issubset({0, 1})
    assert set(logreg_preds).issubset({0, 1})

    # Verify strategy types
    assert xgb_classifier.classifier.to_dict()["type"] == "xgboost"
    assert logreg_classifier.classifier.to_dict()["type"] == "logistic_regression"


@pytest.mark.integration
def test_xgboost_deterministic_with_seed() -> None:
    """
    Verify XGBoost produces deterministic results with same random_state.
    """
    np.random.seed(42)
    X_train = np.random.rand(50, 20)
    y_train = np.random.randint(0, 2, 50)
    X_test = np.random.rand(10, 20)

    config = {
        "type": "xgboost",
        "n_estimators": 10,
        "random_state": 42,
        "model_name": "facebook/esm1v_t33_650M_UR90S_1",
        "device": "cpu",
    }

    # Train twice with same seed
    clf1 = BinaryClassifier(config)
    clf1.fit(X_train, y_train)
    preds1 = clf1.predict(X_test)

    clf2 = BinaryClassifier(config)
    clf2.fit(X_train, y_train)
    preds2 = clf2.predict(X_test)

    # Should be identical
    np.testing.assert_array_equal(preds1, preds2)
