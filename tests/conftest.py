#!/usr/bin/env python3
"""
Pytest Configuration and Shared Fixtures

Provides shared fixtures, test data, and utilities for the entire test suite.
All fixtures are available to all tests without explicit imports.

Philosophy:
- Mock only I/O boundaries (model loading, file system)
- Test behaviors, not implementation details
- DRY: Extract common test data here

Date: 2025-11-07
Author: Claude Code
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

# CRITICAL: Import conf package to trigger ConfigStore registration
# This MUST happen before any Hydra tests call initialize()
import antibody_training_esm.conf  # noqa: F401

# NOTE: sys.path injection removed - uv handles package installation
# If running tests without uv, use: uv run pytest
# sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from tests.fixtures.mock_models import (
    MockClassifier,
    MockESMModel,
    MockTokenizer,
    create_mock_embeddings,
    create_mock_labels,
)
from tests.fixtures.mock_sequences import (
    EMPTY_SEQUENCE,
    LONG_VH,
    MIXED_SEQUENCE_BATCH,
    SEQUENCE_WITH_GAP,
    SEQUENCE_WITH_INVALID_AA,
    SHORT_VH,
    VALID_SEQUENCE_BATCH,
    VALID_VH,
    VALID_VL,
    create_mock_dataframe,
)

# ============================================================================
# ESM Model Fixtures (Mock Heavy Dependencies)
# ============================================================================


@pytest.fixture
def mock_transformers_model(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, Any]:
    """
    Mock Hugging Face transformers ESM-1v model and tokenizer.

    Avoids downloading 650MB model weights during testing.
    Returns mock outputs with correct shape (1280-d embeddings).

    Usage:
        def test_something(mock_transformers_model):
            extractor = ESMEmbeddingExtractor(...)
            embedding = extractor.embed_sequence("QVQLVQSG")
            assert embedding.shape == (1280,)

    Mocked methods:
        - transformers.AutoModel.from_pretrained()
        - transformers.AutoTokenizer.from_pretrained()
    """

    # Mock model
    def mock_automodel(*args: Any, **kwargs: Any) -> MockESMModel:
        return MockESMModel(*args, **kwargs)

    # Mock tokenizer
    def mock_autotokenizer(*args: Any, **kwargs: Any) -> MockTokenizer:
        return MockTokenizer(*args, **kwargs)

    monkeypatch.setattr("transformers.AutoModel.from_pretrained", mock_automodel)
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", mock_autotokenizer
    )

    return mock_automodel, mock_autotokenizer


@pytest.fixture
def mock_sklearn_classifier(monkeypatch: pytest.MonkeyPatch) -> Any:
    """
    Mock sklearn LogisticRegression for deterministic testing.

    Provides fixed predictions for reproducible tests.
    Only use when testing classifier integration, not classifier behavior.

    Usage:
        def test_something(mock_sklearn_classifier):
            classifier = BinaryClassifier(...)
            classifier.fit(X, y)
            predictions = classifier.predict(X)
            # Predictions are deterministic: [0, 1, 0, 1, ...]
    """

    def mock_logistic_regression(*args: Any, **kwargs: Any) -> MockClassifier:
        return MockClassifier(*args, **kwargs)

    monkeypatch.setattr(
        "sklearn.linear_model.LogisticRegression", mock_logistic_regression
    )

    return mock_logistic_regression


# ============================================================================
# Test Data Fixtures
# ============================================================================


@pytest.fixture
def valid_sequences() -> dict[str, Any]:
    """
    Valid antibody sequences for testing.

    Returns:
        dict with VH, VL, short, and long sequences
    """
    return {
        "VH": VALID_VH,
        "VL": VALID_VL,
        "short": SHORT_VH,
        "long": LONG_VH,
        "batch": VALID_SEQUENCE_BATCH,
    }


@pytest.fixture
def invalid_sequences() -> dict[str, Any]:
    """
    Invalid antibody sequences for error testing.

    Returns:
        dict with various invalid sequences (gaps, invalid chars, empty)
    """
    return {
        "gap": SEQUENCE_WITH_GAP,
        "invalid_aa": SEQUENCE_WITH_INVALID_AA,
        "empty": EMPTY_SEQUENCE,
        "mixed_batch": MIXED_SEQUENCE_BATCH,
    }


@pytest.fixture
def mock_embeddings() -> dict[str, np.ndarray]:
    """
    Mock ESM-1v embeddings for testing classifier.

    Returns:
        dict with train/test splits and labels
    """
    X_train = create_mock_embeddings(n_samples=100, seed=42)
    y_train = create_mock_labels(n_samples=100, balanced=True)

    X_test = create_mock_embeddings(n_samples=20, seed=43)
    y_test = create_mock_labels(n_samples=20, balanced=True)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }


@pytest.fixture
def mock_dataset_small() -> pd.DataFrame:
    """
    Small mock dataset (10 samples, balanced).

    Returns:
        pandas.DataFrame with columns: id, VH_sequence, VL_sequence, label
    """
    return create_mock_dataframe(n_samples=10, balanced=True)


@pytest.fixture
def mock_dataset_large() -> pd.DataFrame:
    """
    Larger mock dataset (100 samples, balanced).

    Returns:
        pandas.DataFrame with columns: id, VH_sequence, VL_sequence, label
    """
    return create_mock_dataframe(n_samples=100, balanced=True)


@pytest.fixture
def mock_dataset_imbalanced() -> pd.DataFrame:
    """
    Imbalanced mock dataset (10 samples, all class 0).

    Returns:
        pandas.DataFrame with columns: id, VH_sequence, VL_sequence, label
    """
    return create_mock_dataframe(n_samples=10, balanced=False)


# ============================================================================
# File System Fixtures
# ============================================================================


@pytest.fixture
def fixture_dir() -> Path:
    """
    Path to test fixtures directory.

    Returns:
        pathlib.Path to tests/fixtures/
    """
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def mock_dataset_dir(fixture_dir: Path) -> Path:
    """
    Path to mock dataset CSV files.

    Returns:
        pathlib.Path to tests/fixtures/mock_datasets/
    """
    return fixture_dir / "mock_datasets"


@pytest.fixture
def jain_sample_csv(mock_dataset_dir: Path) -> Path:
    """
    Path to mock Jain dataset CSV.

    Returns:
        pathlib.Path to tests/fixtures/mock_datasets/jain_sample.csv
    """
    csv_path = mock_dataset_dir / "jain_sample.csv"
    assert csv_path.exists(), f"Mock Jain CSV not found: {csv_path}"
    return csv_path


@pytest.fixture
def boughter_sample_csv(mock_dataset_dir: Path) -> Path:
    """
    Path to mock Boughter dataset CSV.

    Returns:
        pathlib.Path to tests/fixtures/mock_datasets/boughter_sample.csv
    """
    csv_path = mock_dataset_dir / "boughter_sample.csv"
    assert csv_path.exists(), f"Mock Boughter CSV not found: {csv_path}"
    return csv_path


# ============================================================================
# Classifier Test Fixtures
# ============================================================================


@pytest.fixture
def default_classifier_params() -> dict[str, Any]:
    """
    Default hyperparameters for BinaryClassifier.

    Returns:
        dict with sklearn LogisticRegression params
    """
    return {
        "model_name": "facebook/esm1v_t33_650M_UR90S_1",
        "device": "cpu",
        "penalty": "l2",
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": 42,
    }


@pytest.fixture
def fitted_classifier(
    mock_embeddings: dict[str, np.ndarray],
    default_classifier_params: dict[str, Any],
    mock_transformers_model: tuple[Any, Any],
) -> Any:
    """
    Pre-fitted BinaryClassifier for testing predictions.

    Returns:
        Fitted BinaryClassifier instance
    """
    from antibody_training_esm.core.classifier import BinaryClassifier

    classifier = BinaryClassifier(params=default_classifier_params)
    classifier.fit(mock_embeddings["X_train"], mock_embeddings["y_train"])

    return classifier


# ============================================================================
# Embedding Extractor Fixtures
# ============================================================================


@pytest.fixture
def embedding_extractor(mock_transformers_model: tuple[Any, Any]) -> Any:
    """
    ESMEmbeddingExtractor with mocked model (no download).

    Returns:
        ESMEmbeddingExtractor instance
    """
    from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor

    return ESMEmbeddingExtractor(
        model_name="facebook/esm1v_t33_650M_UR90S_1", device="cpu", batch_size=8
    )


# ============================================================================
# Utility Functions
# ============================================================================


def assert_valid_predictions(predictions: np.ndarray, expected_len: int) -> None:
    """
    Assert that predictions are valid binary labels.

    Args:
        predictions: Predicted labels
        expected_len: Expected number of predictions

    Raises:
        AssertionError: If predictions are invalid
    """
    assert len(predictions) == expected_len
    assert all(pred in [0, 1] for pred in predictions)
    assert predictions.dtype in [np.int32, np.int64, int]


def assert_valid_embeddings(embeddings: np.ndarray, expected_shape: tuple) -> None:
    """
    Assert that embeddings have correct shape and dtype.

    Args:
        embeddings: ESM-1v embeddings
        expected_shape: Expected shape (n_samples, 1280)

    Raises:
        AssertionError: If embeddings are invalid
    """
    assert embeddings.shape == expected_shape
    assert embeddings.dtype in [np.float32, np.float64]
    assert not np.isnan(embeddings).any()
    assert not np.isinf(embeddings).any()


def assert_valid_dataframe(
    df: pd.DataFrame, required_columns: list[str], min_rows: int = 1
) -> None:
    """
    Assert that DataFrame has required columns and minimum rows.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows

    Raises:
        AssertionError: If DataFrame is invalid
    """
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= min_rows
    for col in required_columns:
        assert col in df.columns, f"Missing column: {col}"


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """
    Configure pytest markers.

    Markers:
        - unit: Unit tests (fast, no I/O)
        - integration: Integration tests (medium speed, some I/O)
        - e2e: End-to-end tests (slow, full pipeline)
        - slow: Tests that take >1s to run
        - gpu: Tests that require GPU (skip in CI)
    """
    config.addinivalue_line("markers", "unit: Unit tests (fast, no I/O)")
    config.addinivalue_line(
        "markers", "integration: Integration tests (medium speed, some I/O)"
    )
    config.addinivalue_line("markers", "e2e: End-to-end tests (slow, full pipeline)")
    config.addinivalue_line("markers", "slow: Tests that take >1s to run")
    config.addinivalue_line("markers", "gpu: Tests that require GPU (skip in CI)")
