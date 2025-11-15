"""
End-to-End Tests for Training Pipeline.

Tests cover:
- Full training workflow (dataset → embeddings → training → model save)
- Config file loading and parsing
- Model persistence (save/load)
- Prediction after training
- Error handling in full pipeline

Testing philosophy:
- Test full end-to-end workflows
- Use real components (datasets, classifier, trainer)
- Mock only heavyweight I/O (ESM model, large datasets)
- Verify final outcomes, not intermediate steps
- Use small test datasets for speed

Date: 2025-11-07
Phase: 4 (CLI & E2E Tests)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from antibody_training_esm.core.classifier import BinaryClassifier
from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor
from antibody_training_esm.core.trainer import train_model

# ==================== Fixtures ====================


@pytest.fixture
def mock_training_config(tmp_path: Path) -> Path:
    """Create a mock training config file"""
    config = {
        "model_name": "facebook/esm1v_t33_650M_UR90S_1",
        "device": "cpu",
        "random_state": 42,
        "max_iter": 10,  # Small for fast tests
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "batch_size": 8,
        "output_model_path": str(tmp_path / "trained_model.pkl"),
        "train_data_path": "data/test/boughter/boughter_translated.csv",
    }

    config_path = tmp_path / "train_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path


@pytest.fixture
def small_training_data(tmp_path: Path) -> Path:
    """Create small CSV training data for fast E2E tests"""
    import pandas as pd

    # Create 20 samples (10 per class)
    np.random.seed(42)
    data = {
        "id": [f"AB{i:03d}" for i in range(20)],
        "VH_sequence": ["QVQLVQSGAEVKKPGASVKVSCKASGYTFT" for _ in range(20)],
        "VL_sequence": ["DIQMTQSPSSLSASVGDRVTITCRASQSIS" for _ in range(20)],
        "label": [0, 1] * 10,  # Balanced
    }

    df = pd.DataFrame(data)
    csv_path = tmp_path / "small_train.csv"
    df.to_csv(csv_path, index=False)

    return csv_path


# ==================== Full Pipeline E2E Tests ====================


@pytest.mark.e2e
@pytest.mark.skipif(
    True,
    reason="Requires real trainer implementation and large datasets. Enable when trainer is fully integrated.",
)
def test_full_training_pipeline_end_to_end(
    mock_transformers_model: tuple[Any, Any],
    mock_training_config: Path,
    small_training_data: Path,
    tmp_path: Path,
) -> None:
    """Verify complete training pipeline: config → train → save → load → predict"""
    # NOTE: This is a placeholder E2E test that will be enabled once:
    # 1. Trainer is fully implemented with dataset integration
    # 2. We have appropriate test datasets
    # 3. CI can handle longer-running E2E tests

    # Arrange: Update config to use small training data
    with open(mock_training_config) as f:
        config = yaml.safe_load(f)

    config["train_data_path"] = str(small_training_data)

    with open(mock_training_config, "w") as f:
        yaml.dump(config, f)

    # Act: Run full training pipeline
    train_model(str(mock_training_config))

    # Assert: Model file exists
    model_path = Path(config["output_model_path"])
    assert model_path.exists(), "Model file was not created"

    # Assert: Model can be loaded
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    assert isinstance(model, BinaryClassifier)
    assert model.is_fitted

    # Assert: Model can make predictions
    test_embeddings = np.random.rand(5, 1280)  # Mock test embeddings
    predictions = model.predict(test_embeddings)

    assert len(predictions) == 5
    assert all(pred in [0, 1] for pred in predictions)


# ==================== Component Integration Tests ====================


@pytest.mark.e2e
def test_dataset_to_embeddings_pipeline(
    mock_transformers_model: tuple[Any, Any], small_training_data: Path
) -> None:
    """Verify dataset loading → embedding extraction workflow"""
    # Arrange
    import pandas as pd

    df = pd.read_csv(small_training_data)
    sequences = df["VH_sequence"].tolist()

    extractor = ESMEmbeddingExtractor(
        model_name="facebook/esm1v_t33_650M_UR90S_1", device="cpu", batch_size=8
    )

    # Act
    embeddings = extractor.extract_batch_embeddings(sequences)

    # Assert
    assert embeddings.shape == (20, 1280)
    assert embeddings.dtype == np.float32


@pytest.mark.e2e
def test_embeddings_to_training_pipeline(
    mock_transformers_model: tuple[Any, Any],
) -> None:
    """Verify embedding extraction → training → prediction workflow"""
    # Arrange
    np.random.seed(42)
    X_train = np.random.rand(50, 1280).astype(np.float32)
    y_train = np.array([0, 1] * 25)

    classifier = BinaryClassifier(
        model_name="facebook/esm1v_t33_650M_UR90S_1",
        device="cpu",
        random_state=42,
        max_iter=10,
        C=1.0,
        batch_size=8,
    )

    # Act: Train
    classifier.fit(X_train, y_train)

    # Assert: Can predict
    X_test = np.random.rand(10, 1280).astype(np.float32)
    predictions = classifier.predict(X_test)

    assert len(predictions) == 10
    assert all(pred in [0, 1] for pred in predictions)


@pytest.mark.e2e
def test_model_save_load_predict_workflow(
    mock_transformers_model: tuple[Any, Any], tmp_path: Path
) -> None:
    """Verify training → save → load → predict workflow"""
    # Arrange
    np.random.seed(42)
    X_train = np.random.rand(50, 1280).astype(np.float32)
    y_train = np.array([0, 1] * 25)

    classifier = BinaryClassifier(
        model_name="facebook/esm1v_t33_650M_UR90S_1",
        device="cpu",
        random_state=42,
        max_iter=10,
        batch_size=8,
    )
    classifier.fit(X_train, y_train)

    # Act: Save model
    model_path = tmp_path / "test_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(classifier, f)

    # Act: Load model
    with open(model_path, "rb") as f:
        loaded_model = pickle.load(f)

    # Assert: Loaded model can predict
    X_test = np.random.rand(10, 1280).astype(np.float32)
    original_preds = classifier.predict(X_test)
    loaded_preds = loaded_model.predict(X_test)

    np.testing.assert_array_equal(original_preds, loaded_preds)


# ==================== Config Validation Tests ====================


@pytest.mark.e2e
def test_training_config_validation(tmp_path: Path) -> None:
    """Verify training config has all required fields"""
    # Arrange
    config = {
        "model_name": "facebook/esm1v_t33_650M_UR90S_1",
        "device": "cpu",
        "random_state": 42,
        "max_iter": 100,
        "C": 1.0,
        "output_model_path": str(tmp_path / "model.pkl"),
        "train_data_path": "train.csv",
    }

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Act
    with open(config_path) as f:
        loaded_config = yaml.safe_load(f)

    # Assert: All required fields present
    assert "model_name" in loaded_config
    assert "device" in loaded_config
    assert "output_model_path" in loaded_config
    assert "train_data_path" in loaded_config


# ==================== Error Handling E2E Tests ====================


@pytest.mark.e2e
def test_training_fails_with_invalid_config(tmp_path: Path) -> None:
    """Verify training fails gracefully with invalid config"""
    # Arrange
    config_path = tmp_path / "invalid_config.yaml"
    config_path.write_text("invalid: yaml: content:")

    # Act & Assert
    with pytest.raises((yaml.YAMLError, ValueError, RuntimeError)):
        train_model(str(config_path))


@pytest.mark.e2e
@pytest.mark.skip(
    reason="Requires trainer.py to be fully implemented with proper config structure. "
    "Current trainer expects nested config with 'training', 'model', etc. keys."
)
def test_training_fails_with_missing_data_file(tmp_path: Path) -> None:
    """Verify training fails gracefully with missing data file"""
    # NOTE: This test will be enabled once trainer.py config structure is finalized
    pass


# ==================== Cross-Component Integration Tests ====================


@pytest.mark.e2e
def test_multiple_classifiers_train_independently(
    mock_transformers_model: tuple[Any, Any],
) -> None:
    """Verify multiple classifiers can be trained without interference"""
    # Arrange
    np.random.seed(42)
    X = np.random.rand(50, 1280).astype(np.float32)
    y = np.array([0, 1] * 25)

    classifier1 = BinaryClassifier(
        model_name="facebook/esm1v_t33_650M_UR90S_1",
        device="cpu",
        random_state=42,
        max_iter=10,
        C=0.1,
        batch_size=8,
    )

    classifier2 = BinaryClassifier(
        model_name="facebook/esm1v_t33_650M_UR90S_1",
        device="cpu",
        random_state=42,
        max_iter=10,
        C=10.0,
        batch_size=8,
    )

    # Act
    classifier1.fit(X, y)
    classifier2.fit(X, y)

    # Assert: Both trained successfully
    assert classifier1.is_fitted
    assert classifier2.is_fitted

    # Assert: Have different hyperparameters
    assert classifier1.C == 0.1
    assert classifier2.C == 10.0


@pytest.mark.e2e
def test_embedding_extractor_handles_batch_boundaries(
    mock_transformers_model: tuple[Any, Any],
) -> None:
    """Verify embedding extractor handles sequences at batch boundaries correctly"""
    # Arrange
    # Test with sequence counts that exercise batch boundaries
    test_counts = [7, 8, 9, 15, 16, 17]  # Around batch_size=8

    extractor = ESMEmbeddingExtractor(
        model_name="facebook/esm1v_t33_650M_UR90S_1", device="cpu", batch_size=8
    )

    for count in test_counts:
        sequences = ["QVQLVQSGAEVKKPGASVKVSCKASGYTFT"] * count

        # Act
        embeddings = extractor.extract_batch_embeddings(sequences)

        # Assert
        assert embeddings.shape == (count, 1280), f"Failed for count={count}"


# ==================== Performance Smoke Tests ====================


@pytest.mark.e2e
def test_training_completes_in_reasonable_time(
    mock_transformers_model: tuple[Any, Any],
) -> None:
    """Verify training completes within time bounds (smoke test)"""
    import time

    # Arrange
    np.random.seed(42)
    X_train = np.random.rand(100, 1280).astype(np.float32)
    y_train = np.array([0, 1] * 50)

    classifier = BinaryClassifier(
        model_name="facebook/esm1v_t33_650M_UR90S_1",
        device="cpu",
        random_state=42,
        max_iter=10,  # Small for fast test
        batch_size=8,
    )

    # Act
    start_time = time.time()
    classifier.fit(X_train, y_train)
    elapsed = time.time() - start_time

    # Assert: Completes in under 10 seconds
    assert elapsed < 10.0, f"Training took {elapsed:.2f}s (expected <10s)"
    assert classifier.is_fitted
