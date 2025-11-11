"""
Unit Tests for Trainer Module

Tests training pipeline components, cross-validation, model evaluation.
Focus: Test behaviors, not implementation details. Mock only I/O boundaries.

Date: 2025-11-07
Phase: 5 Task 3 (Trainer Coverage Gap Closure)
"""

from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pytest
import yaml

from antibody_training_esm.core.trainer import (
    evaluate_model,
    get_or_create_embeddings,
    load_config,
    perform_cross_validation,
    save_model,
    setup_logging,
    train_model,
)

# ==================== Fixtures ====================


@pytest.fixture
def nested_config(tmp_path: Path) -> dict[str, Any]:
    """Proper nested config structure that trainer.py expects"""
    config = {
        "training": {
            "log_level": "INFO",
            "log_file": str(tmp_path / "train.log"),
            "save_model": True,
            "model_name": "test_model",
            "model_save_dir": str(tmp_path / "models"),
            "batch_size": 8,
            "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"],
        },
        "classifier": {
            "C": 1.0,
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 100,
            "random_state": 42,
            "cv_folds": 3,
            "stratify": True,
        },
        "model": {
            "name": "facebook/esm1v_t33_650M_UR90S_1",
            "device": "cpu",
        },
        "data": {
            "source": "local",  # Required by load_data()
            "embeddings_cache_dir": str(tmp_path / "cache"),
            "train_data_path": str(tmp_path / "train.csv"),
            "train_file": str(tmp_path / "train.csv"),
            "sequence_column": "VH_sequence",
            "label_column": "label",
        },
    }
    return config


@pytest.fixture
def config_yaml_path(nested_config: dict[str, Any], tmp_path: Path) -> str:
    """Write config to YAML file"""
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(nested_config, f)
    return str(config_path)


@pytest.fixture
def mock_embeddings() -> np.ndarray:
    """Mock ESM embeddings (1280-dimensional)"""
    return np.random.rand(20, 1280).astype(np.float32)


@pytest.fixture
def mock_labels() -> np.ndarray:
    """Mock binary labels"""
    return np.array([0, 1] * 10)


# ==================== setup_logging Tests ====================


def test_setup_logging_creates_logger(
    nested_config: dict[str, Any], tmp_path: Path
) -> None:
    """Verify setup_logging creates logger with correct config"""
    # Act
    logger = setup_logging(nested_config)

    # Assert
    assert logger is not None
    assert logger.name == "antibody_training_esm.core.trainer"
    # Log file should be created (proves logging is configured)
    assert (tmp_path / "train.log").exists()


def test_setup_logging_creates_log_directory(tmp_path: Path) -> None:
    """Verify setup_logging creates log directory if missing"""
    # Arrange
    nested_log_dir = tmp_path / "logs" / "nested" / "path"
    config = {
        "training": {
            "log_level": "DEBUG",
            "log_file": str(nested_log_dir / "train.log"),
        }
    }

    # Act
    setup_logging(config)

    # Assert: Log directory and file should be created
    assert nested_log_dir.exists()
    assert (nested_log_dir / "train.log").exists()


# ==================== load_config Tests ====================


def test_load_config_loads_yaml_file(
    config_yaml_path: str, nested_config: dict[str, Any]
) -> None:
    """Verify load_config loads YAML file correctly"""
    # Act
    loaded_config = load_config(config_yaml_path)

    # Assert
    assert loaded_config == nested_config
    assert "training" in loaded_config
    assert "classifier" in loaded_config
    assert "model" in loaded_config
    assert "data" in loaded_config


def test_load_config_raises_on_missing_file() -> None:
    """Verify load_config raises FileNotFoundError for missing file"""
    # Act & Assert
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_config.yaml")


def test_load_config_raises_on_invalid_yaml(tmp_path: Path) -> None:
    """Verify load_config raises error for invalid YAML"""
    # Arrange
    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("invalid: yaml: content: [unclosed")

    # Act & Assert
    with pytest.raises(yaml.YAMLError):
        load_config(str(invalid_yaml))


# ==================== get_or_create_embeddings Tests ====================


def test_get_or_create_embeddings_creates_new_embeddings(
    tmp_path: Path, mock_embeddings: np.ndarray
) -> None:
    """Verify embeddings are created when cache doesn't exist"""
    # Arrange
    sequences = ["QVQLVQSG"] * 20
    cache_path = str(tmp_path / "cache")
    mock_extractor = Mock()
    mock_extractor.extract_batch_embeddings.return_value = mock_embeddings
    mock_logger = Mock()

    # Act
    embeddings = get_or_create_embeddings(
        sequences, mock_extractor, cache_path, "test_dataset", mock_logger
    )

    # Assert
    assert np.array_equal(embeddings, mock_embeddings)
    mock_extractor.extract_batch_embeddings.assert_called_once_with(sequences)
    # Cache file should be created
    sequences_str = "|".join(sequences)
    sequences_hash = hashlib.sha256(sequences_str.encode()).hexdigest()[
        :12
    ]  # Changed to SHA-256 (12 chars)
    cache_file = Path(cache_path) / f"test_dataset_{sequences_hash}_embeddings.pkl"
    assert cache_file.exists()


def test_get_or_create_embeddings_loads_from_cache(
    tmp_path: Path, mock_embeddings: np.ndarray
) -> None:
    """Verify embeddings are loaded from cache when available"""
    # Arrange
    sequences = ["QVQLVQSG"] * 20
    cache_path = str(tmp_path / "cache")
    os.makedirs(cache_path)

    # Create cached embeddings
    sequences_str = "|".join(sequences)
    sequences_hash = hashlib.sha256(sequences_str.encode()).hexdigest()[
        :12
    ]  # Changed to SHA-256 (12 chars)
    cache_file = Path(cache_path) / f"test_dataset_{sequences_hash}_embeddings.pkl"

    cache_data = {
        "embeddings": mock_embeddings,
        "sequences_hash": sequences_hash,
        "num_sequences": len(sequences),
        "dataset_name": "test_dataset",
    }
    with open(cache_file, "wb") as f:
        pickle.dump(cache_data, f)

    mock_extractor = Mock()
    mock_logger = Mock()

    # Act
    embeddings = get_or_create_embeddings(
        sequences, mock_extractor, cache_path, "test_dataset", mock_logger
    )

    # Assert
    assert np.array_equal(embeddings, mock_embeddings)
    # Extractor should NOT be called (loaded from cache)
    mock_extractor.extract_batch_embeddings.assert_not_called()
    mock_logger.info.assert_any_call(f"Loading cached embeddings from {cache_file}")


def test_get_or_create_embeddings_recomputes_on_hash_mismatch(
    tmp_path: Path, mock_embeddings: np.ndarray
) -> None:
    """Verify embeddings are recomputed if cache hash doesn't match"""
    # Arrange
    sequences = ["QVQLVQSG"] * 20
    cache_path = str(tmp_path / "cache")
    os.makedirs(cache_path)

    # Create cached embeddings with WRONG hash
    sequences_str = "|".join(sequences)
    sequences_hash = hashlib.sha256(sequences_str.encode()).hexdigest()[
        :12
    ]  # Changed to SHA-256 (12 chars)
    cache_file = Path(cache_path) / f"test_dataset_{sequences_hash}_embeddings.pkl"

    wrong_cache_data = {
        "embeddings": mock_embeddings,
        "sequences_hash": "WRONGHASH",  # Intentionally wrong
        "num_sequences": len(sequences),
        "dataset_name": "test_dataset",
    }
    with open(cache_file, "wb") as f:
        pickle.dump(wrong_cache_data, f)

    mock_extractor = Mock()
    mock_extractor.extract_batch_embeddings.return_value = mock_embeddings
    mock_logger = Mock()

    # Act
    embeddings = get_or_create_embeddings(
        sequences, mock_extractor, cache_path, "test_dataset", mock_logger
    )

    # Assert
    assert np.array_equal(embeddings, mock_embeddings)
    # Extractor SHOULD be called (cache mismatch)
    mock_extractor.extract_batch_embeddings.assert_called_once_with(sequences)
    mock_logger.warning.assert_called_once_with(
        "Cached embeddings hash mismatch, recomputing..."
    )


# ==================== evaluate_model Tests ====================


def test_evaluate_model_computes_all_metrics(
    mock_embeddings: np.ndarray, mock_labels: np.ndarray
) -> None:
    """Verify evaluate_model computes all requested metrics"""
    # Arrange
    from antibody_training_esm.core.classifier import BinaryClassifier

    classifier = BinaryClassifier(
        model_name="facebook/esm1v_t33_650M_UR90S_1",
        device="cpu",
        random_state=42,
        max_iter=10,
        batch_size=8,
    )
    classifier.fit(mock_embeddings, mock_labels)

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    mock_logger = Mock()

    # Act
    results = evaluate_model(
        classifier, mock_embeddings, mock_labels, "Test", metrics, mock_logger
    )

    # Assert
    assert "accuracy" in results
    assert "precision" in results
    assert "recall" in results
    assert "f1" in results
    assert "roc_auc" in results
    # All metrics should be floats between 0 and 1
    for _metric, value in results.items():
        assert isinstance(value, (float, np.floating))
        assert 0.0 <= value <= 1.0


def test_evaluate_model_computes_subset_of_metrics(
    mock_embeddings: np.ndarray, mock_labels: np.ndarray
) -> None:
    """Verify evaluate_model only computes requested metrics"""
    # Arrange
    from antibody_training_esm.core.classifier import BinaryClassifier

    classifier = BinaryClassifier(
        model_name="facebook/esm1v_t33_650M_UR90S_1",
        device="cpu",
        random_state=42,
        max_iter=10,
        batch_size=8,
    )
    classifier.fit(mock_embeddings, mock_labels)

    metrics = ["accuracy", "f1"]  # Only request 2 metrics
    mock_logger = Mock()

    # Act
    results = evaluate_model(
        classifier, mock_embeddings, mock_labels, "Test", metrics, mock_logger
    )

    # Assert
    assert "accuracy" in results
    assert "f1" in results
    assert "precision" not in results  # Not requested
    assert "recall" not in results  # Not requested
    assert "roc_auc" not in results  # Not requested


def test_evaluate_model_logs_results(
    mock_embeddings: np.ndarray, mock_labels: np.ndarray
) -> None:
    """Verify evaluate_model logs results to logger"""
    # Arrange
    from antibody_training_esm.core.classifier import BinaryClassifier

    classifier = BinaryClassifier(
        model_name="facebook/esm1v_t33_650M_UR90S_1",
        device="cpu",
        random_state=42,
        max_iter=10,
        batch_size=8,
    )
    classifier.fit(mock_embeddings, mock_labels)

    metrics = ["accuracy"]
    mock_logger = Mock()

    # Act
    evaluate_model(
        classifier, mock_embeddings, mock_labels, "Test", metrics, mock_logger
    )

    # Assert
    # Logger should be called with results
    assert mock_logger.info.call_count >= 2  # At least dataset name + metric result
    log_messages = [call[0][0] for call in mock_logger.info.call_args_list]
    assert any("Test" in msg for msg in log_messages)
    assert any("accuracy" in msg for msg in log_messages)


# ==================== perform_cross_validation Tests ====================


def test_perform_cross_validation_returns_cv_results(
    mock_embeddings: np.ndarray, mock_labels: np.ndarray, nested_config: dict[str, Any]
) -> None:
    """Verify perform_cross_validation returns CV results for all metrics"""
    # Arrange
    mock_logger = Mock()

    # Act
    results = perform_cross_validation(
        mock_embeddings, mock_labels, nested_config, mock_logger
    )

    # Assert
    assert "cv_accuracy" in results
    assert "cv_f1" in results
    assert "cv_roc_auc" in results
    # Each result should have mean and std
    for _metric, values in results.items():
        assert "mean" in values
        assert "std" in values
        assert isinstance(values["mean"], (float, np.floating))
        assert isinstance(values["std"], (float, np.floating))


def test_perform_cross_validation_uses_stratified_kfold_when_configured(
    mock_embeddings: np.ndarray, mock_labels: np.ndarray, nested_config: dict[str, Any]
) -> None:
    """Verify stratified K-fold is used when stratify=True"""
    # Arrange
    nested_config["classifier"]["stratify"] = True
    nested_config["classifier"]["cv_folds"] = 3
    mock_logger = Mock()

    # Act
    results = perform_cross_validation(
        mock_embeddings, mock_labels, nested_config, mock_logger
    )

    # Assert
    assert results is not None
    # Verify logger was called with CV info
    log_messages = [call[0][0] for call in mock_logger.info.call_args_list]
    assert any("3-fold cross-validation" in msg for msg in log_messages)


def test_perform_cross_validation_uses_regular_kfold_when_stratify_false(
    mock_embeddings: np.ndarray, mock_labels: np.ndarray, nested_config: dict[str, Any]
) -> None:
    """Verify regular K-fold is used when stratify=False"""
    # Arrange
    nested_config["classifier"]["stratify"] = False
    nested_config["classifier"]["cv_folds"] = 3
    mock_logger = Mock()

    # Act
    results = perform_cross_validation(
        mock_embeddings, mock_labels, nested_config, mock_logger
    )

    # Assert
    assert results is not None
    assert "cv_accuracy" in results


# ==================== save_model Tests ====================


def test_save_model_saves_classifier_to_file(
    nested_config: dict[str, Any], tmp_path: Path
) -> None:
    """Verify save_model saves classifier to pickle file (updated for dict return)"""
    # Arrange
    from antibody_training_esm.core.classifier import BinaryClassifier

    classifier = BinaryClassifier(
        model_name="facebook/esm1v_t33_650M_UR90S_1",
        device="cpu",
        random_state=42,
        max_iter=10,
        batch_size=8,
    )
    # Fit with dummy data
    X = np.random.rand(10, 1280)
    y = np.array([0, 1] * 5)
    classifier.fit(X, y)

    nested_config["training"]["save_model"] = True
    nested_config["training"]["model_name"] = "test_model"
    nested_config["training"]["model_save_dir"] = str(tmp_path / "models")
    mock_logger = Mock()

    # Act
    model_paths = save_model(classifier, nested_config, mock_logger)

    # Assert
    assert model_paths is not None
    assert isinstance(model_paths, dict)
    assert "pickle" in model_paths
    assert Path(model_paths["pickle"]).exists()
    assert model_paths["pickle"] == str(tmp_path / "models" / "test_model.pkl")
    # Verify model can be loaded
    with open(model_paths["pickle"], "rb") as f:
        loaded_classifier = pickle.load(f)
    assert hasattr(loaded_classifier, "predict")


def test_save_model_returns_none_when_save_disabled(
    nested_config: dict[str, Any],
) -> None:
    """Verify save_model returns empty dict when save_model=False (updated for dict return)"""
    # Arrange
    from antibody_training_esm.core.classifier import BinaryClassifier

    classifier = BinaryClassifier(
        model_name="facebook/esm1v_t33_650M_UR90S_1",
        device="cpu",
        random_state=42,
        max_iter=10,
        batch_size=8,
    )

    nested_config["training"]["save_model"] = False
    mock_logger = Mock()

    # Act
    model_paths = save_model(classifier, nested_config, mock_logger)

    # Assert
    assert model_paths == {}


def test_save_model_creates_save_directory_if_missing(
    nested_config: dict[str, Any], tmp_path: Path
) -> None:
    """Verify save_model creates model_save_dir if it doesn't exist (updated for dict return)"""
    # Arrange
    from antibody_training_esm.core.classifier import BinaryClassifier

    classifier = BinaryClassifier(
        model_name="facebook/esm1v_t33_650M_UR90S_1",
        device="cpu",
        random_state=42,
        max_iter=10,
        batch_size=8,
    )
    X = np.random.rand(10, 1280)
    y = np.array([0, 1] * 5)
    classifier.fit(X, y)

    # Point to nested directory that doesn't exist
    nested_dir = tmp_path / "models" / "nested" / "path"
    nested_config["training"]["save_model"] = True
    nested_config["training"]["model_name"] = "test_model"
    nested_config["training"]["model_save_dir"] = str(nested_dir)
    mock_logger = Mock()

    # Act
    model_paths = save_model(classifier, nested_config, mock_logger)

    # Assert
    assert model_paths is not None
    assert isinstance(model_paths, dict)
    assert "pickle" in model_paths
    assert Path(model_paths["pickle"]).exists()
    assert nested_dir.exists()


# ==================== save_model Dual-Format Tests (TDD) ====================


def test_save_model_creates_dual_format_files(
    nested_config: dict[str, Any], tmp_path: Path
) -> None:
    """Verify save_model creates both pickle and NPZ+JSON files (TDD)"""
    # Arrange
    from antibody_training_esm.core.classifier import BinaryClassifier

    classifier = BinaryClassifier(
        model_name="facebook/esm1v_t33_650M_UR90S_1",
        device="cpu",
        random_state=42,
        max_iter=10,
        batch_size=8,
    )
    X = np.random.rand(10, 1280)
    y = np.array([0, 1] * 5)
    classifier.fit(X, y)

    nested_config["training"]["save_model"] = True
    nested_config["training"]["model_name"] = "test_model"
    nested_config["training"]["model_save_dir"] = str(tmp_path / "models")
    mock_logger = Mock()

    # Act
    model_paths = save_model(classifier, nested_config, mock_logger)

    # Assert
    assert isinstance(model_paths, dict)
    assert "pickle" in model_paths
    assert "npz" in model_paths
    assert "config" in model_paths

    # Verify all files exist
    assert Path(model_paths["pickle"]).exists()
    assert Path(model_paths["npz"]).exists()
    assert Path(model_paths["config"]).exists()

    # Verify file names
    assert model_paths["pickle"] == str(tmp_path / "models" / "test_model.pkl")
    assert model_paths["npz"] == str(tmp_path / "models" / "test_model.npz")
    assert model_paths["config"] == str(tmp_path / "models" / "test_model_config.json")


def test_save_model_npz_arrays_match_pickle(
    nested_config: dict[str, Any], tmp_path: Path
) -> None:
    """Verify NPZ arrays match pickle model coefficients (TDD)"""
    # Arrange
    from antibody_training_esm.core.classifier import BinaryClassifier

    classifier = BinaryClassifier(
        model_name="facebook/esm1v_t33_650M_UR90S_1",
        device="cpu",
        random_state=42,
        max_iter=10,
        batch_size=8,
        class_weight="balanced",
    )
    X = np.random.rand(10, 1280)
    y = np.array([0, 1] * 5)
    classifier.fit(X, y)

    nested_config["training"]["save_model"] = True
    nested_config["training"]["model_name"] = "test_model"
    nested_config["training"]["model_save_dir"] = str(tmp_path / "models")
    mock_logger = Mock()

    # Act
    model_paths = save_model(classifier, nested_config, mock_logger)

    # Load pickle
    with open(model_paths["pickle"], "rb") as f:
        pkl_classifier = pickle.load(f)

    # Load NPZ
    npz_arrays = np.load(model_paths["npz"])

    # Assert: NPZ arrays match pickle model
    np.testing.assert_array_equal(pkl_classifier.classifier.coef_, npz_arrays["coef"])
    np.testing.assert_array_equal(
        pkl_classifier.classifier.intercept_, npz_arrays["intercept"]
    )
    np.testing.assert_array_equal(
        pkl_classifier.classifier.classes_, npz_arrays["classes"]
    )
    assert pkl_classifier.classifier.n_features_in_ == int(
        npz_arrays["n_features_in"][0]
    )
    np.testing.assert_array_equal(
        pkl_classifier.classifier.n_iter_, npz_arrays["n_iter"]
    )


def test_save_model_json_metadata_complete(
    nested_config: dict[str, Any], tmp_path: Path
) -> None:
    """Verify JSON metadata contains all required fields (TDD)"""
    # Arrange
    import json

    from antibody_training_esm.core.classifier import BinaryClassifier

    classifier = BinaryClassifier(
        model_name="facebook/esm1v_t33_650M_UR90S_1",
        device="cpu",
        random_state=42,
        max_iter=100,
        batch_size=8,
        C=1.0,
        penalty="l2",
        solver="lbfgs",
        class_weight="balanced",
        revision="main",
    )
    X = np.random.rand(10, 1280)
    y = np.array([0, 1] * 5)
    classifier.fit(X, y)

    nested_config["training"]["save_model"] = True
    nested_config["training"]["model_name"] = "test_model"
    nested_config["training"]["model_save_dir"] = str(tmp_path / "models")
    mock_logger = Mock()

    # Act
    model_paths = save_model(classifier, nested_config, mock_logger)

    # Load JSON
    with open(model_paths["config"]) as f:
        metadata = json.load(f)

    # Assert: All required fields present
    assert metadata["model_type"] == "LogisticRegression"
    assert "sklearn_version" in metadata

    # LogisticRegression params
    assert metadata["C"] == 1.0
    assert metadata["penalty"] == "l2"
    assert metadata["solver"] == "lbfgs"
    assert metadata["class_weight"] == "balanced"
    assert metadata["max_iter"] == 100
    assert metadata["random_state"] == 42

    # ESM params
    assert metadata["esm_model"] == "facebook/esm1v_t33_650M_UR90S_1"
    assert metadata["esm_revision"] == "main"
    assert metadata["batch_size"] == 8
    assert metadata["device"] == "cpu"


def test_save_model_returns_empty_dict_when_disabled(
    nested_config: dict[str, Any],
) -> None:
    """Verify save_model returns empty dict when save_model=False (TDD)"""
    # Arrange
    from antibody_training_esm.core.classifier import BinaryClassifier

    classifier = BinaryClassifier(
        model_name="facebook/esm1v_t33_650M_UR90S_1",
        device="cpu",
        random_state=42,
        max_iter=10,
        batch_size=8,
    )

    nested_config["training"]["save_model"] = False
    mock_logger = Mock()

    # Act
    model_paths = save_model(classifier, nested_config, mock_logger)

    # Assert
    assert model_paths == {}


# ==================== train_model Integration Tests ====================


def test_train_model_loads_config_and_completes_pipeline(
    config_yaml_path: str,
    tmp_path: Path,
    mock_transformers_model: tuple[Any, Any],
) -> None:
    """Verify train_model loads config and completes full training pipeline"""
    # Arrange: Create minimal training CSV
    import pandas as pd

    train_csv = tmp_path / "train.csv"
    df = pd.DataFrame(
        {
            "id": [f"AB{i:03d}" for i in range(10)],
            "VH_sequence": ["QVQLVQSGAEVKKPGA"] * 10,
            "label": [0, 1] * 5,
        }
    )
    df.to_csv(train_csv, index=False)

    # Update config to point to train CSV
    config = load_config(config_yaml_path)
    config["data"]["train_data_path"] = str(train_csv)
    config["training"]["save_model"] = False  # Don't save (faster test)
    config["classifier"]["cv_folds"] = 2  # Fewer folds for speed
    with open(config_yaml_path, "w") as f:
        yaml.dump(config, f)

    # Act
    with patch("antibody_training_esm.core.trainer.load_data") as mock_load_data:
        # Mock load_data to return sequences and labels
        mock_load_data.return_value = (
            df["VH_sequence"].tolist(),
            df["label"].tolist(),
        )
        results = train_model(config_yaml_path)

    # Assert
    assert results is not None
    assert "train_metrics" in results
    assert "cv_metrics" in results
    assert "config" in results
    assert "accuracy" in results["train_metrics"]
    assert "cv_accuracy" in results["cv_metrics"]


def test_train_model_raises_on_training_failure(config_yaml_path: str) -> None:
    """Verify train_model raises exception on training failure"""
    # Arrange: Point to non-existent training data
    config = load_config(config_yaml_path)
    config["data"]["train_data_path"] = "nonexistent.csv"
    with open(config_yaml_path, "w") as f:
        yaml.dump(config, f)

    # Act & Assert: Should raise FileNotFoundError or other training error
    with pytest.raises((FileNotFoundError, ValueError, RuntimeError)):
        train_model(config_yaml_path)


def test_train_model_deletes_cache_after_training(
    config_yaml_path: str,
    tmp_path: Path,
    mock_transformers_model: tuple[Any, Any],
) -> None:
    """Verify train_model deletes embeddings cache after training"""
    # Arrange
    import pandas as pd

    train_csv = tmp_path / "train.csv"
    df = pd.DataFrame(
        {
            "id": [f"AB{i:03d}" for i in range(10)],
            "VH_sequence": ["QVQLVQSGAEVKKPGA"] * 10,
            "label": [0, 1] * 5,
        }
    )
    df.to_csv(train_csv, index=False)

    config = load_config(config_yaml_path)
    config["data"]["train_data_path"] = str(train_csv)
    config["data"]["embeddings_cache_dir"] = str(tmp_path / "cache")
    config["training"]["save_model"] = False
    config["classifier"]["cv_folds"] = 2
    with open(config_yaml_path, "w") as f:
        yaml.dump(config, f)

    # Act
    with patch("antibody_training_esm.core.trainer.load_data") as mock_load_data:
        mock_load_data.return_value = (
            df["VH_sequence"].tolist(),
            df["label"].tolist(),
        )
        train_model(config_yaml_path)

    # Assert: Cache directory should be deleted
    assert not (tmp_path / "cache").exists()
