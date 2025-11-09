"""
Integration Tests for Model Persistence

Tests model serialization and deserialization workflows.
Focus: Save/load models, state preservation, compatibility

Testing philosophy:
- Test pickle serialization (Python standard)
- Test model state preservation (hyperparameters, fitted state)
- Test ESM model recreation (not saved, recreated on load)
- Test error handling (corrupt files, missing files)

Date: 2025-11-07
Phase: 3 (Integration Tests)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from antibody_training_esm.core.classifier import BinaryClassifier
from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor

# ==================== Fixtures ====================


@pytest.fixture
def trained_classifier(mock_transformers_model: tuple[Any, Any]) -> BinaryClassifier:
    """Create and train a classifier for persistence tests"""
    # Arrange: Create classifier with specific params
    params = {
        "model_name": "facebook/esm1v_t33_650M_UR90S_1",
        "device": "cpu",
        "random_state": 42,
        "max_iter": 100,
        "C": 0.5,
        "penalty": "l2",
        "solver": "lbfgs",
        "batch_size": 8,
    }
    classifier = BinaryClassifier(params=params)

    # Train on sample data
    np.random.seed(42)
    X_train = np.random.rand(50, 1280).astype(np.float32)
    y_train = np.array([0, 1] * 25)
    classifier.fit(X_train, y_train)

    return classifier


# ==================== Basic Save/Load Tests ====================


@pytest.mark.integration
def test_save_and_load_trained_classifier(
    mock_transformers_model: tuple[Any, Any],
    trained_classifier: BinaryClassifier,
    tmp_path: Path,
) -> None:
    """Verify trained classifier can be saved and loaded"""
    # Arrange
    model_path = tmp_path / "test_classifier.pkl"

    # Act: Save classifier
    with open(model_path, "wb") as f:
        pickle.dump(trained_classifier, f)

    # Assert: File exists
    assert model_path.exists()

    # Act: Load classifier
    with open(model_path, "rb") as f:
        loaded_classifier = pickle.load(f)

    # Assert: Loaded classifier has same properties
    assert loaded_classifier.C == trained_classifier.C
    assert loaded_classifier.penalty == trained_classifier.penalty
    assert loaded_classifier.random_state == trained_classifier.random_state
    assert loaded_classifier.is_fitted is True


@pytest.mark.integration
def test_loaded_classifier_can_predict(
    mock_transformers_model: tuple[Any, Any],
    trained_classifier: BinaryClassifier,
    tmp_path: Path,
) -> None:
    """Verify loaded classifier can make predictions"""
    # Arrange: Save and load classifier
    model_path = tmp_path / "test_classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(trained_classifier, f)

    with open(model_path, "rb") as f:
        loaded_classifier = pickle.load(f)

    # Act: Make predictions with loaded classifier
    np.random.seed(42)
    X_test = np.random.rand(10, 1280).astype(np.float32)
    predictions = loaded_classifier.predict(X_test)

    # Assert: Predictions are valid
    assert len(predictions) == 10
    assert all(pred in [0, 1] for pred in predictions)


@pytest.mark.integration
def test_loaded_classifier_predictions_match_original(
    mock_transformers_model: tuple[Any, Any],
    trained_classifier: BinaryClassifier,
    tmp_path: Path,
) -> None:
    """Verify loaded classifier produces same predictions as original"""
    # Arrange: Create test data
    np.random.seed(42)
    X_test = np.random.rand(10, 1280).astype(np.float32)

    # Act: Get predictions from original classifier
    original_predictions = trained_classifier.predict(X_test)

    # Act: Save and load classifier
    model_path = tmp_path / "test_classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(trained_classifier, f)

    with open(model_path, "rb") as f:
        loaded_classifier = pickle.load(f)

    # Act: Get predictions from loaded classifier
    loaded_predictions = loaded_classifier.predict(X_test)

    # Assert: Predictions match
    np.testing.assert_array_equal(original_predictions, loaded_predictions)


# ==================== ESM Model Recreation Tests ====================


@pytest.mark.integration
def test_esm_model_not_saved_with_classifier(
    mock_transformers_model: tuple[Any, Any],
    trained_classifier: BinaryClassifier,
    tmp_path: Path,
) -> None:
    """Verify ESM embedding extractor is not serialized (recreated on load)"""
    # Arrange: Save classifier
    model_path = tmp_path / "test_classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(trained_classifier, f)

    # Act: Read pickle file size
    file_size = model_path.stat().st_size

    # Assert: File is small (no 650MB ESM model saved)
    assert file_size < 1_000_000  # Less than 1MB (ESM would be ~650MB)


@pytest.mark.integration
def test_loaded_classifier_recreates_embedding_extractor(
    mock_transformers_model: tuple[Any, Any],
    trained_classifier: BinaryClassifier,
    tmp_path: Path,
) -> None:
    """Verify embedding extractor is recreated when classifier is loaded"""
    # Arrange: Save classifier
    model_path = tmp_path / "test_classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(trained_classifier, f)

    # Act: Load classifier
    with open(model_path, "rb") as f:
        loaded_classifier = pickle.load(f)

    # Assert: Embedding extractor exists
    assert hasattr(loaded_classifier, "embedding_extractor")
    assert isinstance(loaded_classifier.embedding_extractor, ESMEmbeddingExtractor)
    assert (
        loaded_classifier.embedding_extractor.model_name
        == trained_classifier.model_name
    )
    assert loaded_classifier.embedding_extractor.device == trained_classifier.device


# ==================== Hyperparameter Preservation Tests ====================


@pytest.mark.integration
def test_all_hyperparameters_preserved_after_load(
    mock_transformers_model: tuple[Any, Any], tmp_path: Path
) -> None:
    """Verify all classifier hyperparameters are preserved through save/load"""
    # Arrange: Create classifier with specific hyperparameters
    original_params = {
        "model_name": "facebook/esm1v_t33_650M_UR90S_1",
        "device": "cpu",
        "random_state": 123,
        "max_iter": 500,
        "C": 0.1,
        "penalty": "l1",
        "solver": "liblinear",
        "class_weight": "balanced",
        "batch_size": 16,
    }
    classifier = BinaryClassifier(params=original_params)

    # Train classifier
    np.random.seed(42)
    X_train = np.random.rand(30, 1280).astype(np.float32)
    y_train = np.array([0, 1] * 15)
    classifier.fit(X_train, y_train)

    # Act: Save and load
    model_path = tmp_path / "test_classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(classifier, f)

    with open(model_path, "rb") as f:
        loaded_classifier = pickle.load(f)

    # Assert: All hyperparameters match
    assert loaded_classifier.C == 0.1
    assert loaded_classifier.penalty == "l1"
    assert loaded_classifier.solver == "liblinear"
    assert loaded_classifier.random_state == 123
    assert loaded_classifier.max_iter == 500
    assert loaded_classifier.class_weight == "balanced"
    assert loaded_classifier.model_name == "facebook/esm1v_t33_650M_UR90S_1"
    assert loaded_classifier.device == "cpu"
    assert loaded_classifier.batch_size == 16


@pytest.mark.integration
def test_fitted_state_preserved_after_load(
    mock_transformers_model: tuple[Any, Any], tmp_path: Path
) -> None:
    """Verify is_fitted flag is preserved through save/load"""
    # Arrange: Create untrained classifier
    params = {
        "model_name": "facebook/esm1v_t33_650M_UR90S_1",
        "device": "cpu",
        "random_state": 42,
        "max_iter": 100,
        "batch_size": 8,
    }
    classifier = BinaryClassifier(params=params)

    # Act: Save unfitted classifier
    model_path_unfitted = tmp_path / "unfitted_classifier.pkl"
    with open(model_path_unfitted, "wb") as f:
        pickle.dump(classifier, f)

    # Act: Train and save fitted classifier
    np.random.seed(42)
    X_train = np.random.rand(30, 1280).astype(np.float32)
    y_train = np.array([0, 1] * 15)
    classifier.fit(X_train, y_train)

    model_path_fitted = tmp_path / "fitted_classifier.pkl"
    with open(model_path_fitted, "wb") as f:
        pickle.dump(classifier, f)

    # Act: Load both classifiers
    with open(model_path_unfitted, "rb") as f:
        loaded_unfitted = pickle.load(f)

    with open(model_path_fitted, "rb") as f:
        loaded_fitted = pickle.load(f)

    # Assert: Fitted state preserved
    assert loaded_unfitted.is_fitted is False
    assert loaded_fitted.is_fitted is True


# ==================== Backward Compatibility Tests ====================


@pytest.mark.integration
def test_load_classifier_without_batch_size_param(
    mock_transformers_model: tuple[Any, Any], tmp_path: Path
) -> None:
    """Verify backward compatibility when loading models without batch_size"""
    # Arrange: Create classifier and remove batch_size (simulate old model)
    params = {
        "model_name": "facebook/esm1v_t33_650M_UR90S_1",
        "device": "cpu",
        "random_state": 42,
        "max_iter": 100,
        "batch_size": 8,
    }
    classifier = BinaryClassifier(params=params)

    # Train classifier
    np.random.seed(42)
    X_train = np.random.rand(30, 1280).astype(np.float32)
    y_train = np.array([0, 1] * 15)
    classifier.fit(X_train, y_train)

    # Manually remove batch_size to simulate old model
    delattr(classifier, "batch_size")

    # Act: Save and load (should use default batch_size=32)
    model_path = tmp_path / "old_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(classifier, f)

    with open(model_path, "rb") as f:
        loaded_classifier = pickle.load(f)

    # Assert: Default batch_size is used
    assert loaded_classifier.embedding_extractor.batch_size == 32  # Default


# ==================== Multiple Model Save/Load Tests ====================


@pytest.mark.integration
def test_save_multiple_models_to_different_files(
    mock_transformers_model: tuple[Any, Any], tmp_path: Path
) -> None:
    """Verify multiple models can be saved independently"""
    # Arrange: Create two different classifiers
    params_1 = {
        "model_name": "facebook/esm1v_t33_650M_UR90S_1",
        "device": "cpu",
        "random_state": 42,
        "max_iter": 100,
        "C": 0.1,
        "batch_size": 8,
    }
    classifier_1 = BinaryClassifier(params=params_1)

    params_2 = {
        "model_name": "facebook/esm1v_t33_650M_UR90S_1",
        "device": "cpu",
        "random_state": 42,
        "max_iter": 100,
        "C": 1.0,
        "batch_size": 8,
    }
    classifier_2 = BinaryClassifier(params=params_2)

    # Train both classifiers
    np.random.seed(42)
    X_train = np.random.rand(30, 1280).astype(np.float32)
    y_train = np.array([0, 1] * 15)

    classifier_1.fit(X_train, y_train)
    classifier_2.fit(X_train, y_train)

    # Act: Save both models
    model_path_1 = tmp_path / "classifier_c01.pkl"
    model_path_2 = tmp_path / "classifier_c10.pkl"

    with open(model_path_1, "wb") as f:
        pickle.dump(classifier_1, f)

    with open(model_path_2, "wb") as f:
        pickle.dump(classifier_2, f)

    # Act: Load both models
    with open(model_path_1, "rb") as f:
        loaded_1 = pickle.load(f)

    with open(model_path_2, "rb") as f:
        loaded_2 = pickle.load(f)

    # Assert: Models maintain different hyperparameters
    assert loaded_1.C == 0.1
    assert loaded_2.C == 1.0


# ==================== Error Handling Tests ====================


@pytest.mark.integration
def test_load_from_nonexistent_file_raises_error(
    mock_transformers_model: tuple[Any, Any],
) -> None:
    """Verify loading from missing file raises FileNotFoundError"""
    # Arrange
    nonexistent_path = Path("/tmp/nonexistent_model.pkl")

    # Act & Assert
    with pytest.raises(FileNotFoundError), open(nonexistent_path, "rb") as f:
        pickle.load(f)


@pytest.mark.integration
def test_load_from_corrupt_file_raises_error(
    mock_transformers_model: tuple[Any, Any], tmp_path: Path
) -> None:
    """Verify loading from corrupt pickle file raises error"""
    # Arrange: Create corrupt file
    corrupt_path = tmp_path / "corrupt_model.pkl"
    with open(corrupt_path, "wb") as f:
        f.write(b"not a valid pickle file")

    # Act & Assert - corrupt pickle raises UnpicklingError or EOFError
    with (
        pytest.raises((pickle.UnpicklingError, EOFError)),
        open(corrupt_path, "rb") as f,
    ):
        pickle.load(f)


# ==================== Probability Preservation Tests ====================


@pytest.mark.integration
def test_loaded_classifier_predict_proba_matches_original(
    mock_transformers_model: tuple[Any, Any],
    trained_classifier: BinaryClassifier,
    tmp_path: Path,
) -> None:
    """Verify predict_proba results match between original and loaded classifier"""
    # Arrange: Create test data
    np.random.seed(42)
    X_test = np.random.rand(10, 1280).astype(np.float32)

    # Act: Get probabilities from original classifier
    original_proba = trained_classifier.predict_proba(X_test)

    # Act: Save and load classifier
    model_path = tmp_path / "test_classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(trained_classifier, f)

    with open(model_path, "rb") as f:
        loaded_classifier = pickle.load(f)

    # Act: Get probabilities from loaded classifier
    loaded_proba = loaded_classifier.predict_proba(X_test)

    # Assert: Probabilities match
    np.testing.assert_array_almost_equal(original_proba, loaded_proba, decimal=6)


@pytest.mark.integration
def test_loaded_classifier_respects_assay_thresholds(
    mock_transformers_model: tuple[Any, Any],
    trained_classifier: BinaryClassifier,
    tmp_path: Path,
) -> None:
    """Verify loaded classifier respects ELISA and PSR thresholds"""
    # Arrange: Save and load classifier
    model_path = tmp_path / "test_classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(trained_classifier, f)

    with open(model_path, "rb") as f:
        loaded_classifier = pickle.load(f)

    # Act: Make predictions with different assay types
    np.random.seed(42)
    X_test = np.random.rand(10, 1280).astype(np.float32)

    predictions_elisa = loaded_classifier.predict(X_test, assay_type="ELISA")
    predictions_psr = loaded_classifier.predict(X_test, assay_type="PSR")

    # Assert: Both prediction types work
    assert len(predictions_elisa) == 10
    assert len(predictions_psr) == 10
    assert all(pred in [0, 1] for pred in predictions_elisa)
    assert all(pred in [0, 1] for pred in predictions_psr)


# ==================== Full Pipeline Persistence Test ====================


@pytest.mark.integration
def test_full_train_save_load_predict_pipeline(
    mock_transformers_model: tuple[Any, Any], tmp_path: Path
) -> None:
    """Verify complete pipeline: train → save → load → predict"""
    # Step 1: Train classifier
    params = {
        "model_name": "facebook/esm1v_t33_650M_UR90S_1",
        "device": "cpu",
        "random_state": 42,
        "max_iter": 100,
        "C": 0.5,
        "batch_size": 8,
    }
    classifier = BinaryClassifier(params=params)

    np.random.seed(42)
    X_train = np.random.rand(50, 1280).astype(np.float32)
    y_train = np.array([0, 1] * 25)
    classifier.fit(X_train, y_train)

    # Step 2: Save model
    model_path = tmp_path / "full_pipeline_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(classifier, f)

    # Step 3: Load model (simulate new session)
    with open(model_path, "rb") as f:
        loaded_classifier = pickle.load(f)

    # Step 4: Predict with loaded model
    X_test = np.random.rand(10, 1280).astype(np.float32)
    predictions = loaded_classifier.predict(X_test)
    probabilities = loaded_classifier.predict_proba(X_test)

    # Step 5: Verify full pipeline works
    assert len(predictions) == 10
    assert probabilities.shape == (10, 2)
    assert all(pred in [0, 1] for pred in predictions)
    assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1
