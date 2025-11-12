"""
Integration tests for ModelTester (antibody-test CLI).

Tests cover:
- Full ModelTester workflow with real BinaryClassifier
- Model loading from pickle files
- Dataset loading from CSV
- Embedding extraction (mocked ESM)
- Metrics computation and reporting
- Confusion matrix plotting
- Results saving
- Multi-model, multi-dataset testing

Testing philosophy:
- Use real BinaryClassifier (not mocked)
- Mock only ESM model downloads (heavyweight I/O)
- Test full workflows end-to-end
- Verify file artifacts (plots, CSVs, logs)

Date: 2025-11-07
Phase: 5 (Coverage Gap Closure)
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from antibody_training_esm.cli.test import ModelTester, TestConfig
from antibody_training_esm.core.classifier import BinaryClassifier

# ==================== Fixtures ====================


@pytest.fixture
def trained_classifier(
    mock_transformers_model: tuple[Any, Any], tmp_path: Path
) -> Path:
    """Create a trained BinaryClassifier for testing"""
    # Train a simple classifier
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

    # Save to pickle
    model_path = tmp_path / "trained_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(classifier, f)

    return model_path


@pytest.fixture
def test_dataset_csv(tmp_path: Path) -> Path:
    """Create a test dataset CSV"""
    data = pd.DataFrame(
        {
            "sequence": ["QVQLVQSGAEVKKPGASVKVSCKASGYTFT"] * 20,
            "label": [0, 1] * 10,
            "id": [f"AB{i:03d}" for i in range(20)],
        }
    )

    csv_path = tmp_path / "test_data.csv"
    data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def test_config(
    trained_classifier: Path, test_dataset_csv: Path, tmp_path: Path
) -> TestConfig:
    """Create ModelTester test configuration"""
    return TestConfig(
        model_paths=[str(trained_classifier)],
        data_paths=[str(test_dataset_csv)],
        sequence_column="sequence",
        label_column="label",
        output_dir=str(tmp_path / "test_results"),
        batch_size=8,
        device="cpu",
    )


# ==================== ModelTester Initialization Tests ====================


@pytest.mark.integration
def test_model_tester_initialization(test_config: TestConfig) -> None:
    """Verify ModelTester initializes correctly"""
    # Act
    tester = ModelTester(test_config)

    # Assert
    assert tester.config == test_config
    assert Path(test_config.output_dir).exists()
    assert hasattr(tester, "logger")
    assert tester.results == {}


@pytest.mark.integration
def test_model_tester_creates_output_directory(tmp_path: Path) -> None:
    """Verify ModelTester creates output directory"""
    # Arrange
    output_dir = tmp_path / "custom_results"
    config = TestConfig(
        model_paths=["model.pkl"],
        data_paths=["data.csv"],
        output_dir=str(output_dir),
    )

    # Act
    ModelTester(config)

    # Assert
    assert output_dir.exists()


# ==================== load_model Tests ====================


@pytest.mark.integration
def test_load_model_succeeds(test_config: TestConfig, trained_classifier: Path) -> None:
    """Verify ModelTester loads trained model successfully"""
    # Arrange
    tester = ModelTester(test_config)

    # Act
    model = tester.load_model(str(trained_classifier))

    # Assert
    assert isinstance(model, BinaryClassifier)
    assert model.is_fitted


@pytest.mark.integration
def test_load_model_raises_on_missing_file(test_config: TestConfig) -> None:
    """Verify load_model raises error for missing file"""
    # Arrange
    tester = ModelTester(test_config)

    # Act & Assert
    with pytest.raises(FileNotFoundError):
        tester.load_model("nonexistent.pkl")


@pytest.mark.integration
def test_load_model_raises_on_invalid_type(
    test_config: TestConfig, tmp_path: Path
) -> None:
    """Verify load_model raises error for invalid model type"""
    # Arrange
    tester = ModelTester(test_config)
    bad_model_path = tmp_path / "bad_model.pkl"
    with open(bad_model_path, "wb") as f:
        pickle.dump({"not": "a_classifier"}, f)

    # Act & Assert
    with pytest.raises(ValueError, match="Expected BinaryClassifier"):
        tester.load_model(str(bad_model_path))


# ==================== load_dataset Tests ====================


@pytest.mark.integration
def test_load_dataset_succeeds(test_config: TestConfig, test_dataset_csv: Path) -> None:
    """Verify ModelTester loads dataset successfully"""
    # Arrange
    tester = ModelTester(test_config)

    # Act
    sequences, labels = tester.load_dataset(str(test_dataset_csv))

    # Assert
    assert len(sequences) == 20
    assert len(labels) == 20
    assert all(isinstance(seq, str) for seq in sequences)
    assert all(label in [0, 1] for label in labels)


@pytest.mark.integration
def test_load_dataset_raises_on_missing_file(test_config: TestConfig) -> None:
    """Verify load_dataset raises error for missing file"""
    # Arrange
    tester = ModelTester(test_config)

    # Act & Assert
    with pytest.raises(FileNotFoundError):
        tester.load_dataset("nonexistent.csv")


@pytest.mark.integration
def test_load_dataset_raises_on_missing_column(
    test_config: TestConfig, tmp_path: Path
) -> None:
    """Verify load_dataset raises error for missing column"""
    # Arrange
    tester = ModelTester(test_config)
    bad_csv = tmp_path / "bad_data.csv"
    pd.DataFrame({"wrong_column": ["A", "B"], "label": [0, 1]}).to_csv(
        bad_csv, index=False
    )

    # Act & Assert
    with pytest.raises(ValueError, match="Sequence column"):
        tester.load_dataset(str(bad_csv))


# ==================== embed_sequences Tests ====================


@pytest.mark.integration
def test_embed_sequences(
    mock_transformers_model: tuple[Any, Any],
    test_config: TestConfig,
    trained_classifier: Path,
) -> None:
    """Verify ModelTester embeds sequences correctly"""
    # Arrange
    tester = ModelTester(test_config)
    model = tester.load_model(str(trained_classifier))
    sequences = ["QVQLVQSGAEVKKPGASVKVSCKASGYTFT"] * 5

    # Act
    embeddings = tester.embed_sequences(
        sequences, model, "test_data", test_config.output_dir
    )

    # Assert
    assert embeddings.shape == (5, 1280)
    assert embeddings.dtype == np.float32


# ==================== evaluate_pretrained Tests ====================


@pytest.mark.integration
def test_evaluate_pretrained(
    mock_transformers_model: tuple[Any, Any],
    test_config: TestConfig,
    trained_classifier: Path,
    test_dataset_csv: Path,
) -> None:
    """Verify ModelTester evaluates model correctly"""
    # Arrange
    tester = ModelTester(test_config)
    model = tester.load_model(str(trained_classifier))
    sequences, labels = tester.load_dataset(str(test_dataset_csv))
    embeddings = tester.embed_sequences(
        sequences, model, "test_data", test_config.output_dir
    )

    # Act
    results = tester.evaluate_pretrained(
        model, embeddings, np.array(labels), "test_model", "test_data"
    )

    # Assert
    assert "test_scores" in results
    assert "confusion_matrix" in results
    assert "accuracy" in results["test_scores"]
    assert "f1" in results["test_scores"]
    assert "roc_auc" in results["test_scores"]


# ==================== run_comprehensive_test Tests ====================


@pytest.mark.integration
def test_run_comprehensive_test_single_model(
    mock_transformers_model: tuple[Any, Any], test_config: TestConfig
) -> None:
    """Verify run_comprehensive_test works with single model/dataset"""
    # Arrange
    tester = ModelTester(test_config)

    # Act
    results = tester.run_comprehensive_test()

    # Assert
    assert isinstance(results, dict)
    assert len(results) > 0


@pytest.mark.integration
def test_run_comprehensive_test_multiple_models(
    mock_transformers_model: tuple[Any, Any],
    trained_classifier: Path,
    test_dataset_csv: Path,
    tmp_path: Path,
) -> None:
    """Verify run_comprehensive_test works with multiple models"""
    # Arrange: Train second model
    np.random.seed(43)
    X_train = np.random.rand(50, 1280).astype(np.float32)
    y_train = np.array([0, 1] * 25)

    classifier2 = BinaryClassifier(
        model_name="facebook/esm1v_t33_650M_UR90S_1",
        device="cpu",
        random_state=43,
        max_iter=10,
        batch_size=8,
    )
    classifier2.fit(X_train, y_train)

    model2_path = tmp_path / "model2.pkl"
    with open(model2_path, "wb") as f:
        pickle.dump(classifier2, f)

    # Arrange: Config with 2 models
    config = TestConfig(
        model_paths=[str(trained_classifier), str(model2_path)],
        data_paths=[str(test_dataset_csv)],
        output_dir=str(tmp_path / "multi_results"),
        device="cpu",
    )

    tester = ModelTester(config)

    # Act
    results = tester.run_comprehensive_test()

    # Assert
    assert len(results) > 0
    # Should test both models
    for dataset_results in results.values():
        assert isinstance(dataset_results, dict)


@pytest.mark.integration
def test_run_comprehensive_test_multiple_datasets(
    mock_transformers_model: tuple[Any, Any],
    trained_classifier: Path,
    test_dataset_csv: Path,
    tmp_path: Path,
) -> None:
    """Verify run_comprehensive_test works with multiple datasets"""
    # Arrange: Create second dataset
    data2 = pd.DataFrame(
        {
            "sequence": ["EVQLVESGGGLVQPGGSLRLSCAASGFTFS"] * 15,
            "label": [0, 1, 0, 1, 0] * 3,
        }
    )
    csv2_path = tmp_path / "test_data2.csv"
    data2.to_csv(csv2_path, index=False)

    config = TestConfig(
        model_paths=[str(trained_classifier)],
        data_paths=[str(test_dataset_csv), str(csv2_path)],
        output_dir=str(tmp_path / "multi_dataset_results"),
        device="cpu",
    )

    tester = ModelTester(config)

    # Act
    results = tester.run_comprehensive_test()

    # Assert
    assert len(results) == 2  # Two datasets


# ==================== Artifact Tests ====================


@pytest.mark.integration
def test_model_tester_creates_log_file(test_config: TestConfig) -> None:
    """Verify ModelTester creates log file"""
    # Act
    ModelTester(test_config)

    # Assert
    log_files = list(Path(test_config.output_dir).glob("test_*.log"))
    assert len(log_files) > 0


@pytest.mark.integration
def test_model_tester_cleanup_embeddings(
    mock_transformers_model: tuple[Any, Any], test_config: TestConfig
) -> None:
    """Verify ModelTester cleanup removes cached embeddings"""
    # Arrange
    tester = ModelTester(test_config)

    # Create fake cached embeddings
    cache_file = Path(test_config.output_dir) / "embeddings_test_model_test_data.npz"
    np.savez(cache_file, embeddings=np.random.rand(10, 1280))
    tester.cached_embedding_files.append(str(cache_file))

    # Act
    tester.cleanup_cached_embeddings()

    # Assert
    assert not cache_file.exists()


# ==================== Edge Cases ====================


@pytest.mark.integration
def test_model_tester_handles_empty_dataset(
    mock_transformers_model: tuple[Any, Any], test_config: TestConfig, tmp_path: Path
) -> None:
    """Verify ModelTester handles empty dataset gracefully"""
    # Arrange
    empty_csv = tmp_path / "empty.csv"
    pd.DataFrame({"sequence": [], "label": []}).to_csv(empty_csv, index=False)

    config = TestConfig(
        model_paths=[str(test_config.model_paths[0])],
        data_paths=[str(empty_csv)],
        output_dir=str(tmp_path / "empty_results"),
        device="cpu",
    )

    tester = ModelTester(config)

    # Act & Assert
    # Should handle gracefully (may log error but not crash)
    try:
        results = tester.run_comprehensive_test()
        # If it completes, verify results are empty or have error markers
        assert isinstance(results, dict)
    except ValueError:
        # Empty dataset may raise ValueError - acceptable
        pass


@pytest.mark.integration
def test_model_tester_custom_metrics(
    mock_transformers_model: tuple[Any, Any],
    trained_classifier: Path,
    test_dataset_csv: Path,
    tmp_path: Path,
) -> None:
    """Verify ModelTester uses custom metrics"""
    # Arrange
    config = TestConfig(
        model_paths=[str(trained_classifier)],
        data_paths=[str(test_dataset_csv)],
        output_dir=str(tmp_path / "custom_metrics"),
        metrics=["accuracy", "f1"],  # Only 2 metrics
        device="cpu",
    )

    tester = ModelTester(config)

    # Act
    results = tester.run_comprehensive_test()

    # Assert
    for dataset_results in results.values():
        for model_results in dataset_results.values():
            if "test_scores" in model_results:
                assert "accuracy" in model_results["test_scores"]
                assert "f1" in model_results["test_scores"]


# ==================== Helper Functions ====================


def _create_model_config(
    model_path: Path, model_name: str, classifier_type: str
) -> None:
    """Create model config JSON for testing hierarchical path detection"""
    config_path = model_path.with_name(model_path.stem + "_config.json")
    config = {
        "model_name": f"facebook/{model_name}_t33_650M_UR90S_1",
        "classifier": {"type": classifier_type},
    }
    with open(config_path, "w") as f:
        json.dump(config, f)


# ==================== Bug Fix Regression Tests (Issue #10) ====================


@pytest.mark.integration
def test_embed_sequences_uses_hierarchical_cache(
    mock_transformers_model: tuple[Any, Any],
    trained_classifier: Path,
    test_dataset_csv: Path,
    tmp_path: Path,
) -> None:
    """Verify embed_sequences caches to hierarchical output_dir (Bug #1 regression test)"""
    # Arrange
    hierarchical_dir = tmp_path / "esm1v" / "logreg" / "jain"
    config = TestConfig(
        model_paths=[str(trained_classifier)],
        data_paths=[str(test_dataset_csv)],
        output_dir=str(tmp_path),
        device="cpu",
    )
    tester = ModelTester(config)
    model = tester.load_model(str(trained_classifier))
    sequences = ["QVQLVQSGAEVKKPGASVKVSCKASGYTFT"] * 5

    # Act
    embeddings = tester.embed_sequences(
        sequences, model, "test_data", str(hierarchical_dir)
    )

    # Assert
    assert hierarchical_dir.exists(), "Hierarchical directory not created"
    cache_file = hierarchical_dir / "test_data_test_embeddings.pkl"
    assert cache_file.exists(), f"Cache file not found at {cache_file}"
    assert embeddings.shape == (5, 1280)


@pytest.mark.integration
def test_run_comprehensive_test_no_model_collision(
    mock_transformers_model: tuple[Any, Any],
    trained_classifier: Path,
    test_dataset_csv: Path,
    tmp_path: Path,
) -> None:
    """Verify multiple models don't overwrite each other's results (Bug #2 regression test)"""
    # Arrange: Create second model
    np.random.seed(99)
    X_train = np.random.rand(50, 1280).astype(np.float32)
    y_train = np.array([0, 1] * 25)

    classifier2 = BinaryClassifier(
        model_name="facebook/esm1v_t33_650M_UR90S_1",
        device="cpu",
        random_state=99,
        max_iter=10,
        batch_size=8,
    )
    classifier2.fit(X_train, y_train)

    model2_path = tmp_path / "model2.pkl"
    with open(model2_path, "wb") as f:
        pickle.dump(classifier2, f)

    # Add config JSONs for hierarchical path detection
    # Use different model types to ensure separate hierarchical paths
    _create_model_config(trained_classifier, "esm1v", "logistic_regression")
    _create_model_config(model2_path, "esm2_650m", "logistic_regression")

    # Arrange: Config with 2 models
    config = TestConfig(
        model_paths=[str(trained_classifier), str(model2_path)],
        data_paths=[str(test_dataset_csv)],
        output_dir=str(tmp_path / "test_results"),
        device="cpu",
    )

    tester = ModelTester(config)

    # Act
    results = tester.run_comprehensive_test()

    # Assert: Verify embedding caches were created (they're cleaned up after test)
    # Check the cached_embedding_files list to see what was cached
    cached_files = tester.cached_embedding_files
    assert len(cached_files) >= 2, (
        f"Expected >=2 cache files to have been created, found {len(cached_files)}. "
        f"Cached files: {cached_files}"
    )

    # Verify cache files are in hierarchical directories (not all in same dir)
    cache_dirs = {str(Path(f).parent) for f in cached_files}
    assert len(cache_dirs) >= 2, (
        f"Expected cache files in >=2 different directories (hierarchical), "
        f"found {len(cache_dirs)} unique directories: {cache_dirs}"
    )

    # Verify results for both models
    assert len(results) > 0
    dataset_name = list(results.keys())[0]
    assert len(results[dataset_name]) == 2, "Should have results for both models"


@pytest.mark.integration
def test_run_comprehensive_test_generates_aggregated_reports(
    mock_transformers_model: tuple[Any, Any],
    trained_classifier: Path,
    test_dataset_csv: Path,
    tmp_path: Path,
) -> None:
    """Verify aggregated multi-model reports are generated (Bug #2 regression test)"""
    # Arrange: Create second model
    np.random.seed(100)
    X_train = np.random.rand(50, 1280).astype(np.float32)
    y_train = np.array([0, 1] * 25)

    classifier2 = BinaryClassifier(
        model_name="facebook/esm1v_t33_650M_UR90S_1",
        device="cpu",
        random_state=100,
        max_iter=10,
        batch_size=8,
    )
    classifier2.fit(X_train, y_train)

    model2_path = tmp_path / "model2.pkl"
    with open(model2_path, "wb") as f:
        pickle.dump(classifier2, f)

    config = TestConfig(
        model_paths=[str(trained_classifier), str(model2_path)],
        data_paths=[str(test_dataset_csv)],
        output_dir=str(tmp_path / "test_results"),
        device="cpu",
    )

    tester = ModelTester(config)

    # Act
    results = tester.run_comprehensive_test()

    # Assert: Aggregated reports exist in root output_dir
    output_root = tmp_path / "test_results"

    # Look for aggregated confusion matrix and results in root (not in subdirs)
    root_files = list(output_root.glob("*"))
    root_files_only = [f for f in root_files if f.is_file()]
    aggregated_yamls = [f for f in root_files_only if f.suffix == ".yaml"]
    aggregated_pngs = [f for f in root_files_only if f.suffix == ".png"]

    assert len(aggregated_yamls) > 0, (
        f"No aggregated YAML report found in root. "
        f"Root files: {[f.name for f in root_files_only]}"
    )
    assert len(aggregated_pngs) > 0, (
        f"No aggregated confusion matrix PNG found in root. "
        f"Root files: {[f.name for f in root_files_only]}"
    )

    # Verify we have results for both models
    assert len(results) > 0
    dataset_name = list(results.keys())[0]
    assert len(results[dataset_name]) == 2, "Should have results for both models"
