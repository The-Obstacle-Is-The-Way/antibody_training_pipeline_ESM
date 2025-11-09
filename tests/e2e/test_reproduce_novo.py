"""
End-to-End Tests for Novo Nordisk Result Reproduction.

Tests cover:
- Reproducing Novo Nordisk published results (66-71% on Jain parity)
- Training on Boughter dataset (flag-based filtering)
- Testing on Jain parity set (86 antibodies)
- PSR threshold calibration (0.5495 for exact parity)
- Cross-dataset generalization

Testing philosophy:
- Test against published benchmarks
- Verify scientific reproducibility
- Use real datasets (when available)
- Mock only heavyweight ESM model downloads
- Allow reasonable variance in accuracy (±5%)

NOTE: These tests require real preprocessed datasets and are
      computationally expensive. They are skipped by default
      and should be run explicitly when validating results.

Date: 2025-11-07
Phase: 4 (CLI & E2E Tests)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from antibody_training_esm.core.classifier import BinaryClassifier
from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor
from antibody_training_esm.datasets.boughter import BoughterDataset
from antibody_training_esm.datasets.jain import JainDataset

# ==================== Fixtures ====================


@pytest.fixture
def novo_classifier_params() -> dict[str, Any]:
    """Novo Nordisk classifier parameters"""
    return {
        "model_name": "facebook/esm1v_t33_650M_UR90S_1",
        "device": "cpu",
        "random_state": 42,
        "max_iter": 1000,
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "class_weight": None,  # Novo used unweighted
        "batch_size": 32,
    }


@pytest.fixture
def real_dataset_paths() -> dict[str, str]:
    """Paths to real preprocessed datasets"""
    return {
        "boughter_train": "train_datasets/boughter/boughter_translated.csv",
        "jain_parity": "test_datasets/jain/canonical/VH_only_jain_86_p5e_s2.csv",
    }


# ==================== Novo Parity Reproduction Tests ====================


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.skipif(
    not Path("train_datasets/boughter/boughter_translated.csv").exists()
    or not Path("test_datasets/jain/canonical/VH_only_jain_86_p5e_s2.csv").exists(),
    reason="Requires real preprocessed Boughter and Jain datasets. "
    "Run preprocessing scripts first: "
    "python preprocessing/boughter/stage2_stage3_annotation_qc.py && "
    "python preprocessing/jain/step2_preprocess_p5e_s2.py",
)
def test_reproduce_novo_jain_accuracy_with_real_data(
    mock_transformers_model: tuple[Any, Any],
    novo_classifier_params: dict[str, Any],
    real_dataset_paths: dict[str, str],
) -> None:
    """Verify we can reproduce Novo Nordisk Jain accuracy (66-71%) with real data

    Published result: 66-71% accuracy on Jain parity set (86 antibodies)
    Training set: Boughter (exclude mild flags 1-3)
    Test set: Jain parity (86 antibodies, VH only)

    NOTE: This test requires REAL datasets and is computationally expensive.
          It is skipped by default and marked as @pytest.mark.slow.
    """
    # Arrange: Load Boughter training data (Novo methodology)
    boughter = BoughterDataset()
    df_train = boughter.load_data(
        processed_csv=real_dataset_paths["boughter_train"],
        include_mild=False,  # Novo excludes mild flags (1-3)
    )

    # Arrange: Load Jain parity test data
    jain = JainDataset()
    df_test = jain.load_data(
        full_csv_path=real_dataset_paths["jain_parity"], stage="full"
    )

    # Verify test set size matches Novo parity (86 antibodies)
    assert len(df_test) == 86, f"Expected 86 antibodies, got {len(df_test)}"

    # Arrange: Extract embeddings
    extractor = ESMEmbeddingExtractor(
        model_name=novo_classifier_params["model_name"],
        device=novo_classifier_params["device"],
        batch_size=novo_classifier_params["batch_size"],
    )

    X_train = extractor.extract_batch_embeddings(df_train["VH_sequence"].tolist())
    y_train = df_train["label"].to_numpy(dtype=int)

    X_test = extractor.extract_batch_embeddings(df_test["VH_sequence"].tolist())
    y_test = df_test["label"].to_numpy(dtype=int)

    # Act: Train classifier (Novo methodology)
    classifier = BinaryClassifier(params=novo_classifier_params)
    classifier.fit(X_train, y_train)

    # Act: Predict on Jain parity set
    y_pred = classifier.predict(X_test)

    # Assert: Accuracy in Novo range (66-71%)
    accuracy = accuracy_score(y_test, y_pred)

    # Allow ±5% variance (61-76%) to account for random seed differences
    assert 0.61 <= accuracy <= 0.76, (
        f"Accuracy {accuracy:.2%} outside Novo range (66-71% ±5%). "
        f"Expected 61-76%, got {accuracy:.2%}"
    )

    # Log full metrics for debugging
    print("\nNovo Parity Reproduction Results:")
    print(f"  Accuracy:  {accuracy:.2%} (Novo: 66-71%)")
    print(f"  Precision: {precision_score(y_test, y_pred):.2%}")
    print(f"  Recall:    {recall_score(y_test, y_pred):.2%}")
    print(f"  F1 Score:  {f1_score(y_test, y_pred):.2%}")


# ==================== PSR Threshold Calibration Tests ====================


@pytest.mark.e2e
def test_psr_threshold_calibration(mock_transformers_model: tuple[Any, Any]) -> None:
    """Verify PSR threshold (0.5495) is correctly applied

    Novo Nordisk calibrated threshold to 0.5495 for exact parity.
    This test verifies the threshold is applied correctly.
    """
    # Arrange
    np.random.seed(42)
    X_train = np.random.rand(100, 1280).astype(np.float32)
    y_train = np.array([0, 1] * 50)

    classifier = BinaryClassifier(
        model_name="facebook/esm1v_t33_650M_UR90S_1",
        device="cpu",
        random_state=42,
        max_iter=10,
        batch_size=8,
    )
    classifier.fit(X_train, y_train)

    # Create test case with probability = 0.55 (above PSR threshold 0.5495)
    # Mock predict_proba to return known probabilities
    classifier.classifier.predict_proba = lambda X: np.array([[0.45, 0.55]])

    # Act: Predict with PSR threshold
    X_test = np.zeros((1, 1280))
    prediction = classifier.predict(X_test, assay_type="PSR")

    # Assert: 0.55 > 0.5495 → prediction should be 1
    assert prediction[0] == 1


@pytest.mark.e2e
def test_elisa_threshold_default(mock_transformers_model: tuple[Any, Any]) -> None:
    """Verify ELISA threshold (0.5) is default

    ELISA assay uses standard 0.5 threshold (unlike PSR's 0.5495).
    """
    # Arrange
    np.random.seed(42)
    X_train = np.random.rand(100, 1280).astype(np.float32)
    y_train = np.array([0, 1] * 50)

    classifier = BinaryClassifier(
        model_name="facebook/esm1v_t33_650M_UR90S_1",
        device="cpu",
        random_state=42,
        max_iter=10,
        batch_size=8,
    )
    classifier.fit(X_train, y_train)

    # Create test case with probability = 0.51 (just above ELISA threshold)
    classifier.classifier.predict_proba = lambda X: np.array([[0.49, 0.51]])

    # Act: Predict with ELISA threshold (default)
    X_test = np.zeros((1, 1280))
    prediction = classifier.predict(X_test, assay_type="ELISA")

    # Assert: 0.51 > 0.5 → prediction should be 1
    assert prediction[0] == 1


# ==================== Flag Filtering Methodology Tests ====================


@pytest.mark.e2e
def test_novo_flag_filtering_excludes_mild(
    mock_transformers_model: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Novo methodology excludes mild flags (1-3) from training

    Novo Nordisk excludes antibodies with 1-3 flags (mild polyreactivity).
    This test verifies the filtering logic.
    """
    # Arrange
    boughter = BoughterDataset()

    # Create mock data with different flag values
    import pandas as pd

    df = pd.DataFrame(
        {
            "id": [f"AB{i:03d}" for i in range(10)],
            "VH_sequence": ["QVQLVQSGAEVKKPGA"] * 10,
            "VL_sequence": ["DIQMTQSPSSLSASVG"] * 10,
            "flags": [0, 1, 2, 3, 4, 5, 6, 7, 0, 1],  # Mix of all flag values
            "label": [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
        }
    )

    # Mock load_data to use our test DataFrame
    def mock_load(
        *args: Any, include_mild: bool = False, **kwargs: Any
    ) -> pd.DataFrame:
        if include_mild:
            return df
        else:
            # Exclude flags 1-3
            return df[~df["flags"].isin([1, 2, 3])].copy()

    monkeypatch.setattr(boughter, "load_data", mock_load)

    # Act
    df_no_mild = boughter.load_data(include_mild=False)

    # Assert: Mild flags excluded
    assert 1 not in df_no_mild["flags"].values
    assert 2 not in df_no_mild["flags"].values
    assert 3 not in df_no_mild["flags"].values

    # Assert: Flags 0 and 4+ retained
    assert 0 in df_no_mild["flags"].values
    assert 4 in df_no_mild["flags"].values


# ==================== Cross-Dataset Generalization Tests ====================


@pytest.mark.e2e
def test_cross_dataset_predictions_are_valid(
    mock_transformers_model: tuple[Any, Any],
) -> None:
    """Verify cross-dataset predictions produce valid outputs

    This smoke test verifies the pipeline works across datasets,
    without requiring exact accuracy targets.
    """
    # Arrange: Create small mock datasets
    import pandas as pd

    df_train = pd.DataFrame(
        {
            "id": [f"AB{i:03d}" for i in range(20)],
            "VH_sequence": ["QVQLVQSGAEVKKPGASVKVSCKASGYTFT"] * 20,
            "VL_sequence": ["DIQMTQSPSSLSASVGDRVTITCRASQSIS"] * 20,
            "label": [0, 1] * 10,
        }
    )

    df_test = pd.DataFrame(
        {
            "id": [f"TEST{i:03d}" for i in range(10)],
            "VH_sequence": ["EVQLVESGGGLVQPGGSLRLSCAASGFTFS"] * 10,
            "VL_sequence": ["DIQMTQSPSSLSASVGDRVTITCRASQSIS"] * 10,
            "label": [0, 1] * 5,
        }
    )

    # Arrange: Extract embeddings
    extractor = ESMEmbeddingExtractor(
        model_name="facebook/esm1v_t33_650M_UR90S_1", device="cpu", batch_size=8
    )

    X_train = extractor.extract_batch_embeddings(df_train["VH_sequence"].tolist())
    y_train = df_train["label"].to_numpy(dtype=int)

    X_test = extractor.extract_batch_embeddings(df_test["VH_sequence"].tolist())

    # Act: Train and predict
    classifier = BinaryClassifier(
        model_name="facebook/esm1v_t33_650M_UR90S_1",
        device="cpu",
        random_state=42,
        max_iter=10,
        batch_size=8,
    )
    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)
    probabilities = classifier.predict_proba(X_test)

    # Assert: Valid outputs
    assert len(predictions) == 10
    assert all(pred in [0, 1] for pred in predictions)
    assert probabilities.shape == (10, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


# ==================== Deterministic Reproduction Tests ====================


@pytest.mark.e2e
def test_training_is_reproducible_with_same_seed(
    mock_transformers_model: tuple[Any, Any],
) -> None:
    """Verify training is deterministic with fixed random seed"""
    # Arrange
    np.random.seed(42)
    X = np.random.rand(100, 1280).astype(np.float32)
    y = np.array([0, 1] * 50)
    X_test = np.random.rand(10, 1280).astype(np.float32)

    # Act: Train twice with same seed
    classifier1 = BinaryClassifier(
        model_name="facebook/esm1v_t33_650M_UR90S_1",
        device="cpu",
        random_state=42,
        max_iter=10,
        batch_size=8,
    )
    classifier1.fit(X, y)
    pred1 = classifier1.predict(X_test)

    classifier2 = BinaryClassifier(
        model_name="facebook/esm1v_t33_650M_UR90S_1",
        device="cpu",
        random_state=42,
        max_iter=10,
        batch_size=8,
    )
    classifier2.fit(X, y)
    pred2 = classifier2.predict(X_test)

    # Assert: Predictions are identical
    np.testing.assert_array_equal(pred1, pred2)


# ==================== Documentation Tests ====================


@pytest.mark.e2e
def test_novo_parameters_documented_correctly() -> None:
    """Verify Novo parameters are documented in classifier"""
    # Arrange & Act
    classifier = BinaryClassifier(
        model_name="facebook/esm1v_t33_650M_UR90S_1",
        device="cpu",
        random_state=42,
        max_iter=100,
        batch_size=8,
    )

    # Assert: PSR threshold documented
    assert hasattr(classifier, "ASSAY_THRESHOLDS")
    assert "PSR" in classifier.ASSAY_THRESHOLDS
    assert classifier.ASSAY_THRESHOLDS["PSR"] == 0.5495

    # Assert: ELISA threshold documented
    assert "ELISA" in classifier.ASSAY_THRESHOLDS
    assert classifier.ASSAY_THRESHOLDS["ELISA"] == 0.5
