"""Unit tests for artifact models."""

import numpy as np
import pytest
from pydantic import ValidationError

from antibody_training_esm.models.artifact import (
    CVResults,
    EvaluationMetrics,
    ModelArtifactMetadata,
)


class TestModelArtifactMetadata:
    """Test ModelArtifactMetadata validation."""

    def test_valid_metadata(self) -> None:
        """Valid metadata constructs correctly."""
        metadata = ModelArtifactMetadata(
            model_name="facebook/esm1v_t33_650M_UR90S_1",
            model_type="logistic_regression",
            sklearn_version="1.3.0",
            classifier={"type": "logistic_regression", "C": 1.0},
            esm_model="facebook/esm1v_t33_650M_UR90S_1",
        )
        assert metadata.model_type == "logistic_regression"

    def test_class_weight_with_int_keys(self) -> None:
        """class_weight dict with int keys is preserved."""
        metadata = ModelArtifactMetadata(
            model_name="facebook/esm1v_t33_650M_UR90S_1",
            model_type="logistic_regression",
            sklearn_version="1.3.0",
            classifier={"type": "logistic_regression"},
            esm_model="facebook/esm1v_t33_650M_UR90S_1",
            class_weight={0: 1.0, 1: 2.0},  # Int keys
        )
        assert metadata.class_weight == {0: 1.0, 1: 2.0}

    def test_to_classifier_params(self) -> None:
        """to_classifier_params() extracts init params."""
        metadata = ModelArtifactMetadata(
            model_name="facebook/esm1v_t33_650M_UR90S_1",
            model_type="logistic_regression",
            sklearn_version="1.3.0",
            classifier={"type": "logistic_regression", "C": 1.0},
            esm_model="facebook/esm1v_t33_650M_UR90S_1",
            esm_revision="abc123",
            batch_size=32,
            device="cuda",
        )

        params = metadata.to_classifier_params()
        assert params["model_name"] == "facebook/esm1v_t33_650M_UR90S_1"
        assert params["device"] == "cuda"
        assert params["batch_size"] == 32


class TestEvaluationMetrics:
    """Test EvaluationMetrics validation."""

    def test_valid_metrics(self) -> None:
        """Valid metrics construct correctly."""
        metrics = EvaluationMetrics(
            accuracy=0.85,
            precision=0.80,
            recall=0.90,
            f1=0.85,
            roc_auc=0.88,
        )
        assert metrics.accuracy == 0.85

    def test_metrics_out_of_range_rejected(self) -> None:
        """Metrics must be 0-1."""
        with pytest.raises(ValidationError):
            EvaluationMetrics(
                accuracy=1.5,  # Out of range
            )

    def test_from_sklearn_metrics(self) -> None:
        """from_sklearn_metrics() constructs from arrays."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        y_proba = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.1, 0.9]])

        metrics = EvaluationMetrics.from_sklearn_metrics(
            y_true, y_pred, y_proba, dataset_name="Test"
        )

        assert metrics.accuracy == 0.75
        assert metrics.dataset_name == "Test"
        assert metrics.n_samples == 4
        assert metrics.confusion_matrix is not None


class TestCVResults:
    """Test CVResults validation."""

    def test_valid_cv_results(self) -> None:
        """Valid CV results construct correctly."""
        cv = CVResults(
            cv_accuracy={"mean": 0.82, "std": 0.05},
            n_splits=10,
        )
        assert cv.cv_accuracy["mean"] == 0.82

    def test_from_sklearn_cv_results(self) -> None:
        """from_sklearn_cv_results() constructs from sklearn output."""
        cv_scores: dict[str, list[float] | np.ndarray] = {
            "test_accuracy": [0.8, 0.82, 0.85, 0.81],
            "test_f1": [0.75, 0.78, 0.80, 0.77],
        }

        cv = CVResults.from_sklearn_cv_results(cv_scores, n_splits=4)

        assert cv.n_splits == 4
        assert cv.cv_accuracy is not None
        assert 0.81 < cv.cv_accuracy["mean"] < 0.83
