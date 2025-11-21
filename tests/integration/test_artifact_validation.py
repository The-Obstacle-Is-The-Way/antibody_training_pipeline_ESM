"""
Integration tests for artifact validation.

Verifies that the training pipeline produces valid artifacts that conform to
Pydantic schemas:
- ExperimentMetadata
- TrainingMetrics (EvaluationMetrics)
- EvaluationResults (CVResults)
- CheckpointMetadata (ModelArtifactMetadata)
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from antibody_training_esm.core.classifier import BinaryClassifier
from antibody_training_esm.core.trainer import save_model
from antibody_training_esm.models.artifact import (
    CVResults,
    EvaluationMetrics,
    ModelArtifactMetadata,
)
from antibody_training_esm.models.config import TrainingPipelineConfig


@pytest.fixture
def mock_training_artifacts(tmp_path: Path) -> dict[str, Any]:
    """Generate mock artifacts for testing validation."""
    # Create a dummy classifier
    classifier = BinaryClassifier(
        model_name="facebook/esm1v_t33_650M_UR90S_1",
        device="cpu",
        random_state=42,
        max_iter=100,
        C=1.0,
        penalty="l2",
        solver="lbfgs",
        class_weight={0: 1.0, 1: 2.0},
    )
    X = np.random.rand(10, 1280)
    y = np.array([0, 1] * 5)
    classifier.fit(X, y)

    # Create config
    config_dict = {
        "model": {"name": "facebook/esm1v_t33_650M_UR90S_1", "device": "cpu"},
        "data": {
            "train_file": str(tmp_path / "train.csv"),
            "test_file": str(tmp_path / "test.csv"),
            "embeddings_cache_dir": str(tmp_path / "cache"),
        },
        "classifier": {"C": 1.0},
        "training": {"model_name": "test_model", "model_save_dir": str(tmp_path)},
        "experiment": {"name": "test_experiment"},
    }
    # Create dummy files for config validation
    (tmp_path / "train.csv").touch()
    (tmp_path / "test.csv").touch()

    config = TrainingPipelineConfig.model_validate(config_dict)

    return {
        "classifier": classifier,
        "config": config,
        "logger": pytest.importorskip("logging").getLogger("test"),
        "output_dir": tmp_path,
    }


@pytest.mark.integration
def test_save_model_produces_valid_metadata(
    mock_training_artifacts: dict[str, Any],
) -> None:
    """Verify save_model produces a valid ModelArtifactMetadata JSON."""
    classifier = mock_training_artifacts["classifier"]
    config = mock_training_artifacts["config"]
    logger = mock_training_artifacts["logger"]

    # Attach mock metrics
    mock_metrics = EvaluationMetrics(accuracy=0.95).model_dump(
        mode="json", exclude_none=True
    )
    config.train_metrics = mock_metrics

    paths = save_model(classifier, config, logger)
    json_path = paths["config"]

    # Validate
    with open(json_path) as f:
        data = json.load(f)

    metadata = ModelArtifactMetadata.model_validate(data)
    assert metadata.model_type == "logistic_regression"
    assert metadata.training_metrics is not None
    assert metadata.training_metrics["accuracy"] == 0.95
    assert metadata.class_weight == {0: 1.0, 1: 2.0}


@pytest.mark.integration
def test_cv_results_serialization(tmp_path: Path) -> None:
    """Verify CV results can be saved and reloaded as valid Pydantic model."""
    import logging

    from antibody_training_esm.core.training.metrics import save_cv_results

    cv_results = CVResults(
        cv_accuracy={"mean": 0.85, "std": 0.05},
        n_splits=5,
        fold_results=[
            EvaluationMetrics(accuracy=0.80),
            EvaluationMetrics(accuracy=0.90),
        ],
    )

    output_dir = tmp_path / "results"
    save_cv_results(cv_results, output_dir, "test_exp", logging.getLogger("test"))

    yaml_path = output_dir / "cv_results.yaml"
    assert yaml_path.exists()

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    # Re-validate from loaded data (extract inner metrics)
    loaded_results = CVResults.model_validate(data["cv_metrics"])
    assert loaded_results.cv_accuracy["mean"] == 0.85
    assert loaded_results.fold_results is not None
    assert len(loaded_results.fold_results) == 2


@pytest.mark.integration
def test_evaluation_metrics_from_real_predictions() -> None:
    """Verify EvaluationMetrics handles real sklearn outputs correctly."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0])  # 75% accuracy
    y_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.6, 0.4]])

    metrics = EvaluationMetrics.from_sklearn_metrics(
        y_true, y_pred, y_proba, dataset_name="IntegrationTest"
    )

    assert metrics.accuracy == 0.75
    assert metrics.confusion_matrix == [[2, 0], [1, 1]]
    assert metrics.dataset_name == "IntegrationTest"
