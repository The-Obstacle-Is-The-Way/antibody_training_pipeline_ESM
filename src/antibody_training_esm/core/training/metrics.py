"""
Evaluation metrics and cross-validation logic.

Computes accuracy, F1, ROC-AUC, and other classification metrics.
Handles logging and result storage.
"""

import logging
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate

from antibody_training_esm.core.classifier import BinaryClassifier
from antibody_training_esm.core.config import DEFAULT_BATCH_SIZE
from antibody_training_esm.models.artifact import CVResults, EvaluationMetrics

if TYPE_CHECKING:
    from antibody_training_esm.models.config import TrainingPipelineConfig


def evaluate_model(
    classifier: BinaryClassifier,
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    _metrics: Sequence[str] | set[str],
    logger: logging.Logger,
) -> EvaluationMetrics:
    """
    Evaluate model performance

    Args:
        classifier: Trained classifier
        X: Embeddings array
        y: Labels array
        dataset_name: Name of dataset being evaluated
        _metrics: List/Set of metrics to compute (ignored, computes all standard metrics)
        logger: Logger instance

    Returns:
        EvaluationMetrics Pydantic model
    """
    logger.info(f"Evaluating model on {dataset_name} set")

    # Get predictions
    y_pred = classifier.predict(X)
    y_pred_proba = classifier.predict_proba(X)

    # Create metrics object using Pydantic factory
    eval_metrics = EvaluationMetrics.from_sklearn_metrics(
        y_true=y,
        y_pred=y_pred,
        y_proba=y_pred_proba,
        dataset_name=dataset_name,
    )

    # Log results
    logger.info(f"{dataset_name} Results:")
    logger.info(f"  Accuracy:  {eval_metrics.accuracy:.4f}")
    if eval_metrics.precision is not None:
        logger.info(f"  Precision: {eval_metrics.precision:.4f}")
    if eval_metrics.recall is not None:
        logger.info(f"  Recall:    {eval_metrics.recall:.4f}")
    if eval_metrics.f1 is not None:
        logger.info(f"  F1:        {eval_metrics.f1:.4f}")
    if eval_metrics.roc_auc is not None:
        logger.info(f"  ROC-AUC:   {eval_metrics.roc_auc:.4f}")

    # Log classification report (useful for detailed class-wise metrics)
    logger.info(f"\n{dataset_name} Classification Report:")
    logger.info(f"\n{classification_report(y, y_pred)}")

    return eval_metrics


def perform_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    config: "TrainingPipelineConfig | dict[str, Any]",
    logger: logging.Logger,
) -> CVResults:
    """
    Perform cross-validation

    Args:
        X: Embeddings array
        y: Labels array
        config: Configuration (Pydantic object or legacy dict)
        logger: Logger instance

    Returns:
        CVResults Pydantic model
    """
    from antibody_training_esm.models.config import TrainingPipelineConfig

    # Extract parameters based on config type
    if isinstance(config, TrainingPipelineConfig):
        cv_folds = config.training.n_splits
        random_state = config.training.random_state
        stratify = config.training.stratify
        model_name = config.model.name
        device = config.model.device
        batch_size = config.model.batch_size

        clf_params = config.classifier.model_dump()
    else:
        training_conf = config.get("training", {})
        classifier_conf = config.get("classifier", {})

        cv_folds = training_conf.get("n_splits", classifier_conf.get("cv_folds", 10))
        stratify = training_conf.get("stratify", True)
        random_state = training_conf.get(
            "random_state", classifier_conf.get("random_state", 42)
        )

        model_cfg = config.get("model", {})
        model_name = model_cfg.get("name", "")
        device = model_cfg.get("device", "cpu")
        batch_size = training_conf.get(
            "batch_size", model_cfg.get("batch_size", DEFAULT_BATCH_SIZE)
        )
        clf_params = classifier_conf.copy()

    logger.info(f"Performing {cv_folds}-fold cross-validation")

    # Setup cross-validation
    if stratify:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # Create a new classifier instance for CV
    cv_params = clf_params.copy()
    cv_params["model_name"] = model_name
    cv_params["device"] = device
    cv_params["batch_size"] = batch_size

    cv_classifier = BinaryClassifier(cv_params)

    # Define metrics to compute
    scoring = {
        "accuracy": "accuracy",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "roc_auc": "roc_auc",
    }

    # Perform cross-validation using cross_validate (more efficient than multiple cross_val_score calls)
    cv_scores = cross_validate(
        cv_classifier, X, y, cv=cv, scoring=scoring, return_train_score=False
    )

    # Create CVResults object using Pydantic factory
    cv_results = CVResults.from_sklearn_cv_results(cv_scores, n_splits=cv_folds)

    # Log results
    logger.info("Cross-validation Results:")
    logger.info(
        f"  Accuracy: {cv_results.cv_accuracy['mean']:.4f} (+/- {cv_results.cv_accuracy['std'] * 2:.4f})"
    )
    if cv_results.cv_f1:
        logger.info(
            f"  F1:       {cv_results.cv_f1['mean']:.4f} (+/- {cv_results.cv_f1['std'] * 2:.4f})"
        )
    if cv_results.cv_roc_auc:
        logger.info(
            f"  ROC-AUC:  {cv_results.cv_roc_auc['mean']:.4f} (+/- {cv_results.cv_roc_auc['std'] * 2:.4f})"
        )

    return cv_results


def save_cv_results(
    cv_results: CVResults,
    output_dir: Path,
    experiment_name: str,
    logger: logging.Logger,
) -> None:
    """
    Save cross-validation results to structured YAML file.

    Args:
        cv_results: CVResults Pydantic model
        output_dir: Directory to save CV results file
        experiment_name: Name of the experiment
        logger: Logger instance
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    cv_file = output_dir / "cv_results.yaml"

    # Use Pydantic's model_dump for clean serialization
    results_dict = cv_results.model_dump(mode="json")

    with open(cv_file, "w") as f:
        yaml.dump(
            {
                "experiment": experiment_name,
                "timestamp": datetime.now().isoformat(),
                "cv_metrics": results_dict,
            },
            f,
            default_flow_style=False,
        )

    logger.info(f"CV results saved to {cv_file}")
