import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from antibody_training_esm.core.classifier import BinaryClassifier
from antibody_training_esm.core.config import DEFAULT_BATCH_SIZE


def evaluate_model(
    classifier: BinaryClassifier,
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    metrics: list[str],
    logger: logging.Logger,
) -> dict[str, float]:
    """
    Evaluate model performance

    Args:
        classifier: Trained classifier
        X: Embeddings array
        y: Labels array
        dataset_name: Name of dataset being evaluated
        metrics: List of metrics to compute
        logger: Logger instance

    Returns:
        Dictionary of metric results
    """
    logger.info(f"Evaluating model on {dataset_name} set")

    # Get predictions
    y_pred = classifier.predict(X)
    y_pred_proba = classifier.predict_proba(X)[:, 1]  # Probability of positive class

    # Calculate metrics
    results = {}

    if "accuracy" in metrics:
        results["accuracy"] = accuracy_score(y, y_pred)

    if "precision" in metrics:
        results["precision"] = precision_score(y, y_pred, average="binary")

    if "recall" in metrics:
        results["recall"] = recall_score(y, y_pred, average="binary")

    if "f1" in metrics:
        results["f1"] = f1_score(y, y_pred, average="binary")

    if "roc_auc" in metrics:
        results["roc_auc"] = roc_auc_score(y, y_pred_proba)

    # Log results
    logger.info(f"{dataset_name} Results:")
    for metric, value in results.items():
        logger.info(f"  {metric}: {value:.4f}")

    # Log classification report
    logger.info(f"\n{dataset_name} Classification Report:")
    logger.info(f"\n{classification_report(y, y_pred)}")

    return results


def perform_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    config: dict[str, Any],
    logger: logging.Logger,
) -> dict[str, dict[str, float]]:
    """
    Perform cross-validation

    Args:
        X: Embeddings array
        y: Labels array
        config: Configuration dictionary
        logger: Logger instance

    Returns:
        Dictionary of cross-validation results
    """
    cv_config = config["classifier"]
    cv_folds = cv_config["cv_folds"]
    random_state = cv_config["random_state"]
    stratify = cv_config["stratify"]

    logger.info(f"Performing {cv_folds}-fold cross-validation")

    # Setup cross-validation
    if stratify:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # Perform cross-validation for different metrics
    cv_results = {}

    # Create a new classifier instance for CV (to avoid fitting on full data)
    cv_params = config["classifier"].copy()
    cv_params["model_name"] = config["model"]["name"]
    cv_params["device"] = config["model"]["device"]
    cv_params["batch_size"] = config["training"].get("batch_size", DEFAULT_BATCH_SIZE)
    cv_classifier = BinaryClassifier(cv_params)

    # Use full BinaryClassifier for CV (no StandardScaler - matches Novo methodology)

    # Accuracy
    scores = cross_val_score(cv_classifier, X, y, cv=cv, scoring="accuracy")
    cv_results["cv_accuracy"] = {"mean": scores.mean(), "std": scores.std()}

    # F1 score
    scores = cross_val_score(cv_classifier, X, y, cv=cv, scoring="f1")
    cv_results["cv_f1"] = {"mean": scores.mean(), "std": scores.std()}

    # ROC AUC
    scores = cross_val_score(cv_classifier, X, y, cv=cv, scoring="roc_auc")
    cv_results["cv_roc_auc"] = {"mean": scores.mean(), "std": scores.std()}

    # Log results
    logger.info("Cross-validation Results:")
    for metric, values in cv_results.items():
        logger.info(f"  {metric}: {values['mean']:.4f} (+/- {values['std'] * 2:.4f})")

    return cv_results


def save_cv_results(
    cv_results: dict[str, dict[str, float]],
    output_dir: Path,
    experiment_name: str,
    logger: logging.Logger,
) -> None:
    """
    Save cross-validation results to structured YAML file.

    Args:
        cv_results: Dictionary of CV metrics with mean/std
        output_dir: Directory to save CV results file
        experiment_name: Name of the experiment
        logger: Logger instance

    Example output:
        experiment: novo_replication
        timestamp: 2025-11-15T17:30:00
        cv_metrics:
          cv_accuracy:
            mean: 0.6413
            std: 0.0972
          cv_f1:
            mean: 0.6604
            std: 0.0994
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    cv_file = output_dir / "cv_results.yaml"

    # Convert numpy types to native Python floats for clean YAML
    cv_results_clean = {}
    for metric, values in cv_results.items():
        cv_results_clean[metric] = {
            "mean": float(values["mean"]),
            "std": float(values["std"]),
        }

    with open(cv_file, "w") as f:
        yaml.dump(
            {
                "experiment": experiment_name,
                "timestamp": datetime.now().isoformat(),
                "cv_metrics": cv_results_clean,
            },
            f,
            default_flow_style=False,
        )

    logger.info(f"CV results saved to {cv_file}")
