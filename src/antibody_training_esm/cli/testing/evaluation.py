"""Metric calculation and model evaluation utilities."""

import logging
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from antibody_training_esm.core.classifier import BinaryClassifier

logger = logging.getLogger(__name__)


def detect_assay_type(dataset_name: str) -> str | None:
    """
    Auto-detect assay type from dataset name for threshold selection

    Args:
        dataset_name: Name of the dataset (e.g., "VH_only_jain", "VHH_only_harvey")

    Returns:
        'ELISA' for ELISA-based datasets (Boughter, Jain)
        'PSR' for PSR-based datasets (Harvey, Shehata)
        None if unable to detect

    Notes:
        Novo Nordisk (Sakhnini et al. 2025, Section 2.7):
        "Antibodies characterised by the PSR assay appear to be on a different
        non-specificity spectrum than that from the non-specificity ELISA assay."

        PSR datasets require threshold=0.5495 for optimal performance.
        ELISA datasets use standard threshold=0.5.
    """
    dataset_lower = dataset_name.lower()

    # PSR-based datasets (Harvey, Shehata)
    if any(marker in dataset_lower for marker in ["harvey", "shehata"]):
        return "PSR"

    # ELISA-based datasets (Boughter, Jain)
    if any(marker in dataset_lower for marker in ["boughter", "jain"]):
        return "ELISA"

    # Unable to detect - will use default threshold
    return None


def evaluate_pretrained(
    model: BinaryClassifier,
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    dataset_name: str,
    metrics_list: list[str] | None = None,
    threshold_override: float | None = None,
) -> dict[str, Any]:
    """
    Evaluate pretrained model directly on test set (no retraining)

    Args:
        model: The trained BinaryClassifier.
        X: Embeddings (features).
        y: True labels.
        model_name: Name of the model for logging.
        dataset_name: Name of the dataset for logging.
        metrics_list: List of metrics to calculate (default: all).
        threshold_override: Optional manual threshold.

    Returns:
        Dictionary of results including scores, predictions, and reports.
    """
    logger.info(f"Evaluating pretrained model {model_name} on {dataset_name}")

    # Determine threshold: manual override > auto-detect > default 0.5
    if threshold_override is not None:
        # Manual override via CLI
        threshold = threshold_override
        logger.info(f"Using manual threshold override: {threshold}")
    else:
        # Auto-detect assay type from dataset name
        assay_type = detect_assay_type(dataset_name)
        if assay_type is not None:
            threshold = model.ASSAY_THRESHOLDS[assay_type]
            logger.info(
                f"Auto-detected assay type: {assay_type} → threshold={threshold} "
                f"(Dataset: {dataset_name})"
            )
        else:
            threshold = 0.5
            logger.warning(
                f"Unable to auto-detect assay type for '{dataset_name}'. "
                f"Using default threshold={threshold}. "
                f"For optimal results, specify --threshold or use standard dataset names."
            )

    # Get predictions using the pretrained model with appropriate threshold
    y_pred = model.predict(
        X, threshold=threshold, assay_type=None
    )  # threshold already determined
    y_proba = model.predict_proba(X)[:, 1]

    # Calculate metrics
    results = {
        "test_scores": {},
        "predictions": {"y_true": y, "y_pred": y_pred, "y_proba": y_proba},
        "confusion_matrix": confusion_matrix(y, y_pred),
        "classification_report": classification_report(y, y_pred, output_dict=True),
    }

    # Calculate all requested metrics
    if metrics_list is not None:
        if "accuracy" in metrics_list:
            results["test_scores"]["accuracy"] = accuracy_score(y, y_pred)
        if "precision" in metrics_list:
            results["test_scores"]["precision"] = precision_score(
                y, y_pred, zero_division=0
            )
        if "recall" in metrics_list:
            results["test_scores"]["recall"] = recall_score(y, y_pred, zero_division=0)
        if "f1" in metrics_list:
            results["test_scores"]["f1"] = f1_score(y, y_pred, zero_division=0)
        if "roc_auc" in metrics_list:
            results["test_scores"]["roc_auc"] = roc_auc_score(y, y_proba)
        if "pr_auc" in metrics_list:
            results["test_scores"]["pr_auc"] = average_precision_score(y, y_proba)

    # Log results
    logger.info(f"Test results for {model_name} on {dataset_name}:")
    for metric, value in results["test_scores"].items():
        logger.info(f"  {metric}: {value:.4f}")

    return results
