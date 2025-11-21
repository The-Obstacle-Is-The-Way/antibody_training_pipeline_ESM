"""
Pydantic models for model artifacts and metrics.

This module defines the schema for:
1. Saved model metadata (JSON sidecar)
2. Evaluation metrics (accuracy, F1, etc.)
3. Cross-validation results
"""

from typing import Any, Literal, cast

import numpy as np
from pydantic import BaseModel, Field


class ModelArtifactMetadata(BaseModel):
    """
    Metadata for saved model artifacts.

    This model structures the JSON sidecar file that accompanies
    NPZ/XGB model files. It enables version checking and parameter
    reconstruction.
    """

    # Model architecture
    model_name: str = Field(
        ...,
        description="HuggingFace ESM model ID",
        examples=["facebook/esm1v_t33_650M_UR90S_1"],
    )

    model_type: Literal["logistic_regression", "xgboost", "random_forest"] = Field(
        ...,
        description="Classifier type",
    )

    sklearn_version: str = Field(
        ...,
        description="scikit-learn version used for training",
        examples=["1.3.0"],
    )

    # Classifier configuration (strategy-specific)
    classifier: dict[str, Any] = Field(
        ...,
        description="Full classifier config from to_dict() method",
    )

    # ESM embedding extractor params
    esm_model: str = Field(
        ...,
        description="ESM model name (redundant with model_name, kept for compat)",
    )

    esm_revision: str = Field(
        default="main",
        description="HuggingFace model revision (commit hash)",
    )

    batch_size: int = Field(
        default=16,
        ge=1,
        description="Batch size for embedding extraction",
    )

    device: str = Field(
        default="cpu",
        description="Device used during training",
    )

    # Legacy flat fields (LogReg only, for backward compatibility)
    C: float | None = Field(
        default=None,
        description="LogReg: Inverse regularization strength",
    )

    penalty: Literal["l1", "l2"] | None = Field(
        default=None,
        description="LogReg: Regularization type",
    )

    solver: str | None = Field(
        default=None,
        description="LogReg: Optimization algorithm",
    )

    # Pydantic handles dict[int, float] keys automatically (converts string keys back to int)
    class_weight: Literal["balanced"] | dict[int, float] | None = Field(
        default=None,
        description="Class weighting strategy",
    )

    max_iter: int | None = Field(
        default=None,
        description="LogReg: Maximum iterations",
    )

    random_state: int | None = Field(
        default=None,
        description="Random seed",
    )

    # Optional metrics from training
    training_metrics: dict[str, float] | None = Field(
        default=None,
        description="Metrics from final training run",
    )

    @classmethod
    def from_classifier(cls, classifier: Any) -> "ModelArtifactMetadata":
        """
        Construct metadata from BinaryClassifier instance.

        Args:
            classifier: Trained BinaryClassifier

        Returns:
            ModelArtifactMetadata
        """
        import sklearn

        strategy_config = classifier.classifier.to_dict()
        classifier_type = strategy_config.get("type", "logistic_regression")

        metadata_dict = {
            # Model architecture
            "model_name": classifier.model_name,
            "model_type": classifier_type,
            "sklearn_version": sklearn.__version__,
            # Classifier config (strategy-specific)
            "classifier": strategy_config,
            # ESM params
            "esm_model": classifier.model_name,
            "esm_revision": classifier.revision,
            "batch_size": classifier.batch_size,
            "device": classifier.device,
        }

        # Add legacy flat fields for LogReg (backward compat)
        if classifier_type == "logistic_regression":
            metadata_dict.update(
                {
                    "C": classifier.C,
                    "penalty": classifier.penalty,
                    "solver": classifier.solver,
                    "class_weight": classifier.class_weight,
                    "max_iter": classifier.max_iter,
                    "random_state": classifier.random_state,
                }
            )

        return cls.model_validate(metadata_dict)

    def to_classifier_params(self) -> dict[str, Any]:
        """
        Extract parameters for BinaryClassifier reconstruction.

        Returns:
            Dict of parameters for BinaryClassifier(...) init
        """
        params = {
            # ESM params
            "model_name": self.esm_model,
            "device": self.device,
            "batch_size": self.batch_size,
            "revision": self.esm_revision,
            # Classifier params
            **self.classifier,
        }

        # Overwrite with typed fields for LogReg to ensure correct types (e.g. int keys in dict)
        if self.model_type == "logistic_regression":
            params.update(
                {
                    "C": self.C,
                    "penalty": self.penalty,
                    "solver": self.solver,
                    "class_weight": self.class_weight,
                    "max_iter": self.max_iter,
                    "random_state": self.random_state,
                }
            )

        return params


class EvaluationMetrics(BaseModel):
    """
    Evaluation metrics for a single dataset.

    Used for training set, test set, and cross-validation fold results.
    """

    accuracy: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Classification accuracy (0-1)",
    )

    precision: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Precision (positive predictive value)",
    )

    recall: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Recall (sensitivity, true positive rate)",
    )

    f1: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="F1 score (harmonic mean of precision and recall)",
    )

    roc_auc: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Area under ROC curve",
    )

    # Optional confusion matrix
    confusion_matrix: list[list[int]] | None = Field(
        default=None,
        description="Confusion matrix [[TN, FP], [FN, TP]]",
    )

    # Dataset metadata
    dataset_name: str | None = Field(
        default=None,
        description="Name of evaluated dataset (e.g., 'Jain', 'Training')",
    )

    n_samples: int | None = Field(
        default=None,
        ge=0,
        description="Number of samples in dataset",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "accuracy": 0.6628,
                    "precision": 0.47,
                    "recall": 0.63,
                    "f1": 0.54,
                    "roc_auc": 0.68,
                    "confusion_matrix": [[40, 19], [10, 17]],
                    "dataset_name": "Jain",
                    "n_samples": 86,
                }
            ]
        }
    }

    @classmethod
    def from_sklearn_metrics(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None = None,
        dataset_name: str | None = None,
    ) -> "EvaluationMetrics":
        """
        Construct metrics from sklearn predictions.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (for ROC-AUC)
            dataset_name: Name of dataset

        Returns:
            EvaluationMetrics
        """
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        metrics_dict = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "dataset_name": dataset_name,
            "n_samples": len(y_true),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }

        # ROC-AUC requires probabilities
        if y_proba is not None:
            try:
                # Check if y_proba has 2 columns (binary classification)
                if y_proba.ndim == 2 and y_proba.shape[1] >= 2:
                    score = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    # Fallback for 1D array if passed incorrectly
                    score = roc_auc_score(y_true, y_proba)
                metrics_dict["roc_auc"] = float(score)
            except ValueError:
                # ROC AUC might fail if only one class is present in y_true
                metrics_dict["roc_auc"] = None

        return cls.model_validate(metrics_dict)


class CVResults(BaseModel):
    """
    Cross-validation results with mean and std for each metric.

    Aggregates metrics across all CV folds.
    """

    cv_accuracy: dict[Literal["mean", "std"], float] = Field(
        ...,
        description="Mean and std of accuracy across folds",
    )

    cv_precision: dict[Literal["mean", "std"], float] | None = Field(
        default=None,
        description="Mean and std of precision",
    )

    cv_recall: dict[Literal["mean", "std"], float] | None = Field(
        default=None,
        description="Mean and std of recall",
    )

    cv_f1: dict[Literal["mean", "std"], float] | None = Field(
        default=None,
        description="Mean and std of F1 score",
    )

    cv_roc_auc: dict[Literal["mean", "std"], float] | None = Field(
        default=None,
        description="Mean and std of ROC-AUC",
    )

    n_splits: int = Field(
        ...,
        ge=2,
        description="Number of cross-validation folds",
    )

    # Optional: per-fold results
    fold_results: list[EvaluationMetrics] | None = Field(
        default=None,
        description="Metrics for each individual fold",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "cv_accuracy": {"mean": 0.82, "std": 0.05},
                    "cv_precision": {"mean": 0.78, "std": 0.06},
                    "cv_recall": {"mean": 0.85, "std": 0.04},
                    "cv_f1": {"mean": 0.81, "std": 0.05},
                    "cv_roc_auc": {"mean": 0.87, "std": 0.03},
                    "n_splits": 10,
                }
            ]
        }
    }

    @classmethod
    def from_sklearn_cv_results(
        cls,
        cv_scores: dict[str, list[float] | np.ndarray],
        n_splits: int,
    ) -> "CVResults":
        """
        Construct CVResults from sklearn cross_validate output.

        Args:
            cv_scores: Dict like {"test_accuracy": [...], "test_f1": [...]}
            n_splits: Number of folds

        Returns:
            CVResults
        """
        results_dict: dict[str, Any] = {"n_splits": n_splits}

        # Map sklearn metric names to our field names
        metric_map = {
            "test_accuracy": "cv_accuracy",
            "test_precision": "cv_precision",
            "test_recall": "cv_recall",
            "test_f1": "cv_f1",
            "test_roc_auc": "cv_roc_auc",
        }

        for sklearn_name, pydantic_name in metric_map.items():
            if sklearn_name in cv_scores:
                scores = cv_scores[sklearn_name]
                # Handle potential NaN in scores
                valid_scores = [s for s in scores if not np.isnan(s)]

                if valid_scores:
                    results_dict[pydantic_name] = {
                        "mean": float(np.mean(valid_scores)),
                        "std": float(np.std(valid_scores)),
                    }
                else:
                    results_dict[pydantic_name] = {
                        "mean": 0.0,
                        "std": 0.0,
                    }

        return cast(CVResults, cls.model_validate(results_dict))
