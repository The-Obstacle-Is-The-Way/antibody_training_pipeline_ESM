"""
Binary Classifier Module

Professional binary classifier for antibody sequences using ESM-1V embeddings.
Includes sklearn compatibility, assay-specific thresholds, and model serialization.
"""

import logging
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from antibody_training_esm.core.config import DEFAULT_BATCH_SIZE
from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor

logger = logging.getLogger(__name__)


class BinaryClassifier:
    """Binary classifier for protein sequences using ESM-1V embeddings"""

    # Assay-specific thresholds (Novo Nordisk methodology)
    ASSAY_THRESHOLDS = {
        "ELISA": 0.5,  # Training data type (Boughter, Jain)
        "PSR": 0.5495,  # PSR assay type (Shehata, Harvey) - EXACT Novo parity
    }

    def __init__(self, params: dict[str, Any] | None = None, **kwargs: Any):
        """
        Initialize the binary classifier

        Args:
            params: Dictionary containing classifier parameters (legacy API)
            **kwargs: Individual parameters (for sklearn compatibility)

        Notes:
            Supports both dict-based (legacy) and kwargs-based (sklearn) initialization
        """
        # Support both dict-based (legacy) and kwargs-based (sklearn) initialization
        if params is None:
            params = kwargs

        random_state = params["random_state"]
        batch_size = params.get(
            "batch_size", DEFAULT_BATCH_SIZE
        )  # Default if not provided
        revision = params.get("revision", "main")  # HF model revision (default: "main")

        self.embedding_extractor = ESMEmbeddingExtractor(
            params["model_name"], params["device"], batch_size, revision=revision
        )

        # Get all LogisticRegression hyperparameters from config (with defaults)
        class_weight = params.get("class_weight", None)
        C = params.get("C", 1.0)
        penalty = params.get("penalty", "l2")
        solver = params.get("solver", "lbfgs")

        self.classifier = LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            random_state=params["random_state"],
            max_iter=params["max_iter"],
            class_weight=class_weight,
        )

        logger.info(
            "Classifier initialized: C=%s, penalty=%s, solver=%s, random_state=%s, class_weight=%s",
            C,
            penalty,
            solver,
            random_state,
            class_weight,
        )
        logger.info(
            "VERIFICATION: LogisticRegression config = C=%s, penalty=%s, solver=%s, class_weight=%s",
            self.classifier.C,
            self.classifier.penalty,
            self.classifier.solver,
            self.classifier.class_weight,
        )

        # Store all hyperparameters for recreation and sklearn compatibility
        self.random_state = random_state
        self.is_fitted = False
        self.device = self.embedding_extractor.device
        self.model_name = params["model_name"]
        self.max_iter = params["max_iter"]
        self.class_weight = class_weight
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.batch_size = batch_size
        self.revision = revision  # Store HF model revision for reproducibility

        # Store all params for sklearn compatibility
        self._params = params

    def get_params(self, deep: bool = True) -> dict[str, Any]:  # noqa: ARG002
        """
        Get parameters for sklearn compatibility (required for cross_val_score)

        Args:
            deep: If True, return parameters for sub-estimators (unused but required by sklearn API)

        Returns:
            Dictionary of parameters
        """
        # Return only valid constructor parameters (exclude 'type', 'cv_folds', 'stratify', etc.)
        valid_params = {
            "random_state": self.random_state,
            "max_iter": self.max_iter,
            "class_weight": self.class_weight,
            "C": self.C,
            "penalty": self.penalty,
            "solver": self.solver,
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
        }
        return valid_params

    def set_params(self, **params: Any) -> "BinaryClassifier":
        """
        Set parameters for sklearn compatibility (required for cross_val_score)

        Args:
            **params: Parameters to set

        Returns:
            self
        """
        self._params.update(params)
        # Reinitialize with new parameters (sklearn compatibility pattern)
        self.__init__(self._params)  # type: ignore[misc]
        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the classifier to the data

        Args:
            X: Array of ESM-1V embeddings
            y: Array of labels
        """
        # Fit the classifier directly on embeddings (no scaling per Novo methodology)
        self.classifier.fit(X, y)
        self.is_fitted = True
        logger.info(f"Classifier fitted on {len(X)} samples")

    def predict(
        self, X: np.ndarray, threshold: float = 0.5, assay_type: str | None = None
    ) -> np.ndarray:
        """
        Predict labels for the data with optional assay-specific thresholds

        Args:
            X: Array of ESM-1V embeddings
            threshold: Decision threshold for classification (default: 0.5)
                      Ignored if assay_type is specified
            assay_type: Type of assay for dataset-specific thresholds. Options:
                       - 'ELISA': Use threshold=0.5 (for Jain, Boughter datasets)
                       - 'PSR': Use threshold=0.549 (for Shehata, Harvey datasets)
                       - None: Use the threshold parameter

        Returns:
            Predicted labels

        Raises:
            ValueError: If classifier is not fitted or assay_type is unknown

        Notes:
            The model was trained on ELISA data (Boughter dataset). Different assay types
            measure different "spectrums" of non-specificity (Sakhnini et al. 2025, Section 2.7).
            Use assay_type='PSR' for PSR-based datasets to get calibrated predictions.
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before making predictions")

        # Determine which threshold to use
        if assay_type is not None:
            if assay_type not in self.ASSAY_THRESHOLDS:
                raise ValueError(
                    f"Unknown assay_type '{assay_type}'. Must be one of: {list(self.ASSAY_THRESHOLDS.keys())}"
                )
            threshold = self.ASSAY_THRESHOLDS[assay_type]

        # Get probabilities and apply threshold
        probabilities = self.classifier.predict_proba(X)
        predictions: np.ndarray = (probabilities[:, 1] > threshold).astype(int)

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for the data

        Args:
            X: Array of ESM-1V embeddings

        Returns:
            Predicted probabilities

        Raises:
            ValueError: If classifier is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before making predictions")

        result: np.ndarray = self.classifier.predict_proba(X)
        return result

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the mean accuracy on the given test data and labels

        Args:
            X: Array of ESM-1V embeddings
            y: Array of true labels

        Returns:
            Mean accuracy

        Raises:
            ValueError: If classifier is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before scoring")

        score: float = self.classifier.score(X, y)
        return score

    def __getstate__(self) -> dict[str, Any]:
        """Custom pickle method - don't save the ESM model"""
        state = self.__dict__.copy()
        # Remove the embedding_extractor (it will be recreated on load)
        state.pop("embedding_extractor", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Custom unpickle method - recreate ESM model with correct config"""
        self.__dict__.update(state)
        # Recreate embedding extractor with fixed configuration
        batch_size = getattr(
            self, "batch_size", DEFAULT_BATCH_SIZE
        )  # Default if not stored (backwards compatibility)
        revision = getattr(
            self, "revision", "main"
        )  # Default if not stored (backwards compatibility)
        self.embedding_extractor = ESMEmbeddingExtractor(
            self.model_name, self.device, batch_size, revision=revision
        )
