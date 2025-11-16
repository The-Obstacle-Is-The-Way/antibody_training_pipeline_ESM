"""
Binary Classifier Module

Professional binary classifier for antibody sequences using ESM-1V embeddings.
Includes sklearn compatibility, assay-specific thresholds, and model serialization.
"""

import logging
from typing import Any

import numpy as np

from antibody_training_esm.core.classifier_factory import create_classifier
from antibody_training_esm.core.classifier_strategy import ClassifierStrategy
from antibody_training_esm.core.config import DEFAULT_BATCH_SIZE
from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor

logger = logging.getLogger(__name__)


class BinaryClassifier:
    """Binary classifier for protein sequences using ESM-1V embeddings"""

    # sklearn 1.7+ requires explicit estimator type for cross_val_score
    # This tells sklearn's validation logic that we're a classifier, not a regressor
    _estimator_type = "classifier"

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

        # Validate required parameters
        REQUIRED_PARAMS = ["random_state", "model_name", "device", "max_iter"]
        missing = [p for p in REQUIRED_PARAMS if p not in params]
        if missing:
            raise ValueError(
                f"Missing required parameters: {missing}. "
                f"BinaryClassifier requires: {REQUIRED_PARAMS}"
            )

        random_state = params["random_state"]
        batch_size = params.get(
            "batch_size", DEFAULT_BATCH_SIZE
        )  # Default if not provided
        revision = params.get("revision", "main")  # HF model revision (default: "main")

        self.embedding_extractor = ESMEmbeddingExtractor(
            params["model_name"], params["device"], batch_size, revision=revision
        )

        # Use factory to create classifier strategy (supports LogReg, XGBoost, etc.)
        self.classifier: ClassifierStrategy = create_classifier(params)

        logger.info(
            "Classifier initialized: type=%s, params=%s",
            params.get("type", "logistic_regression"),
            self.classifier.get_params(),
        )

        # Store hyperparameters for recreation and sklearn compatibility
        self.random_state = random_state
        self.is_fitted = False
        self.device = self.embedding_extractor.device
        self.model_name = params["model_name"]
        self.batch_size = batch_size
        self.revision = revision  # Store HF model revision for reproducibility

        # Store all params for sklearn compatibility
        self._params = params

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """
        Get parameters for sklearn compatibility (required for cross_val_score)

        Args:
            deep: If True, return parameters for sub-estimators

        Returns:
            Dictionary of parameters (embedding params + classifier params)
        """
        # Merge embedding extractor params + classifier strategy params
        params = {
            "random_state": self.random_state,
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "revision": self.revision,
        }
        # Add classifier-specific params from strategy
        params.update(self.classifier.get_params(deep=deep))
        return params

    def set_params(self, **params: Any) -> "BinaryClassifier":
        """
        Set parameters for sklearn compatibility (required for cross_val_score)

        Args:
            **params: Parameters to set

        Returns:
            self

        Notes:
            This method updates parameters without destroying fitted state.
            If model_name or device changes, the embedding extractor is recreated.
            If classifier type changes, the classifier strategy is recreated.
        """
        # Update internal params dict
        self._params.update(params)

        # Track if we need to recreate components
        needs_extractor_reload = False
        needs_classifier_reload = False

        # Check which components need reloading
        embedding_params = {"model_name", "device", "batch_size", "revision"}
        if any(key in params for key in embedding_params):
            needs_extractor_reload = True
            # Update instance attributes
            self.model_name = self._params.get("model_name", self.model_name)
            self.device = self._params.get("device", self.device)
            self.batch_size = self._params.get("batch_size", self.batch_size)
            self.revision = self._params.get("revision", self.revision)

        if "type" in params:
            needs_classifier_reload = True

        # Update random_state (used by both components)
        if "random_state" in params:
            self.random_state = params["random_state"]

        # Recreate embedding extractor if needed
        if needs_extractor_reload:
            logger.info(
                f"Recreating embedding extractor: model_name={self.model_name}, "
                f"device={self.device}, batch_size={self.batch_size}"
            )
            self.embedding_extractor = ESMEmbeddingExtractor(
                self.model_name, self.device, self.batch_size, revision=self.revision
            )

        # Recreate classifier strategy if type changed
        if needs_classifier_reload:
            logger.info(f"Recreating classifier: type={params.get('type')}")
            self.classifier = create_classifier(self._params)
            self.is_fitted = False  # New classifier is unfitted
        else:
            # Update existing classifier params (e.g., C, penalty, solver)
            classifier_params = {
                k: v
                for k, v in params.items()
                if k not in embedding_params and k not in {"random_state", "type"}
            }
            if classifier_params:
                # For LogReg and other sklearn estimators, update attributes directly
                for key, value in classifier_params.items():
                    if hasattr(self.classifier, key):
                        setattr(self.classifier, key, value)
                        # Also update underlying sklearn classifier
                        if hasattr(self.classifier, "classifier") and hasattr(
                            self.classifier.classifier, key
                        ):
                            setattr(self.classifier.classifier, key, value)

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

        # sklearn 1.7+ requires classes_ attribute for cross_val_score compatibility
        self.classes_ = self.classifier.classes_

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

    # ========================================================================
    # Backward Compatibility Properties (delegate to strategy)
    # ========================================================================

    @property
    def C(self) -> float:
        """Regularization parameter (LogReg only, for backward compatibility)"""
        return getattr(self.classifier, "C", 1.0)

    @property
    def penalty(self) -> str:
        """Regularization type (LogReg only, for backward compatibility)"""
        return getattr(self.classifier, "penalty", "l2")

    @property
    def solver(self) -> str:
        """Optimization algorithm (LogReg only, for backward compatibility)"""
        return getattr(self.classifier, "solver", "lbfgs")

    @property
    def class_weight(self) -> Any:
        """Class weights (LogReg only, for backward compatibility)"""
        return getattr(self.classifier, "class_weight", None)

    @property
    def max_iter(self) -> int:
        """Maximum iterations (LogReg only, for backward compatibility)"""
        return getattr(self.classifier, "max_iter", 1000)

    def __getstate__(self) -> dict[str, Any]:
        """Custom pickle method - don't save the ESM model"""
        state = self.__dict__.copy()
        # Remove the embedding_extractor (it will be recreated on load)
        state.pop("embedding_extractor", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Custom unpickle method - recreate ESM model with correct config"""
        self.__dict__.update(state)

        # Check for missing attributes from old model versions
        warnings_issued = []
        if not hasattr(self, "batch_size"):
            warnings_issued.append(f"batch_size (using default: {DEFAULT_BATCH_SIZE})")
        if not hasattr(self, "revision"):
            warnings_issued.append("revision (using default: 'main')")

        if warnings_issued:
            import warnings

            warnings.warn(
                f"Loading old model missing attributes: {', '.join(warnings_issued)}. "
                "Predictions may differ from original model. Consider retraining with current version.",
                UserWarning,
                stacklevel=2,
            )

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
