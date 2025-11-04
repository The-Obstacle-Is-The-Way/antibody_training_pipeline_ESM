import logging
from typing import Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

from model import ESMEmbeddingExtractor

logger = logging.getLogger(__name__)


class BinaryClassifier:
    """Binary classifier for protein sequences using ESM-1V embeddings"""

    def __init__(self, params: Dict = None, **kwargs):
        """
        Initialize the binary classifier

        Args:
            params: Dictionary containing the parameters for the classifier (legacy API)
            **kwargs: Individual parameters (for sklearn compatibility)
        """
        # Support both dict-based (legacy) and kwargs-based (sklearn) initialization
        if params is None:
            params = kwargs

        random_state = params["random_state"]
        batch_size = params.get("batch_size", 32)  # Default to 32 if not provided

        self.embedding_extractor = ESMEmbeddingExtractor(
            params["model_name"], params["device"], batch_size
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
        print(
            f"Classifier initialized: C={C}, penalty={penalty}, solver={solver}, "
            f"random_state={random_state}, class_weight={class_weight}"
        )
        print(
            f"  VERIFICATION: LogisticRegression config = "
            f"C={self.classifier.C}, penalty={self.classifier.penalty}, "
            f"solver={self.classifier.solver}, class_weight={self.classifier.class_weight}"
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

        # Store all params for sklearn compatibility
        self._params = params

    def get_params(self, deep=True):
        """
        Get parameters for sklearn compatibility (required for cross_val_score)

        Args:
            deep: If True, return parameters for sub-estimators

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

    def set_params(self, **params):
        """
        Set parameters for sklearn compatibility (required for cross_val_score)

        Args:
            **params: Parameters to set

        Returns:
            self
        """
        self._params.update(params)
        # Reinitialize with new parameters
        self.__init__(self._params)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray):
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
        self, X: np.ndarray, threshold: float = 0.5, assay_type: Optional[str] = None
    ) -> np.ndarray:
        """
        Predict the labels for the data

        Args:
            X: Array of ESM-1V embeddings
            threshold: Decision threshold for classification (default: 0.5)
                      If assay_type is specified, this parameter is ignored
            assay_type: Type of assay for dataset-specific thresholds. Options:
                       - 'ELISA': Use threshold=0.5 (for Jain, Boughter datasets)
                       - 'PSR': Use threshold=0.549 (for Shehata, Harvey datasets)
                       - None: Use the threshold parameter

        Returns:
            Predicted labels

        Notes:
            The model was trained on ELISA data (Boughter dataset). Different assay types
            measure different "spectrums" of non-specificity (Sakhnini et al. 2025, Section 2.7).
            Use assay_type='PSR' for PSR-based datasets to get calibrated predictions.
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before making predictions")

        # Dataset-specific threshold mapping
        ASSAY_THRESHOLDS = {
            "ELISA": 0.5,  # Training data type (Boughter, Jain)
            "PSR": 0.5495,  # PSR assay type (Shehata, Harvey) - EXACT Novo parity
        }

        # Determine which threshold to use
        if assay_type is not None:
            if assay_type not in ASSAY_THRESHOLDS:
                raise ValueError(
                    f"Unknown assay_type '{assay_type}'. Must be one of: {list(ASSAY_THRESHOLDS.keys())}"
                )
            threshold = ASSAY_THRESHOLDS[assay_type]

        # Get probabilities and apply threshold
        probabilities = self.classifier.predict_proba(X)
        predictions = (probabilities[:, 1] > threshold).astype(int)

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for the data

        Args:
            X: Array of ESM-1V embeddings

        Returns:
            Predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before making predictions")

        return self.classifier.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the mean accuracy on the given test data and labels

        Args:
            X: Array of ESM-1V embeddings
            y: Array of true labels

        Returns:
            Mean accuracy
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before scoring")

        return self.classifier.score(X, y)

    def __getstate__(self):
        """Custom pickle method - don't save the ESM model"""
        state = self.__dict__.copy()
        # Remove the embedding_extractor (it will be recreated on load)
        state.pop("embedding_extractor", None)
        return state

    def __setstate__(self, state):
        """Custom unpickle method - recreate ESM model with correct config"""
        self.__dict__.update(state)
        # Recreate embedding extractor with fixed configuration
        batch_size = getattr(
            self, "batch_size", 32
        )  # Default to 32 if not stored (backwards compatibility)
        self.embedding_extractor = ESMEmbeddingExtractor(
            self.model_name, self.device, batch_size
        )
