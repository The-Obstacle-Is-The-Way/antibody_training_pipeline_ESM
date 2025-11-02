import logging
from typing import Dict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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
        self.scaler = StandardScaler()

        # Get class_weight parameter if provided, otherwise use None (default)
        class_weight = params.get("class_weight", None)

        self.classifier = LogisticRegression(
            random_state=params["random_state"],
            max_iter=params["max_iter"],
            class_weight=class_weight,  # FIXED: Use the parameter from config!
        )
        print(
            f"Classifier initialized with random state: {random_state}, class_weight: {class_weight}"
        )
        print(
            f"  VERIFICATION: LogisticRegression.class_weight = {self.classifier.class_weight}"
        )
        self.random_state = random_state
        self.is_fitted = False
        self.device = self.embedding_extractor.device
        self.model_name = params["model_name"]  # Store for recreation
        self.max_iter = params["max_iter"]
        self.class_weight = class_weight
        self.batch_size = batch_size  # Store for recreation

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
        # Scale the embeddings
        X_scaled = self.scaler.fit_transform(X)

        # Fit the classifier
        self.classifier.fit(X_scaled, y)
        self.is_fitted = True
        logger.info(f"Classifier fitted on {len(X)} samples")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the labels for the data

        Args:
            X: Array of ESM-1V embeddings

        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before making predictions")

        X_scaled = self.scaler.transform(X)
        return self.classifier.predict(X_scaled)

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

        X_scaled = self.scaler.transform(X)
        return self.classifier.predict_proba(X_scaled)

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

        X_scaled = self.scaler.transform(X)
        return self.classifier.score(X_scaled, y)

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
