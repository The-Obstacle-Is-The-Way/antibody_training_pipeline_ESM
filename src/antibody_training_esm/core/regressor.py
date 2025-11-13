"""Regression model for continuous antibody property prediction.

This module provides AntibodyRegressor, a regression model for predicting
continuous antibody properties (e.g., polyreactivity scores) using ESM embeddings
and Ridge regression.

Example:
    >>> regressor = AntibodyRegressor(alpha=1.0, device='cpu')
    >>> regressor.fit(sequences, continuous_labels)
    >>> predictions = regressor.predict(test_sequences)
"""

import pickle
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor


class AntibodyRegressor:
    """
    Regression model for predicting continuous antibody properties.

    This class combines ESM protein language model embeddings with Ridge regression
    to predict continuous values (e.g., polyreactivity scores, thermostability).

    Architecture:
        1. Extract ESM embeddings from VH sequences
        2. Train Ridge regression on embeddings
        3. Predict continuous values (not binary)

    This is similar to BinaryClassifier but for regression tasks.

    Example:
        >>> # Train regressor
        >>> regressor = AntibodyRegressor(alpha=1.0, device='cpu')
        >>> regressor.fit(
        ...     sequences=['QVQLQQ...', 'EVQLVE...'],
        ...     labels=np.array([0.3, 0.7]),
        ...     cache_path='embeddings_cache/train.npy'
        ... )
        >>>
        >>> # Predict continuous values
        >>> preds = regressor.predict(['QVKLQE...'])
        >>> print(f"Predicted value: {preds[0]:.3f}")
        Predicted value: 0.45

    Attributes:
        model_name: HuggingFace ESM model name
        device: Device for ESM model (cpu/cuda/mps)
        batch_size: Batch size for embedding extraction
        alpha: Ridge regularization strength (higher = more regularization)
        embedding_extractor: ESM embedding extractor
        regressor: Fitted Ridge regression model
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        device: str = "cpu",
        batch_size: int = 8,
        alpha: float = 1.0,
        **regressor_kwargs: Any,
    ) -> None:
        """
        Initialize regression model.

        Args:
            model_name: HuggingFace ESM model name (e.g., facebook/esm2_t33_650M_UR50D)
            device: Device for ESM model (cpu/cuda/mps)
            batch_size: Batch size for embedding extraction
            alpha: Ridge regularization strength (like 1/C in LogisticRegression)
                Higher alpha = more regularization (simpler model)
                Common values: 0.1, 0.5, 1.0, 5.0, 10.0
            **regressor_kwargs: Additional kwargs for Ridge (fit_intercept, max_iter, etc.)

        Example:
            >>> # Default ESM2-650M model with alpha=1.0
            >>> regressor = AntibodyRegressor()
            >>>
            >>> # Custom regularization
            >>> regressor = AntibodyRegressor(alpha=0.5, fit_intercept=False)
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.alpha = alpha
        self.regressor_kwargs = regressor_kwargs

        # Initialize ESM embedding extractor
        self.embedding_extractor = ESMEmbeddingExtractor(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
        )

        # Initialize Ridge regressor
        self.regressor = Ridge(alpha=alpha, **regressor_kwargs)
        self._is_fitted = False

    def fit(
        self,
        sequences: list[str] | np.ndarray,
        labels: np.ndarray,
        _cache_path: str | None = None,
    ) -> "AntibodyRegressor":
        """
        Train regressor on sequences OR pre-computed embeddings.

        Args:
            sequences: List of VH protein sequences OR pre-computed embeddings (np.ndarray)
            labels: Continuous target values (e.g., polyreactivity scores)
            _cache_path: UNUSED - Kept for API compatibility. Caching handled by trainer.

        Returns:
            Self (fitted regressor)

        Raises:
            ValueError: If labels are not continuous floats

        Example:
            >>> # From sequences
            >>> regressor = AntibodyRegressor(alpha=1.0)
            >>> regressor.fit(
            ...     sequences=['QVQLQQ...', 'EVQLVE...'],
            ...     labels=np.array([0.337, 0.205])
            ... )
            >>> # From pre-computed embeddings
            >>> embeddings = np.random.randn(2, 1280)
            >>> regressor.fit(embeddings, labels)
        """
        # Check if input is pre-computed embeddings or sequences
        if isinstance(sequences, np.ndarray):
            embeddings = sequences
        else:
            embeddings = self.embedding_extractor.extract_batch_embeddings(sequences)

        # Train Ridge regressor
        self.regressor.fit(embeddings, labels)
        self._is_fitted = True

        return self

    def predict(
        self,
        sequences: list[str] | np.ndarray,
        _cache_path: str | None = None,
    ) -> np.ndarray:
        """
        Predict continuous values from sequences OR pre-computed embeddings.

        Args:
            sequences: List of VH protein sequences OR pre-computed embeddings (np.ndarray)
            _cache_path: UNUSED - Kept for API compatibility. Caching handled by trainer.

        Returns:
            Array of continuous predictions (floats)

        Raises:
            ValueError: If model not fitted

        Example:
            >>> # From sequences
            >>> regressor = AntibodyRegressor.load('model.pkl')
            >>> predictions = regressor.predict(['QVKLQE...'])
            >>> # From pre-computed embeddings
            >>> embeddings = np.random.randn(1, 1280)
            >>> predictions = regressor.predict(embeddings)
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before predict()")

        # Check if input is pre-computed embeddings or sequences
        if isinstance(sequences, np.ndarray):
            embeddings = sequences
        else:
            embeddings = self.embedding_extractor.extract_batch_embeddings(sequences)

        # Predict continuous values
        predictions: np.ndarray = self.regressor.predict(embeddings)

        return predictions

    def save(self, path: str) -> None:
        """
        Save fitted model to disk as pickle file.

        Args:
            path: Path to save model (should end in .pkl)

        Raises:
            ValueError: If model not fitted

        Example:
            >>> regressor.save('models/ginkgo_2025/pr_cho_regressor.pkl')
        """
        if not self._is_fitted:
            raise ValueError("Cannot save unfitted model")

        model_data = {
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "alpha": self.alpha,
            "regressor_kwargs": self.regressor_kwargs,
            "regressor": self.regressor,
        }

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

    @classmethod
    def load(cls, path: str) -> "AntibodyRegressor":
        """
        Load fitted model from disk.

        Args:
            path: Path to saved model (.pkl file)

        Returns:
            Loaded AntibodyRegressor instance

        Example:
            >>> regressor = AntibodyRegressor.load('models/pr_cho_regressor.pkl')
            >>> predictions = regressor.predict(test_sequences)
        """
        with open(path, "rb") as f:
            model_data = pickle.load(f)

        instance = cls(
            model_name=model_data["model_name"],
            device=model_data["device"],
            batch_size=model_data["batch_size"],
            alpha=model_data["alpha"],
            **model_data["regressor_kwargs"],
        )

        instance.regressor = model_data["regressor"]
        instance._is_fitted = True

        return instance
