"""
Logistic Regression Classifier Strategy

Wraps sklearn.linear_model.LogisticRegression as a ClassifierStrategy.
This is the EXISTING classifier, refactored to use the strategy pattern.

The LogisticRegressionStrategy implements both the ClassifierStrategy protocol
(for training/prediction) and the SerializableClassifier protocol (for production
deployment without pickle files).

Design Pattern: Strategy (Gang of Four)
Type System: Protocol-based structural subtyping

Examples:
    >>> # Basic usage
    >>> config = {"C": 1.0, "random_state": 42}
    >>> strategy = LogisticRegressionStrategy(config)
    >>> strategy.fit(X_train, y_train)
    >>> predictions = strategy.predict(X_test)

    >>> # Production serialization (pickle-free)
    >>> config_dict = strategy.to_dict()
    >>> arrays_dict = strategy.to_arrays()
    >>> json.dump(config_dict, open("model.json", "w"))
    >>> np.savez("model.npz", **arrays_dict)
    >>>
    >>> # Load model
    >>> config = json.load(open("model.json"))
    >>> arrays = dict(np.load("model.npz"))
    >>> loaded = LogisticRegressionStrategy.from_dict(config, arrays)
"""

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression


class LogisticRegressionStrategy:
    """
    Logistic Regression classifier strategy.

    Wraps sklearn LogisticRegression with SerializableClassifier interface.
    Implements both training (fit/predict) and serialization (to_dict/from_dict).

    This class refactors the EXISTING LogisticRegression classifier from
    BinaryClassifier into a separate strategy, enabling the Strategy Pattern
    for supporting multiple classifier backends (XGBoost, MLP, etc.).

    Attributes:
        classifier: sklearn LogisticRegression instance
        C: Inverse regularization strength (default: 1.0)
        penalty: Regularization type: 'l1', 'l2', 'elasticnet', 'none' (default: 'l2')
        solver: Optimization algorithm: 'lbfgs', 'liblinear', 'saga', etc. (default: 'lbfgs')
        max_iter: Maximum iterations for optimization (default: 1000)
        random_state: Random seed for reproducibility (default: 42)
        class_weight: Class weights: 'balanced', dict, or None (default: None)

    Notes:
        - Default hyperparameters match the EXISTING BinaryClassifier behavior
        - Implements ClassifierStrategy protocol for sklearn compatibility
        - Implements SerializableClassifier protocol for production deployment
        - No scaling applied (matches Novo Nordisk methodology)

    See Also:
        - sklearn.linear_model.LogisticRegression
        - docs/research/novo-parity.md (methodology)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize LogisticRegression strategy.

        Args:
            config: Configuration dictionary with hyperparameters.
                   All keys are optional (defaults provided).

        Configuration Keys:
            - C: Inverse regularization strength (default: 1.0)
            - penalty: Regularization type (default: 'l2')
            - solver: Optimization algorithm (default: 'lbfgs')
            - max_iter: Maximum iterations (default: 1000)
            - random_state: Random seed (default: 42)
            - class_weight: Class weights (default: None)

        Examples:
            >>> # Default config
            >>> strategy = LogisticRegressionStrategy({})
            >>> strategy.C
            1.0

            >>> # Custom config
            >>> config = {"C": 0.5, "penalty": "l1", "solver": "liblinear"}
            >>> strategy = LogisticRegressionStrategy(config)
            >>> strategy.C
            0.5
        """
        # Extract hyperparameters with defaults
        # Defaults match EXISTING BinaryClassifier behavior
        self.C = config.get("C", 1.0)
        self.penalty = config.get("penalty", "l2")
        self.solver = config.get("solver", "lbfgs")
        self.max_iter = config.get("max_iter", 1000)
        self.random_state = config.get("random_state", 42)
        self.class_weight = config.get("class_weight", None)

        # Create sklearn LogisticRegression estimator
        self.classifier = LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=self.random_state,
            class_weight=self.class_weight,
        )

    # ========================================================================
    # ClassifierStrategy Protocol Methods
    # ========================================================================

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit LogisticRegression on embeddings.

        Args:
            X: Embeddings array, shape (n_samples, n_features)
            y: Labels array, shape (n_samples,)

        Notes:
            No scaling is applied (matches Novo Nordisk methodology).
            After fitting, the classes_ attribute is available.

        Examples:
            >>> X_train = np.random.rand(100, 1280)
            >>> y_train = np.array([0, 1] * 50)
            >>> strategy.fit(X_train, y_train)
            >>> strategy.classes_
            array([0, 1])
        """
        self.classifier.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Embeddings array, shape (n_samples, n_features)

        Returns:
            Predicted labels, shape (n_samples,)

        Examples:
            >>> X_test = np.random.rand(20, 1280)
            >>> predictions = strategy.predict(X_test)
            >>> predictions.shape
            (20,)
        """
        result: np.ndarray = self.classifier.predict(X)
        return result

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Embeddings array, shape (n_samples, n_features)

        Returns:
            Probability array, shape (n_samples, n_classes)

        Examples:
            >>> X_test = np.random.rand(20, 1280)
            >>> probs = strategy.predict_proba(X_test)
            >>> probs.shape
            (20, 2)
            >>> np.allclose(probs.sum(axis=1), 1.0)
            True
        """
        result: np.ndarray = self.classifier.predict_proba(X)
        return result

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return mean accuracy on test data.

        Args:
            X: Embeddings array, shape (n_samples, n_features)
            y: True labels, shape (n_samples,)

        Returns:
            Mean accuracy score

        Examples:
            >>> X_test = np.random.rand(20, 1280)
            >>> y_test = np.array([0, 1] * 10)
            >>> acc = strategy.score(X_test, y_test)
            >>> 0.0 <= acc <= 1.0
            True
        """
        result: float = self.classifier.score(X, y)
        return result

    def get_params(self, deep: bool = True) -> dict[str, Any]:  # noqa: ARG002
        """
        Get hyperparameters (sklearn API).

        Args:
            deep: If True, return params for nested estimators (unused)

        Returns:
            Dictionary of hyperparameters

        Examples:
            >>> params = strategy.get_params()
            >>> params['C']
            1.0
        """
        return {
            "C": self.C,
            "penalty": self.penalty,
            "solver": self.solver,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
            "class_weight": self.class_weight,
        }

    @property
    def classes_(self) -> np.ndarray:
        """
        Class labels discovered during fit.

        Returns:
            Array of class labels, shape (n_classes,)

        Raises:
            AttributeError: If classifier not fitted

        Examples:
            >>> strategy.fit(X_train, y_train)
            >>> strategy.classes_
            array([0, 1])
        """
        result: np.ndarray = self.classifier.classes_
        return result

    # ========================================================================
    # SerializableClassifier Protocol Methods
    # ========================================================================

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize hyperparameters to dict (for JSON).

        Returns:
            Dictionary with all hyperparameters and metadata.
            Does NOT include fitted state (arrays) - use to_arrays() for that.

        Examples:
            >>> config = strategy.to_dict()
            >>> config['type']
            'logistic_regression'
            >>> config['C']
            1.0

            >>> # Save to JSON
            >>> import json
            >>> json.dump(config, open("model_config.json", "w"))
        """
        return {
            "type": "logistic_regression",
            "C": self.C,
            "penalty": self.penalty,
            "solver": self.solver,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
            "class_weight": self.class_weight,
        }

    def to_arrays(self) -> dict[str, np.ndarray]:
        """
        Extract fitted state as arrays (for NPZ).

        Returns:
            Dictionary of arrays representing the fitted model.

        Raises:
            ValueError: If classifier not fitted

        Examples:
            >>> strategy.fit(X_train, y_train)
            >>> arrays = strategy.to_arrays()
            >>> arrays.keys()
            dict_keys(['coef', 'intercept', 'classes', 'n_features_in', 'n_iter'])

            >>> # Save to NPZ
            >>> np.savez("model.npz", **arrays)
        """
        if not hasattr(self.classifier, "coef_"):
            raise ValueError("Classifier must be fitted before serialization")

        return {
            "coef": self.classifier.coef_,
            "intercept": self.classifier.intercept_,
            "classes": self.classifier.classes_,
            "n_features_in": np.array([self.classifier.n_features_in_]),
            "n_iter": self.classifier.n_iter_,
        }

    @classmethod
    def from_dict(
        cls, config: dict[str, Any], arrays: dict[str, np.ndarray] | None = None
    ) -> "LogisticRegressionStrategy":
        """
        Deserialize from dict + arrays.

        Args:
            config: Dictionary with hyperparameters (from JSON)
            arrays: Dictionary with fitted state (from NPZ), None if unfitted

        Returns:
            Reconstructed LogisticRegressionStrategy instance

        Examples:
            >>> # Load from JSON + NPZ
            >>> import json
            >>> config = json.load(open("model_config.json"))
            >>> arrays = dict(np.load("model.npz"))
            >>> strategy = LogisticRegressionStrategy.from_dict(config, arrays)
            >>> strategy.predict(X_test)
            array([0, 1, 0, ...])
        """
        # Create unfitted classifier
        strategy = cls(config)

        # Restore fitted state if arrays provided
        if arrays is not None:
            strategy.classifier.coef_ = arrays["coef"]
            strategy.classifier.intercept_ = arrays["intercept"]
            strategy.classifier.classes_ = arrays["classes"]
            strategy.classifier.n_features_in_ = int(arrays["n_features_in"][0])
            strategy.classifier.n_iter_ = arrays["n_iter"]

        return strategy
