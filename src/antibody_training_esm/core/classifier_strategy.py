"""
Classifier Strategy Protocol

Defines the interface for classifier backends (LogReg, XGBoost, MLP, etc.)
Uses Protocol for structural subtyping - sklearn compatibility without inheritance.

This module implements the Strategy Pattern for classifier algorithms, enabling
runtime swapping of different classifier backends while maintaining a consistent
interface.

Design Pattern: Strategy (Gang of Four)
Type System: Protocol-based structural subtyping (PEP 544)

Examples:
    >>> # Any class implementing these methods can be used as a classifier
    >>> from sklearn.linear_model import LogisticRegression
    >>> clf = LogisticRegression()
    >>> isinstance(clf, ClassifierStrategy)  # True (runtime_checkable)
    True

    >>> # Custom strategy implementation
    >>> class MyStrategy:
    ...     def fit(self, X, y): ...
    ...     def predict(self, X): ...
    ...     def predict_proba(self, X): ...
    ...     def get_params(self, deep=True): ...
    ...     @property
    ...     def classes_(self): ...
    >>> isinstance(MyStrategy(), ClassifierStrategy)  # True
    True
"""

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ClassifierStrategy(Protocol):
    """
    Protocol for classifier strategies.

    Any class implementing these methods can be used as a classifier backend,
    including sklearn estimators (LogisticRegression, XGBClassifier, etc.)

    This protocol defines the minimal interface required by BinaryClassifier.
    It follows the sklearn estimator API for maximum compatibility.

    Notes:
        - Uses Protocol for duck typing (PEP 544)
        - runtime_checkable enables isinstance() checks
        - Minimal interface - only what BinaryClassifier needs
        - Compatible with sklearn cross_val_score and GridSearchCV

    See Also:
        - sklearn.base.BaseEstimator
        - sklearn.base.ClassifierMixin
        - PEP 544: https://www.python.org/dev/peps/pep-0544/
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the classifier on embeddings.

        Args:
            X: Embeddings array, shape (n_samples, n_features)
               Each row is an ESM embedding vector for one antibody sequence.
            y: Labels array, shape (n_samples,)
               Binary labels (0 = non-polyreactive, 1 = polyreactive)

        Raises:
            ValueError: If X or y have invalid shapes, contain NaN/inf, or
                       have mismatched dimensions

        Notes:
            After calling fit(), the classifier must set the classes_ attribute
            to enable sklearn compatibility (required for cross_val_score).

        Examples:
            >>> X_train = np.random.rand(100, 1280)  # 100 ESM1v embeddings
            >>> y_train = np.array([0, 1] * 50)      # Binary labels
            >>> clf.fit(X_train, y_train)
            >>> hasattr(clf, 'classes_')
            True
        """
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for embeddings.

        Args:
            X: Embeddings array, shape (n_samples, n_features)
               Each row is an ESM embedding vector for one antibody sequence.

        Returns:
            Predicted labels, shape (n_samples,)
            Binary labels (0 = non-polyreactive, 1 = polyreactive)

        Raises:
            ValueError: If classifier not fitted (must call fit() first)
            ValueError: If X has invalid shape or contains NaN/inf
            ValueError: If X.shape[1] != n_features_in_ (mismatched dimensions)

        Examples:
            >>> X_test = np.random.rand(20, 1280)
            >>> predictions = clf.predict(X_test)
            >>> predictions.shape
            (20,)
            >>> set(predictions)
            {0, 1}
        """
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for embeddings.

        Args:
            X: Embeddings array, shape (n_samples, n_features)
               Each row is an ESM embedding vector for one antibody sequence.

        Returns:
            Probability array, shape (n_samples, n_classes)
            For binary classification:
            - [:, 0] = P(non-polyreactive)
            - [:, 1] = P(polyreactive)
            Probabilities sum to 1.0 for each row.

        Raises:
            ValueError: If classifier not fitted (must call fit() first)
            ValueError: If X has invalid shape or contains NaN/inf

        Notes:
            Used by BinaryClassifier for threshold-based prediction:
            - ELISA assay: threshold = 0.5
            - PSR assay: threshold = 0.5495 (Novo Nordisk exact parity)

        Examples:
            >>> X_test = np.random.rand(20, 1280)
            >>> probs = clf.predict_proba(X_test)
            >>> probs.shape
            (20, 2)
            >>> np.allclose(probs.sum(axis=1), 1.0)
            True
        """
        ...

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """
        Get classifier hyperparameters (sklearn API).

        Args:
            deep: If True, return parameters for nested estimators
                 (e.g., for sklearn Pipelines or ensemble methods)

        Returns:
            Dictionary of hyperparameters

        Notes:
            Required for sklearn compatibility:
            - cross_val_score needs get_params() for cloning estimators
            - GridSearchCV needs get_params() for hyperparameter tuning
            - set_params() needs get_params() for validation

        Examples:
            >>> params = clf.get_params()
            >>> 'C' in params  # LogisticRegression
            True
            >>> 'n_estimators' in params  # XGBoost
            True
        """
        ...

    @property
    def classes_(self) -> np.ndarray:
        """
        Class labels discovered during fit().

        Returns:
            Array of class labels, shape (n_classes,)
            For binary classification: np.array([0, 1])

        Raises:
            AttributeError: If classifier not fitted (must call fit() first)

        Notes:
            Required for sklearn compatibility:
            - cross_val_score checks classes_ to determine estimator type
            - sklearn 1.7+ requires _estimator_type and classes_ for CV

        Examples:
            >>> clf.fit(X_train, y_train)
            >>> clf.classes_
            array([0, 1])
        """
        ...


@runtime_checkable
class SerializableClassifier(Protocol):
    """
    Extended protocol for classifiers that support production serialization.

    This is OPTIONAL - only needed for production deployment (NPZ+JSON format).
    Pickle-based serialization works for any ClassifierStrategy.

    The SerializableClassifier protocol enables pickle-free deployment to
    production environments where loading untrusted pickle files is a security risk.

    Design Rationale:
        - to_dict(): Serialize hyperparameters to JSON (human-readable, safe)
        - to_arrays(): Serialize fitted state to NPZ (efficient, NumPy-native)
        - from_dict(): Deserialize from JSON+NPZ (production deployment)

    Security:
        - JSON is safe to load (no arbitrary code execution)
        - NPZ is NumPy's native format (no pickle dependency)
        - Enables HuggingFace Hub publishing (no pickle files)

    See Also:
        - docs/developer-guide/security.md
        - SECURITY_REMEDIATION_PLAN.md
    """

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize classifier hyperparameters to dictionary (for JSON).

        Returns:
            Dictionary with all hyperparameters and metadata.
            Does NOT include fitted state (arrays) - use to_arrays() for that.

        Examples:
            >>> config = clf.to_dict()
            >>> config
            {
                "type": "logistic_regression",
                "C": 1.0,
                "penalty": "l2",
                "solver": "lbfgs",
                "max_iter": 1000,
                "random_state": 42,
                "class_weight": None
            }

            >>> # Save to JSON
            >>> import json
            >>> with open("model_config.json", "w") as f:
            ...     json.dump(config, f)
        """
        ...

    def to_arrays(self) -> dict[str, np.ndarray]:
        """
        Extract fitted state as numpy arrays (for NPZ).

        Returns:
            Dictionary of arrays representing the fitted model.

        Raises:
            ValueError: If classifier not fitted (must call fit() first)

        Examples:
            >>> # LogisticRegression
            >>> arrays = logreg.to_arrays()
            >>> arrays.keys()
            dict_keys(['coef', 'intercept', 'classes', 'n_features_in', 'n_iter'])

            >>> # Save to NPZ
            >>> np.savez("model.npz", **arrays)
        """
        ...

    @classmethod
    def from_dict(
        cls, config: dict[str, Any], arrays: dict[str, np.ndarray] | None = None
    ) -> "SerializableClassifier":
        """
        Deserialize classifier from dict + arrays.

        Args:
            config: Dictionary with hyperparameters (from JSON)
            arrays: Dictionary with fitted state (from NPZ), None if unfitted

        Returns:
            Reconstructed classifier instance

        Examples:
            >>> # Load from JSON + NPZ
            >>> import json
            >>> import numpy as np
            >>>
            >>> with open("model_config.json") as f:
            ...     config = json.load(f)
            >>> arrays = dict(np.load("model.npz"))
            >>>
            >>> clf = LogisticRegressionStrategy.from_dict(config, arrays)
            >>> clf.predict(X_test)  # Ready to use
            array([0, 1, 0, ...])
        """
        ...
