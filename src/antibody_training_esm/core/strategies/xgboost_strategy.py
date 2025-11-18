"""
XGBoost Classifier Strategy

Wraps xgboost.XGBClassifier as a ClassifierStrategy.
Enables gradient boosting for nonlinear decision boundaries.

The XGBoostStrategy implements the ClassifierStrategy protocol for training/prediction.
Uses XGBoost's native .xgb format for production serialization (pickle-free).

Design Pattern: Strategy (Gang of Four)
Type System: Protocol-based structural subtyping

Examples:
    >>> # Basic usage
    >>> config = {"n_estimators": 100, "random_state": 42}
    >>> strategy = XGBoostStrategy(config)
    >>> strategy.fit(X_train, y_train)
    >>> predictions = strategy.predict(X_test)

    >>> # Production serialization (native .xgb format)
    >>> strategy.save_model("model.xgb")
    >>> config_dict = strategy.to_dict()
    >>> json.dump(config_dict, open("model_config.json", "w"))
    >>>
    >>> # Load model
    >>> config = json.load(open("model_config.json"))
    >>> loaded = XGBoostStrategy.load_model("model.xgb", config)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xgboost as xgb


class XGBoostStrategy:
    """
    XGBoost classifier strategy.

    Wraps xgboost.XGBClassifier with ClassifierStrategy interface.
    Implements both training (fit/predict) and native serialization (save_model/load_model).

    XGBoost provides gradient boosting trees capable of learning nonlinear decision
    boundaries, which can outperform linear models like LogisticRegression on
    complex antibody polyreactivity patterns.

    Attributes:
        classifier: xgboost.XGBClassifier instance
        n_estimators: Number of boosting rounds (default: 100)
        max_depth: Maximum tree depth (default: 6)
        learning_rate: Boosting learning rate (default: 0.3)
        subsample: Subsample ratio of training instances (default: 1.0)
        colsample_bytree: Subsample ratio of features (default: 1.0)
        reg_alpha: L1 regularization on weights (default: 0.0)
        reg_lambda: L2 regularization on weights (default: 1.0)
        random_state: Random seed for reproducibility (default: 42)
        objective: Learning objective (default: "binary:logistic")

    Notes:
        - Uses XGBoost's native .xgb serialization format (no pickle dependency)
        - Supports GPU acceleration via device parameter
        - Default hyperparameters are XGBoost defaults (good starting point)
        - For production deployment, use save_model() + to_dict() (JSON + .xgb)

    See Also:
        - xgboost.XGBClassifier
        - docs/developer-guide/xgboost-integration-spec.md
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize XGBoost strategy.

        Args:
            config: Configuration dictionary with hyperparameters.
                   All keys are optional (defaults provided).

        Configuration Keys:
            - n_estimators: Number of trees (default: 100)
            - max_depth: Maximum tree depth (default: 6)
            - learning_rate: Boosting learning rate (default: 0.3)
            - subsample: Subsample ratio (default: 1.0)
            - colsample_bytree: Feature subsample ratio (default: 1.0)
            - reg_alpha: L1 regularization (default: 0.0)
            - reg_lambda: L2 regularization (default: 1.0)
            - random_state: Random seed (default: 42)
            - objective: Learning objective (default: "binary:logistic")

        Examples:
            >>> # Default config
            >>> strategy = XGBoostStrategy({})
            >>> strategy.n_estimators
            100

            >>> # Custom config
            >>> config = {"n_estimators": 50, "max_depth": 4, "learning_rate": 0.1}
            >>> strategy = XGBoostStrategy(config)
            >>> strategy.n_estimators
            50
        """
        # Extract hyperparameters with defaults
        # Defaults are XGBoost defaults (good starting point)
        self.n_estimators = config.get("n_estimators", 100)
        self.max_depth = config.get("max_depth", 6)
        self.learning_rate = config.get("learning_rate", 0.3)
        self.subsample = config.get("subsample", 1.0)
        self.colsample_bytree = config.get("colsample_bytree", 1.0)
        self.reg_alpha = config.get("reg_alpha", 0.0)
        self.reg_lambda = config.get("reg_lambda", 1.0)
        self.random_state = config.get("random_state", 42)
        self.objective = config.get("objective", "binary:logistic")

        # Create XGBClassifier estimator
        self.classifier = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            objective=self.objective,
        )

    # ========================================================================
    # ClassifierStrategy Protocol Methods
    # ========================================================================

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit XGBoost on embeddings.

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
            >>> params['n_estimators']
            100
        """
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "random_state": self.random_state,
            "objective": self.objective,
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
    # Native Serialization Methods (XGBoost .xgb format)
    # ========================================================================

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize hyperparameters to dict (for JSON).

        Returns:
            Dictionary with all hyperparameters and metadata.
            Does NOT include fitted state - use save_model() for that.

        Examples:
            >>> config = strategy.to_dict()
            >>> config['type']
            'xgboost'
            >>> config['n_estimators']
            100

            >>> # Save to JSON
            >>> import json
            >>> json.dump(config, open("model_config.json", "w"))
        """
        return {
            "type": "xgboost",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "random_state": self.random_state,
            "objective": self.objective,
        }

    def save_model(self, path: str) -> None:
        """
        Save fitted model to XGBoost native .xgb format.

        Args:
            path: File path for .xgb model file

        Raises:
            ValueError: If classifier not fitted

        Examples:
            >>> strategy.fit(X_train, y_train)
            >>> strategy.save_model("model.xgb")
        """
        if not hasattr(self.classifier, "_Booster"):
            raise ValueError("Classifier must be fitted before saving")

        self.classifier.save_model(path)

    @classmethod
    def load_model(cls, path: str, config: dict[str, Any]) -> XGBoostStrategy:
        """
        Load fitted model from XGBoost native .xgb format.

        Args:
            path: File path to .xgb model file
            config: Configuration dictionary with hyperparameters

        Returns:
            XGBoostStrategy with loaded model

        Examples:
            >>> # Load from .xgb + JSON
            >>> import json
            >>> config = json.load(open("model_config.json"))
            >>> strategy = XGBoostStrategy.load_model("model.xgb", config)
            >>> strategy.predict(X_test)
            array([0, 1, 0, ...])
        """
        # Create unfitted classifier
        strategy = cls(config)

        # Load fitted model from .xgb file
        strategy.classifier.load_model(path)

        return strategy

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> XGBoostStrategy:
        """
        Create XGBoostStrategy from configuration dictionary.

        Args:
            config: Configuration dictionary with hyperparameters

        Returns:
            Unfitted XGBoostStrategy instance

        Examples:
            >>> config = {"type": "xgboost", "n_estimators": 50, "random_state": 42}
            >>> strategy = XGBoostStrategy.from_dict(config)
            >>> strategy.n_estimators
            50
        """
        return cls(config)
