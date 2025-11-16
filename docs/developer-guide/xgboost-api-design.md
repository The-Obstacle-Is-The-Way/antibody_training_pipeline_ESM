# XGBoost Classifier - API Design Document

**Status:** Draft
**Version:** 1.0
**Date:** 2025-11-15
**Related:** `xgboost-integration-spec.md`

---

## Table of Contents

1. [Design Philosophy](#design-philosophy)
2. [Architecture Overview](#architecture-overview)
3. [ClassifierStrategy Protocol](#classifierstrategy-protocol)
4. [Concrete Implementations](#concrete-implementations)
5. [Factory Pattern](#factory-pattern)
6. [BinaryClassifier Updates](#binaryclassifier-updates)
7. [Serialization API](#serialization-api)
8. [Usage Examples](#usage-examples)

---

## Design Philosophy

### Core Principles

1. **SOLID:**
   - **S**ingle Responsibility: Each strategy handles one classifier type
   - **O**pen/Closed: Open for extension (new classifiers), closed for modification
   - **L**iskov Substitution: All strategies interchangeable via protocol
   - **I**nterface Segregation: Minimal protocol, no bloated interfaces
   - **D**ependency Inversion: Depend on abstractions (protocol), not concretions

2. **DRY:** Shared code in base class, no duplication between strategies

3. **Gang of Four:**
   - **Strategy Pattern:** Swap classifier algorithms at runtime
   - **Factory Pattern:** Create classifiers based on config

4. **Type Safety:**
   - Protocol-based typing (duck typing with type hints)
   - Full mypy strict mode compliance
   - No `Any` types in public APIs

5. **Backward Compatibility:**
   - Existing `BinaryClassifier` API unchanged
   - Default to LogReg if `type` not specified
   - All existing tests pass without modification

---

## Architecture Overview

### Before (Current)

```
┌─────────────────────────────────────┐
│       BinaryClassifier              │
│  ┌────────────────────────────────┐ │
│  │  ESMEmbeddingExtractor         │ │
│  └────────────────────────────────┘ │
│  ┌────────────────────────────────┐ │
│  │  LogisticRegression (hardcoded)│ │  ← PROBLEM: No abstraction
│  └────────────────────────────────┘ │
└─────────────────────────────────────┘
```

### After (Refactored)

```
┌─────────────────────────────────────────────────────┐
│              BinaryClassifier                        │
│  ┌────────────────────────────────────────────────┐ │
│  │         ESMEmbeddingExtractor                  │ │
│  └────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────┐ │
│  │      ClassifierStrategy (Protocol)             │ │  ← Abstraction
│  │                                                 │ │
│  │  ┌──────────────────┐  ┌──────────────────┐  │ │
│  │  │ LogRegStrategy   │  │ XGBoostStrategy  │  │ │  ← Concrete
│  │  └──────────────────┘  └──────────────────┘  │ │
│  └────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

**Key Changes:**
- `ClassifierStrategy` protocol defines interface
- `create_classifier()` factory selects implementation
- `BinaryClassifier` delegates to strategy
- Strategies are **swappable at runtime**

---

## ClassifierStrategy Protocol

### Why Protocol Instead of ABC?

**Protocols** (PEP 544) enable **structural subtyping** (duck typing with type hints):
- sklearn estimators don't inherit from common base class
- Protocols let us type-check sklearn compatibility WITHOUT forcing inheritance
- More Pythonic than ABCs for interoperability

### Protocol Definition

**File:** `src/antibody_training_esm/core/classifier_strategy.py`

```python
"""
Classifier Strategy Protocol

Defines the interface for classifier backends (LogReg, XGBoost, MLP, etc.)
Uses Protocol for structural subtyping - sklearn compatibility without inheritance.
"""

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ClassifierStrategy(Protocol):
    """
    Protocol for classifier strategies.

    Any class implementing these methods can be used as a classifier backend,
    including sklearn estimators (LogisticRegression, XGBClassifier, etc.)

    Notes:
        - Uses Protocol for duck typing (PEP 544)
        - runtime_checkable enables isinstance() checks
        - Minimal interface - only what BinaryClassifier needs
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the classifier on embeddings.

        Args:
            X: Embeddings array, shape (n_samples, n_features)
            y: Labels array, shape (n_samples,)

        Raises:
            ValueError: If X or y have invalid shapes or contain NaN/inf
        """
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for embeddings.

        Args:
            X: Embeddings array, shape (n_samples, n_features)

        Returns:
            Predicted labels, shape (n_samples,)

        Raises:
            ValueError: If classifier not fitted or X has invalid shape
        """
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for embeddings.

        Args:
            X: Embeddings array, shape (n_samples, n_features)

        Returns:
            Probability array, shape (n_samples, n_classes)
            For binary classification: [:, 0] = P(class=0), [:, 1] = P(class=1)

        Raises:
            ValueError: If classifier not fitted or X has invalid shape
        """
        ...

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """
        Get classifier hyperparameters (sklearn API).

        Args:
            deep: If True, return params for nested estimators

        Returns:
            Dictionary of hyperparameters
        """
        ...

    @property
    def classes_(self) -> np.ndarray:
        """
        Class labels discovered during fit().

        Returns:
            Array of class labels, shape (n_classes,)
            For binary classification: [0, 1]

        Raises:
            AttributeError: If classifier not fitted
        """
        ...


class SerializableClassifier(Protocol):
    """
    Extended protocol for classifiers that support serialization.

    This is OPTIONAL - only needed for production deployment (NPZ+JSON format).
    Pickle-based serialization works for any ClassifierStrategy.
    """

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize classifier to dictionary (for JSON).

        Returns:
            Dictionary with all hyperparameters and metadata.
            Does NOT include fitted state (arrays) - use to_arrays() for that.

        Example:
            {
                "type": "xgboost",
                "n_estimators": 100,
                "max_depth": 6,
                ...
            }
        """
        ...

    def to_arrays(self) -> dict[str, np.ndarray]:
        """
        Extract fitted state as numpy arrays (for NPZ).

        Returns:
            Dictionary of arrays representing the fitted model.
            For XGBoost: trees serialized as arrays or save to .xgb file separately.

        Example (LogReg):
            {
                "coef": np.array([...]),      # (n_features,)
                "intercept": np.array([...])  # scalar
            }
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

        Example:
            config = json.load(f)
            arrays = np.load("model.npz")
            classifier = XGBoostStrategy.from_dict(config, dict(arrays))
        """
        ...
```

**Key Design Decisions:**

1. **Minimal Interface:** Only 5 methods (fit, predict, predict_proba, get_params, classes_)
   - Matches sklearn estimator API
   - No bloat - only what `BinaryClassifier` actually uses

2. **Separate Serialization Protocol:** `SerializableClassifier` is OPTIONAL
   - Pickle works for everything (research/debugging)
   - Production deployment can implement `to_dict()` / `from_dict()`

3. **Type Safety:** Full type annotations, no `Any` in return types

---

## Concrete Implementations

### LogisticRegressionStrategy

**File:** `src/antibody_training_esm/core/strategies/logistic_regression.py`

```python
"""
Logistic Regression Classifier Strategy

Wraps sklearn.linear_model.LogisticRegression as a ClassifierStrategy.
This is the EXISTING classifier, refactored to use the strategy pattern.
"""

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from antibody_training_esm.core.classifier_strategy import (
    ClassifierStrategy,
    SerializableClassifier,
)


class LogisticRegressionStrategy:
    """
    Logistic Regression classifier strategy.

    Wraps sklearn LogisticRegression with SerializableClassifier interface.
    Implements both training (fit/predict) and serialization (to_dict/from_dict).

    Attributes:
        classifier: sklearn LogisticRegression instance
        C: Inverse regularization strength
        penalty: Regularization type ('l1', 'l2', 'elasticnet', 'none')
        solver: Optimization algorithm
        max_iter: Maximum iterations
        random_state: Random seed
        class_weight: Class weights ('balanced', dict, or None)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize LogisticRegression strategy.

        Args:
            config: Configuration dictionary with hyperparameters.
                   Required keys: None (all have defaults)
                   Optional keys: C, penalty, solver, max_iter, random_state, class_weight

        Example:
            config = {
                "C": 1.0,
                "penalty": "l2",
                "solver": "lbfgs",
                "max_iter": 1000,
                "random_state": 42,
                "class_weight": None
            }
            strategy = LogisticRegressionStrategy(config)
        """
        # Extract hyperparameters with defaults
        self.C = config.get("C", 1.0)
        self.penalty = config.get("penalty", "l2")
        self.solver = config.get("solver", "lbfgs")
        self.max_iter = config.get("max_iter", 1000)
        self.random_state = config.get("random_state", 42)
        self.class_weight = config.get("class_weight", None)

        # Create sklearn estimator
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
        """Fit LogisticRegression on embeddings."""
        self.classifier.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        result: np.ndarray = self.classifier.predict(X)
        return result

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        result: np.ndarray = self.classifier.predict_proba(X)
        return result

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get hyperparameters (sklearn API)."""
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
        """Class labels discovered during fit."""
        result: np.ndarray = self.classifier.classes_
        return result

    # ========================================================================
    # SerializableClassifier Protocol Methods
    # ========================================================================

    def to_dict(self) -> dict[str, Any]:
        """Serialize hyperparameters to dict (for JSON)."""
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
        """Extract fitted state as arrays (for NPZ)."""
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
        """Deserialize from dict + arrays."""
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
```

### XGBoostStrategy

**File:** `src/antibody_training_esm/core/strategies/xgboost_strategy.py`

```python
"""
XGBoost Classifier Strategy

Wraps xgboost.XGBClassifier as a ClassifierStrategy.
Supports both sklearn API (fit/predict) and XGBoost native serialization.
"""

from typing import Any

import numpy as np

try:
    from xgboost import XGBClassifier
except ImportError:
    raise ImportError(
        "xgboost is required for XGBoostStrategy. Install with: pip install xgboost>=2.0.0"
    )

from antibody_training_esm.core.classifier_strategy import (
    ClassifierStrategy,
    SerializableClassifier,
)


class XGBoostStrategy:
    """
    XGBoost classifier strategy.

    Wraps XGBClassifier with SerializableClassifier interface.
    Uses sklearn API for training, native .xgb format for serialization.

    Attributes:
        classifier: XGBClassifier instance
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Step size shrinkage (eta)
        subsample: Row subsampling ratio
        colsample_bytree: Column subsampling ratio
        reg_alpha: L1 regularization
        reg_lambda: L2 regularization
        random_state: Random seed
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize XGBoost strategy.

        Args:
            config: Configuration dictionary with hyperparameters.
                   Optional keys: n_estimators, max_depth, learning_rate, etc.

        Example:
            config = {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.3,
                "random_state": 42
            }
            strategy = XGBoostStrategy(config)
        """
        # Extract hyperparameters with defaults
        self.n_estimators = config.get("n_estimators", 100)
        self.max_depth = config.get("max_depth", 6)
        self.learning_rate = config.get("learning_rate", 0.3)
        self.subsample = config.get("subsample", 1.0)
        self.colsample_bytree = config.get("colsample_bytree", 1.0)
        self.reg_alpha = config.get("reg_alpha", 0.0)
        self.reg_lambda = config.get("reg_lambda", 1.0)
        self.random_state = config.get("random_state", 42)

        # XGBoost-specific params
        self.objective = config.get("objective", "binary:logistic")
        self.eval_metric = config.get("eval_metric", "logloss")

        # Create XGBClassifier
        self.classifier = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            objective=self.objective,
            eval_metric=self.eval_metric,
            use_label_encoder=False,  # Deprecated in xgboost>=1.6
            enable_categorical=False,  # We use embeddings (continuous)
        )

    # ========================================================================
    # ClassifierStrategy Protocol Methods
    # ========================================================================

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit XGBoost on embeddings."""
        self.classifier.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        result: np.ndarray = self.classifier.predict(X)
        return result

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        result: np.ndarray = self.classifier.predict_proba(X)
        return result

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get hyperparameters (sklearn API)."""
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
            "eval_metric": self.eval_metric,
        }

    @property
    def classes_(self) -> np.ndarray:
        """Class labels discovered during fit."""
        result: np.ndarray = self.classifier.classes_
        return result

    # ========================================================================
    # SerializableClassifier Protocol Methods
    # ========================================================================

    def to_dict(self) -> dict[str, Any]:
        """Serialize hyperparameters to dict (for JSON)."""
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
            "eval_metric": self.eval_metric,
        }

    def save_model(self, path: str) -> None:
        """
        Save XGBoost model to native .xgb format.

        Args:
            path: File path for .xgb file

        Notes:
            Uses XGBoost native format (ubj or json).
            This is the PREFERRED serialization for production.
        """
        if not hasattr(self.classifier, "_Booster"):
            raise ValueError("Classifier must be fitted before saving")

        self.classifier.save_model(path)

    @classmethod
    def load_model(cls, path: str, config: dict[str, Any]) -> "XGBoostStrategy":
        """
        Load XGBoost model from native .xgb format.

        Args:
            path: File path to .xgb file
            config: Configuration dict with hyperparameters

        Returns:
            XGBoostStrategy with loaded model

        Example:
            config = json.load(open("model_config.json"))
            strategy = XGBoostStrategy.load_model("model.xgb", config)
        """
        # Create unfitted classifier
        strategy = cls(config)

        # Load fitted model
        strategy.classifier.load_model(path)

        return strategy

    @classmethod
    def from_dict(
        cls, config: dict[str, Any], arrays: dict[str, np.ndarray] | None = None
    ) -> "XGBoostStrategy":
        """
        Deserialize from dict.

        Args:
            config: Dictionary with hyperparameters (from JSON)
            arrays: Not used for XGBoost (use .xgb file instead)

        Returns:
            Unfitted XGBoostStrategy instance

        Notes:
            For XGBoost, use save_model() / load_model() instead of arrays.
            This method creates an UNFITTED classifier.
        """
        return cls(config)
```

---

## Factory Pattern

**File:** `src/antibody_training_esm/core/classifier_factory.py`

```python
"""
Classifier Factory

Creates classifier strategies based on configuration.
Implements Factory Pattern for runtime strategy selection.
"""

from typing import Any

from antibody_training_esm.core.classifier_strategy import ClassifierStrategy
from antibody_training_esm.core.strategies.logistic_regression import (
    LogisticRegressionStrategy,
)
from antibody_training_esm.core.strategies.xgboost_strategy import XGBoostStrategy


def create_classifier(config: dict[str, Any]) -> ClassifierStrategy:
    """
    Factory function for creating classifier strategies.

    Args:
        config: Configuration dictionary with "type" key and hyperparameters

    Returns:
        ClassifierStrategy instance (LogReg, XGBoost, etc.)

    Raises:
        ValueError: If classifier type is unknown

    Example:
        # Logistic Regression
        config = {"type": "logistic_regression", "C": 1.0, ...}
        clf = create_classifier(config)

        # XGBoost
        config = {"type": "xgboost", "n_estimators": 100, ...}
        clf = create_classifier(config)
    """
    # Default to logistic_regression for backward compatibility
    classifier_type = config.get("type", "logistic_regression")

    if classifier_type == "logistic_regression":
        return LogisticRegressionStrategy(config)
    elif classifier_type == "xgboost":
        return XGBoostStrategy(config)
    else:
        raise ValueError(
            f"Unknown classifier type: '{classifier_type}'. "
            f"Supported types: logistic_regression, xgboost"
        )


# Registry pattern for extensibility
CLASSIFIER_REGISTRY: dict[str, type[ClassifierStrategy]] = {
    "logistic_regression": LogisticRegressionStrategy,
    "xgboost": XGBoostStrategy,
}


def register_classifier(name: str, strategy_class: type[ClassifierStrategy]) -> None:
    """
    Register a new classifier strategy.

    Args:
        name: Classifier type name (e.g., "mlp", "svm")
        strategy_class: ClassifierStrategy subclass

    Example:
        class MLPStrategy(ClassifierStrategy):
            ...

        register_classifier("mlp", MLPStrategy)
    """
    CLASSIFIER_REGISTRY[name] = strategy_class


def create_classifier_from_registry(config: dict[str, Any]) -> ClassifierStrategy:
    """
    Create classifier using registry (extensible version).

    Args:
        config: Configuration with "type" key

    Returns:
        ClassifierStrategy instance

    Raises:
        ValueError: If type not in registry
    """
    classifier_type = config.get("type", "logistic_regression")

    if classifier_type not in CLASSIFIER_REGISTRY:
        raise ValueError(
            f"Unknown classifier type: '{classifier_type}'. "
            f"Registered types: {list(CLASSIFIER_REGISTRY.keys())}"
        )

    strategy_class = CLASSIFIER_REGISTRY[classifier_type]
    return strategy_class(config)
```

---

## BinaryClassifier Updates

**File:** `src/antibody_training_esm/core/classifier.py` (MODIFIED)

```python
"""
Binary Classifier Module (REFACTORED)

Uses Strategy Pattern for classifier backends.
Supports LogisticRegression, XGBoost, and future classifiers.
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
    """Binary classifier for protein sequences using ESM embeddings"""

    # sklearn 1.7+ requires explicit estimator type
    _estimator_type = "classifier"

    # Assay-specific thresholds (unchanged)
    ASSAY_THRESHOLDS = {
        "ELISA": 0.5,
        "PSR": 0.5495,
    }

    def __init__(self, params: dict[str, Any] | None = None, **kwargs: Any):
        """
        Initialize binary classifier with strategy pattern.

        Args:
            params: Dictionary containing classifier parameters (legacy API)
            **kwargs: Individual parameters (for sklearn compatibility)

        Notes:
            Classifier type determined by params["type"]:
            - "logistic_regression" (default)
            - "xgboost"
        """
        # Support both dict-based and kwargs-based initialization
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

        # Initialize embedding extractor (UNCHANGED)
        batch_size = params.get("batch_size", DEFAULT_BATCH_SIZE)
        revision = params.get("revision", "main")

        self.embedding_extractor = ESMEmbeddingExtractor(
            params["model_name"], params["device"], batch_size, revision=revision
        )

        # NEW: Use factory to create classifier strategy
        self.classifier: ClassifierStrategy = create_classifier(params)

        logger.info(
            "Classifier initialized: type=%s, params=%s",
            params.get("type", "logistic_regression"),
            self.classifier.get_params(),
        )

        # Store attributes for sklearn compatibility (UNCHANGED)
        self.random_state = params["random_state"]
        self.is_fitted = False
        self.device = self.embedding_extractor.device
        self.model_name = params["model_name"]
        self.batch_size = batch_size
        self.revision = revision
        self._params = params

    # ========================================================================
    # sklearn API Methods (UNCHANGED - delegate to strategy)
    # ========================================================================

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit classifier on embeddings."""
        self.classifier.fit(X, y)
        self.is_fitted = True
        self.classes_ = self.classifier.classes_
        logger.info(f"Classifier fitted on {len(X)} samples")

    def predict(
        self, X: np.ndarray, threshold: float = 0.5, assay_type: str | None = None
    ) -> np.ndarray:
        """Predict labels with optional assay-specific thresholds."""
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before making predictions")

        # Determine threshold (unchanged)
        if assay_type is not None:
            if assay_type not in self.ASSAY_THRESHOLDS:
                raise ValueError(
                    f"Unknown assay_type '{assay_type}'. "
                    f"Must be one of: {list(self.ASSAY_THRESHOLDS.keys())}"
                )
            threshold = self.ASSAY_THRESHOLDS[assay_type]

        # Get probabilities and apply threshold
        probabilities = self.classifier.predict_proba(X)
        predictions: np.ndarray = (probabilities[:, 1] > threshold).astype(int)

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before making predictions")

        result: np.ndarray = self.classifier.predict_proba(X)
        return result

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return mean accuracy."""
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before scoring")

        predictions = self.predict(X)
        accuracy: float = (predictions == y).mean()
        return accuracy

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for sklearn compatibility."""
        # Merge embedding params + classifier params
        params = {
            "random_state": self.random_state,
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "revision": self.revision,
        }
        params.update(self.classifier.get_params(deep=deep))
        return params

    def set_params(self, **params: Any) -> "BinaryClassifier":
        """Set parameters for sklearn compatibility."""
        # Update internal params dict
        self._params.update(params)

        # Recreate embedding extractor if needed
        embedding_params = {"model_name", "device", "batch_size", "revision"}
        if any(key in params for key in embedding_params):
            self.embedding_extractor = ESMEmbeddingExtractor(
                self._params["model_name"],
                self._params["device"],
                self._params.get("batch_size", DEFAULT_BATCH_SIZE),
                revision=self._params.get("revision", "main"),
            )

        # Recreate classifier strategy if type changed
        if "type" in params:
            self.classifier = create_classifier(self._params)
            self.is_fitted = False  # New classifier is unfitted

        return self
```

**Key Changes:**
1. `self.classifier` now has type `ClassifierStrategy` (not hardcoded `LogisticRegression`)
2. Uses `create_classifier(params)` factory
3. All methods delegate to `self.classifier` (unchanged behavior)
4. No breaking changes to public API

---

## Serialization API

### Dual Serialization Format

**1. Pickle (Research/Debugging):**
```python
# Save
with open("model.pkl", "wb") as f:
    pickle.dump(binary_classifier, f)

# Load
with open("model.pkl", "rb") as f:
    classifier = pickle.load(f)
```

**2. Production Format (NPZ + JSON or .xgb + JSON):**

**LogisticRegression:**
```python
# Save
config = classifier.to_dict()  # Hyperparameters
arrays = classifier.to_arrays()  # Fitted state

json.dump(config, open("model_config.json", "w"))
np.savez("model.npz", **arrays)

# Load
config = json.load(open("model_config.json"))
arrays = dict(np.load("model.npz"))
classifier = LogisticRegressionStrategy.from_dict(config, arrays)
```

**XGBoost:**
```python
# Save
config = classifier.to_dict()  # Hyperparameters
classifier.save_model("model.xgb")  # Native XGBoost format

json.dump(config, open("model_config.json", "w"))

# Load
config = json.load(open("model_config.json"))
classifier = XGBoostStrategy.load_model("model.xgb", config)
```

---

## Usage Examples

### Example 1: Train with LogReg (Backward Compatible)

```bash
# CLI (unchanged)
antibody-train classifier=logreg
```

```python
# Python API (unchanged)
params = {
    "type": "logistic_regression",  # Can omit - defaults to logreg
    "model_name": "facebook/esm1v_t33_650M_UR90S_1",
    "device": "cpu",
    "random_state": 42,
    "max_iter": 1000,
    "C": 1.0,
}

classifier = BinaryClassifier(params)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
```

### Example 2: Train with XGBoost (New Feature)

```bash
# CLI
antibody-train classifier=xgboost
```

```python
# Python API
params = {
    "type": "xgboost",  # NEW: Specify XGBoost
    "model_name": "facebook/esm1v_t33_650M_UR90S_1",
    "device": "cpu",
    "random_state": 42,
    "max_iter": 1000,  # Still required (for embedding extractor)
    "n_estimators": 100,
    "max_depth": 6,
}

classifier = BinaryClassifier(params)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
```

### Example 3: Compare Models

```python
# Train both on same embeddings
logreg_params = {"type": "logistic_regression", ...}
xgboost_params = {"type": "xgboost", ...}

logreg = BinaryClassifier(logreg_params)
xgboost = BinaryClassifier(xgboost_params)

# Same embeddings (cached)
logreg.fit(X_train, y_train)
xgboost.fit(X_train, y_train)

# Compare performance
logreg_acc = logreg.score(X_test, y_test)
xgboost_acc = xgboost.score(X_test, y_test)

print(f"LogReg: {logreg_acc:.4f}")
print(f"XGBoost: {xgboost_acc:.4f}")
```

---

## Next Steps

1. **Read Test Plan:** `xgboost-test-plan.md`
2. **Implement Phase 1:** Refactoring (LogReg strategy)
3. **Implement Phase 2:** XGBoost strategy
4. **Implement Phase 3:** Integration tests & benchmarking

---

**Document Status:** Draft → Ready for Review
**Next Action:** Write Test Plan
**Review Required:** Yes
