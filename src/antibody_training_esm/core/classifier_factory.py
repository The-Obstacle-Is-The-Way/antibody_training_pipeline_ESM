"""
Classifier Factory

Creates classifier strategies based on configuration.
Implements Factory Pattern for runtime strategy selection.

Design Pattern: Factory (Gang of Four)
Purpose: Decouple classifier creation from BinaryClassifier

Examples:
    >>> # Logistic Regression (default)
    >>> config = {"C": 1.0, "random_state": 42}
    >>> clf = create_classifier(config)
    >>> isinstance(clf, LogisticRegressionStrategy)
    True

    >>> # XGBoost (Phase 2)
    >>> config = {"type": "xgboost", "n_estimators": 100}
    >>> clf = create_classifier(config)
    >>> isinstance(clf, XGBoostStrategy)
    True
"""

from typing import Any

from antibody_training_esm.core.classifier_strategy import ClassifierStrategy
from antibody_training_esm.core.strategies.logistic_regression import (
    LogisticRegressionStrategy,
)


def create_classifier(config: dict[str, Any]) -> ClassifierStrategy:
    """
    Factory function for creating classifier strategies.

    Args:
        config: Configuration dictionary with "type" key and hyperparameters

    Returns:
        ClassifierStrategy instance (LogReg, XGBoost, etc.)

    Raises:
        ValueError: If classifier type is unknown

    Notes:
        - Defaults to "logistic_regression" if "type" not specified (backward compat)
        - Supported types: "logistic_regression", "xgboost" (Phase 2)

    Examples:
        >>> # Logistic Regression (explicit)
        >>> config = {"type": "logistic_regression", "C": 1.0}
        >>> clf = create_classifier(config)

        >>> # Logistic Regression (implicit default)
        >>> config = {"C": 1.0}  # No "type" field
        >>> clf = create_classifier(config)

        >>> # XGBoost (Phase 2)
        >>> config = {"type": "xgboost", "n_estimators": 100}
        >>> clf = create_classifier(config)
    """
    # Default to logistic_regression for backward compatibility
    classifier_type = config.get("type", "logistic_regression")

    if classifier_type == "logistic_regression":
        return LogisticRegressionStrategy(config)
    elif classifier_type == "xgboost":
        # Phase 2: XGBoost implementation
        try:
            from antibody_training_esm.core.strategies.xgboost_strategy import (
                XGBoostStrategy,
            )

            return XGBoostStrategy(config)
        except ImportError as e:
            raise ImportError(
                "XGBoost classifier requested but xgboost not installed. "
                "Install with: pip install xgboost>=2.0.0"
            ) from e
    else:
        raise ValueError(
            f"Unknown classifier type: '{classifier_type}'. "
            f"Supported types: logistic_regression, xgboost"
        )


# Registry pattern for extensibility (future: plugin system)
CLASSIFIER_REGISTRY: dict[str, type[ClassifierStrategy]] = {
    "logistic_regression": LogisticRegressionStrategy,
}


def register_classifier(name: str, strategy_class: type[ClassifierStrategy]) -> None:
    """
    Register a new classifier strategy (for plugins/extensions).

    Args:
        name: Classifier type name (e.g., "mlp", "svm")
        strategy_class: ClassifierStrategy implementation

    Examples:
        >>> # Future: MLP classifier
        >>> class MLPStrategy(ClassifierStrategy):
        ...     pass
        >>> register_classifier("mlp", MLPStrategy)
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

    Notes:
        This function enables plugin-based classifier registration.
        Use for third-party classifiers or experimental implementations.
    """
    classifier_type = config.get("type", "logistic_regression")

    if classifier_type not in CLASSIFIER_REGISTRY:
        raise ValueError(
            f"Unknown classifier type: '{classifier_type}'. "
            f"Registered types: {list(CLASSIFIER_REGISTRY.keys())}"
        )

    strategy_class = CLASSIFIER_REGISTRY[classifier_type]
    return strategy_class(config)
