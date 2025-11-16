"""
Classifier Strategy Implementations

This package contains concrete implementations of the ClassifierStrategy protocol.

Available Strategies:
    - LogisticRegressionStrategy: Wrapper for sklearn LogisticRegression
    - XGBoostStrategy: Wrapper for xgboost XGBClassifier (requires xgboost>=2.0.0)

Usage:
    >>> from antibody_training_esm.core.strategies import LogisticRegressionStrategy
    >>> config = {"C": 1.0, "random_state": 42}
    >>> clf = LogisticRegressionStrategy(config)
    >>> clf.fit(X_train, y_train)
    >>> predictions = clf.predict(X_test)
"""

from antibody_training_esm.core.strategies.logistic_regression import (
    LogisticRegressionStrategy,
)

__all__ = [
    "LogisticRegressionStrategy",
]

# XGBoostStrategy will be added in Phase 2
try:
    from antibody_training_esm.core.strategies.xgboost_strategy import XGBoostStrategy

    __all__.append("XGBoostStrategy")
except ImportError:
    # xgboost not installed - skip XGBoostStrategy
    pass
