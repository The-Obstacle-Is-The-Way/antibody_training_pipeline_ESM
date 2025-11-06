"""
Binary Classifier Module (BACKWARDS COMPATIBILITY SHIM)

This module is deprecated. Import from antibody_training_esm.core.classifier instead.

For backwards compatibility, this module re-exports BinaryClassifier from the new package.
"""

import warnings

# Re-export from new package for backwards compatibility
from antibody_training_esm.core.classifier import BinaryClassifier

warnings.warn(
    "Importing from 'classifier' is deprecated. Use 'from antibody_training_esm.core.classifier import BinaryClassifier' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["BinaryClassifier"]
