"""
ESM Model Module (BACKWARDS COMPATIBILITY SHIM)

This module is deprecated. Import from antibody_training_esm.core.embeddings instead.

For backwards compatibility, this module re-exports ESMEmbeddingExtractor from the new package.
"""

import warnings

# Re-export from new package for backwards compatibility
from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor

warnings.warn(
    "Importing from 'model' is deprecated. Use 'from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["ESMEmbeddingExtractor"]
