"""
Data Loading Module (BACKWARDS COMPATIBILITY SHIM)

This module is deprecated. Import from antibody_training_esm.data.loaders instead.

For backwards compatibility, this module re-exports all functions from the new package.
"""

import warnings

# Re-export from new package for backwards compatibility
from antibody_training_esm.data.loaders import (
    load_data,
    load_hf_dataset,
    load_local_data,
    load_preprocessed_data,
    preprocess_raw_data,
    store_preprocessed_data,
)

warnings.warn(
    "Importing from 'data' is deprecated. Use 'from antibody_training_esm.data.loaders import ...' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "load_data",
    "load_hf_dataset",
    "load_local_data",
    "load_preprocessed_data",
    "preprocess_raw_data",
    "store_preprocessed_data",
]
