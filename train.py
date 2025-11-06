"""
Training Script (BACKWARDS COMPATIBILITY SHIM)

This module is deprecated. Use the new CLI interface instead:
    antibody-train --config configs/config.yaml

For backwards compatibility, this script delegates to the new package.
"""

import warnings

# Re-export from new package for backwards compatibility
from antibody_training_esm.core.trainer import train_model

warnings.warn(
    "Running 'python train.py' is deprecated. Use 'antibody-train' CLI command instead.",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"
    results = train_model(config_path)

    print("Training completed successfully!")
