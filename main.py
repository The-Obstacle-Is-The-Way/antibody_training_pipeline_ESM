"""
Main Entry Point (BACKWARDS COMPATIBILITY SHIM)

This script is deprecated. Use the new CLI commands instead:
    antibody-train --config configs/config.yaml

For backwards compatibility, this script delegates to the new CLI.
"""

import sys
import warnings

from antibody_training_esm.cli.train import main as train_main

warnings.warn(
    "Running 'python main.py' is deprecated. Use 'antibody-train' CLI command instead.",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    sys.exit(train_main())
