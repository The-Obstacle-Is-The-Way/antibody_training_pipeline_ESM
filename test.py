"""
Test Script (BACKWARDS COMPATIBILITY SHIM)

This script is deprecated. Use the new CLI command instead:
    antibody-test --model models/antibody_classifier.pkl --data test_datasets/jain/test.csv

    # Multi-model and multi-dataset testing
    antibody-test --model m1.pkl m2.pkl --data d1.csv d2.csv

    # Using config files
    antibody-test --config test_config.yaml

For backwards compatibility, this script delegates to the new CLI.
"""

import sys
import warnings

from antibody_training_esm.cli.test import main as test_main

warnings.warn(
    "Running 'python test.py' is deprecated. Use 'antibody-test' CLI command instead.\n"
    "Examples:\n"
    "  antibody-test --model models/classifier.pkl --data test_datasets/jain/test.csv\n"
    "  antibody-test --model m1.pkl m2.pkl --data d1.csv d2.csv\n"
    "  antibody-test --config test_config.yaml",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    sys.exit(test_main())
