"""Dataset loading and validation utilities."""

import logging
import os

import pandas as pd

from antibody_training_esm.cli.testing.config import TestConfig

logger = logging.getLogger(__name__)


def load_dataset(data_path: str, config: TestConfig) -> tuple[list[str], list[int]]:
    """
    Load dataset from CSV file using configured column names.

    Args:
        data_path: Path to the CSV file.
        config: Test configuration object containing column names.

    Returns:
        Tuple of (sequences, labels).
    """
    logger.info(f"Loading dataset from {data_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    # Defensive: Handle legacy files with comment headers
    # New files (post-HF cleanup) are standard CSVs without comments
    df = pd.read_csv(data_path, comment="#")

    sequence_col = config.sequence_column
    label_col = config.label_column

    if sequence_col not in df.columns:
        raise ValueError(
            f"Sequence column '{sequence_col}' not found in dataset. Available columns: {list(df.columns)}"
        )
    if label_col not in df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found in dataset. Available columns: {list(df.columns)}"
        )

    # CRITICAL VALIDATION: Check for NaN labels (P0 bug fix)
    nan_count = df[label_col].isna().sum()
    if nan_count > 0:
        raise ValueError(
            f"CRITICAL: Dataset contains {nan_count} NaN labels! "
            f"This will corrupt evaluation metrics. "
            f"Please use the curated canonical test file (e.g., "
            f"data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv with no NaNs)."
        )

    # For Jain test sets, validate expected size (allow legacy 94 + canonical 86)
    if "jain" in data_path.lower() and "test" in data_path.lower():
        expected_sizes = {94, 86}
        if len(df) not in expected_sizes:
            raise ValueError(
                f"Jain test set has {len(df)} antibodies but expected one of {sorted(expected_sizes)}. "
                f"Using the wrong test set will produce invalid metrics. "
                f"Please use the correct curated file (preferred: "
                f"data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv)."
            )

    sequences = df[sequence_col].tolist()
    labels = df[label_col].tolist()

    logger.info(
        f"Loaded {len(sequences)} samples from {data_path} (sequence_col='{sequence_col}', label_col='{label_col}')"
    )
    logger.info(f"  Label distribution: {pd.Series(labels).value_counts().to_dict()}")
    return sequences, labels
