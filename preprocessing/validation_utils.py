"""
Shared validation utilities for preprocessing pipelines.

Consolidates common validation logic used across different dataset pipelines
(Boughter, Jain, Harvey, Shehata) to ensure consistency and reduce duplication.

Functions:
- File/Directory validation
- DataFrame structure validation (columns, nulls)
- Sequence validation (amino acids, gaps, empty)
- Label statistics calculation
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from preprocessing.logging_config import setup_logger

logger = setup_logger(__name__)

# Standard valid amino acids (20 standard + X for unknown if permitted)
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def validate_directory_exists(path: Path, name: str = "Directory") -> bool:
    """Check if a directory exists and log result."""
    if not path.exists():
        logger.error(f"✗ {name} not found: {path}")
        return False
    return True


def validate_file_exists(path: Path, name: str = "File") -> bool:
    """Check if a file exists and log result."""
    if not path.exists():
        logger.error(f"✗ {name} not found: {path}")
        return False
    return True


def validate_dataframe_columns(
    df: pd.DataFrame, required_columns: set[str], file_name: str
) -> list[str]:
    """
    Check if DataFrame contains all required columns.

    Args:
        df: DataFrame to check
        required_columns: Set of column names that must exist
        file_name: Name of the file for error messaging

    Returns:
        List of error messages (empty if valid)
    """
    missing_cols = required_columns - set(df.columns)
    errors = []
    if missing_cols:
        msg = f"{file_name}: Missing required columns {missing_cols}"
        errors.append(msg)
        logger.error(f"  ✗ {msg}")
    return errors


def validate_no_nulls(
    df: pd.DataFrame, columns: list[str], file_name: str
) -> list[str]:
    """
    Check for null values in specified columns.

    Args:
        df: DataFrame to check
        columns: List of columns to check for nulls
        file_name: Name of file for error messaging

    Returns:
        List of error messages
    """
    errors = []
    for col in columns:
        if col in df.columns:
            nulls = df[col].isna().sum()
            if nulls > 0:
                msg = f"{file_name}: {nulls} null values in '{col}'"
                errors.append(msg)
                logger.error(f"  ✗ {msg}")
    return errors


def validate_no_empty_sequences(
    df: pd.DataFrame, sequence_col: str, file_name: str
) -> list[str]:
    """
    Check for empty strings in sequence column.

    Args:
        df: DataFrame to check
        sequence_col: Name of sequence column
        file_name: Name of file for error messaging

    Returns:
        List of error messages
    """
    errors = []
    if sequence_col in df.columns:
        # Check for empty strings (len == 0)
        empty_seqs = (df[sequence_col].fillna("").astype(str).str.len() == 0).sum()
        if empty_seqs > 0:
            msg = f"{file_name}: {empty_seqs} empty sequences in '{sequence_col}'"
            errors.append(msg)
            logger.error(f"  ✗ {msg}")
    return errors


def validate_no_gaps(df: pd.DataFrame, sequence_col: str, file_name: str) -> list[str]:
    """
    Check for gap characters (-, *, .) in sequences.
    ESM models cannot handle gaps.

    Args:
        df: DataFrame to check
        sequence_col: Name of sequence column
        file_name: Name of file for error messaging

    Returns:
        List of error messages
    """
    errors = []
    if sequence_col in df.columns:
        # Check for common gap characters
        gap_pattern = r"[-*.]"
        gap_count = (
            df[sequence_col].astype(str).str.contains(gap_pattern, regex=True).sum()
        )

        if gap_count > 0:
            msg = f"{file_name}: {gap_count} sequences contain gaps/invalid chars (-, *, .)"
            errors.append(msg)
            logger.error(f"  ✗ {msg}")

    return errors


def validate_amino_acids(
    df: pd.DataFrame, sequence_col: str, file_name: str
) -> list[str]:
    """
    Check that sequences contain only valid standard amino acids.

    Args:
        df: DataFrame to check
        sequence_col: Name of sequence column
        file_name: Name of file for error messaging

    Returns:
        List of error messages
    """
    errors = []
    if sequence_col in df.columns:
        invalid_count = 0
        for seq in df[sequence_col].dropna().astype(str):
            if set(seq.upper()) - VALID_AA:
                invalid_count += 1

        if invalid_count > 0:
            msg = f"{file_name}: {invalid_count} sequences contain invalid amino acids"
            errors.append(msg)
            logger.error(f"  ✗ {msg}")

    return errors


def calculate_label_stats(df: pd.DataFrame, label_col: str = "label") -> dict[str, Any]:
    """
    Calculate distribution statistics for binary labels.

    Args:
        df: DataFrame with labels
        label_col: Name of label column

    Returns:
        Dictionary with counts and percentages
    """
    stats: dict[str, Any] = {
        "total": len(df),
        "specific": 0,
        "non_specific": 0,
        "specific_pct": 0.0,
        "non_specific_pct": 0.0,
        "nulls": 0,
    }

    if label_col in df.columns:
        stats["specific"] = int((df[label_col] == 0).sum())
        stats["non_specific"] = int((df[label_col] == 1).sum())
        stats["nulls"] = int(df[label_col].isna().sum())

        if len(df) > 0:
            stats["specific_pct"] = (stats["specific"] / len(df)) * 100
            stats["non_specific_pct"] = (stats["non_specific"] / len(df)) * 100

    return stats


def log_label_stats(stats: dict[str, Any], dataset_name: str) -> None:
    """Log label statistics in a standardized format."""
    logger.info(f"\n{dataset_name} Label Distribution:")
    logger.info(f"  Total: {stats['total']}")
    logger.info(
        f"  Specific (0):     {stats['specific']:4d} ({stats['specific_pct']:.1f}%)"
    )
    logger.info(
        f"  Non-specific (1): {stats['non_specific']:4d} ({stats['non_specific_pct']:.1f}%)"
    )
    if stats["nulls"] > 0:
        logger.info(f"  Null/Excluded:    {stats['nulls']:4d}")
