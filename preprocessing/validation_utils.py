"""
Shared validation utilities for preprocessing pipelines.

Consolidates common validation logic used across different dataset pipelines
(Boughter, Jain, Harvey, Shehata) to ensure consistency and reduce duplication.

Functions:
- File/Directory validation
- DataFrame validation with Pandera schemas
- Label statistics calculation
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError

from preprocessing.logging_config import setup_logger

logger = setup_logger(__name__)


def calculate_checksum(file_path: str | Path) -> str:
    """Calculate SHA256 checksum of a file."""
    sha = hashlib.sha256()
    path = Path(file_path)
    if not path.exists():
        return ""

    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


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


def validate_dataframe_with_schema(
    df: pd.DataFrame,
    schema: pa.DataFrameSchema,
    dataset_name: str,
) -> list[str]:
    """
    Validate DataFrame against Pandera schema.

    Args:
        df: DataFrame to validate
        schema: Pandera schema
        dataset_name: Name for error messages

    Returns:
        List of error messages (empty if valid)
    """
    try:
        schema.validate(df, lazy=False)
        return []
    except SchemaError as e:
        msg = f"{dataset_name} validation failed: {e}"
        logger.error(f"  ✗ {msg}")
        return [msg]


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
