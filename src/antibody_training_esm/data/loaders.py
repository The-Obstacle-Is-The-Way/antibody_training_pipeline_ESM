"""
Data Loading Module

Professional data loading utilities for antibody sequence datasets.
Supports Hugging Face datasets, local CSV files, and preprocessing pipelines.
"""

import logging
import pickle  # nosec B403 - Used only for local trusted data (preprocessed datasets)
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from datasets import load_dataset

logger = logging.getLogger(__name__)
type Label = int | float | bool | str


def preprocess_raw_data(
    X: Sequence[str],
    y: Sequence[Label],
    embedding_extractor: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Embed sequences using ESM model

    Args:
        X: List of input protein sequences (strings)
        y: List or array of labels
        embedding_extractor: Instance with 'embed_sequence' or 'extract_batch_embeddings' method

    Returns:
        X_embedded: Embedded input sequences
        y: Labels as numpy array

    Notes:
        No StandardScaler used - matches Novo Nordisk methodology
    """
    logger.info(f"Embedding {len(X)} sequences...")

    # Try to use batch embedding if available (more efficient)
    if hasattr(embedding_extractor, "extract_batch_embeddings"):
        X_embedded = embedding_extractor.extract_batch_embeddings(X)
    else:
        X_embedded = np.array([embedding_extractor.embed_sequence(seq) for seq in X])

    return X_embedded, np.array(y)


def store_preprocessed_data(
    X: Sequence[str] | None = None,
    y: Sequence[Label] | None = None,
    X_embedded: np.ndarray | None = None,
    filename: str | None = None,
) -> None:
    """
    Store preprocessed data to pickle file

    Args:
        X: Raw sequences (optional)
        y: Labels (optional)
        X_embedded: Embedded data (optional)
        filename: Output file path (required)

    Raises:
        ValueError: If filename is not provided
    """
    if filename is None:
        raise ValueError("filename is required")

    data: dict[str, Sequence[str] | Sequence[Label] | np.ndarray] = {}
    if X_embedded is not None:
        data["X_embedded"] = X_embedded
    if X is not None:
        data["X"] = X
    if y is not None:
        data["y"] = y

    with open(filename, "wb") as f:
        pickle.dump(data, f)


def load_preprocessed_data(
    filename: str,
) -> dict[str, list[str] | list[Label] | np.ndarray]:
    """
    Load preprocessed data from pickle file

    Args:
        filename: Path to pickle file

    Returns:
        Dictionary with keys: 'X', 'y', and/or 'X_embedded'
    """
    with open(filename, "rb") as f:
        data = cast(dict[str, list[str] | list[Label] | np.ndarray], pickle.load(f))  # nosec B301 - Loading our own preprocessed dataset from local file
        return data


def load_hf_dataset(
    dataset_name: str,
    split: str,
    text_column: str,
    label_column: str,
    revision: str = "main",
) -> tuple[list[str], list[Label]]:
    """
    Load dataset from Hugging Face datasets library

    Args:
        dataset_name: Name of the dataset to load
        split: Which split to load (e.g., 'train', 'test', 'validation')
        text_column: Name of the column containing the sequences
        label_column: Name of the column containing the labels
        revision: HuggingFace dataset revision (commit SHA or branch name) for reproducibility

    Returns:
        X: List of input sequences
        y: List of labels
    """
    dataset = load_dataset(
        dataset_name,
        split=split,
        revision=revision,  # nosec B615 - Pinned to specific version for scientific reproducibility
    )
    X = list(dataset[text_column])
    y = cast(list[Label], list(dataset[label_column]))

    return X, y


def load_local_data(
    file_path: str | Path, text_column: str, label_column: str
) -> tuple[list[str], list[Label]]:
    """
    Load training data from local CSV file

    Args:
        file_path: Path to the local data file (CSV format)
        text_column: Name of the column containing the sequences
        label_column: Name of the column containing the labels

    Returns:
        X: List of input sequences
        y: List of labels

    Raises:
        ValueError: If required columns are missing from CSV
    """
    train_df = pd.read_csv(file_path, comment="#")  # Handle comment lines in CSV

    # Validate required columns exist
    available_columns = list(train_df.columns)
    if text_column not in train_df.columns:
        raise ValueError(
            f"Sequence column '{text_column}' not found in {file_path}. "
            f"Available columns: {available_columns}"
        )
    if label_column not in train_df.columns:
        raise ValueError(
            f"Label column '{label_column}' not found in {file_path}. "
            f"Available columns: {available_columns}"
        )

    X_train = train_df[text_column].tolist()
    y_train = cast(list[Label], train_df[label_column].tolist())

    return X_train, y_train


def load_data(config: dict[str, Any]) -> tuple[list[str], list[Label]]:
    """
    Load training data from either Hugging Face or local file based on config

    Args:
        config: Configuration dictionary containing data parameters

    Returns:
        X_train: List of training sequences
        y_train: List of training labels

    Raises:
        ValueError: If data source is unknown
    """
    data_config = config["data"]

    if data_config["source"] == "hf":
        return load_hf_dataset(
            dataset_name=data_config["dataset_name"],
            split=data_config["train_split"],
            text_column=data_config["sequence_column"],
            label_column=data_config["label_column"],
        )
    elif data_config["source"] == "local":
        return load_local_data(
            data_config["train_file"],
            text_column=data_config["sequence_column"],
            label_column=data_config["label_column"],
        )
    else:
        raise ValueError(f"Unknown data source: {data_config['source']}")
