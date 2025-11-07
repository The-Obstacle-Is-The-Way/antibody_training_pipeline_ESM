"""
Data Loading Module

Professional data loading utilities for antibody sequence datasets.
Supports Hugging Face datasets, local CSV files, and preprocessing pipelines.
"""

import logging
import pickle
from typing import Any

import numpy as np
import pandas as pd
from datasets import load_dataset

logger = logging.getLogger(__name__)


def preprocess_raw_data(
    X: list[str],
    y: list[Any],
    embedding_extractor,
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
    X: list[str] | None = None,
    y: list[Any] | None = None,
    X_embedded: np.ndarray | None = None,
    filename: str | None = None,
):
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

    data: dict[str, list[str] | list[Any] | np.ndarray] = {}
    if X_embedded is not None:
        data["X_embedded"] = X_embedded
    if X is not None:
        data["X"] = X
    if y is not None:
        data["y"] = y

    with open(filename, "wb") as f:
        pickle.dump(data, f)


def load_preprocessed_data(filename: str) -> dict[str, Any]:
    """
    Load preprocessed data from pickle file

    Args:
        filename: Path to pickle file

    Returns:
        Dictionary with keys: 'X', 'y', and/or 'X_embedded'
    """
    with open(filename, "rb") as f:
        data: dict[str, Any] = pickle.load(f)
        return data


def load_hf_dataset(
    dataset_name: str,
    split: str,
    text_column: str,
    label_column: str,
) -> tuple[list[str], list[Any]]:
    """
    Load dataset from Hugging Face datasets library

    Args:
        dataset_name: Name of the dataset to load
        split: Which split to load (e.g., 'train', 'test', 'validation')
        text_column: Name of the column containing the sequences
        label_column: Name of the column containing the labels

    Returns:
        X: List of input sequences
        y: List of labels
    """
    dataset = load_dataset(dataset_name, split=split)
    X = dataset[text_column]
    y = dataset[label_column]

    return list(X), list(y)


def load_local_data(
    file_path: str, text_column: str, label_column: str
) -> tuple[list[str], list[Any]]:
    """
    Load training data from local CSV file

    Args:
        file_path: Path to the local data file (CSV format)
        text_column: Name of the column containing the sequences
        label_column: Name of the column containing the labels

    Returns:
        X: List of input sequences
        y: List of labels
    """
    train_df = pd.read_csv(file_path, comment="#")  # Handle comment lines in CSV
    X_train = train_df[text_column].tolist()
    y_train = train_df[label_column].tolist()

    return X_train, y_train


def load_data(config: dict) -> tuple[list[str], list[int]]:
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
