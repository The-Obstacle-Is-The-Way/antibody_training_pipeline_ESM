"""
Embedding cache management.

Handles loading, saving, and validating ESM/AMPLIFY embeddings to disk to avoid
redundant computation.
"""

import hashlib
import logging
import os
import pickle  # nosec B403
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from antibody_training_esm.core.classifier import EmbeddingExtractorProtocol


def validate_embeddings(
    embeddings: np.ndarray,
    num_sequences: int,
    logger: logging.Logger,
    source: str = "cache",
) -> None:
    """
    Validate embeddings are not corrupted.

    Args:
        embeddings: Embedding array to validate
        num_sequences: Expected number of sequences
        logger: Logger instance
        source: Where embeddings came from (for error messages)

    Raises:
        ValueError: If embeddings are invalid (wrong shape, NaN, all zeros)
    """
    # Check shape
    if embeddings.shape[0] != num_sequences:
        raise ValueError(
            f"Embeddings from {source} have wrong shape: expected {num_sequences} sequences, "
            f"got {embeddings.shape[0]}"
        )

    if len(embeddings.shape) != 2:
        raise ValueError(
            f"Embeddings from {source} must be 2D array, got shape {embeddings.shape}"
        )

    # Check for NaN values
    if np.isnan(embeddings).any():
        nan_count = np.isnan(embeddings).sum()
        raise ValueError(
            f"Embeddings from {source} contain {nan_count} NaN values. "
            "This indicates corrupted embeddings - cannot train on invalid data."
        )

    # Check for all-zero rows (corrupted/failed embeddings)
    zero_rows = np.all(embeddings == 0, axis=1)
    if zero_rows.any():
        zero_count = zero_rows.sum()
        raise ValueError(
            f"Embeddings from {source} contain {zero_count} all-zero rows. "
            "This indicates corrupted embeddings from failed batch processing. "
            "Delete the cache file and recompute."
        )

    logger.debug(
        f"Embeddings validation passed: shape={embeddings.shape}, no NaN, no zero rows"
    )


def get_or_create_embeddings(
    sequences: list[str],
    embedding_extractor: "EmbeddingExtractorProtocol",
    cache_path: str | Path,
    dataset_name: str,
    logger: logging.Logger,
) -> np.ndarray:
    """
    Get embeddings from cache or create them

    Args:
        sequences: List of protein sequences
        embedding_extractor: ESM or AMPLIFY embedding extractor
        cache_path: Directory for caching embeddings
        dataset_name: Name of dataset (for cache filename)
        logger: Logger instance

    Returns:
        Array of embeddings

    Raises:
        ValueError: If cached or computed embeddings are invalid
    """
    # Ensure cache_path is string for os.path.join/os.makedirs compatibility
    # (os.path supports Path in 3.6+, but for safety/consistency with type hint)
    cache_path_str = str(cache_path)

    # P2.3 fix: Use streaming hash to avoid creating giant string in memory
    # For large datasets (100k+ sequences), joining all sequences into one
    # string can consume 10s of MB and cause memory pressure.
    hasher = hashlib.sha256()

    # Hash model metadata first
    hasher.update(
        f"{embedding_extractor.model_name}|"
        f"{embedding_extractor.revision}|"
        f"{embedding_extractor.max_length}|".encode()
    )

    # Stream sequences through hash (no giant string!)
    for seq in sequences:
        hasher.update(seq.encode())
        hasher.update(b"|")  # Separator

    sequences_hash = hasher.hexdigest()[:12]
    cache_file = os.path.join(
        cache_path_str, f"{dataset_name}_{sequences_hash}_embeddings.pkl"
    )

    if os.path.exists(cache_file):
        logger.info(f"Loading cached embeddings from {cache_file}")
        with open(cache_file, "rb") as f:
            cached_data_raw = pickle.load(f)  # nosec B301 - Hash-validated local cache

        # Validate loaded data type and structure
        if not isinstance(cached_data_raw, dict):
            logger.warning(
                f"Invalid cache file format (expected dict, got {type(cached_data_raw).__name__}). "
                "Recomputing embeddings..."
            )
        elif (
            "embeddings" not in cached_data_raw
            or "sequences_hash" not in cached_data_raw
        ):
            missing_keys = {"embeddings", "sequences_hash"} - set(
                cached_data_raw.keys()
            )
            logger.warning(
                f"Corrupt cache file (missing keys: {missing_keys}). "
                "Recomputing embeddings..."
            )
        else:
            cached_data: dict[str, Any] = cached_data_raw

            # Verify the cached sequences and model metadata match exactly
            # This prevents ESM2 from reusing ESM-1v embeddings, etc.
            model_metadata_matches = (
                cached_data.get("model_name") == embedding_extractor.model_name
                and cached_data.get("revision") == embedding_extractor.revision
                and cached_data.get("max_length") == embedding_extractor.max_length
            )

            if (
                len(cached_data["embeddings"]) == len(sequences)
                and cached_data["sequences_hash"] == sequences_hash
                and model_metadata_matches
            ):
                logger.info(
                    f"Using cached embeddings for {len(sequences)} sequences "
                    f"(model: {embedding_extractor.model_name}, hash: {sequences_hash})"
                )
                embeddings_result: np.ndarray = cached_data["embeddings"]

                # Validate cached embeddings before using them
                validate_embeddings(
                    embeddings_result, len(sequences), logger, source="cache"
                )

                return embeddings_result
            elif not model_metadata_matches:
                logger.warning(
                    f"Cached embeddings model mismatch "
                    f"(cached: {cached_data.get('model_name')}, "
                    f"current: {embedding_extractor.model_name}). "
                    "Recomputing..."
                )
            else:
                logger.warning("Cached embeddings hash mismatch, recomputing...")

    logger.info(f"Computing embeddings for {len(sequences)} sequences...")
    embeddings = embedding_extractor.extract_batch_embeddings(sequences)

    # Validate newly computed embeddings before caching
    validate_embeddings(embeddings, len(sequences), logger, source="computed")

    # Cache the embeddings with metadata for verification
    # Include model metadata to prevent cache collisions between different backbones
    os.makedirs(cache_path_str, exist_ok=True)
    cache_data = {
        "embeddings": embeddings,
        "sequences_hash": sequences_hash,
        "num_sequences": len(sequences),
        "dataset_name": dataset_name,
        "model_name": embedding_extractor.model_name,
        "revision": embedding_extractor.revision,
        "max_length": embedding_extractor.max_length,
    }
    with open(cache_file, "wb") as f:
        pickle.dump(cache_data, f)
    logger.info(
        f"Cached embeddings to {cache_file} "
        f"(model: {embedding_extractor.model_name}, hash: {sequences_hash})"
    )

    return embeddings
