"""
Training Module

Professional training pipeline for antibody classification models.
Includes cross-validation, embedding caching, and comprehensive evaluation.
"""

import hashlib
import json
import logging
import os
import pickle  # nosec B403 - Used only for local trusted data (models, caches)
import warnings
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
import sklearn  # For sklearn.__version__
import yaml
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from antibody_training_esm.core.classifier import BinaryClassifier
from antibody_training_esm.core.config import DEFAULT_BATCH_SIZE
from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor
from antibody_training_esm.data.loaders import load_data


def validate_config(config: dict[str, Any]) -> None:
    """
    Validate that config dictionary contains all required keys.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If any required keys are missing or invalid
    """
    # Define required config structure
    required_keys = {
        "data": ["train_file", "test_file", "embeddings_cache_dir"],
        "model": ["name", "device"],
        "classifier": [],  # Nested validation happens in BinaryClassifier
        "training": ["log_level", "metrics", "n_splits"],
        "experiment": ["name"],
    }

    missing_sections = []
    missing_keys = []

    # Check top-level sections exist
    for section in required_keys:
        if section not in config:
            missing_sections.append(section)
            continue

        # Check keys within each section
        if not isinstance(config[section], dict):
            raise ValueError(
                f"Config section '{section}' must be a dictionary, "
                f"got {type(config[section]).__name__}"
            )

        for key in required_keys[section]:
            if key not in config[section]:
                missing_keys.append(f"{section}.{key}")

    # Construct helpful error message
    if missing_sections or missing_keys:
        error_parts = []
        if missing_sections:
            error_parts.append(
                f"Missing config sections: {', '.join(missing_sections)}"
            )
        if missing_keys:
            error_parts.append(f"Missing config keys: {', '.join(missing_keys)}")
        raise ValueError("Config validation failed:\n  - " + "\n  - ".join(error_parts))


def setup_logging(config: dict[str, Any] | DictConfig) -> logging.Logger:
    """
    Setup logging configuration (Hydra-aware)

    If running under Hydra (@hydra.main decorator), uses Hydra's output directory.
    If running in legacy mode, uses absolute path from config.

    Args:
        config: Configuration dictionary or DictConfig

    Returns:
        Configured logger

    Raises:
        ValueError: If log_level is invalid
    """
    from hydra.core.hydra_config import HydraConfig

    # Convert DictConfig to dict if needed for uniform access
    if isinstance(config, DictConfig):
        config_dict = cast(dict[str, Any], OmegaConf.to_container(config, resolve=True))
    else:
        config_dict = config

    # Validate log level
    VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    level_str = config_dict["training"]["log_level"].upper()
    if level_str not in VALID_LEVELS:
        raise ValueError(
            f"Invalid log_level '{level_str}' in config. Must be one of: {VALID_LEVELS}"
        )

    log_level = getattr(logging, level_str)
    log_file_str = config_dict["training"]["log_file"]

    # Determine log file path (Hydra-aware)
    try:
        # Try to get Hydra's output directory (only works when @hydra.main is active)
        hydra_cfg = HydraConfig.get()
        output_dir = Path(hydra_cfg.runtime.output_dir)
        log_file = output_dir / log_file_str  # log_file is relative to Hydra output dir
    except Exception:
        # Running in legacy mode (no Hydra decorator)
        # Fall back to absolute path from config
        log_file = Path(log_file_str)
        if not log_file.is_absolute():
            # If relative path, route to logs/ directory (not repo root!)
            log_file = Path.cwd() / "logs" / log_file_str
        # Create log directory if it doesn't exist
        log_file.parent.mkdir(parents=True, exist_ok=True)

    # Configure logging
    # force=True prevents duplicate log lines when Hydra has already configured logging
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,  # Python 3.8+ - replaces existing handlers
    )

    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict[str, Any]:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If YAML is invalid
    """
    try:
        with open(config_path) as f:
            config: dict[str, Any] = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            "Please create it or specify a valid path with --config"
        ) from None
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config file {config_path}: {e}") from e


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
    embedding_extractor: ESMEmbeddingExtractor,
    cache_path: str,
    dataset_name: str,
    logger: logging.Logger,
) -> np.ndarray:
    """
    Get embeddings from cache or create them

    Args:
        sequences: List of protein sequences
        embedding_extractor: ESM embedding extractor
        cache_path: Directory for caching embeddings
        dataset_name: Name of dataset (for cache filename)
        logger: Logger instance

    Returns:
        Array of embeddings

    Raises:
        ValueError: If cached or computed embeddings are invalid
    """
    # Create a hash of the sequences to ensure cache validity
    sequences_str = "|".join(sequences)
    # Use SHA-256 (non-cryptographic usage) to satisfy security scanners and
    # prevent weak-hash findings while keeping deterministic cache keys.
    sequences_hash = hashlib.sha256(sequences_str.encode()).hexdigest()[:12]
    cache_file = os.path.join(
        cache_path, f"{dataset_name}_{sequences_hash}_embeddings.pkl"
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

            # Verify the cached sequences match exactly
            if (
                len(cached_data["embeddings"]) == len(sequences)
                and cached_data["sequences_hash"] == sequences_hash
            ):
                logger.info(
                    f"Using cached embeddings for {len(sequences)} sequences (hash: {sequences_hash})"
                )
                embeddings_result: np.ndarray = cached_data["embeddings"]

                # Validate cached embeddings before using them
                validate_embeddings(
                    embeddings_result, len(sequences), logger, source="cache"
                )

                return embeddings_result
            else:
                logger.warning("Cached embeddings hash mismatch, recomputing...")

    logger.info(f"Computing embeddings for {len(sequences)} sequences...")
    embeddings = embedding_extractor.extract_batch_embeddings(sequences)

    # Validate newly computed embeddings before caching
    validate_embeddings(embeddings, len(sequences), logger, source="computed")

    # Cache the embeddings with metadata for verification
    os.makedirs(cache_path, exist_ok=True)
    cache_data = {
        "embeddings": embeddings,
        "sequences_hash": sequences_hash,
        "num_sequences": len(sequences),
        "dataset_name": dataset_name,
    }
    with open(cache_file, "wb") as f:
        pickle.dump(cache_data, f)
    logger.info(f"Cached embeddings to {cache_file} (hash: {sequences_hash})")

    return embeddings


def evaluate_model(
    classifier: BinaryClassifier,
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    metrics: list[str],
    logger: logging.Logger,
) -> dict[str, float]:
    """
    Evaluate model performance

    Args:
        classifier: Trained classifier
        X: Embeddings array
        y: Labels array
        dataset_name: Name of dataset being evaluated
        metrics: List of metrics to compute
        logger: Logger instance

    Returns:
        Dictionary of metric results
    """
    logger.info(f"Evaluating model on {dataset_name} set")

    # Get predictions
    y_pred = classifier.predict(X)
    y_pred_proba = classifier.predict_proba(X)[:, 1]  # Probability of positive class

    # Calculate metrics
    results = {}

    if "accuracy" in metrics:
        results["accuracy"] = accuracy_score(y, y_pred)

    if "precision" in metrics:
        results["precision"] = precision_score(y, y_pred, average="binary")

    if "recall" in metrics:
        results["recall"] = recall_score(y, y_pred, average="binary")

    if "f1" in metrics:
        results["f1"] = f1_score(y, y_pred, average="binary")

    if "roc_auc" in metrics:
        results["roc_auc"] = roc_auc_score(y, y_pred_proba)

    # Log results
    logger.info(f"{dataset_name} Results:")
    for metric, value in results.items():
        logger.info(f"  {metric}: {value:.4f}")

    # Log classification report
    logger.info(f"\n{dataset_name} Classification Report:")
    logger.info(f"\n{classification_report(y, y_pred)}")

    return results


def perform_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    config: dict[str, Any],
    logger: logging.Logger,
) -> dict[str, dict[str, float]]:
    """
    Perform cross-validation

    Args:
        X: Embeddings array
        y: Labels array
        config: Configuration dictionary
        logger: Logger instance

    Returns:
        Dictionary of cross-validation results
    """
    cv_config = config["classifier"]
    cv_folds = cv_config["cv_folds"]
    random_state = cv_config["random_state"]
    stratify = cv_config["stratify"]

    logger.info(f"Performing {cv_folds}-fold cross-validation")

    # Setup cross-validation
    if stratify:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    else:
        from sklearn.model_selection import KFold

        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # Perform cross-validation for different metrics
    cv_results = {}

    # Create a new classifier instance for CV (to avoid fitting on full data)
    cv_params = config["classifier"].copy()
    cv_params["model_name"] = config["model"]["name"]
    cv_params["device"] = config["model"]["device"]
    cv_params["batch_size"] = config["training"].get("batch_size", DEFAULT_BATCH_SIZE)
    cv_classifier = BinaryClassifier(cv_params)

    # Use full BinaryClassifier for CV (no StandardScaler - matches Novo methodology)

    # Accuracy
    scores = cross_val_score(cv_classifier, X, y, cv=cv, scoring="accuracy")
    cv_results["cv_accuracy"] = {"mean": scores.mean(), "std": scores.std()}

    # F1 score
    scores = cross_val_score(cv_classifier, X, y, cv=cv, scoring="f1")
    cv_results["cv_f1"] = {"mean": scores.mean(), "std": scores.std()}

    # ROC AUC
    scores = cross_val_score(cv_classifier, X, y, cv=cv, scoring="roc_auc")
    cv_results["cv_roc_auc"] = {"mean": scores.mean(), "std": scores.std()}

    # Log results
    logger.info("Cross-validation Results:")
    for metric, values in cv_results.items():
        logger.info(f"  {metric}: {values['mean']:.4f} (+/- {values['std'] * 2:.4f})")

    return cv_results


def save_model(
    classifier: BinaryClassifier, config: dict[str, Any], logger: logging.Logger
) -> dict[str, str]:
    """
    Save trained model in dual format (pickle + NPZ+JSON)

    Args:
        classifier: Trained classifier
        config: Configuration dictionary
        logger: Logger instance

    Returns:
        Dictionary with paths to saved files:
        {
            "pickle": "models/model.pkl",
            "npz": "models/model.npz",
            "config": "models/model_config.json"
        }
        Empty dict if saving is disabled.
    """
    if not config["training"]["save_model"]:
        return {}

    model_name = config["training"]["model_name"]
    model_save_dir = config["training"]["model_save_dir"]
    os.makedirs(model_save_dir, exist_ok=True)

    base_path = os.path.join(model_save_dir, model_name)

    # Format 1: Pickle checkpoint (research/debugging)
    pickle_path = f"{base_path}.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(classifier, f)
    logger.info(f"Saved pickle checkpoint: {pickle_path}")

    # Format 2: NPZ (production arrays - all sklearn LogisticRegression fitted attributes)
    npz_path = f"{base_path}.npz"
    np.savez(
        npz_path,
        coef=classifier.classifier.coef_,
        intercept=classifier.classifier.intercept_,
        classes=classifier.classifier.classes_,
        n_features_in=np.array([classifier.classifier.n_features_in_]),
        n_iter=classifier.classifier.n_iter_,
    )
    logger.info(f"Saved NPZ arrays: {npz_path}")

    # Format 3: JSON (production metadata - all BinaryClassifier params)
    json_path = f"{base_path}_config.json"
    metadata = {
        # Model architecture
        "model_type": "LogisticRegression",
        "sklearn_version": sklearn.__version__,
        # LogisticRegression hyperparameters
        "C": classifier.C,
        "penalty": classifier.penalty,
        "solver": classifier.solver,
        "class_weight": classifier.class_weight,  # JSON handles None, str, dict natively
        "max_iter": classifier.max_iter,
        "random_state": classifier.random_state,
        # ESM embedding extractor params
        "esm_model": classifier.model_name,
        "esm_revision": classifier.revision,
        "batch_size": classifier.batch_size,
        "device": classifier.device,
    }

    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved JSON config: {json_path}")

    logger.info("Model saved successfully (dual-format: pickle + NPZ+JSON)")
    return {"pickle": pickle_path, "npz": npz_path, "config": json_path}


def load_model_from_npz(npz_path: str, json_path: str) -> BinaryClassifier:
    """
    Load model from NPZ+JSON format (production deployment)

    Args:
        npz_path: Path to .npz file with arrays
        json_path: Path to .json file with metadata

    Returns:
        Reconstructed BinaryClassifier instance

    Notes:
        This function enables production deployment without pickle files.
        It reconstructs a fully functional BinaryClassifier from NPZ+JSON format.
    """
    # Load arrays
    arrays = np.load(npz_path)
    coef = arrays["coef"]
    intercept = arrays["intercept"]
    classes = arrays["classes"]
    n_features_in = int(arrays["n_features_in"][0])
    n_iter = arrays["n_iter"]

    # Load metadata
    with open(json_path) as f:
        metadata = json.load(f)

    # Handle class_weight: JSON converts int keys to strings, convert back
    class_weight = metadata["class_weight"]
    if isinstance(class_weight, dict):
        # Convert string keys back to int keys (JSON forces keys to strings)
        class_weight = {int(k): v for k, v in class_weight.items()}

    # Reconstruct BinaryClassifier with ALL required params
    params = {
        # ESM params
        "model_name": metadata["esm_model"],
        "device": metadata.get("device", "cpu"),  # Use saved device or default to CPU
        "batch_size": metadata["batch_size"],
        "revision": metadata["esm_revision"],
        # LogisticRegression hyperparameters
        "C": metadata["C"],
        "penalty": metadata["penalty"],
        "solver": metadata["solver"],
        "max_iter": metadata["max_iter"],
        "random_state": metadata["random_state"],
        "class_weight": class_weight,  # Restored with int keys (if dict)
    }

    # Create classifier (initializes with unfitted LogisticRegression)
    classifier = BinaryClassifier(params)

    # Restore fitted LogisticRegression state
    classifier.classifier.coef_ = coef
    classifier.classifier.intercept_ = intercept
    classifier.classifier.classes_ = classes
    classifier.classifier.n_features_in_ = n_features_in
    classifier.classifier.n_iter_ = n_iter
    classifier.is_fitted = True

    return classifier


def train_pipeline(cfg: DictConfig) -> dict[str, Any]:
    """
    Core training pipeline - accepts Hydra DictConfig

    This is the main entry point for Hydra-based training. It accepts a composed
    DictConfig from Hydra and executes the full training pipeline.

    Args:
        cfg: Hydra DictConfig (composed from YAML + overrides)

    Returns:
        Dictionary containing training results and metrics:
        {
            "train_metrics": {...},
            "cv_metrics": {...},
            "config": {...},
            "model_paths": {...}
        }

    Raises:
        Exception: If training fails

    Examples:
        >>> with initialize(config_path="conf"):
        ...     cfg = compose(config_name="config")
        ...     results = train_pipeline(cfg)
    """
    # Resolve all interpolations (e.g., ${hardware.device})
    OmegaConf.resolve(cfg)

    # Convert to dict for legacy code compatibility
    # Cast to dict[str, Any] since OmegaConf.to_container can return various types
    config: dict[str, Any] = cast(
        dict[str, Any], OmegaConf.to_container(cfg, resolve=True)
    )

    # Validate config structure
    validate_config(config)

    # Setup logging (Hydra-aware)
    logger = setup_logging(cfg)
    logger.info("Starting antibody classification training (Hydra pipeline)")
    logger.info(f"Experiment: {config['experiment']['name']}")

    try:
        # Load data
        X_train, y_train = load_data(config)

        logger.info(f"Loaded {len(X_train)} training samples")

        # Initialize embedding extractor and classifier
        logger.info("Initializing ESM embedding extractor and classifier")
        classifier_params = config["classifier"].copy()
        classifier_params["model_name"] = config["model"]["name"]
        classifier_params["device"] = config["model"]["device"]
        classifier_params["batch_size"] = config["training"].get(
            "batch_size", DEFAULT_BATCH_SIZE
        )
        classifier = BinaryClassifier(classifier_params)

        # Get or create embeddings
        cache_dir = config["data"]["embeddings_cache_dir"]

        X_train_embedded = get_or_create_embeddings(
            X_train, classifier.embedding_extractor, cache_dir, "train", logger
        )

        # Convert labels to numpy array
        y_train_array: np.ndarray = np.array(y_train)

        # Perform cross-validation on full training data
        logger.info("Performing cross-validation on training data...")
        cv_results = perform_cross_validation(
            X_train_embedded, y_train_array, config, logger
        )

        # Train final model on full training set
        logger.info("Training final model on full training set...")
        classifier.fit(X_train_embedded, y_train_array)
        logger.info("Training completed")

        # Evaluate final model on training set
        metrics = config["training"]["metrics"]
        train_results = evaluate_model(
            classifier, X_train_embedded, y_train_array, "Training", metrics, logger
        )

        # Save model
        model_paths = save_model(classifier, config, logger)

        # Compile results
        results = {
            "train_metrics": train_results,
            "cv_metrics": cv_results,
            "config": config,
            "model_paths": model_paths,
        }

        logger.info("Training pipeline completed successfully")

        # Cache preserved for reuse in hyperparameter sweeps
        logger.info(
            f"Embedding cache preserved at {cache_dir} for future training runs"
        )

        return results

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise


def train_model(config_path: str = "configs/config.yaml") -> dict[str, Any]:
    """
    Legacy training function (DEPRECATED)

    DEPRECATED: Use train_pipeline(cfg) with Hydra instead.
    This function will be removed in v0.4.0.

    Args:
        config_path: Path to configuration YAML file

    Returns:
        Dictionary containing training results and metrics

    Raises:
        Exception: If training fails
    """
    # Emit deprecation warning
    warnings.warn(
        "train_model(config_path) is deprecated and will be removed in v0.4.0. "
        "Use train_pipeline(cfg) with Hydra instead. "
        "See docs for migration guide: https://docs.hydra.cc",
        DeprecationWarning,
        stacklevel=2,
    )

    # Load configuration
    config = load_config(config_path)

    # Convert to DictConfig and delegate to train_pipeline
    cfg = OmegaConf.create(config)

    return train_pipeline(cfg)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point for CLI - DO NOT call directly in tests

    This is the CLI entry point decorated with @hydra.main. It:
    - Automatically parses command-line overrides
    - Creates Hydra output directories
    - Saves composed config to .hydra/config.yaml
    - Delegates to train_pipeline() for core logic

    Usage:
        # Default config
        python -m antibody_training_esm.core.trainer

        # With overrides
        python -m antibody_training_esm.core.trainer model.batch_size=16

        # Multi-run sweep
        python -m antibody_training_esm.core.trainer --multirun model=esm1v,esm2

    Note:
        Tests should call train_pipeline() directly, not this function.
        This function is only for CLI usage with sys.argv parsing.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting training with Hydra (experiment: {cfg.experiment.name})")

    try:
        # Call core training pipeline
        results = train_pipeline(cfg)

        # Log final results
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Train Accuracy: {results['train_metrics']['accuracy']:.4f}")
        logger.info(
            f"CV Accuracy: {results['cv_metrics']['cv_accuracy']['mean']:.4f} "
            f"(+/- {results['cv_metrics']['cv_accuracy']['std'] * 2:.4f})"
        )

        if results.get("model_paths"):
            logger.info(f"Model saved to: {results['model_paths']['pickle']}")

        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Use Hydra main entry point (parses sys.argv automatically)
    main()
