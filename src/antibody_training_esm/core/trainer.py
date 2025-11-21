"""
Training Module

Professional training pipeline for antibody classification models.
Includes cross-validation, embedding caching, and comprehensive evaluation.
"""

import logging
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from antibody_training_esm.core.classifier import BinaryClassifier
from antibody_training_esm.core.config import DEFAULT_BATCH_SIZE

# Import and re-export from submodules
from antibody_training_esm.core.training.cache import (
    get_or_create_embeddings,
    validate_embeddings,
)
from antibody_training_esm.core.training.metrics import (
    evaluate_model,
    perform_cross_validation,
    save_cv_results,
)
from antibody_training_esm.core.training.serialization import (
    load_config,
    load_model_from_npz,
    save_model,
)
from antibody_training_esm.data.loaders import load_data

__all__ = [
    "validate_config",
    "setup_logging",
    "load_config",
    "validate_embeddings",
    "get_or_create_embeddings",
    "evaluate_model",
    "perform_cross_validation",
    "save_cv_results",
    "save_model",
    "load_model_from_npz",
    "train_pipeline",
    "main",
]


def validate_config(config: dict[str, Any] | DictConfig) -> None:
    """
    Validate config structure and semantics (Hydra-aware).

    Performs two levels of validation:
    1. Schema validation: Required keys exist (structural)
    2. Semantic validation: Files exist, metrics/devices valid (semantic)

    Args:
        config: Configuration dictionary or DictConfig to validate

    Raises:
        ValueError: If any required keys are missing or invalid
        FileNotFoundError: If required files don't exist
    """
    # Convert DictConfig to dict for uniform access
    if isinstance(config, DictConfig):
        config_dict = cast(dict[str, Any], OmegaConf.to_container(config, resolve=True))
    else:
        config_dict = config

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
        if section not in config_dict:
            missing_sections.append(section)
            continue

        # Check keys within each section
        if not isinstance(config_dict[section], dict):
            raise ValueError(
                f"Config section '{section}' must be a dictionary, "
                f"got {type(config_dict[section]).__name__}"
            )

        for key in required_keys[section]:
            if key not in config_dict[section]:
                missing_keys.append(f"{section}.{key}")

    # Construct helpful error message for structural errors
    if missing_sections or missing_keys:
        error_parts = []
        if missing_sections:
            error_parts.append(
                f"Missing config sections: {', '.join(missing_sections)}"
            )
        if missing_keys:
            error_parts.append(f"Missing config keys: {', '.join(missing_keys)}")
        raise ValueError("Config validation failed:\n  - " + "\n  - ".join(error_parts))

    # Semantic validation (beyond structure)
    errors = []

    # Validate files exist
    train_file = Path(config_dict["data"]["train_file"])
    if not train_file.exists():
        errors.append(f"Training file not found: {train_file}")

    test_file = Path(config_dict["data"]["test_file"])
    if not test_file.exists():
        errors.append(f"Test file not found: {test_file}")

    # Validate metrics are valid
    VALID_METRICS = {"accuracy", "precision", "recall", "f1", "roc_auc"}
    metrics = set(config_dict["training"]["metrics"])
    invalid_metrics = metrics - VALID_METRICS
    if invalid_metrics:
        errors.append(
            f"Invalid metrics: {invalid_metrics}. Valid metrics: {VALID_METRICS}"
        )

    # Validate device is valid
    VALID_DEVICES = {"cpu", "cuda", "mps", "auto"}
    device = config_dict["model"]["device"]
    if device not in VALID_DEVICES:
        errors.append(f"Invalid device: {device}. Valid devices: {VALID_DEVICES}")

    # Raise if any semantic validation failed
    if errors:
        raise ValueError(
            "Config semantic validation failed:\n  - " + "\n  - ".join(errors)
        )


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
        # Create log directory if it doesn't exist (even in Hydra mode)
        log_file.parent.mkdir(parents=True, exist_ok=True)
    except (ValueError, AttributeError, OSError) as e:
        logging.getLogger(__name__).warning(
            "Hydra output dir not available, falling back to config log path: %s", e
        )
        log_file = Path(log_file_str)
        if not log_file.is_absolute():
            log_file = Path.cwd() / log_file_str
        log_file.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        logging.getLogger(__name__).exception(
            "Unexpected error determining log file path"
        )
        raise

    # Configure logging
    # force=True prevents duplicate log lines when Hydra has already configured logging
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,  # Python 3.8+ - replaces existing handlers
    )

    return logging.getLogger(__name__)


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

    # Validate config structure (accepts DictConfig)
    validate_config(cfg)

    # Setup logging (Hydra-aware, accepts DictConfig)
    logger = setup_logging(cfg)

    # Convert to dict for legacy code that requires dict access
    # Keep DictConfig as long as possible to preserve type safety and validation
    config: dict[str, Any] = cast(
        dict[str, Any], OmegaConf.to_container(cfg, resolve=True)
    )
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

        # Save CV results to file (with Hydra/legacy mode fallback)
        try:
            # Try Hydra output directory first (production mode)
            from hydra.core.hydra_config import HydraConfig

            hydra_cfg = HydraConfig.get()
            cv_output_dir = Path(hydra_cfg.runtime.output_dir)
            experiment_name = cfg.experiment.name
            logger.info(f"Saving CV results to Hydra output dir: {cv_output_dir}")
        except (ImportError, AttributeError, OSError, ValueError) as e:
            model_save_dir = config.get("training", {}).get(
                "model_save_dir", "./outputs"
            )
            cv_output_dir = Path(model_save_dir)
            experiment_name = config.get("experiment", {}).get("name", "training")
            logger.info(
                "Running without Hydra, saving CV results to %s (reason: %s)",
                cv_output_dir,
                e,
            )
        except Exception:
            logger.exception("Unexpected error determining CV output directory")
            raise

        # Save CV results (both Hydra and legacy modes)
        save_cv_results(cv_results, cv_output_dir, experiment_name, logger)

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
