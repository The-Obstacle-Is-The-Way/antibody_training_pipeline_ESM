"""
Training Module

Professional training pipeline for antibody classification models.
Includes cross-validation, embedding caching, and comprehensive evaluation.
"""

import logging
from pathlib import Path
from typing import Any, Literal, cast

import hydra
import numpy as np
from omegaconf import DictConfig

from antibody_training_esm.core.classifier import BinaryClassifier
from antibody_training_esm.core.config import LOG_SEPARATOR_WIDTH
from antibody_training_esm.core.device import resolve_device

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


from antibody_training_esm.models.config import TrainingPipelineConfig


def validate_config(config: dict[str, Any] | DictConfig) -> TrainingPipelineConfig:
    """
    Validate config with Pydantic models.

    Args:
        config: Raw dict or Hydra DictConfig

    Returns:
        Validated TrainingPipelineConfig

    Raises:
        ValidationError: If config is invalid
    """
    if isinstance(config, DictConfig):
        return TrainingPipelineConfig.from_hydra(config)
    result: TrainingPipelineConfig = TrainingPipelineConfig.model_validate(config)
    return result


def setup_logging(config: TrainingPipelineConfig) -> logging.Logger:
    """
    Setup logging from Pydantic config.

    Args:
        config: Validated TrainingPipelineConfig

    Returns:
        Configured logger
    """
    from hydra.core.hydra_config import HydraConfig

    log_level = getattr(logging, config.training.log_level.upper())
    log_file = config.training.log_file

    # Hydra-aware path resolution (same as before)
    try:
        hydra_cfg = HydraConfig.get()
        output_dir = Path(hydra_cfg.runtime.output_dir)
        log_path = output_dir / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except (ValueError, AttributeError):
        log_path = Path(log_file)
        if not log_path.is_absolute():
            log_path = Path.cwd() / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
        force=True,
    )

    return logging.getLogger(__name__)


def train_pipeline(cfg: DictConfig) -> dict[str, Any]:
    """Core training pipeline with Pydantic validation."""
    # Validate config (now returns Pydantic model)
    config = validate_config(cfg)

    # Setup logging (accepts Pydantic model now)
    logger = setup_logging(config)

    logger.info("Starting antibody classification training")
    logger.info(f"Experiment: {config.experiment.name}")

    try:
        X_train, y_train = load_data(config)

        logger.info(f"Loaded {len(X_train)} training samples")

        # Phase B (Biophysical) filtering: remove sequences with ambiguous AAs ('X')
        # and stop codons ('*'). Biopython cannot handle these, unlike ESM.
        # Filter X and y together to keep labels aligned.
        if config.model.model_type == "biophysical":
            valid_indices = [
                i for i, seq in enumerate(X_train) if "X" not in seq and "*" not in seq
            ]
            dropped_count = len(X_train) - len(valid_indices)
            if dropped_count > 0:
                logger.warning(
                    f"Biophysical model requires strict amino acids. "
                    f"Dropping {dropped_count} sequences containing 'X' or '*'."
                )
                X_train = [X_train[i] for i in valid_indices]
                y_train = [y_train[i] for i in valid_indices]
                logger.info(f"Remaining samples after filtering: {len(X_train)}")

        # Resolve device (handles auto + explicit availability validation)
        device = resolve_device(config.model.device)
        config.model.device = cast(
            Literal["cpu", "cuda", "mps", "auto"], device
        )  # Persist resolved device
        if config.hardware and isinstance(config.hardware, dict):
            # Keep hardware section in sync when present
            config.hardware["device"] = device
        logger.info(f"Using device: {device}")

        # Initialize classifier
        classifier_params = {
            "model_name": config.model.name,
            "device": device,  # Use resolved device
            "batch_size": config.model.batch_size,
            "revision": config.model.revision,
            "model_type": config.model.model_type,  # ESM or AMPLIFY
            # Classifier strategy params
            "strategy": config.classifier.strategy,
            "C": config.classifier.C,
            "penalty": config.classifier.penalty,
            "solver": config.classifier.solver,
            "class_weight": config.classifier.class_weight,
            "max_iter": config.classifier.max_iter,
            "random_state": config.classifier.random_state,
            "n_estimators": config.classifier.n_estimators,
            "max_depth": config.classifier.max_depth,
            "learning_rate": config.classifier.learning_rate,
        }

        classifier = BinaryClassifier(classifier_params)

        # Get embeddings (cache_dir from config)
        cache_dir = config.data.embeddings_cache_dir
        X_train_embedded = get_or_create_embeddings(
            X_train, classifier.embedding_extractor, cache_dir, "train", logger
        )

        # Convert labels to numpy array
        y_train_array: np.ndarray = np.array(y_train)

        # Perform CV (returns CVResults Pydantic model)
        cv_results = perform_cross_validation(
            X_train_embedded,
            y_train_array,
            config,  # Passing Pydantic model
            logger,
        )

        # Save CV results
        try:
            from hydra.core.hydra_config import HydraConfig

            hydra_cfg = HydraConfig.get()
            cv_output_dir = Path(hydra_cfg.runtime.output_dir)
            experiment_name = config.experiment.name
            logger.info(f"Saving CV results to Hydra output dir: {cv_output_dir}")
        except (ValueError, AttributeError, ImportError):
            cv_output_dir = config.training.model_save_dir
            experiment_name = config.experiment.name
            logger.info(f"Running without Hydra, saving CV results to {cv_output_dir}")

        save_cv_results(cv_results, cv_output_dir, experiment_name, logger)

        # Train final model
        classifier.fit(X_train_embedded, y_train_array)

        # Evaluate (returns EvaluationMetrics Pydantic model)
        train_results = evaluate_model(
            classifier,
            X_train_embedded,
            y_train_array,
            "Training",
            list(config.training.metrics),  # Cast to list for type safety
            logger,
        )

        # Save model
        if config.training.save_model:
            # save_model expects config dict or object.
            # We'll pass Pydantic config.
            # Attach metrics to config for metadata saving
            config.train_metrics = train_results.model_dump(
                mode="json", exclude_none=True
            )
            model_paths = save_model(classifier, config, logger)
        else:
            model_paths = {}

        return {
            "train_metrics": train_results,
            "cv_metrics": cv_results,
            "config": config.model_dump(),  # Convert back to dict for serialization
            "model_paths": model_paths,
        }

    except Exception as e:
        logger.error(f"Training failed: {e}")
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

        # Log final results (access Pydantic fields)
        train_metrics = results["train_metrics"]
        cv_metrics = results["cv_metrics"]

        logger.info("=" * LOG_SEPARATOR_WIDTH)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * LOG_SEPARATOR_WIDTH)
        logger.info(f"Train Accuracy: {train_metrics.accuracy:.4f}")
        logger.info(
            f"CV Accuracy: {cv_metrics.cv_accuracy['mean']:.4f} "
            f"(+/- {cv_metrics.cv_accuracy['std'] * 2:.4f})"
        )

        if results.get("model_paths"):
            logger.info(f"Model saved to: {results['model_paths']['pickle']}")

        logger.info("=" * LOG_SEPARATOR_WIDTH)

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Use Hydra main entry point (parses sys.argv automatically)
    main()
