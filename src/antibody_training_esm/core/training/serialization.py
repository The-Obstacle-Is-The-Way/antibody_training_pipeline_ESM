"""
Model serialization utilities.

Handles saving/loading models in dual format (pickle for dev, NPZ+JSON for production).
Manages configuration loading and directory structure.
"""

import json
import logging
import pickle  # nosec B403
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import sklearn
import yaml

from antibody_training_esm.core.classifier import BinaryClassifier
from antibody_training_esm.core.directory_utils import get_hierarchical_model_dir

if TYPE_CHECKING:
    from antibody_training_esm.models.config import TrainingPipelineConfig


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


def save_model(
    classifier: BinaryClassifier,
    config: "TrainingPipelineConfig | dict[str, Any]",
    logger: logging.Logger,
) -> dict[str, str]:
    """
    Save trained model in dual format (pickle + NPZ+JSON)

    Models are saved in hierarchical directory structure:
        {model_save_dir}/{model_shortname}/{classifier_type}/{model_name}.*

    Example:
        experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl

    Args:
        classifier: Trained classifier
        config: Configuration dictionary or Pydantic model
        logger: Logger instance

    Returns:
        Dictionary with paths to saved files:
        {
            "pickle": "experiments/checkpoints/esm1v/logreg/model.pkl",
            "npz": "experiments/checkpoints/esm1v/logreg/model.npz",
            "config": "experiments/checkpoints/esm1v/logreg/model_config.json"
        }
        Empty dict if saving is disabled.
    """
    from antibody_training_esm.models.config import TrainingPipelineConfig

    if isinstance(config, TrainingPipelineConfig):
        # Pydantic path
        if not config.training.save_model:
            return {}
        model_name = config.training.model_name
        base_save_dir = config.training.model_save_dir
        model_shortname = config.model.name
        classifier_config = config.classifier.model_dump()
    else:
        # Legacy dict path
        if not config["training"]["save_model"]:
            return {}
        model_name = config["training"]["model_name"]
        base_save_dir = config["training"]["model_save_dir"]
        model_shortname = config["model"]["name"]
        classifier_config = config["classifier"]

    # Generate hierarchical directory path
    hierarchical_dir = get_hierarchical_model_dir(
        str(base_save_dir),
        model_shortname,
        classifier_config,
    )
    hierarchical_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using hierarchical model directory: {hierarchical_dir}")

    base_path = hierarchical_dir / model_name

    # Format 1: Pickle checkpoint (research/debugging)
    pickle_path = f"{base_path}.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(classifier, f)
    logger.info(f"Saved pickle checkpoint: {pickle_path}")

    # Format 2: Strategy-specific production serialization
    # Use duck typing to detect serialization method
    saved_paths = {"pickle": str(pickle_path)}

    if hasattr(classifier.classifier, "save_model"):
        # XGBoost native .xgb format (pickle-free)
        xgb_path = f"{base_path}.xgb"
        classifier.classifier.save_model(str(xgb_path))
        logger.info(f"Saved XGBoost native model: {xgb_path}")
        saved_paths["xgb"] = str(xgb_path)
    elif hasattr(classifier.classifier, "to_arrays"):
        # LogReg NPZ format (sklearn arrays)
        npz_path = f"{base_path}.npz"
        arrays = classifier.classifier.to_arrays()
        np.savez(npz_path, **cast(dict[str, Any], arrays))
        logger.info(f"Saved NPZ arrays: {npz_path}")
        saved_paths["npz"] = str(npz_path)
    else:
        # Fallback: legacy LogReg direct attribute access
        # Cast to Any because protocol doesn't enforce LogReg attributes
        inner_clf = cast(Any, classifier.classifier)
        npz_path = f"{base_path}.npz"
        np.savez(
            npz_path,
            coef=inner_clf.coef_,
            intercept=inner_clf.intercept_,
            classes=inner_clf.classes_,
            n_features_in=np.array([inner_clf.n_features_in_]),
            n_iter=inner_clf.n_iter_,
        )
        logger.info(f"Saved NPZ arrays (legacy): {npz_path}")
        saved_paths["npz"] = str(npz_path)

    # Format 3: JSON metadata (universal across all strategies)
    json_path = f"{base_path}_config.json"

    # Get strategy config via to_dict() method (all strategies implement this)
    strategy_config = classifier.classifier.to_dict()
    classifier_type = strategy_config.get("type", "logistic_regression")

    metadata = {
        # Model architecture
        "model_name": classifier.model_name,  # HuggingFace model ID
        "model_type": classifier_type,  # Dynamic: logistic_regression, xgboost, etc.
        "sklearn_version": sklearn.__version__,
        # Classifier configuration block (from strategy's to_dict())
        "classifier": strategy_config,
        # ESM embedding extractor params
        "esm_model": classifier.model_name,
        "esm_revision": classifier.revision,
        "batch_size": classifier.batch_size,
        "device": classifier.device,
    }

    # Legacy flat fields for backward compatibility (LogReg only)
    if classifier_type == "logistic_regression":
        metadata.update(
            {
                "C": classifier.C,
                "penalty": classifier.penalty,
                "solver": classifier.solver,
                "class_weight": classifier.class_weight,
                "max_iter": classifier.max_iter,
                "random_state": classifier.random_state,
            }
        )

    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved JSON config: {json_path}")

    saved_paths["config"] = str(json_path)
    logger.info(f"Model saved successfully ({classifier_type} format)")
    return saved_paths


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
    # Cast to Any because protocol doesn't enforce LogReg attributes
    inner_clf = cast(Any, classifier.classifier)
    inner_clf.classifier.coef_ = coef
    inner_clf.classifier.intercept_ = intercept
    inner_clf.classifier.classes_ = classes
    inner_clf.classifier.n_features_in_ = n_features_in
    inner_clf.classifier.n_iter_ = n_iter
    classifier.is_fitted = True

    return classifier
