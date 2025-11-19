"""
Model Serialization Manager

Handles saving and loading of trained models in dual format:
1. Pickle (development/debugging)
2. NPZ + JSON (production/deployment)
"""

import json
import logging
import pickle  # nosec B403 - Used only for local trusted data
from typing import Any, cast

import numpy as np
import sklearn  # type: ignore

from antibody_training_esm.core.classifier import BinaryClassifier
from antibody_training_esm.core.directory_utils import get_hierarchical_model_dir


class ModelSerializer:
    """Manages model saving and loading operations."""

    def __init__(self, logger: logging.Logger | None = None):
        """
        Initialize ModelSerializer.

        Args:
            logger: Logger instance (defaults to module logger)
        """
        self.logger = logger or logging.getLogger(__name__)

    def save_model(
        self,
        classifier: BinaryClassifier,
        config: dict[str, Any],
    ) -> dict[str, str]:
        """
        Save trained model in dual format (pickle + NPZ+JSON).

        Args:
            classifier: Trained classifier
            config: Configuration dictionary

        Returns:
            Dictionary with paths to saved files
        """
        if not config["training"]["save_model"]:
            return {}

        model_name = config["training"]["model_name"]
        base_save_dir = config["training"]["model_save_dir"]

        # Generate hierarchical directory path
        hierarchical_dir = get_hierarchical_model_dir(
            base_save_dir,
            config["model"]["name"],
            config["classifier"],
        )
        hierarchical_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Using hierarchical model directory: {hierarchical_dir}")

        base_path = hierarchical_dir / model_name

        # Format 1: Pickle checkpoint (research/debugging)
        pickle_path = f"{base_path}.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(classifier, f)
        self.logger.info(f"Saved pickle checkpoint: {pickle_path}")

        # Format 2: Strategy-specific production serialization
        saved_paths = {"pickle": str(pickle_path)}

        if hasattr(classifier.classifier, "save_model"):
            # XGBoost native .xgb format (pickle-free)
            xgb_path = f"{base_path}.xgb"
            classifier.classifier.save_model(str(xgb_path))
            self.logger.info(f"Saved XGBoost native model: {xgb_path}")
            saved_paths["xgb"] = str(xgb_path)
        elif hasattr(classifier.classifier, "to_arrays"):
            # LogReg NPZ format (sklearn arrays)
            npz_path = f"{base_path}.npz"
            arrays = classifier.classifier.to_arrays()
            np.savez(npz_path, **cast(dict[str, Any], arrays))
            self.logger.info(f"Saved NPZ arrays: {npz_path}")
            saved_paths["npz"] = str(npz_path)
        else:
            # Fallback: legacy LogReg direct attribute access
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
            self.logger.info(f"Saved NPZ arrays (legacy): {npz_path}")
            saved_paths["npz"] = str(npz_path)

        # Format 3: JSON metadata (universal across all strategies)
        json_path = f"{base_path}_config.json"

        # Get strategy config via to_dict() method
        strategy_config = classifier.classifier.to_dict()
        classifier_type = strategy_config.get("type", "logistic_regression")

        metadata = {
            # Model architecture
            "model_name": classifier.model_name,
            "model_type": classifier_type,
            "sklearn_version": sklearn.__version__,
            # Classifier configuration block
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
        self.logger.info(f"Saved JSON config: {json_path}")

        saved_paths["config"] = str(json_path)
        self.logger.info(f"Model saved successfully ({classifier_type} format)")
        return saved_paths

    def load_model_from_npz(self, npz_path: str, json_path: str) -> BinaryClassifier:
        """
        Load model from NPZ+JSON format (production deployment).

        Args:
            npz_path: Path to .npz file with arrays
            json_path: Path to .json file with metadata

        Returns:
            Reconstructed BinaryClassifier instance
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
            class_weight = {int(k): v for k, v in class_weight.items()}

        # Reconstruct BinaryClassifier with ALL required params
        params = {
            # ESM params
            "model_name": metadata["esm_model"],
            "device": metadata.get("device", "cpu"),
            "batch_size": metadata["batch_size"],
            "revision": metadata["esm_revision"],
            # LogisticRegression hyperparameters
            "C": metadata["C"],
            "penalty": metadata["penalty"],
            "solver": metadata["solver"],
            "max_iter": metadata["max_iter"],
            "random_state": metadata["random_state"],
            "class_weight": class_weight,
        }

        # Create classifier (initializes with unfitted LogisticRegression)
        classifier = BinaryClassifier(params)

        # Restore fitted LogisticRegression state
        inner_clf = cast(Any, classifier.classifier)
        inner_clf.classifier.coef_ = coef
        inner_clf.classifier.intercept_ = intercept
        inner_clf.classifier.classes_ = classes
        inner_clf.classifier.n_features_in_ = n_features_in
        inner_clf.classifier.n_iter_ = n_iter
        classifier.is_fitted = True

        return classifier
