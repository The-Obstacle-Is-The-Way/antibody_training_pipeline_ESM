"""
Training Script for Antibody Classification Pipeline
Trains a binary classifier on ESM-1V embeddings of antibody sequences.
"""

import logging
import os
import pickle
import shutil
from typing import Any

import numpy as np
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from classifier import BinaryClassifier
from data import load_data
from model import ESMEmbeddingExtractor


def setup_logging(config: dict) -> logging.Logger:
    """Setup logging configuration"""
    log_level = getattr(logging, config["training"]["log_level"].upper())
    log_file = config["training"]["log_file"]

    # Create log directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path) as f:
        config: dict = yaml.safe_load(f)
    return config


def get_or_create_embeddings(
    sequences: list[str],
    embedding_extractor: ESMEmbeddingExtractor,
    cache_path: str,
    dataset_name: str,
    logger: logging.Logger,
) -> np.ndarray:
    """Get embeddings from cache or create them"""
    import hashlib

    # Create a hash of the sequences to ensure cache validity
    # Now that we have fixed data splits, this ensures we cache the right sequences
    sequences_str = "|".join(sequences)  # Keep original order since split is now fixed
    sequences_hash = hashlib.md5(sequences_str.encode()).hexdigest()[:8]
    cache_file = os.path.join(
        cache_path, f"{dataset_name}_{sequences_hash}_embeddings.pkl"
    )

    if os.path.exists(cache_file):
        logger.info(f"Loading cached embeddings from {cache_file}")
        with open(cache_file, "rb") as f:
            cached_data: dict = pickle.load(f)

        # Verify the cached sequences match exactly
        if (
            len(cached_data["embeddings"]) == len(sequences)
            and cached_data["sequences_hash"] == sequences_hash
        ):
            logger.info(
                f"Using cached embeddings for {len(sequences)} sequences (hash: {sequences_hash})"
            )
            embeddings_result: np.ndarray = cached_data["embeddings"]
            return embeddings_result
        else:
            logger.warning("Cached embeddings hash mismatch, recomputing...")

    logger.info(f"Computing embeddings for {len(sequences)} sequences...")
    embeddings = embedding_extractor.extract_batch_embeddings(sequences)

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
    """Evaluate model performance"""
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
    config: dict,
    logger: logging.Logger,
) -> dict[str, dict[str, float]]:
    """Perform cross-validation"""
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
    cv_params["batch_size"] = config["training"].get("batch_size", 32)
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
    classifier: BinaryClassifier, config: dict, logger: logging.Logger
) -> str | None:
    """Save trained model

    Returns:
        Path to the saved model
    """
    if not config["training"]["save_model"]:
        return None

    # Build model path from model_name and model_save_dir
    model_name = config["training"]["model_name"]
    model_save_dir = config["training"]["model_save_dir"]
    model_path = os.path.join(model_save_dir, f"{model_name}.pkl")

    os.makedirs(model_save_dir, exist_ok=True)

    logger.info(f"Saving model to {model_path}")

    # Save the entire classifier (including scaler and fitted model)
    with open(model_path, "wb") as f:
        pickle.dump(classifier, f)

    logger.info("Model saved successfully")
    return model_path


def train_model(config_path: str = "configs/config.yaml") -> dict[str, Any]:
    """
    Main training function

    Args:
        config_path: Path to configuration YAML file

    Returns:
        Dictionary containing training results and metrics
    """
    # Load configuration
    config = load_config(config_path)

    # Setup logging
    logger = setup_logging(config)
    logger.info("Starting antibody classification training")
    logger.info(f"Configuration loaded from {config_path}")

    try:
        # Load data
        X_train, y_train = load_data(config)

        logger.info(f"Loaded {len(X_train)} training samples")

        # Initialize embedding extractor and classifier
        logger.info("Initializing ESM embedding extractor and classifier")
        classifier_params = config["classifier"].copy()
        classifier_params["model_name"] = config["model"][
            "name"
        ]  # Map 'name' to 'model_name'
        classifier_params["device"] = config["model"]["device"]
        classifier_params["batch_size"] = config["training"].get(
            "batch_size", 32
        )  # Get batch_size from training config
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
        model_path = save_model(classifier, config, logger)

        # Delete cached embeddings
        shutil.rmtree(cache_dir)

        # Compile results
        results = {
            "train_metrics": train_results,
            "cv_metrics": cv_results,
            "config": config,
            "model_path": model_path,
        }

        logger.info("Training pipeline completed successfully")
        return results

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"
    results = train_model(config_path)

    print("Training completed successfully!")
