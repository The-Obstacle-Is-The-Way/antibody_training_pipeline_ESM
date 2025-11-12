"""
Directory utilities for organizing model artifacts hierarchically

Implements standardized directory structure:
    models/{model_shortname}/{classifier_type}/{model_files}
    test_results/{model_shortname}/{classifier_type}/{dataset}/{results}
"""

import re
from pathlib import Path
from typing import Any


def extract_model_shortname(model_name: str) -> str:
    """
    Extract short model identifier from HuggingFace model name

    Examples:
        facebook/esm1v_t33_650M_UR90S_1 -> esm1v
        facebook/esm2_t33_650M_UR50D -> esm2_650m
        alchemab/antiberta2 -> antiberta

    Args:
        model_name: Full HuggingFace model name

    Returns:
        Short model identifier
    """
    # Handle facebook/esm models
    if "esm1v" in model_name.lower():
        return "esm1v"
    elif "esm2" in model_name.lower():
        # Extract size info (e.g., 650M)
        match = re.search(r"esm2.*?(\d+M)", model_name, re.IGNORECASE)
        if match:
            size = match.group(1).lower()
            return f"esm2_{size}"
        return "esm2"
    elif "antiberta" in model_name.lower():
        return "antiberta"
    elif "protbert" in model_name.lower():
        return "protbert"
    elif "ablang" in model_name.lower():
        return "ablang"
    else:
        # Fallback: use the last part of the model path
        return model_name.split("/")[-1].lower()


def extract_classifier_shortname(classifier_config: dict[str, Any]) -> str:
    """
    Extract short classifier identifier from classifier config

    Examples:
        {"type": "logistic_regression", ...} -> logreg
        {"type": "xgboost", ...} -> xgboost
        {"type": "mlp", ...} -> mlp

    Args:
        classifier_config: Classifier configuration dictionary

    Returns:
        Short classifier identifier
    """
    classifier_type = classifier_config.get("type", "unknown")

    # Map full names to short names
    shortname_map = {
        "logistic_regression": "logreg",
        "xgboost": "xgboost",
        "mlp": "mlp",
        "svm": "svm",
        "random_forest": "rf",
    }

    return shortname_map.get(classifier_type, classifier_type)


def get_hierarchical_model_dir(
    base_dir: str,
    model_name: str,
    classifier_config: dict[str, Any],
) -> Path:
    """
    Generate hierarchical model directory path

    Structure: {base_dir}/{model_shortname}/{classifier_type}/

    Args:
        base_dir: Base models directory (e.g., "./models")
        model_name: Full HuggingFace model name
        classifier_config: Classifier configuration dictionary

    Returns:
        Path to hierarchical model directory

    Examples:
        >>> get_hierarchical_model_dir(
        ...     "./models",
        ...     "facebook/esm1v_t33_650M_UR90S_1",
        ...     {"type": "logistic_regression"}
        ... )
        PosixPath('models/esm1v/logreg')
    """
    model_short = extract_model_shortname(model_name)
    classifier_short = extract_classifier_shortname(classifier_config)

    return Path(base_dir) / model_short / classifier_short


def get_hierarchical_test_results_dir(
    base_dir: str,
    model_name: str,
    classifier_config: dict[str, Any],
    dataset_name: str,
) -> Path:
    """
    Generate hierarchical test results directory path

    Structure: {base_dir}/{model_shortname}/{classifier_type}/{dataset}/

    Args:
        base_dir: Base test results directory (e.g., "./test_results")
        model_name: Full HuggingFace model name
        classifier_config: Classifier configuration dictionary
        dataset_name: Dataset name (e.g., "jain", "harvey")

    Returns:
        Path to hierarchical test results directory

    Examples:
        >>> get_hierarchical_test_results_dir(
        ...     "./test_results",
        ...     "facebook/esm1v_t33_650M_UR90S_1",
        ...     {"type": "logistic_regression"},
        ...     "jain"
        ... )
        PosixPath('test_results/esm1v/logreg/jain')
    """
    model_short = extract_model_shortname(model_name)
    classifier_short = extract_classifier_shortname(classifier_config)

    return Path(base_dir) / model_short / classifier_short / dataset_name
