"""Plotting and result serialization utilities."""

import logging
import os
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

# Configure matplotlib
plt.style.use("seaborn-v0_8" if "seaborn-v0_8" in plt.style.available else "default")
sns.set_palette("husl")

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    results: dict[str, dict[str, Any]],
    dataset_name: str,
    output_dir: str,
) -> None:
    """
    Create confusion matrix visualization (individual files per model).

    Args:
        results: Dictionary mapping model names to result dictionaries.
        dataset_name: Name of the dataset.
        output_dir: Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Creating confusion matrices for {dataset_name} in {output_dir}")

    # Create individual confusion matrix for each model to prevent overrides
    for model_name, model_results in results.items():
        if "confusion_matrix" not in model_results:
            logger.warning(f"No confusion matrix found for {model_name}, skipping plot")
            continue

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        cm = model_results["confusion_matrix"]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
            ax=ax,
        )
        ax.set_title(f"Confusion Matrix - {model_name} on {dataset_name}")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")

        plt.tight_layout()

        # Save plot with model name to prevent overrides when testing multiple backbones
        plot_file = os.path.join(
            output_dir,
            f"confusion_matrix_{model_name}_{dataset_name}.png",
        )
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Confusion matrix saved to {plot_file}")


def save_detailed_results(
    results: dict[str, dict[str, Any]],
    dataset_name: str,
    config_dict: dict[str, Any],
    output_dir: str,
    save_predictions: bool = True,
) -> None:
    """
    Save detailed results to files (individual files per model).

    Args:
        results: Dictionary mapping model names to result dictionaries.
        dataset_name: Name of the dataset.
        config_dict: Configuration dictionary to embed in YAML.
        output_dir: Directory to save results.
        save_predictions: Whether to save prediction CSVs.
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save individual YAML for each model to prevent overrides
    for model_name, model_results in results.items():
        results_file = os.path.join(
            output_dir,
            f"detailed_results_{model_name}_{dataset_name}_{timestamp}.yaml",
        )
        with open(results_file, "w") as f:
            yaml.dump(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "config": config_dict,
                    "results": model_results,
                },
                f,
                default_flow_style=False,
            )
        logger.info(f"Detailed results saved to {results_file}")

    # Save predictions if requested
    if save_predictions:
        for model_name, model_results in results.items():
            if "predictions" in model_results:
                pred_file = os.path.join(
                    output_dir,
                    f"predictions_{model_name}_{dataset_name}_{timestamp}.csv",
                )
                pred_df = pd.DataFrame(
                    {
                        "y_true": model_results["predictions"]["y_true"],
                        "y_pred": model_results["predictions"]["y_pred"],
                        "y_proba": model_results["predictions"]["y_proba"],
                    }
                )
                pred_df.to_csv(pred_file, index=False)
                logger.info(f"Predictions saved to {pred_file}")
