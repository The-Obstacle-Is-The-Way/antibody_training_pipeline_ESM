#!/usr/bin/env python3
"""
Test CLI for Antibody Classification Pipeline

Professional command-line interface for testing trained antibody classifiers:
1. Load trained models from pickle files
2. Evaluate on test datasets with performance metrics
3. Generate confusion matrices and comprehensive logging

Usage:
    antibody-test --model models/antibody_classifier.pkl --data sample_data.csv
    antibody-test --config test_config.yaml
    antibody-test --model m1.pkl m2.pkl --data d1.csv d2.csv
"""

import argparse
import json
import logging
import os
import pickle  # nosec B403 - Used only for local trusted data (models, caches)
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml

# Scikit-learn imports
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Package imports (using professional package paths)
from antibody_training_esm.core.classifier import BinaryClassifier
from antibody_training_esm.core.config import DEFAULT_BATCH_SIZE
from antibody_training_esm.core.directory_utils import (
    extract_classifier_shortname,
    extract_model_shortname,
    get_hierarchical_test_results_dir,
)
from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor

# Configure matplotlib for better plots
plt.style.use("seaborn-v0_8" if "seaborn-v0_8" in plt.style.available else "default")
sns.set_palette("husl")


# ============================================================================
# NOTE: Hierarchical organization uses shared utilities from directory_utils
# See: src/antibody_training_esm/core/directory_utils.py
# ============================================================================


@dataclass
class TestConfig:
    """Configuration for testing pipeline"""

    model_paths: list[str]
    data_paths: list[str]
    sequence_column: str = "sequence"  # Column name for sequences in dataset
    label_column: str = "label"  # Column name for labels in dataset
    output_dir: str = "./test_results"
    metrics: list[str] | None = None
    save_predictions: bool = True
    batch_size: int = DEFAULT_BATCH_SIZE  # Batch size for embedding extraction
    device: str = "mps"  # Device to use for inference [cuda, cpu, mps] - MUST match training config

    def __post_init__(self) -> None:
        if self.metrics is None:
            self.metrics = [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
            ]


class ModelTester:
    """Model testing class"""

    def __init__(self, config: TestConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.results: dict[str, Any] = {}
        self.cached_embedding_files: list[str] = []  # Track cached files for cleanup

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        # Create output directory if it doesn't exist
        os.makedirs(self.config.output_dir, exist_ok=True)

        log_file = os.path.join(
            self.config.output_dir,
            f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        )

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

        return logging.getLogger(__name__)

    def load_model(self, model_path: str) -> BinaryClassifier:
        """Load trained model from pickle file"""
        self.logger.info(f"Loading model from {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with open(model_path, "rb") as f:
            model = pickle.load(f)  # nosec B301 - Loading our own trained model from local file

        if not isinstance(model, BinaryClassifier):
            raise ValueError(f"Expected BinaryClassifier, got {type(model)}")

        # Update device if different from config
        if (
            hasattr(model, "embedding_extractor")
            and model.embedding_extractor.device != self.config.device
        ):
            self.logger.warning(
                f"Device mismatch: model trained on {model.embedding_extractor.device}, "
                f"test config specifies {self.config.device}. Recreating extractor..."
            )

            # CRITICAL: Explicit cleanup to prevent semaphore leaks (P0 bug fix)
            # See P0_SEMAPHORE_LEAK.md for details
            old_device = str(model.embedding_extractor.device)
            old_extractor = model.embedding_extractor

            # Delete old extractor before creating new one
            del model.embedding_extractor
            del old_extractor

            # Clear device-specific GPU cache
            if old_device.startswith("cuda"):
                torch.cuda.empty_cache()
            elif old_device.startswith("mps"):
                torch.mps.empty_cache()

            self.logger.info(f"Cleaned up old extractor on {old_device}")

            # NOW create new extractor (no leak)
            batch_size = getattr(model, "batch_size", DEFAULT_BATCH_SIZE)
            revision = getattr(model, "revision", "main")  # Backwards compatibility
            model.embedding_extractor = ESMEmbeddingExtractor(
                model.model_name, self.config.device, batch_size, revision=revision
            )
            model.device = self.config.device

            self.logger.info(f"Created new extractor on {self.config.device}")

        # Update batch_size if different from config
        if (
            hasattr(model, "embedding_extractor")
            and model.embedding_extractor.batch_size != self.config.batch_size
        ):
            self.logger.info(
                f"Updating batch_size from {model.embedding_extractor.batch_size} to {self.config.batch_size}"
            )
            model.embedding_extractor.batch_size = self.config.batch_size

        self.logger.info(
            f"Model loaded successfully: {model_path} on device: {model.embedding_extractor.device}"
        )
        return model

    def load_dataset(self, data_path: str) -> tuple[list[str], list[int]]:
        """Load dataset from CSV file using configured column names"""
        self.logger.info(f"Loading dataset from {data_path}")

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset file not found: {data_path}")

        # Defensive: Handle legacy files with comment headers
        # New files (post-HF cleanup) are standard CSVs without comments
        df = pd.read_csv(data_path, comment="#")

        sequence_col = self.config.sequence_column
        label_col = self.config.label_column

        if sequence_col not in df.columns:
            raise ValueError(
                f"Sequence column '{sequence_col}' not found in dataset. Available columns: {list(df.columns)}"
            )
        if label_col not in df.columns:
            raise ValueError(
                f"Label column '{label_col}' not found in dataset. Available columns: {list(df.columns)}"
            )

        # CRITICAL VALIDATION: Check for NaN labels (P0 bug fix)
        nan_count = df[label_col].isna().sum()
        if nan_count > 0:
            raise ValueError(
                f"CRITICAL: Dataset contains {nan_count} NaN labels! "
                f"This will corrupt evaluation metrics. "
                f"Please use the curated canonical test file (e.g., "
                f"data/test/jain/canonical/VH_only_jain_test_PARITY_86.csv with no NaNs)."
            )

        # For Jain test sets, validate expected size (allow legacy 94 + canonical 86)
        if "jain" in data_path.lower() and "test" in data_path.lower():
            expected_sizes = {94, 86}
            if len(df) not in expected_sizes:
                raise ValueError(
                    f"Jain test set has {len(df)} antibodies but expected one of {sorted(expected_sizes)}. "
                    f"Using the wrong test set will produce invalid metrics. "
                    f"Please use the correct curated file (preferred: "
                    f"data/test/jain/canonical/VH_only_jain_test_PARITY_86.csv)."
                )

        sequences = df[sequence_col].tolist()
        labels = df[label_col].tolist()

        self.logger.info(
            f"Loaded {len(sequences)} samples from {data_path} (sequence_col='{sequence_col}', label_col='{label_col}')"
        )
        self.logger.info(
            f"  Label distribution: {pd.Series(labels).value_counts().to_dict()}"
        )
        return sequences, labels

    def embed_sequences(
        self,
        sequences: list[str],
        model: BinaryClassifier,
        dataset_name: str,
        output_dir: str,
    ) -> np.ndarray:
        """Extract embeddings for sequences using the model's embedding extractor"""
        # Ensure output directory exists before file I/O
        os.makedirs(output_dir, exist_ok=True)

        cache_file = os.path.join(output_dir, f"{dataset_name}_test_embeddings.pkl")

        # Track this file for cleanup
        if cache_file not in self.cached_embedding_files:
            self.cached_embedding_files.append(cache_file)

        # Try to load from cache
        if os.path.exists(cache_file):
            try:
                self.logger.info(f"Loading cached embeddings from {cache_file}")
                with open(cache_file, "rb") as f:
                    embeddings: np.ndarray = pickle.load(f)  # nosec B301 - Loading our own cached embeddings for performance

                # Validate shape and type
                if not isinstance(embeddings, np.ndarray):
                    raise ValueError(f"Invalid cache data type: {type(embeddings)}")
                if embeddings.ndim != 2:
                    raise ValueError(f"Invalid embedding shape: {embeddings.shape}")

                if len(embeddings) == len(sequences):
                    self.logger.info(f"Loaded {len(embeddings)} cached embeddings")
                    return embeddings
                else:
                    self.logger.warning(
                        "Cached embeddings size mismatch, recomputing..."
                    )

            except (pickle.UnpicklingError, EOFError, ValueError, AttributeError) as e:
                self.logger.warning(
                    f"Failed to load cached embeddings from {cache_file}: {e}. "
                    "Recomputing embeddings..."
                )
                # Fall through to recomputation below

        # Extract embeddings
        self.logger.info(f"Extracting embeddings for {len(sequences)} sequences...")
        embeddings = model.embedding_extractor.extract_batch_embeddings(sequences)

        # Cache embeddings
        with open(cache_file, "wb") as f:
            pickle.dump(embeddings, f)
        self.logger.info(f"Embeddings cached to {cache_file}")

        return embeddings

    def evaluate_pretrained(
        self,
        model: BinaryClassifier,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str,
        dataset_name: str,
    ) -> dict[str, Any]:
        """Evaluate pretrained model directly on test set (no retraining)"""
        self.logger.info(f"Evaluating pretrained model {model_name} on {dataset_name}")

        # Get predictions using the pretrained model
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]

        # Calculate metrics
        results = {
            "test_scores": {},
            "predictions": {"y_true": y, "y_pred": y_pred, "y_proba": y_proba},
            "confusion_matrix": confusion_matrix(y, y_pred),
            "classification_report": classification_report(y, y_pred, output_dict=True),
        }

        # Calculate all requested metrics
        metrics = self.config.metrics
        if metrics is not None:
            if "accuracy" in metrics:
                results["test_scores"]["accuracy"] = accuracy_score(y, y_pred)
            if "precision" in metrics:
                results["test_scores"]["precision"] = precision_score(
                    y, y_pred, zero_division=0
                )
            if "recall" in metrics:
                results["test_scores"]["recall"] = recall_score(
                    y, y_pred, zero_division=0
                )
            if "f1" in metrics:
                results["test_scores"]["f1"] = f1_score(y, y_pred, zero_division=0)
            if "roc_auc" in metrics:
                results["test_scores"]["roc_auc"] = roc_auc_score(y, y_proba)
            if "pr_auc" in metrics:
                results["test_scores"]["pr_auc"] = average_precision_score(y, y_proba)

        # Log results
        self.logger.info(f"Test results for {model_name} on {dataset_name}:")
        for metric, value in results["test_scores"].items():
            self.logger.info(f"  {metric}: {value:.4f}")

        return results

    def plot_confusion_matrix(
        self,
        results: dict[str, dict[str, Any]],
        dataset_name: str,
        output_dir: str | None = None,
    ) -> None:
        """Create confusion matrix visualization (individual files per model).

        Args:
            results: Dictionary mapping model names to result dictionaries
            dataset_name: Name of the dataset
            output_dir: Directory to save plots (defaults to self.config.output_dir)
        """
        # Use provided output_dir or fall back to config default
        target_dir = output_dir if output_dir is not None else self.config.output_dir
        os.makedirs(target_dir, exist_ok=True)

        self.logger.info(
            f"Creating confusion matrices for {dataset_name} in {target_dir}"
        )

        # Create individual confusion matrix for each model to prevent overrides
        for model_name, model_results in results.items():
            if "confusion_matrix" not in model_results:
                self.logger.warning(
                    f"No confusion matrix found for {model_name}, skipping plot"
                )
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
                target_dir,
                f"confusion_matrix_{model_name}_{dataset_name}.png",
            )
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(f"Confusion matrix saved to {plot_file}")

    def save_detailed_results(
        self,
        results: dict[str, dict[str, Any]],
        dataset_name: str,
        output_dir: str | None = None,
    ) -> None:
        """Save detailed results to files (individual files per model).

        Args:
            results: Dictionary mapping model names to result dictionaries
            dataset_name: Name of the dataset
            output_dir: Directory to save results (defaults to self.config.output_dir)
        """
        # Use provided output_dir or fall back to config default
        target_dir = output_dir if output_dir is not None else self.config.output_dir
        os.makedirs(target_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save individual YAML for each model to prevent overrides
        for model_name, model_results in results.items():
            results_file = os.path.join(
                target_dir,
                f"detailed_results_{model_name}_{dataset_name}_{timestamp}.yaml",
            )
            with open(results_file, "w") as f:
                yaml.dump(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "config": self.config.__dict__,
                        "results": model_results,
                    },
                    f,
                    default_flow_style=False,
                )
            self.logger.info(f"Detailed results saved to {results_file}")

        # Save predictions if requested
        if self.config.save_predictions:
            for model_name, model_results in results.items():
                if "predictions" in model_results:
                    pred_file = os.path.join(
                        target_dir,
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
                    self.logger.info(f"Predictions saved to {pred_file}")

    def cleanup_cached_embeddings(self) -> None:
        """Delete cached embedding files"""
        self.logger.info("Cleaning up cached embedding files...")
        for cache_file in self.cached_embedding_files:
            if os.path.exists(cache_file):
                try:
                    os.remove(cache_file)
                    self.logger.info(f"Deleted cached embeddings: {cache_file}")
                except Exception as e:
                    self.logger.warning(f"Failed to delete {cache_file}: {e}")

    def _compute_output_directory(
        self, model_path: str | None, dataset_name: str
    ) -> str:
        """Compute output directory (hierarchical if model config available, else flat).

        Uses shared directory_utils.get_hierarchical_test_results_dir for consistency.

        Args:
            model_path: Path to the model file
            dataset_name: Name of the dataset

        Returns:
            Output directory path (hierarchical or flat)
        """
        if model_path is None:
            self.logger.warning("No model path provided, using flat output structure")
            return self.config.output_dir

        # Try to load model config JSON to determine hierarchical path
        model_config_path = (
            Path(model_path)
            .with_suffix("")
            .with_name(Path(model_path).stem + "_config.json")
        )

        if not model_config_path.exists():
            self.logger.info(
                f"Model config not found at {model_config_path}, using flat output structure"
            )
            return self.config.output_dir

        try:
            with open(model_config_path) as f:
                model_config = json.load(f)

            # Extract model name and classifier config from JSON
            # Try 'model_name' first (preferred), fallback to 'esm_model' (legacy)
            model_name = model_config.get("model_name") or model_config.get(
                "esm_model", ""
            )
            if not model_name:
                raise ValueError("Model config missing 'model_name' or 'esm_model'")

            classifier_config = model_config.get("classifier", {})

            # Use shared utility for hierarchical path generation
            hierarchical_path = get_hierarchical_test_results_dir(
                base_dir=self.config.output_dir,
                model_name=model_name,
                classifier_config=classifier_config,
                dataset_name=dataset_name,
            )

            # Extract shortnames for logging
            model_short = extract_model_shortname(model_name)
            classifier_short = extract_classifier_shortname(classifier_config)

            self.logger.info(
                f"Using hierarchical output: {hierarchical_path} "
                f"(model={model_short}, classifier={classifier_short})"
            )
            return str(hierarchical_path)

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.warning(
                f"Could not determine hierarchical path from model config: {e}. "
                "Using flat structure."
            )
            return self.config.output_dir

    def run_comprehensive_test(self) -> dict[str, dict[str, Any]]:
        """Run testing pipeline"""
        self.logger.info("Starting model testing")
        self.logger.info(f"Models to test: {self.config.model_paths}")
        self.logger.info(f"Datasets to test: {self.config.data_paths}")

        all_results = {}
        failed_datasets = []
        failed_models = []

        try:
            # Test each dataset
            for data_path in self.config.data_paths:
                dataset_name = Path(data_path).stem
                self.logger.info(f"\n{'=' * 60}")
                self.logger.info(f"Testing on dataset: {dataset_name}")
                self.logger.info(f"{'=' * 60}")

                # Load dataset
                try:
                    sequences, labels_list = self.load_dataset(data_path)
                    labels: np.ndarray = np.array(labels_list)
                except Exception as e:
                    self.logger.error(f"Failed to load dataset {data_path}: {e}")
                    failed_datasets.append((dataset_name, str(e)))
                    continue

                dataset_results = {}

                # Test each model
                for model_path in self.config.model_paths:
                    model_name = Path(model_path).stem
                    self.logger.info(f"\nTesting model: {model_name}")

                    # Determine output directory (hierarchical or flat)
                    output_dir_for_dataset = self._compute_output_directory(
                        model_path, dataset_name
                    )

                    try:
                        # Load model
                        model = self.load_model(model_path)

                        # Extract embeddings
                        X_embedded = self.embed_sequences(
                            sequences,
                            model,
                            f"{dataset_name}_{model_name}",
                            output_dir_for_dataset,
                        )

                        # Direct evaluation on test set using pretrained model
                        test_results = self.evaluate_pretrained(
                            model, X_embedded, labels, model_name, dataset_name
                        )
                        dataset_results[model_name] = test_results

                        # Save this model's results to its hierarchical directory
                        single_model_results = {model_name: test_results}
                        self.plot_confusion_matrix(
                            single_model_results,
                            dataset_name,
                            output_dir=output_dir_for_dataset,
                        )
                        self.save_detailed_results(
                            single_model_results,
                            dataset_name,
                            output_dir=output_dir_for_dataset,
                        )

                    except Exception as e:
                        self.logger.error(f"Failed to test model {model_path}: {e}")
                        failed_models.append((f"{dataset_name}_{model_name}", str(e)))
                        continue

                # Generate aggregated multi-model report (after all models tested)
                if dataset_results:  # Only if we have successful results
                    # Use flat output_dir for aggregated reports (dataset root)
                    aggregated_output_dir = self.config.output_dir
                    self.logger.info(
                        f"Generating aggregated multi-model report for {dataset_name} "
                        f"in {aggregated_output_dir}"
                    )

                    self.plot_confusion_matrix(
                        dataset_results,  # All models
                        dataset_name,
                        output_dir=aggregated_output_dir,
                    )
                    self.save_detailed_results(
                        dataset_results,  # All models
                        dataset_name,
                        output_dir=aggregated_output_dir,
                    )

                all_results[dataset_name] = dataset_results

            # Check if all tests failed
            if not all_results:
                error_msg = "All tests failed:\n"
                if failed_datasets:
                    error_msg += (
                        f"  Failed datasets: {[name for name, _ in failed_datasets]}\n"
                    )
                if failed_models:
                    error_msg += (
                        f"  Failed models: {[name for name, _ in failed_models]}\n"
                    )
                raise RuntimeError(error_msg + "No successful test results to report.")

            # Warn about partial failures
            if failed_datasets or failed_models:
                self.logger.warning(
                    f"\nSome tests failed (datasets: {len(failed_datasets)}, "
                    f"models: {len(failed_models)}). Check logs for details."
                )

            self.results = all_results
            self.logger.info(
                f"\nTesting completed. Results saved to: {self.config.output_dir}"
            )

        finally:
            # Always cleanup cached embeddings
            self.cleanup_cached_embeddings()

        return all_results


def load_config_file(config_path: str) -> TestConfig:
    """Load test configuration from YAML file"""
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    return TestConfig(**config_dict)


def create_sample_test_config() -> None:
    """Create a sample test configuration file"""
    sample_config = {
        "model_paths": ["./models/antibody_classifier.pkl"],
        "data_paths": ["./sample_data.csv"],
        "sequence_column": "sequence",
        "label_column": "label",
        "output_dir": "./test_results",
        "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"],
        "save_predictions": True,
    }

    with open("test_config.yaml", "w") as f:
        yaml.dump(sample_config, f, default_flow_style=False)

    print("Sample test configuration created: test_config.yaml")


def main() -> int:
    """Main entry point for antibody-test CLI"""
    parser = argparse.ArgumentParser(
        description="Testing for antibody classification models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test single model on single dataset
    antibody-test --model models/antibody_classifier.pkl --data sample_data.csv

    # Test multiple models on multiple datasets
    antibody-test --model models/model1.pkl models/model2.pkl --data dataset1.csv dataset2.csv

    # Use configuration file
    antibody-test --config test_config.yaml

    # Override device and batch size
    antibody-test --config test_config.yaml --device cuda --batch-size 64

    # Create sample configuration
    antibody-test --create-config
        """,
    )

    parser.add_argument(
        "--model", nargs="+", help="Path(s) to trained model pickle files"
    )
    parser.add_argument("--data", nargs="+", help="Path(s) to test dataset CSV files")
    parser.add_argument("--config", help="Path to test configuration YAML file")
    parser.add_argument(
        "--output-dir", default="./test_results", help="Output directory for results"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        help="Device to use for inference (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for embedding extraction (overrides config)",
    )
    parser.add_argument(
        "--create-config", action="store_true", help="Create sample configuration file"
    )

    args = parser.parse_args()

    # Create sample config if requested
    if args.create_config:
        create_sample_test_config()
        return 0

    # Load configuration
    if args.config:
        config = load_config_file(args.config)
    else:
        if not args.model or not args.data:
            parser.error("Either --config or both --model and --data must be specified")

        config = TestConfig(
            model_paths=args.model, data_paths=args.data, output_dir=args.output_dir
        )

    # Override config with command line arguments
    if args.device:
        config.device = args.device
    if args.batch_size:
        config.batch_size = args.batch_size

    # Run testing
    try:
        tester = ModelTester(config)
        results = tester.run_comprehensive_test()

        print(f"\n{'=' * 60}")
        print("TESTING COMPLETED SUCCESSFULLY!")
        print(f"{'=' * 60}")
        print(f"Results saved to: {config.output_dir}")

        # Print summary
        for dataset_name, dataset_results in results.items():
            print(f"\nDataset: {dataset_name}")
            print("-" * 40)
            for model_name, model_results in dataset_results.items():
                print(f"Model: {model_name}")
                if "test_scores" in model_results:
                    for metric, value in model_results["test_scores"].items():
                        print(f"  {metric}: {value:.4f}")

        return 0

    except KeyboardInterrupt:
        print("Error during testing: Interrupted by user", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error during testing: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
