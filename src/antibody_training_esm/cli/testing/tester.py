"Model orchestration logic."

import json
import logging
import os
import pickle  # nosec B403
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from antibody_training_esm.cli.testing.config import TestConfig
from antibody_training_esm.cli.testing.data import load_dataset
from antibody_training_esm.cli.testing.evaluation import evaluate_pretrained
from antibody_training_esm.cli.testing.visualization import (
    plot_confusion_matrix,
    save_detailed_results,
)
from antibody_training_esm.core.classifier import BinaryClassifier
from antibody_training_esm.core.config import DEFAULT_BATCH_SIZE
from antibody_training_esm.core.device import resolve_device
from antibody_training_esm.core.directory_utils import (
    extract_classifier_shortname,
    extract_model_shortname,
    get_hierarchical_test_results_dir,
)
from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor


class ModelTester:
    """Model testing orchestrator"""

    def __init__(self, config: TestConfig):
        self.config = config

        # Resolve device (handles "auto" and validates explicit devices)
        self.config.device = resolve_device(self.config.device)

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
            model = pickle.load(f)  # nosec B301

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
            revision = getattr(model, "revision", "main")
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
                    embeddings: np.ndarray = pickle.load(f)  # nosec B301

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
        self,
        model_path: str | None,
        dataset_name: str,
    ) -> str:
        """Compute output directory (hierarchical if model config available, else flat)."""
        if model_path is None:
            self.logger.warning("No model path provided, using flat output structure")
            return self.config.output_dir

        # Try to load model config JSON
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
                    sequences, labels_list = load_dataset(data_path, self.config)
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

                        # Evaluation (delegated to evaluation module)
                        test_results = evaluate_pretrained(
                            model,
                            X_embedded,
                            labels,
                            model_name,
                            dataset_name,
                            self.config.metrics,
                            self.config.threshold,
                        )
                        dataset_results[model_name] = test_results

                        # Visualization (delegated to visualization module)
                        single_model_results = {model_name: test_results}
                        plot_confusion_matrix(
                            single_model_results,
                            dataset_name,
                            output_dir=output_dir_for_dataset,
                        )
                        save_detailed_results(
                            single_model_results,
                            dataset_name,
                            self.config.__dict__,
                            output_dir=output_dir_for_dataset,
                            save_predictions=self.config.save_predictions,
                        )

                    except Exception as e:
                        self.logger.error(f"Failed to test model {model_path}: {e}")
                        failed_models.append((f"{dataset_name}_{model_name}", str(e)))
                        continue

                # Generate aggregated multi-model report
                if dataset_results:
                    aggregated_output_dir = self.config.output_dir
                    self.logger.info(
                        f"Generating aggregated multi-model report for {dataset_name} "
                        f"in {aggregated_output_dir}"
                    )

                    plot_confusion_matrix(
                        dataset_results,
                        dataset_name,
                        output_dir=aggregated_output_dir,
                    )
                    save_detailed_results(
                        dataset_results,
                        dataset_name,
                        self.config.__dict__,
                        output_dir=aggregated_output_dir,
                        save_predictions=self.config.save_predictions,
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
            self.cleanup_cached_embeddings()

        return all_results
