import logging
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from antibody_training_esm.core.config import DEFAULT_BATCH_SIZE
from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor
from antibody_training_esm.core.trainer import load_model_from_npz
from antibody_training_esm.models.prediction import (
    AssayType,
    PredictionRequest,
    PredictionResult,
)

logger = logging.getLogger(__name__)


class Predictor:
    """
    A class to handle the antibody non-specificity prediction pipeline.

    This class encapsulates the model loading, embedding extraction, and prediction logic.
    It follows the principle of 'prepare once, execute many' (though for CLI it's usually once).
    """

    def __init__(
        self,
        model_name: str,
        classifier_path: str,
        device: str | None = None,
        config_path: str | None = None,
    ):
        """
        Initialize the Predictor with model configurations.

        Args:
            model_name: The name of the ESM model to use (e.g. 'facebook/esm1v_t33_650M_UR90S_1').
            classifier_path: Path to the trained scikit-learn classifier (pickle/joblib file) or NPZ weights.
            device: The device to run the model on ('cpu' or 'cuda'). If None, auto-detects.
            config_path: Path to the JSON config file (required if classifier_path is .npz).
        """
        self.device = self._select_device(device)
        self.model_name = model_name
        self.classifier_path = classifier_path
        self.config_path = config_path

        self._embedder: ESMEmbeddingExtractor | None = None
        self._classifier: Any = None

    @property
    def classifier(self) -> Any:
        """
        Lazy loads the classifier.

        Supports:
        1. Legacy Pickle (.pkl): Loaded via joblib.
        2. Production NPZ (.npz): Loaded via load_model_from_npz using accompanying JSON config.
        """
        if self._classifier is None:
            path_obj = Path(self.classifier_path)

            if path_obj.suffix == ".npz":
                # NPZ loading path
                if self.config_path:
                    json_path = Path(self.config_path)
                else:
                    # Infer JSON path: model.npz -> model_config.json
                    json_path = path_obj.with_name(f"{path_obj.stem}_config.json")

                if not json_path.exists():
                    raise FileNotFoundError(
                        f"JSON config not found at {json_path}. "
                        "For .npz models, a corresponding JSON config is required. "
                        "Specify it explicitly with config_path if the naming convention differs."
                    )

                logger.info(f"Loading model from NPZ: {path_obj} (Config: {json_path})")
                self._classifier = load_model_from_npz(str(path_obj), str(json_path))

            else:
                # Legacy/Pickle loading path
                logger.info(f"Loading model from Pickle: {path_obj}")
                self._classifier = joblib.load(self.classifier_path)

        return self._classifier

    @property
    def embedder(self) -> ESMEmbeddingExtractor:
        """
        Lazy loads the ESM embedding extractor.

        Optimization:
            If the loaded classifier is a BinaryClassifier instance (which contains
            its own embedding_extractor), we reuse it to avoid double-loading
            the 650MB model into GPU/CPU memory.
        """
        if self._embedder is None:
            # First ensure classifier is loaded (it might have the embedder)
            clf = self.classifier

            # Check if it's our BinaryClassifier wrapper that has an embedder
            if (
                hasattr(clf, "embedding_extractor")
                and clf.embedding_extractor is not None
            ):
                embedder = clf.embedding_extractor

                # If the persisted embedder device doesn't match requested device,
                # recreate it to avoid MPS/CUDA mismatches (common segfault source on macOS).
                if self.device and str(embedder.device) != self.device:
                    batch_size = getattr(embedder, "batch_size", DEFAULT_BATCH_SIZE)
                    revision = getattr(embedder, "revision", "main")
                    logger.info(
                        "Recreating embedder on requested device %s (was %s)",
                        self.device,
                        embedder.device,
                    )
                    embedder = ESMEmbeddingExtractor(
                        model_name=self.model_name,
                        device=self.device,
                        batch_size=batch_size,
                        revision=revision,
                    )

                self._embedder = embedder
            else:
                # Fallback: Create a new one (e.g., if using raw sklearn model)
                self._embedder = ESMEmbeddingExtractor(
                    model_name=self.model_name,
                    device=self.device,
                )
        return self._embedder

    def predict(
        self,
        sequences: list[str],
        threshold: float = 0.5,
        assay_type: AssayType | None = None,
    ) -> pd.DataFrame:
        """
        Predict specificity for a list of sequences.

        Args:
            sequences: A list of antibody amino acid sequences.
            threshold: Decision threshold (default: 0.5).
            assay_type: 'PSR' or 'ELISA' to use calibrated thresholds (overrides threshold).

        Returns:
            A DataFrame containing 'prediction' (string) and 'probability' (float) columns.
        """
        if not sequences:
            return pd.DataFrame(columns=["prediction", "probability"])

        # Generate embeddings
        embeddings = self.embedder.extract_batch_embeddings(sequences)

        # Make predictions
        # Check if the classifier supports the custom 'predict' signature with assay_type
        # (Our BinaryClassifier does, standard sklearn does not)
        if (
            hasattr(self.classifier, "predict")
            and "assay_type" in self.classifier.predict.__code__.co_varnames
        ):
            predictions = self.classifier.predict(
                embeddings, threshold=threshold, assay_type=assay_type
            )
        else:
            # Standard sklearn behavior
            probabilities = self.classifier.predict_proba(embeddings)
            predictions = (probabilities[:, 1] > threshold).astype(int)

        # Get probabilities (universal)
        probabilities = self.classifier.predict_proba(embeddings)

        # Ensure probabilities is a numpy array
        if isinstance(probabilities, list):
            probabilities = np.array(probabilities)

        # Format results
        results = pd.DataFrame(
            {
                "prediction": [
                    "non-specific" if p == 1 else "specific" for p in predictions
                ],
                "probability": probabilities[
                    :, 1
                ],  # Probability of class 1 (non-specific)
            }
        )

        return results

    def predict_dataframe(
        self,
        df: pd.DataFrame,
        sequence_col: str = "sequence",
        threshold: float = 0.5,
        assay_type: AssayType | None = None,
    ) -> pd.DataFrame:
        """
        Predict specificity for sequences in a DataFrame and append results.

        Args:
            df: Input DataFrame.
            sequence_col: Name of the column containing sequences.
            threshold: Decision threshold.
            assay_type: 'PSR' or 'ELISA' (overrides threshold).

        Returns:
            A copy of the input DataFrame with 'prediction' and 'probability' columns appended.
        """
        if sequence_col not in df.columns:
            raise ValueError(f"Input DataFrame must contain a '{sequence_col}' column.")

        sequences = df[sequence_col].tolist()
        results = self.predict(sequences, threshold=threshold, assay_type=assay_type)

        output_df = df.copy()
        output_df["prediction"] = results["prediction"].values
        output_df["probability"] = results["probability"].values

        return output_df

    def predict_single(
        self,
        sequence: str | PredictionRequest,
        threshold: float = 0.5,
        assay_type: AssayType | None = None,
    ) -> PredictionResult:
        """
        Predict single sequence with Pydantic validation.

        Args:
            sequence: Raw string OR PredictionRequest model
            threshold: Decision threshold (ignored if PredictionRequest passed)
            assay_type: Assay type (ignored if PredictionRequest passed)

        Returns:
            PredictionResult model
        """
        # Normalize input to PredictionRequest
        if isinstance(sequence, str):
            request = PredictionRequest(
                sequence=sequence,
                threshold=threshold,
                assay_type=assay_type,
            )
        else:
            request = sequence

        # Extract validated sequence
        cleaned_seq = request.sequence

        # Run prediction (existing logic)
        results_df = self.predict(
            [cleaned_seq],
            threshold=request.threshold,
            assay_type=request.assay_type,
        )

        # Convert to PredictionResult
        return PredictionResult(
            sequence=cleaned_seq,
            prediction=results_df["prediction"].iloc[0],
            probability=float(results_df["probability"].iloc[0]),
            threshold=request.threshold,
            assay_type=request.assay_type,
        )

    @staticmethod
    def _select_device(device: str | None) -> str:
        """
        Select the best available device.

        Prioritizes CUDA, then MPS (macOS), then CPU.
        """
        if device:
            return device

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"


def run_prediction(input_df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """
    Helper function to run prediction using Hydra config.

    Args:
        input_df: DataFrame containing an sequence column.
        cfg: The Hydra configuration object.

    Returns:
        DataFrame with 'prediction' and 'probability' columns added.
    """
    config_path = getattr(cfg.classifier, "config_path", None)

    predictor = Predictor(
        model_name=cfg.model.name,
        classifier_path=cfg.classifier.path,
        config_path=config_path,
    )

    # Extract config parameters with defaults
    sequence_col = getattr(cfg, "sequence_column", "sequence")
    threshold = getattr(cfg, "threshold", 0.5)
    assay_type = cast(AssayType | None, getattr(cfg, "assay_type", None))

    return predictor.predict_dataframe(
        input_df, sequence_col=sequence_col, threshold=threshold, assay_type=assay_type
    )
