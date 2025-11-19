import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from antibody_training_esm.core.config import DEFAULT_BATCH_SIZE
from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor

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
    ):
        """
        Initialize the Predictor with model configurations.

        Args:
            model_name: The name of the ESM model to use (e.g. 'facebook/esm1v_t33_650M_UR90S_1').
            classifier_path: Path to the trained scikit-learn classifier (pickle/joblib file).
            device: The device to run the model on ('cpu' or 'cuda'). If None, auto-detects.
        """
        self.device = self._select_device(device)
        self.model_name = model_name
        self.classifier_path = classifier_path

        self._embedder: ESMEmbeddingExtractor | None = None
        self._classifier: Any = None

    @property
    def classifier(self) -> Any:
        """Lazy loads the classifier."""
        if self._classifier is None:
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
        assay_type: str | None = None,
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
        assay_type: str | None = None,
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

    def predict_single(self, sequence: str) -> dict[str, Any]:
        """
        Convenience method for single sequence prediction (e.g., for Gradio/API).

        Args:
            sequence: Amino acid sequence string.

        Returns:
            Dictionary with keys 'prediction' and 'probability'.
        """
        results = self.predict([sequence])
        return {
            "prediction": results["prediction"].iloc[0],
            "probability": float(results["probability"].iloc[0]),
        }

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
    predictor = Predictor(
        model_name=cfg.model.name,
        classifier_path=cfg.classifier.path,
    )

    # Extract config parameters with defaults
    sequence_col = getattr(cfg, "sequence_column", "sequence")
    threshold = getattr(cfg, "threshold", 0.5)
    assay_type = getattr(cfg, "assay_type", None)

    return predictor.predict_dataframe(
        input_df, sequence_col=sequence_col, threshold=threshold, assay_type=assay_type
    )
