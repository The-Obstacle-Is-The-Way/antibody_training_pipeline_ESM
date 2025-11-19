from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor


class Predictor:
    """
    A class to handle the antibody non-specificity prediction pipeline.

    This class encapsulates the model loading, embedding extraction, and prediction logic.
    It follows the principle of 'prepare once, execute many' (though for CLI it's usually once).
    """

    def __init__(
        self, model_name: str, classifier_path: str, device: str | None = None
    ):
        """
        Initialize the Predictor with model configurations.

        Args:
            model_name: The name of the ESM model to use (e.g. 'facebook/esm1v_t33_650M_UR90S_1').
            classifier_path: Path to the trained scikit-learn classifier (pickle/joblib file).
            device: The device to run the model on ('cpu' or 'cuda'). If None, auto-detects.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.classifier_path = classifier_path

        self._embedder: ESMEmbeddingExtractor | None = None
        self._classifier: Any = None

    @property
    def embedder(self) -> ESMEmbeddingExtractor:
        """Lazy loads the ESM embedding extractor."""
        if self._embedder is None:
            self._embedder = ESMEmbeddingExtractor(
                model_name=self.model_name,
                device=self.device,
            )
        return self._embedder

    @property
    def classifier(self) -> Any:
        """Lazy loads the classifier."""
        if self._classifier is None:
            self._classifier = joblib.load(self.classifier_path)
        return self._classifier

    def predict(self, sequences: list[str]) -> pd.DataFrame:
        """
        Predict specificity for a list of sequences.

        Args:
            sequences: A list of antibody amino acid sequences.

        Returns:
            A DataFrame containing 'prediction' (string) and 'probability' (float) columns.
        """
        if not sequences:
            return pd.DataFrame(columns=["prediction", "probability"])

        # Generate embeddings
        embeddings = self.embedder.extract_batch_embeddings(sequences)

        # Make predictions
        predictions = self.classifier.predict(embeddings)
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
        self, df: pd.DataFrame, sequence_col: str = "sequence"
    ) -> pd.DataFrame:
        """
        Predict specificity for sequences in a DataFrame and append results.

        Args:
            df: Input DataFrame.
            sequence_col: Name of the column containing sequences.

        Returns:
            A copy of the input DataFrame with 'prediction' and 'probability' columns appended.
        """
        if sequence_col not in df.columns:
            raise ValueError(f"Input DataFrame must contain a '{sequence_col}' column.")

        sequences = df[sequence_col].tolist()
        results = self.predict(sequences)

        output_df = df.copy()
        output_df["prediction"] = results["prediction"].values
        output_df["probability"] = results["probability"].values

        return output_df


def run_prediction(input_df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """
    Legacy/Helper function to run prediction using Hydra config.

    Args:
        input_df: DataFrame containing an 'sequence' column.
        cfg: The Hydra configuration object.

    Returns:
        DataFrame with 'prediction' and 'probability' columns added.
    """
    predictor = Predictor(
        model_name=cfg.model.name,
        classifier_path=cfg.classifier.path,
    )
    return predictor.predict_dataframe(input_df)
