
from pathlib import Path

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
import joblib
import numpy as np

from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor


def run_prediction(input_df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """
    Runs the antibody non-specificity prediction pipeline on a given DataFrame.

    Args:
        input_df: DataFrame containing an 'sequence' column with antibody sequences.
        cfg: The Hydra configuration object.

    Returns:
        DataFrame with 'prediction' and 'probability' columns added.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the embedding extractor
    embedder = ESMEmbeddingExtractor(
        model_name=cfg.model.name,
        device=device,
    )

    # Load classifier
    classifier = joblib.load(cfg.classifier.path)

    if "sequence" not in input_df.columns:
        raise ValueError("Input CSV must contain a 'sequence' column.")
    sequences = input_df["sequence"].tolist()

    # Generate embeddings
    embeddings = embedder.extract_batch_embeddings(sequences)

    # Make predictions
    predictions = classifier.predict(embeddings)
    probabilities = classifier.predict_proba(embeddings)
    if isinstance(probabilities, list):
        probabilities = np.array(probabilities)

    output_df = input_df.copy()
    output_df["prediction"] = ["non-specific" if p == 1 else "specific" for p in predictions]
    output_df["probability"] = probabilities[:, 1]

    return output_df
