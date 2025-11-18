
from pathlib import Path

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from transformers import AutoModelForMaskedLM, AutoTokenizer
import joblib
import numpy as np


def get_embeddings(sequences: list[str], model, tokenizer, device: str) -> np.ndarray:
    """Generates embeddings for a list of sequences."""
    inputs = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-1]
    return hidden_states.mean(dim=1).cpu().numpy()


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

    # Load ESM model and tokenizer
    model_name = cfg.model.name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    model.eval()

    # Load classifier
    classifier = joblib.load(cfg.classifier.path)

    if "sequence" not in input_df.columns:
        raise ValueError("Input CSV must contain a 'sequence' column.")
    sequences = input_df["sequence"].tolist()

    # Generate embeddings
    embeddings = get_embeddings(sequences, model, tokenizer, device)

    # Make predictions
    predictions = classifier.predict(embeddings)
    probabilities = classifier.predict_proba(embeddings)
    if isinstance(probabilities, list):
        probabilities = np.array(probabilities)

    output_df = input_df.copy()
    output_df["prediction"] = ["non-specific" if p == 1 else "specific" for p in predictions]
    output_df["probability"] = probabilities[:, 1]

    return output_df
