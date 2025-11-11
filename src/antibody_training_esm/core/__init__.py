"""
Core ML Module

Professional ML components for antibody classification:
- ESM embedding extraction
- Binary classification
- Training pipelines
- Model serialization (pickle + NPZ+JSON)
"""

from antibody_training_esm.core.classifier import BinaryClassifier
from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor
from antibody_training_esm.core.trainer import load_model_from_npz, train_model

__all__ = [
    "BinaryClassifier",
    "ESMEmbeddingExtractor",
    "train_model",
    "load_model_from_npz",
]
