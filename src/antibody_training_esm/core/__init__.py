"""
Core ML Module

Professional ML components for antibody classification:
- ESM embedding extraction
- Binary classification
- Training pipelines
"""

from antibody_training_esm.core.classifier import BinaryClassifier
from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor
from antibody_training_esm.core.trainer import train_model

__all__ = ["BinaryClassifier", "ESMEmbeddingExtractor", "train_model"]
