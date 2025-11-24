"""
Core ML Module

Professional ML components for antibody classification:
- ESM embedding extraction
- AMPLIFY embedding extraction
- Binary classification
- Training pipelines
- Model serialization (pickle + NPZ+JSON)
"""

from antibody_training_esm.core.classifier import (
    BinaryClassifier,
    EmbeddingExtractorProtocol,
)
from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor
from antibody_training_esm.core.embeddings_amplify import AMPLIFYEmbeddingExtractor
from antibody_training_esm.core.training.serialization import load_model_from_npz

__all__ = [
    "AMPLIFYEmbeddingExtractor",
    "BinaryClassifier",
    "EmbeddingExtractorProtocol",
    "ESMEmbeddingExtractor",
    "load_model_from_npz",
]
