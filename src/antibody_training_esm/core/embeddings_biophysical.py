"""
Biophysical Embedding Extractor

Wrapper for BiophysicalExtractor that conforms to the embedding extractor
protocol used by BinaryClassifier. This enables Track B to use the same
training infrastructure as Track A (ESM).

Note: "embeddings" is a misnomer for biophysical features, but we use this
naming for consistency with the existing codebase architecture.
"""

import sys
from typing import Any

import numpy as np
from numpy.typing import NDArray

from antibody_training_esm.core.biophysical import BiophysicalExtractor


class BiophysicalEmbeddingExtractor:
    """
    Embedding-like interface for biophysical feature extraction.

    Wraps BiophysicalExtractor to match the interface expected by
    BinaryClassifier, enabling Track B to use Hydra pipeline.

    Attributes:
        model_name: Always "biophysical" for this extractor.
        device: Always "cpu" (Biopython is CPU-only).
        batch_size: Ignored (single-sequence processing).
        revision: Version string for cache invalidation.
        max_length: No limit for biophysical features.
    """

    def __init__(
        self,
        _model_name: str,
        _device: str,
        _batch_size: int,
        revision: str = "1.0.0",
        **_kwargs: Any,
    ) -> None:
        """
        Initialize the biophysical extractor.

        Args:
            _model_name: Model identifier (should be "biophysical").
            _device: Device to use (ignored, always CPU).
            _batch_size: Batch size (ignored, single-sequence processing).
            revision: Version for cache key generation.
            **_kwargs: Additional arguments (ignored for compatibility).
        """
        self.biophysical = BiophysicalExtractor()
        self.model_name = "biophysical"
        self.device = "cpu"  # Force CPU - Biopython has no GPU support
        self.batch_size = 1  # Not batched
        self.revision = revision
        self.max_length = sys.maxsize  # No sequence length limit

    def embed_sequence(self, sequence: str) -> NDArray[np.float32]:
        """
        Extract biophysical features for a single sequence.

        Args:
            sequence: Amino acid sequence (VH domain).

        Returns:
            1D array of shape (3,) containing:
                - charge_ph6: Charge at pH 6.0
                - charge_ph7_4: Charge at pH 7.4
                - theoretical_pi: Isoelectric point
        """
        return self.biophysical.extract_features(sequence)

    def extract_batch_embeddings(self, sequences: list[str]) -> NDArray[np.float32]:
        """
        Extract biophysical features for a batch of sequences.

        Args:
            sequences: List of amino acid sequences.

        Returns:
            2D array of shape (n_sequences, 3).
        """
        return self.biophysical.extract_batch_features(sequences)
