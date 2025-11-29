"""
Unit tests for BiophysicalEmbeddingExtractor wrapper
"""

import sys
from unittest.mock import patch

import numpy as np
import pytest

from antibody_training_esm.core.embeddings_biophysical import (
    BiophysicalEmbeddingExtractor,
)


@pytest.fixture
def extractor() -> BiophysicalEmbeddingExtractor:
    return BiophysicalEmbeddingExtractor(
        _model_name="biophysical",
        _device="cpu",
        _batch_size=1,
        revision="1.0.0",
    )


def test_initialization(extractor: BiophysicalEmbeddingExtractor) -> None:
    assert extractor.model_name == "biophysical"
    assert extractor.device == "cpu"
    assert extractor.batch_size == 1
    assert extractor.revision == "1.0.0"
    assert extractor.max_length == sys.maxsize


def test_embed_sequence_returns_correct_shape(
    extractor: BiophysicalEmbeddingExtractor,
) -> None:
    # Mock the internal biophysical extractor
    with patch.object(
        extractor.biophysical,
        "extract_features",
        return_value=np.array([1.0, 2.0, 3.0], dtype=np.float32),
    ) as mock_extract:
        seq = "ACDEF"
        embedding = extractor.embed_sequence(seq)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (3,)
        assert embedding.dtype == np.float32
        mock_extract.assert_called_once_with(seq)


def test_extract_batch_embeddings_returns_correct_shape(
    extractor: BiophysicalEmbeddingExtractor,
) -> None:
    # Mock the internal biophysical extractor
    mock_batch_result = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    with patch.object(
        extractor.biophysical,
        "extract_batch_features",
        return_value=mock_batch_result,
    ) as mock_extract_batch:
        seqs = ["ACDEF", "GHIKL"]
        embeddings = extractor.extract_batch_embeddings(seqs)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 3)
        assert embeddings.dtype == np.float32
        mock_extract_batch.assert_called_once_with(seqs)


def test_integration_with_biophysical_extractor(
    extractor: BiophysicalEmbeddingExtractor,
) -> None:
    # Test with actual calculation (no mocking)
    seq = "ACDEF"  # Valid sequence
    embedding = extractor.embed_sequence(seq)
    assert embedding.shape == (3,)
    assert not np.isnan(embedding).any()


def test_integration_batch_with_biophysical_extractor(
    extractor: BiophysicalEmbeddingExtractor,
) -> None:
    # Test with actual calculation (no mocking)
    seqs = ["ACDEF", "GHIKL"]
    embeddings = extractor.extract_batch_embeddings(seqs)
    assert embeddings.shape == (2, 3)
    assert not np.isnan(embeddings).any()
