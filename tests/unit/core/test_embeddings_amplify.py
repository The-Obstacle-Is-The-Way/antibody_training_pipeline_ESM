#!/usr/bin/env python3
"""
Unit Tests for AMPLIFYEmbeddingExtractor

Tests AMPLIFY-specific functionality: batch_size=1 enforcement, trust_remote_code, etc.
Philosophy: TDD - write tests first, then implement.

Date: 2025-11-23
Coverage Target: 90%+
"""

import logging
from typing import Any

import numpy as np
import pytest

# This import will fail initially, which is expected in TDD
from antibody_training_esm.core.embeddings_amplify import AMPLIFYEmbeddingExtractor

# ============================================================================
# Initialization Tests
# ============================================================================


@pytest.mark.unit
def test_amplify_initializes_with_model_name(mock_transformers_model: Any) -> None:
    """Verify AMPLIFY extractor initializes with model name"""
    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M", device="cpu", batch_size=1
    )

    assert extractor.model_name == "chandar-lab/AMPLIFY_350M"
    assert extractor.device == "cpu"
    assert extractor.batch_size == 1


@pytest.mark.unit
def test_amplify_forces_batch_size_one(
    mock_transformers_model: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """CRITICAL: Verify batch_size > 1 is forced to 1 with warning"""
    with caplog.at_level(logging.WARNING):
        extractor = AMPLIFYEmbeddingExtractor(
            model_name="chandar-lab/AMPLIFY_350M",
            device="cpu",
            batch_size=8,  # Try to set > 1
        )

    # Should be forced to 1
    assert extractor.batch_size == 1

    # Should log warning about padding bug
    assert "AMPLIFY PADDING BUG" in caplog.text
    assert "batch_size=8 requested" in caplog.text


@pytest.mark.unit
def test_amplify_accepts_batch_size_one_without_warning(
    mock_transformers_model: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify batch_size=1 doesn't trigger warning"""
    with caplog.at_level(logging.WARNING):
        extractor = AMPLIFYEmbeddingExtractor(
            model_name="chandar-lab/AMPLIFY_350M", device="cpu", batch_size=1
        )

    assert extractor.batch_size == 1
    assert "PADDING BUG" not in caplog.text


# ============================================================================
# Embedding Dimension Tests (960d vs ESM's 1280d)
# ============================================================================


@pytest.mark.unit
def test_amplify_returns_960_dim_vector(
    mock_transformers_model: Any, valid_sequences: dict[str, str]
) -> None:
    """Verify AMPLIFY returns 960-d embeddings (not 1280-d like ESM)"""
    # Mock needs to return 960-d for AMPLIFY
    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M", device="cpu"
    )

    embedding = extractor.embed_sequence(valid_sequences["VH"])

    # AMPLIFY returns 960-d (not 1280-d)
    assert embedding.shape == (960,)
    assert isinstance(embedding, np.ndarray)


@pytest.mark.unit
def test_amplify_batch_returns_correct_shape(
    mock_transformers_model: Any, valid_sequences: dict[str, Any]
) -> None:
    """Verify batch extraction returns (n, 960) array"""
    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M", device="cpu"
    )

    sequences: list[str] = valid_sequences["batch"]  # 5 sequences
    embeddings = extractor.extract_batch_embeddings(sequences)

    assert embeddings.shape == (5, 960)


# ============================================================================
# Device-Specific Attention Implementation Tests
# ============================================================================


@pytest.mark.unit
def test_amplify_uses_sdpa_for_mps(
    mock_transformers_model: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify MPS device triggers SDPA attention (not Flash Attention)"""
    with caplog.at_level(logging.INFO):
        _extractor = AMPLIFYEmbeddingExtractor(
            model_name="chandar-lab/AMPLIFY_350M", device="mps"
        )

    assert "Using SDPA attention for MPS" in caplog.text
    # Silence unused variable check by referencing it or just _ prefix
    assert _extractor.device == "mps"


@pytest.mark.unit
def test_amplify_uses_eager_for_cpu(
    mock_transformers_model: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify CPU device triggers eager attention"""
    with caplog.at_level(logging.INFO):
        _extractor = AMPLIFYEmbeddingExtractor(
            model_name="chandar-lab/AMPLIFY_350M", device="cpu"
        )

    assert "Using eager attention for CPU" in caplog.text
    assert _extractor.device == "cpu"


@pytest.mark.unit
def test_amplify_uses_auto_for_cuda(mock_transformers_model: Any) -> None:
    """Verify CUDA device uses auto-detection (Flash Attention if available)"""
    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M", device="cuda"
    )

    # attn_impl should be None (auto-detect) or whatever transformers defaults to
    # Just check it doesn't crash
    assert extractor.device == "cuda"


# ============================================================================
# Validation Tests (Reuse ESM patterns)
# ============================================================================


@pytest.mark.unit
def test_amplify_rejects_invalid_amino_acids(mock_transformers_model: Any) -> None:
    """Verify AMPLIFY rejects sequences with invalid characters"""
    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M", device="cpu"
    )

    with pytest.raises(ValueError, match="Invalid amino acid"):
        extractor.embed_sequence(
            "QVQLBZX"
        )  # B and Z are invalid (X is usually valid in some contexts but let's stick to standard)


@pytest.mark.unit
def test_amplify_rejects_empty_sequence(mock_transformers_model: Any) -> None:
    """Verify AMPLIFY rejects empty sequences"""
    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M", device="cpu"
    )

    with pytest.raises(ValueError, match="too short"):
        extractor.embed_sequence("")


@pytest.mark.unit
def test_amplify_logs_progress_every_100_sequences(
    mock_transformers_model: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify progress logging for large batches"""
    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M", device="cpu"
    )

    # Create 150 sequences
    sequences = ["QVQL"] * 150

    with caplog.at_level(logging.INFO):
        _embeddings = extractor.extract_batch_embeddings(sequences)

    # Should log at 100
    assert "Processed 100/150" in caplog.text
    assert _embeddings.shape == (150, 960)
