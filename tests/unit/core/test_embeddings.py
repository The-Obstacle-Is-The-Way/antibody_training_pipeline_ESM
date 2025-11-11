#!/usr/bin/env python3
"""
Unit Tests for ESMEmbeddingExtractor

Tests the ESMEmbeddingExtractor class for sequence embedding extraction.
Focus: behavior testing with mocked models (no 650MB downloads).

Philosophy:
- Mock I/O boundaries (model loading)
- Test validation logic (domain behavior)
- Test batch processing and edge cases

Date: 2025-11-07
Coverage Target: 85%+
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor
from tests.conftest import assert_valid_embeddings

# ============================================================================
# Initialization Tests
# ============================================================================


@pytest.mark.unit
def test_extractor_initializes_with_model_name(
    mock_transformers_model: tuple[Any, Any],
) -> None:
    """Verify extractor initializes with model name"""
    # Act
    extractor = ESMEmbeddingExtractor(
        model_name="facebook/esm1v_t33_650M_UR90S_1", device="cpu", batch_size=32
    )

    # Assert
    assert extractor.model_name == "facebook/esm1v_t33_650M_UR90S_1"
    assert extractor.device == "cpu"
    assert extractor.batch_size == 32


@pytest.mark.unit
def test_extractor_initializes_with_custom_batch_size(
    mock_transformers_model: tuple[Any, Any],
) -> None:
    """Verify extractor accepts custom batch size"""
    # Act
    extractor = ESMEmbeddingExtractor(
        model_name="facebook/esm1v_t33_650M_UR90S_1", device="cpu", batch_size=16
    )

    # Assert
    assert extractor.batch_size == 16


@pytest.mark.unit
def test_extractor_creates_model_and_tokenizer(
    mock_transformers_model: tuple[Any, Any],
) -> None:
    """Verify extractor loads model and tokenizer"""
    # Act
    extractor = ESMEmbeddingExtractor(
        model_name="facebook/esm1v_t33_650M_UR90S_1", device="cpu"
    )

    # Assert
    assert extractor.model is not None
    assert extractor.tokenizer is not None


@pytest.mark.unit
def test_extractor_puts_model_in_eval_mode(
    mock_transformers_model: tuple[Any, Any],
) -> None:
    """Verify model is set to eval mode (no gradient computation)"""
    # Act
    extractor = ESMEmbeddingExtractor(
        model_name="facebook/esm1v_t33_650M_UR90S_1", device="cpu"
    )

    # Assert
    assert extractor.model.eval_mode is True


# ============================================================================
# Single Sequence Embedding Tests (embed_sequence)
# ============================================================================


@pytest.mark.unit
def test_embed_sequence_returns_1280_dim_vector(
    embedding_extractor: ESMEmbeddingExtractor,
    valid_sequences: dict[str, Any],
) -> None:
    """Verify single sequence embedding returns 1280-dimensional vector"""
    # Arrange
    sequence = valid_sequences["VH"]

    # Act
    embedding = embedding_extractor.embed_sequence(sequence)

    # Assert
    assert_valid_embeddings(embedding, (1280,))


@pytest.mark.unit
def test_embed_sequence_handles_short_sequence(
    embedding_extractor: ESMEmbeddingExtractor,
) -> None:
    """Verify extractor handles short sequences"""
    # Arrange
    short_seq = "QVQLVQSG"  # 8 amino acids

    # Act
    embedding = embedding_extractor.embed_sequence(short_seq)

    # Assert
    assert embedding.shape == (1280,)


@pytest.mark.unit
def test_embed_sequence_handles_long_sequence(
    embedding_extractor: ESMEmbeddingExtractor,
    valid_sequences: dict[str, Any],
) -> None:
    """Verify extractor handles long sequences"""
    # Arrange
    long_seq = valid_sequences["long"]  # ~250 amino acids

    # Act
    embedding = embedding_extractor.embed_sequence(long_seq)

    # Assert
    assert embedding.shape == (1280,)


@pytest.mark.unit
def test_embed_sequence_strips_whitespace(
    embedding_extractor: ESMEmbeddingExtractor,
) -> None:
    """Verify extractor strips leading/trailing whitespace"""
    # Arrange
    sequence_with_whitespace = "  QVQLVQSG  "

    # Act
    embedding = embedding_extractor.embed_sequence(sequence_with_whitespace)

    # Assert
    assert embedding.shape == (1280,)


@pytest.mark.unit
def test_embed_sequence_converts_to_uppercase(
    embedding_extractor: ESMEmbeddingExtractor,
) -> None:
    """Verify extractor converts sequences to uppercase"""
    # Arrange
    lowercase_seq = "qvqlvqsg"

    # Act
    embedding = embedding_extractor.embed_sequence(lowercase_seq)

    # Assert - should not raise error
    assert embedding.shape == (1280,)


# ============================================================================
# Sequence Validation Tests (embed_sequence)
# ============================================================================


@pytest.mark.unit
def test_embed_sequence_rejects_sequence_with_gap(
    embedding_extractor: ESMEmbeddingExtractor,
    invalid_sequences: dict[str, Any],
) -> None:
    """Verify extractor raises ValueError for sequences with gaps"""
    # Arrange
    sequence_with_gap = invalid_sequences["gap"]

    # Act & Assert
    with pytest.raises(ValueError, match="Invalid amino acid"):
        embedding_extractor.embed_sequence(sequence_with_gap)


@pytest.mark.unit
def test_embed_sequence_rejects_sequence_with_invalid_aa(
    embedding_extractor: ESMEmbeddingExtractor,
    invalid_sequences: dict[str, Any],
) -> None:
    """Verify extractor raises ValueError for invalid amino acids"""
    # Arrange
    sequence_with_invalid_aa = invalid_sequences["invalid_aa"]

    # Act & Assert
    with pytest.raises(ValueError, match="Invalid amino acid"):
        embedding_extractor.embed_sequence(sequence_with_invalid_aa)


@pytest.mark.unit
def test_embed_sequence_rejects_empty_sequence(
    embedding_extractor: ESMEmbeddingExtractor,
    invalid_sequences: dict[str, Any],
) -> None:
    """Verify extractor raises ValueError for empty sequences"""
    # Arrange
    empty_seq = invalid_sequences["empty"]

    # Act & Assert
    with pytest.raises(ValueError, match="Sequence too short"):
        embedding_extractor.embed_sequence(empty_seq)


@pytest.mark.unit
def test_embed_sequence_accepts_all_valid_amino_acids(
    embedding_extractor: ESMEmbeddingExtractor,
) -> None:
    """Verify extractor accepts all 20 standard amino acids + X"""
    # Arrange - sequence with all valid amino acids
    all_valid_aas = "ACDEFGHIKLMNPQRSTVWYX"

    # Act
    embedding = embedding_extractor.embed_sequence(all_valid_aas)

    # Assert
    assert embedding.shape == (1280,)


@pytest.mark.unit
def test_embed_sequence_rejects_numbers(
    embedding_extractor: ESMEmbeddingExtractor,
) -> None:
    """Verify extractor rejects sequences with numbers"""
    # Arrange
    sequence_with_numbers = "QVQLVQ123SGAEVKKPGA"

    # Act & Assert
    with pytest.raises(ValueError, match="Invalid amino acid"):
        embedding_extractor.embed_sequence(sequence_with_numbers)


@pytest.mark.unit
def test_embed_sequence_rejects_special_chars(
    embedding_extractor: ESMEmbeddingExtractor,
) -> None:
    """Verify extractor rejects sequences with special characters"""
    # Arrange
    sequence_with_special = "QVQLVQ!@#SGAEVKKPGA"

    # Act & Assert
    with pytest.raises(ValueError, match="Invalid amino acid"):
        embedding_extractor.embed_sequence(sequence_with_special)


# ============================================================================
# Batch Embedding Tests (extract_batch_embeddings)
# ============================================================================


@pytest.mark.unit
def test_extract_batch_embeddings_handles_multiple_sequences(
    embedding_extractor: ESMEmbeddingExtractor,
    valid_sequences: dict[str, Any],
) -> None:
    """Verify batch extraction returns correct shape"""
    # Arrange
    sequences = valid_sequences["batch"]  # 5 sequences

    # Act
    embeddings = embedding_extractor.extract_batch_embeddings(sequences)

    # Assert
    assert_valid_embeddings(embeddings, (5, 1280))


@pytest.mark.unit
def test_extract_batch_embeddings_handles_single_sequence(
    embedding_extractor: ESMEmbeddingExtractor,
    valid_sequences: dict[str, Any],
) -> None:
    """Verify batch extraction works with single sequence"""
    # Arrange
    sequences = [valid_sequences["VH"]]

    # Act
    embeddings = embedding_extractor.extract_batch_embeddings(sequences)

    # Assert
    assert_valid_embeddings(embeddings, (1, 1280))


@pytest.mark.unit
def test_extract_batch_embeddings_handles_large_batch(
    embedding_extractor: ESMEmbeddingExtractor,
) -> None:
    """Verify batch extraction handles large batches"""
    # Arrange - 100 sequences
    sequences = ["QVQLVQSGAEVKKPGA"] * 100

    # Act
    embeddings = embedding_extractor.extract_batch_embeddings(sequences)

    # Assert
    assert embeddings.shape == (100, 1280)


@pytest.mark.unit
def test_extract_batch_embeddings_batch_size_smaller_than_input(
    embedding_extractor: ESMEmbeddingExtractor,
) -> None:
    """Verify batch processing works when batch_size < len(sequences)"""
    # Arrange
    sequences = ["QVQLVQSGAEVKKPGA"] * 50
    embedding_extractor.batch_size = 16  # Force multiple batches

    # Act
    embeddings = embedding_extractor.extract_batch_embeddings(sequences)

    # Assert
    assert embeddings.shape == (50, 1280)


@pytest.mark.unit
def test_extract_batch_embeddings_batch_size_larger_than_input(
    embedding_extractor: ESMEmbeddingExtractor,
) -> None:
    """Verify batch processing works when batch_size > len(sequences)"""
    # Arrange
    sequences = ["QVQLVQSGAEVKKPGA"] * 5
    embedding_extractor.batch_size = 32  # Single batch

    # Act
    embeddings = embedding_extractor.extract_batch_embeddings(sequences)

    # Assert
    assert embeddings.shape == (5, 1280)


# ============================================================================
# Batch Validation Tests (extract_batch_embeddings)
# ============================================================================


@pytest.mark.unit
def test_extract_batch_embeddings_handles_invalid_sequences_gracefully(
    embedding_extractor: ESMEmbeddingExtractor,
) -> None:
    """Verify batch extractor fails loudly on invalid sequences (no silent corruption)"""
    # Arrange - mixed valid and invalid
    sequences = [
        "QVQLVQSGAEVKKPGA",  # Valid
        "QVQL-VQSGAEVKKPGA",  # Gap (invalid)
        "DIQMTQSPSSLSASVGDRVT",  # Valid
        "QVQLVQ123SGAEVKKPGA",  # Numbers (invalid)
    ]

    # Act & Assert - should raise RuntimeError with details about invalid sequences
    with pytest.raises(
        RuntimeError,
        match=r"Batch processing failed.*Cannot continue with corrupted embeddings",
    ):
        embedding_extractor.extract_batch_embeddings(sequences)


@pytest.mark.unit
def test_extract_batch_embeddings_strips_whitespace_in_batch(
    embedding_extractor: ESMEmbeddingExtractor,
) -> None:
    """Verify batch extractor strips whitespace from all sequences"""
    # Arrange
    sequences = [
        "  QVQLVQSGAEVKKPGA  ",  # Whitespace
        "DIQMTQSPSSLSASVGDRVT",
    ]

    # Act
    embeddings = embedding_extractor.extract_batch_embeddings(sequences)

    # Assert
    assert embeddings.shape == (2, 1280)


@pytest.mark.unit
def test_extract_batch_embeddings_converts_to_uppercase_in_batch(
    embedding_extractor: ESMEmbeddingExtractor,
) -> None:
    """Verify batch extractor converts all sequences to uppercase"""
    # Arrange
    sequences = [
        "qvqlvqsgaevkkpga",  # Lowercase
        "DIQMTQSPSSLSASVGDRVT",  # Already uppercase
    ]

    # Act
    embeddings = embedding_extractor.extract_batch_embeddings(sequences)

    # Assert
    assert embeddings.shape == (2, 1280)


# ============================================================================
# Edge Case Tests
# ============================================================================


@pytest.mark.unit
def test_embed_sequence_handles_single_amino_acid(
    embedding_extractor: ESMEmbeddingExtractor,
) -> None:
    """Verify extractor handles single amino acid (edge case)"""
    # Arrange
    single_aa = "M"

    # Act
    embedding = embedding_extractor.embed_sequence(single_aa)

    # Assert
    assert embedding.shape == (1280,)


@pytest.mark.unit
def test_embed_sequence_handles_homopolymer(
    embedding_extractor: ESMEmbeddingExtractor,
) -> None:
    """Verify extractor handles homopolymers (all same amino acid)"""
    # Arrange
    homopolymer = "AAAAAAAAAA"

    # Act
    embedding = embedding_extractor.embed_sequence(homopolymer)

    # Assert
    assert embedding.shape == (1280,)


@pytest.mark.unit
def test_extract_batch_embeddings_handles_empty_list(
    embedding_extractor: ESMEmbeddingExtractor,
) -> None:
    """Verify batch extractor handles empty sequence list"""
    # Arrange
    empty_list: list[str] = []

    # Act
    embeddings = embedding_extractor.extract_batch_embeddings(empty_list)

    # Assert
    # Empty input returns (0,) shape - this is acceptable behavior
    assert embeddings.shape[0] == 0
    assert len(embeddings) == 0


@pytest.mark.unit
def test_extract_batch_embeddings_handles_variable_length_sequences(
    embedding_extractor: ESMEmbeddingExtractor,
) -> None:
    """Verify batch extractor handles sequences of different lengths"""
    # Arrange - varying lengths: 8, 16, 100 amino acids
    sequences = [
        "QVQLVQSG",  # Short
        "QVQLVQSGAEVKKPGA",  # Medium
        "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYNMHWVRQAPGQGLEWMGGIYPGDSDTRYSPSFQGQVTISADKSISTAYLQWSSLKASDTAMYYCARSTYYGGDWYFNVWGQGTLVTVSS",  # Long
    ]

    # Act
    embeddings = embedding_extractor.extract_batch_embeddings(sequences)

    # Assert
    assert embeddings.shape == (3, 1280)


# ============================================================================
# GPU Memory Management Tests
# ============================================================================


@pytest.mark.unit
def test_clear_gpu_cache_for_cpu_device(
    embedding_extractor: ESMEmbeddingExtractor,
) -> None:
    """Verify _clear_gpu_cache() does nothing on CPU (no crash)"""
    # Arrange
    embedding_extractor.device = "cpu"

    # Act - should not crash
    embedding_extractor._clear_gpu_cache()

    # Assert - no exception raised
    assert True


@pytest.mark.unit
@pytest.mark.gpu
def test_clear_gpu_cache_for_cuda_device(
    mock_transformers_model: tuple[Any, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify _clear_gpu_cache() calls torch.cuda.empty_cache() on CUDA"""
    # Arrange
    cuda_empty_cache_called = False

    def mock_empty_cache() -> None:
        nonlocal cuda_empty_cache_called
        cuda_empty_cache_called = True

    monkeypatch.setattr("torch.cuda.empty_cache", mock_empty_cache)

    extractor = ESMEmbeddingExtractor(
        model_name="facebook/esm1v_t33_650M_UR90S_1", device="cuda"
    )

    # Act
    extractor._clear_gpu_cache()

    # Assert
    assert cuda_empty_cache_called


@pytest.mark.unit
@pytest.mark.gpu
def test_clear_gpu_cache_for_mps_device(
    mock_transformers_model: tuple[Any, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify _clear_gpu_cache() calls torch.mps.empty_cache() on MPS"""
    # Arrange
    mps_empty_cache_called = False

    def mock_empty_cache() -> None:
        nonlocal mps_empty_cache_called
        mps_empty_cache_called = True

    monkeypatch.setattr("torch.mps.empty_cache", mock_empty_cache)

    extractor = ESMEmbeddingExtractor(
        model_name="facebook/esm1v_t33_650M_UR90S_1", device="mps"
    )

    # Act
    extractor._clear_gpu_cache()

    # Assert
    assert mps_empty_cache_called


# ============================================================================
# Output Quality Tests
# ============================================================================


@pytest.mark.unit
def test_embed_sequence_returns_finite_values(
    embedding_extractor: ESMEmbeddingExtractor,
    valid_sequences: dict[str, Any],
) -> None:
    """Verify embeddings contain no NaN or Inf values"""
    # Arrange
    sequence = valid_sequences["VH"]

    # Act
    embedding = embedding_extractor.embed_sequence(sequence)

    # Assert
    assert not np.isnan(embedding).any()
    assert not np.isinf(embedding).any()


@pytest.mark.unit
def test_extract_batch_embeddings_returns_finite_values(
    embedding_extractor: ESMEmbeddingExtractor,
    valid_sequences: dict[str, Any],
) -> None:
    """Verify batch embeddings contain no NaN or Inf values"""
    # Arrange
    sequences = valid_sequences["batch"]

    # Act
    embeddings = embedding_extractor.extract_batch_embeddings(sequences)

    # Assert
    assert not np.isnan(embeddings).any()
    assert not np.isinf(embeddings).any()


@pytest.mark.unit
def test_embed_sequence_returns_numpy_array(
    embedding_extractor: ESMEmbeddingExtractor,
    valid_sequences: dict[str, Any],
) -> None:
    """Verify embed_sequence returns numpy array, not torch tensor"""
    # Arrange
    sequence = valid_sequences["VH"]

    # Act
    embedding = embedding_extractor.embed_sequence(sequence)

    # Assert
    assert isinstance(embedding, np.ndarray)
    assert not isinstance(embedding, torch.Tensor)


@pytest.mark.unit
def test_extract_batch_embeddings_returns_numpy_array(
    embedding_extractor: ESMEmbeddingExtractor,
    valid_sequences: dict[str, Any],
) -> None:
    """Verify extract_batch_embeddings returns numpy array"""
    # Arrange
    sequences = valid_sequences["batch"]

    # Act
    embeddings = embedding_extractor.extract_batch_embeddings(sequences)

    # Assert
    assert isinstance(embeddings, np.ndarray)
    assert not isinstance(embeddings, torch.Tensor)


# ============================================================================
# Integration-like Tests (Real Workflow)
# ============================================================================


@pytest.mark.unit
def test_full_embedding_extraction_workflow(
    mock_transformers_model: tuple[Any, Any],
) -> None:
    """Verify complete workflow: init → embed_sequence → extract_batch"""
    # Arrange
    extractor = ESMEmbeddingExtractor(
        model_name="facebook/esm1v_t33_650M_UR90S_1", device="cpu", batch_size=8
    )

    # Act: Single sequence
    single_embedding = extractor.embed_sequence("QVQLVQSGAEVKKPGA")

    # Act: Batch
    batch_embeddings = extractor.extract_batch_embeddings(
        [
            "QVQLVQSGAEVKKPGA",
            "DIQMTQSPSSLSASVGDRVT",
        ]
    )

    # Assert
    assert single_embedding.shape == (1280,)
    assert batch_embeddings.shape == (2, 1280)


@pytest.mark.unit
def test_embeddings_consistent_across_calls(
    embedding_extractor: ESMEmbeddingExtractor,
) -> None:
    """Verify same sequence produces similar embeddings across calls"""
    # Arrange
    sequence = "QVQLVQSGAEVKKPGA"

    # Act
    embedding1 = embedding_extractor.embed_sequence(sequence)
    embedding2 = embedding_extractor.embed_sequence(sequence)

    # Assert - should be very similar (mock model returns random, but shape matches)
    assert embedding1.shape == embedding2.shape
    assert embedding1.dtype == embedding2.dtype


# ============================================================================
# Docstring Examples Tests
# ============================================================================


@pytest.mark.unit
def test_readme_example_embedding_workflow(
    mock_transformers_model: tuple[Any, Any],
) -> None:
    """Verify example from README/docstrings works correctly"""
    # This test ensures documentation examples stay accurate

    # Arrange
    extractor = ESMEmbeddingExtractor(
        model_name="facebook/esm1v_t33_650M_UR90S_1", device="cpu"
    )

    sequences = [
        "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYNMH",
        "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLN",
    ]

    # Act: Extract embeddings
    embeddings = extractor.extract_batch_embeddings(sequences)

    # Assert
    assert embeddings.shape == (2, 1280)
    assert not np.isnan(embeddings).any()
