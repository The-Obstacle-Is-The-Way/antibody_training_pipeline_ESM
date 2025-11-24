#!/usr/bin/env python3
"""
Unit Tests for AMPLIFYEmbeddingExtractor

Tests AMPLIFY-specific functionality: batch_size=1 enforcement, trust_remote_code, etc.
Philosophy: TDD - write tests first, then implement.

CRITICAL: AMPLIFY has a padding/batching bug that causes non-reproducible embeddings
when batch_size > 1. See: https://www.nature.com/articles/s41598-025-05674-x

Date: 2025-11-24
Coverage Target: 90%+
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytest

# Import will fail until we implement the class (TDD red phase)
# This is expected - tests are written first!


# ============================================================================
# 1. Initialization Tests
# ============================================================================


@pytest.mark.unit
def test_amplify_initializes_with_model_name(
    mock_transformers_model: tuple[Any, Any],
) -> None:
    """Verify AMPLIFY extractor initializes with model name"""
    from antibody_training_esm.core.embeddings_amplify import AMPLIFYEmbeddingExtractor

    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M",
        device="cpu",
        batch_size=1,
    )

    assert extractor.model_name == "chandar-lab/AMPLIFY_350M"
    assert extractor.device == "cpu"
    assert extractor.batch_size == 1
    assert extractor.revision == "main"


@pytest.mark.unit
def test_amplify_creates_model_and_tokenizer(
    mock_transformers_model: tuple[Any, Any],
) -> None:
    """Verify extractor loads model and tokenizer with trust_remote_code"""
    from antibody_training_esm.core.embeddings_amplify import AMPLIFYEmbeddingExtractor

    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M",
        device="cpu",
    )

    assert extractor.model is not None
    assert extractor.tokenizer is not None
    # Model should be in eval mode
    assert extractor.model.training is False


@pytest.mark.unit
def test_amplify_accepts_custom_revision(
    mock_transformers_model: tuple[Any, Any],
) -> None:
    """Verify extractor accepts custom HuggingFace revision"""
    from antibody_training_esm.core.embeddings_amplify import AMPLIFYEmbeddingExtractor

    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M",
        device="cpu",
        revision="v1.0.0",
    )

    assert extractor.revision == "v1.0.0"


# ============================================================================
# 2. CRITICAL: Batch Size Enforcement Tests (Padding Bug)
# ============================================================================


@pytest.mark.unit
def test_amplify_forces_batch_size_one_with_warning(
    mock_transformers_model: tuple[Any, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """CRITICAL: Verify batch_size > 1 is forced to 1 with warning"""
    from antibody_training_esm.core.embeddings_amplify import AMPLIFYEmbeddingExtractor

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
    assert "Forcing batch_size=1" in caplog.text


@pytest.mark.unit
def test_amplify_accepts_batch_size_one_without_warning(
    mock_transformers_model: tuple[Any, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify batch_size=1 doesn't trigger padding bug warning"""
    from antibody_training_esm.core.embeddings_amplify import AMPLIFYEmbeddingExtractor

    with caplog.at_level(logging.WARNING):
        extractor = AMPLIFYEmbeddingExtractor(
            model_name="chandar-lab/AMPLIFY_350M",
            device="cpu",
            batch_size=1,
        )

    assert extractor.batch_size == 1
    assert "PADDING BUG" not in caplog.text


@pytest.mark.unit
def test_amplify_defaults_to_batch_size_one(
    mock_transformers_model: tuple[Any, Any],
) -> None:
    """Verify default batch_size is 1 (not like ESM's 32)"""
    from antibody_training_esm.core.embeddings_amplify import AMPLIFYEmbeddingExtractor

    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M",
        device="cpu",
    )

    assert extractor.batch_size == 1


# ============================================================================
# 3. Embedding Dimension Tests (960d vs ESM's 1280d)
# ============================================================================


@pytest.mark.unit
def test_amplify_returns_960_dim_vector(
    mock_transformers_model: tuple[Any, Any],
    valid_sequences: dict[str, Any],
) -> None:
    """Verify AMPLIFY returns 960-d embeddings (not 1280-d like ESM)"""
    from antibody_training_esm.core.embeddings_amplify import AMPLIFYEmbeddingExtractor

    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M",
        device="cpu",
    )

    embedding = extractor.embed_sequence(valid_sequences["VH"])

    # AMPLIFY returns 960-d (not 1280-d)
    assert embedding.shape == (960,)
    assert isinstance(embedding, np.ndarray)


@pytest.mark.unit
def test_amplify_batch_returns_correct_shape(
    mock_transformers_model: tuple[Any, Any],
    valid_sequences: dict[str, Any],
) -> None:
    """Verify batch extraction returns (n, 960) array"""
    from antibody_training_esm.core.embeddings_amplify import AMPLIFYEmbeddingExtractor

    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M",
        device="cpu",
    )

    sequences = valid_sequences["batch"]  # 5 sequences
    embeddings = extractor.extract_batch_embeddings(sequences)

    assert embeddings.shape == (5, 960)
    assert isinstance(embeddings, np.ndarray)


@pytest.mark.unit
def test_amplify_single_sequence_batch_returns_correct_shape(
    mock_transformers_model: tuple[Any, Any],
) -> None:
    """Verify single sequence in batch returns (1, 960)"""
    from antibody_training_esm.core.embeddings_amplify import AMPLIFYEmbeddingExtractor

    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M",
        device="cpu",
    )

    embeddings = extractor.extract_batch_embeddings(["QVQLVQSG"])

    assert embeddings.shape == (1, 960)


# ============================================================================
# 4. Device-Specific Attention Implementation Tests
# ============================================================================


@pytest.mark.unit
def test_amplify_uses_sdpa_for_mps(
    mock_transformers_model: tuple[Any, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify MPS device triggers SDPA attention (not Flash Attention)"""
    from antibody_training_esm.core.embeddings_amplify import AMPLIFYEmbeddingExtractor

    with caplog.at_level(logging.INFO):
        extractor = AMPLIFYEmbeddingExtractor(
            model_name="chandar-lab/AMPLIFY_350M",
            device="mps",
        )

    assert "Using SDPA attention for MPS" in caplog.text
    assert extractor.device == "mps"


@pytest.mark.unit
def test_amplify_uses_eager_for_cpu(
    mock_transformers_model: tuple[Any, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify CPU device triggers eager attention"""
    from antibody_training_esm.core.embeddings_amplify import AMPLIFYEmbeddingExtractor

    with caplog.at_level(logging.INFO):
        _extractor = AMPLIFYEmbeddingExtractor(
            model_name="chandar-lab/AMPLIFY_350M",
            device="cpu",
        )

    assert "Using eager attention for CPU" in caplog.text
    assert _extractor is not None  # Silence unused variable warning


@pytest.mark.unit
def test_amplify_allows_auto_for_cuda(
    mock_transformers_model: tuple[Any, Any],
) -> None:
    """Verify CUDA device uses auto-detection (Flash Attention if available)"""
    from antibody_training_esm.core.embeddings_amplify import AMPLIFYEmbeddingExtractor

    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M",
        device="cuda",
    )

    # Should initialize without error (attention auto-selected)
    assert extractor.device == "cuda"


# ============================================================================
# 5. Sequence Validation Tests
# ============================================================================


@pytest.mark.unit
def test_amplify_rejects_invalid_amino_acids(
    mock_transformers_model: tuple[Any, Any],
) -> None:
    """Verify AMPLIFY rejects sequences with invalid characters"""
    from antibody_training_esm.core.embeddings_amplify import AMPLIFYEmbeddingExtractor

    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M",
        device="cpu",
    )

    with pytest.raises(ValueError, match="Invalid amino acid"):
        extractor.embed_sequence("QVQLBZJ")  # B, Z, J are invalid


@pytest.mark.unit
def test_amplify_rejects_empty_sequence(
    mock_transformers_model: tuple[Any, Any],
) -> None:
    """Verify AMPLIFY rejects empty sequences"""
    from antibody_training_esm.core.embeddings_amplify import AMPLIFYEmbeddingExtractor

    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M",
        device="cpu",
    )

    with pytest.raises(ValueError, match="too short"):
        extractor.embed_sequence("")


@pytest.mark.unit
def test_amplify_accepts_valid_amino_acids(
    mock_transformers_model: tuple[Any, Any],
) -> None:
    """Verify AMPLIFY accepts all 20 standard amino acids + X"""
    from antibody_training_esm.core.embeddings_amplify import AMPLIFYEmbeddingExtractor

    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M",
        device="cpu",
    )

    # All 20 standard + X (unknown)
    valid_seq = "ACDEFGHIKLMNPQRSTVWYX"
    embedding = extractor.embed_sequence(valid_seq)

    assert embedding.shape == (960,)


@pytest.mark.unit
def test_amplify_handles_lowercase_sequences(
    mock_transformers_model: tuple[Any, Any],
) -> None:
    """Verify AMPLIFY converts lowercase to uppercase"""
    from antibody_training_esm.core.embeddings_amplify import AMPLIFYEmbeddingExtractor

    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M",
        device="cpu",
    )

    # Both should work without error
    embedding_upper = extractor.embed_sequence("QVQLVQSG")
    embedding_lower = extractor.embed_sequence("qvqlvqsg")

    # Shapes should match (values differ due to mock randomness)
    assert embedding_upper.shape == embedding_lower.shape == (960,)


# ============================================================================
# 6. Edge Cases & Error Handling
# ============================================================================


@pytest.mark.unit
def test_amplify_handles_very_long_sequence(
    mock_transformers_model: tuple[Any, Any],
) -> None:
    """Verify AMPLIFY truncates sequences longer than max_length"""
    from antibody_training_esm.core.embeddings_amplify import AMPLIFYEmbeddingExtractor

    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M",
        device="cpu",
        max_length=512,
    )

    # Create sequence longer than max_length
    long_seq = "A" * 1000

    # Should truncate without error
    embedding = extractor.embed_sequence(long_seq)
    assert embedding.shape == (960,)


@pytest.mark.unit
def test_amplify_batch_processes_one_at_a_time(
    mock_transformers_model: tuple[Any, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify batch extraction processes sequences individually (batch_size=1)"""
    from antibody_training_esm.core.embeddings_amplify import AMPLIFYEmbeddingExtractor

    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M",
        device="cpu",
    )

    sequences = ["QVQL", "ACDG", "EFGH"]

    with caplog.at_level(logging.INFO):
        embeddings = extractor.extract_batch_embeddings(sequences)

    # Should log that it's processing with batch_size=1
    assert "batch_size=1 due to padding bug" in caplog.text
    assert embeddings.shape == (3, 960)


@pytest.mark.unit
def test_amplify_batch_fails_gracefully_on_invalid_sequence(
    mock_transformers_model: tuple[Any, Any],
) -> None:
    """Verify batch extraction fails with clear error on invalid sequence"""
    from antibody_training_esm.core.embeddings_amplify import AMPLIFYEmbeddingExtractor

    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M",
        device="cpu",
    )

    sequences = ["QVQL", "INVALID123", "ACDG"]

    with pytest.raises(RuntimeError, match="Embedding extraction failed at sequence 1"):
        extractor.extract_batch_embeddings(sequences)


# ============================================================================
# 7. Progress Logging Tests
# ============================================================================


@pytest.mark.unit
def test_amplify_logs_progress_every_100_sequences(
    mock_transformers_model: tuple[Any, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify progress logging for large batches"""
    from antibody_training_esm.core.embeddings_amplify import AMPLIFYEmbeddingExtractor

    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M",
        device="cpu",
    )

    # Create 150 sequences
    sequences = ["QVQLVQSG"] * 150

    with caplog.at_level(logging.INFO):
        embeddings = extractor.extract_batch_embeddings(sequences)

    # Should log at 100
    assert "Processed 100/150" in caplog.text
    assert embeddings.shape == (150, 960)


# ============================================================================
# 8. GPU Cache Management Tests
# ============================================================================


@pytest.mark.unit
def test_amplify_clear_gpu_cache_for_cpu_device(
    mock_transformers_model: tuple[Any, Any],
) -> None:
    """Verify _clear_gpu_cache() does nothing on CPU (no crash)"""
    from antibody_training_esm.core.embeddings_amplify import AMPLIFYEmbeddingExtractor

    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M",
        device="cpu",
    )

    # Should not crash
    extractor._clear_gpu_cache()


@pytest.mark.unit
@pytest.mark.gpu
def test_amplify_clear_gpu_cache_for_cuda_device(
    mock_transformers_model: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify _clear_gpu_cache() calls torch.cuda.empty_cache() on CUDA"""
    from antibody_training_esm.core.embeddings_amplify import AMPLIFYEmbeddingExtractor

    cuda_empty_cache_called = False

    def mock_empty_cache() -> None:
        nonlocal cuda_empty_cache_called
        cuda_empty_cache_called = True

    monkeypatch.setattr("torch.cuda.empty_cache", mock_empty_cache)

    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M",
        device="cuda",
    )

    extractor._clear_gpu_cache()

    assert cuda_empty_cache_called


@pytest.mark.unit
@pytest.mark.gpu
def test_amplify_clear_gpu_cache_for_mps_device(
    mock_transformers_model: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify _clear_gpu_cache() calls torch.mps.empty_cache() on MPS"""
    from antibody_training_esm.core.embeddings_amplify import AMPLIFYEmbeddingExtractor

    mps_empty_cache_called = False

    def mock_empty_cache() -> None:
        nonlocal mps_empty_cache_called
        mps_empty_cache_called = True

    monkeypatch.setattr("torch.mps.empty_cache", mock_empty_cache)

    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M",
        device="mps",
    )

    extractor._clear_gpu_cache()

    assert mps_empty_cache_called


# ============================================================================
# 9. Output Quality Tests
# ============================================================================


@pytest.mark.unit
def test_amplify_embed_sequence_returns_finite_values(
    mock_transformers_model: tuple[Any, Any],
    valid_sequences: dict[str, Any],
) -> None:
    """Verify embeddings contain no NaN or Inf values"""
    from antibody_training_esm.core.embeddings_amplify import AMPLIFYEmbeddingExtractor

    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M",
        device="cpu",
    )

    embedding = extractor.embed_sequence(valid_sequences["VH"])

    assert not np.isnan(embedding).any()
    assert not np.isinf(embedding).any()


@pytest.mark.unit
def test_amplify_extract_batch_embeddings_returns_finite_values(
    mock_transformers_model: tuple[Any, Any],
    valid_sequences: dict[str, Any],
) -> None:
    """Verify batch embeddings contain no NaN or Inf values"""
    from antibody_training_esm.core.embeddings_amplify import AMPLIFYEmbeddingExtractor

    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M",
        device="cpu",
    )

    embeddings = extractor.extract_batch_embeddings(valid_sequences["batch"])

    assert not np.isnan(embeddings).any()
    assert not np.isinf(embeddings).any()


@pytest.mark.unit
def test_amplify_embed_sequence_returns_numpy_array(
    mock_transformers_model: tuple[Any, Any],
    valid_sequences: dict[str, Any],
) -> None:
    """Verify embed_sequence returns numpy array, not torch tensor"""
    import torch

    from antibody_training_esm.core.embeddings_amplify import AMPLIFYEmbeddingExtractor

    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M",
        device="cpu",
    )

    embedding = extractor.embed_sequence(valid_sequences["VH"])

    assert isinstance(embedding, np.ndarray)
    assert not isinstance(embedding, torch.Tensor)


# ============================================================================
# Total Tests: 27 tests covering all AMPLIFY-specific requirements
# Expected Coverage: ≥90%
# ============================================================================
