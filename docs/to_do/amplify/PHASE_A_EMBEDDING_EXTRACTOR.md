# Phase A: AMPLIFY Embedding Extractor - Vertical Slice Specification

**Date**: 2025-11-23
**Author**: Claude Code (Sonnet 4.5)
**Status**: 🔴 **PENDING SENIOR APPROVAL**
**Methodology**: TDD + Single Responsibility Principle
**Duration**: 2 hours
**Dependencies**: None (fully independent)

---

## 1. Objective

Create `AMPLIFYEmbeddingExtractor` class that handles AMPLIFY 350M protein language model embedding extraction with AMPLIFY-specific requirements (batch_size=1, trust_remote_code, attention workarounds).

**Success Criteria**: Class works in isolation with ≥90% test coverage, passes all unit tests, mypy strict mode clean.

---

## 2. Requirements

| Requirement | Source | Priority | Acceptance Test |
|-------------|--------|----------|-----------------|
| **Enforce batch_size=1** | [Nature Sci Rep](https://www.nature.com/articles/s41598-025-05674-x) | CRITICAL | `test_amplify_forces_batch_size_one()` |
| **Require trust_remote_code=True** | [HuggingFace](https://huggingface.co/chandar-lab/AMPLIFY_350M) | CRITICAL | Model loads without error |
| **MPS: use attn_implementation="sdpa"** | [HuggingFace Forums](https://discuss.huggingface.co/t/97562) | HIGH | `test_amplify_uses_sdpa_for_mps()` |
| **Return 960d embeddings** | [NVIDIA BioNeMo](https://docs.nvidia.com/bionemo-framework/latest/models/amplify/) | HIGH | `test_amplify_returns_960_dim_vector()` |
| **Same interface as ESMEmbeddingExtractor** | Existing codebase | CRITICAL | API compatibility tests |
| **Validate amino acid sequences** | Existing codebase | MEDIUM | `test_amplify_rejects_invalid_amino_acids()` |

---

## 3. Design

### 3.1 Class Diagram

```
┌────────────────────────────────────────────────────┐
│ AMPLIFYEmbeddingExtractor                          │
├────────────────────────────────────────────────────┤
│ + model_name: str                                  │
│ + device: str                                      │
│ + batch_size: int  (ALWAYS 1)                      │
│ + revision: str                                    │
│ + max_length: int                                  │
├────────────────────────────────────────────────────┤
│ + __init__(model_name, device, batch_size, ...)    │
│ + embed_sequence(sequence: str) -> np.ndarray      │
│ + extract_batch_embeddings(sequences) -> np.ndarray│
│ - _clear_gpu_cache() -> None                       │
└────────────────────────────────────────────────────┘
```

### 3.2 Key Differences from ESMEmbeddingExtractor

| Aspect | ESM | AMPLIFY |
|--------|-----|---------|
| **batch_size** | Configurable (default 8) | FORCED to 1 (padding bug) |
| **trust_remote_code** | Not required | **Required** |
| **attn_implementation** | Not specified | Device-dependent (sdpa/eager) |
| **Embedding dimension** | 1280 | **960** |
| **Padding behavior** | Safe with batching | **BUG: embeddings corrupt with batching** |

---

## 4. Implementation (TDD: Write Tests First!)

### 4.1 Test File (Write This FIRST)

**File**: `tests/unit/core/test_embeddings_amplify.py`

```python
#!/usr/bin/env python3
"""
Unit Tests for AMPLIFYEmbeddingExtractor

TDD Approach: Write ALL tests before implementing the class.

Test Categories:
1. Initialization tests (model loading, device selection)
2. Batch size enforcement tests (CRITICAL padding bug)
3. Embedding dimension tests (960d validation)
4. Device-specific attention tests (MPS/CPU/CUDA)
5. Validation tests (invalid sequences, edge cases)

Date: 2025-11-23
Coverage Target: ≥90%
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from antibody_training_esm.core.embeddings_amplify import AMPLIFYEmbeddingExtractor


# ============================================================================
# 1. Initialization Tests
# ============================================================================

@pytest.mark.unit
def test_amplify_initializes_with_model_name(mock_transformers_model):
    """Verify AMPLIFY extractor initializes with model name"""
    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M",
        device="cpu",
        batch_size=1
    )

    assert extractor.model_name == "chandar-lab/AMPLIFY_350M"
    assert extractor.device == "cpu"
    assert extractor.batch_size == 1
    assert extractor.revision == "main"


@pytest.mark.unit
def test_amplify_creates_model_and_tokenizer(mock_transformers_model):
    """Verify extractor loads model and tokenizer with trust_remote_code"""
    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M",
        device="cpu"
    )

    assert extractor.model is not None
    assert extractor.tokenizer is not None
    # Model should be in eval mode
    assert extractor.model.training is False


@pytest.mark.unit
def test_amplify_accepts_custom_revision(mock_transformers_model):
    """Verify extractor accepts custom HuggingFace revision"""
    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M",
        device="cpu",
        revision="v1.0.0"
    )

    assert extractor.revision == "v1.0.0"


# ============================================================================
# 2. CRITICAL: Batch Size Enforcement Tests (Padding Bug)
# ============================================================================

@pytest.mark.unit
def test_amplify_forces_batch_size_one_with_warning(mock_transformers_model, caplog):
    """CRITICAL: Verify batch_size > 1 is forced to 1 with warning"""
    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M",
        device="cpu",
        batch_size=8  # Try to set > 1
    )

    # Should be forced to 1
    assert extractor.batch_size == 1

    # Should log warning about padding bug
    assert "AMPLIFY PADDING BUG" in caplog.text
    assert "batch_size=8 requested" in caplog.text
    assert "Forcing batch_size=1" in caplog.text


@pytest.mark.unit
def test_amplify_accepts_batch_size_one_without_warning(mock_transformers_model, caplog):
    """Verify batch_size=1 doesn't trigger warning"""
    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M",
        device="cpu",
        batch_size=1
    )

    assert extractor.batch_size == 1
    assert "PADDING BUG" not in caplog.text


@pytest.mark.unit
def test_amplify_defaults_to_batch_size_one(mock_transformers_model):
    """Verify default batch_size is 1 (not 8 like ESM)"""
    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M",
        device="cpu"
    )

    assert extractor.batch_size == 1


# ============================================================================
# 3. Embedding Dimension Tests (960d vs ESM's 1280d)
# ============================================================================

@pytest.mark.unit
def test_amplify_returns_960_dim_vector(mock_transformers_model, valid_sequences):
    """Verify AMPLIFY returns 960-d embeddings (not 1280-d like ESM)"""
    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M",
        device="cpu"
    )

    embedding = extractor.embed_sequence(valid_sequences["VH"])

    # AMPLIFY returns 960-d (not 1280-d)
    assert embedding.shape == (960,)
    assert isinstance(embedding, np.ndarray)
    assert embedding.dtype == np.float32 or embedding.dtype == np.float64


@pytest.mark.unit
def test_amplify_batch_returns_correct_shape(mock_transformers_model, valid_sequences):
    """Verify batch extraction returns (n, 960) array"""
    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M",
        device="cpu"
    )

    sequences = valid_sequences["batch"]  # 5 sequences
    embeddings = extractor.extract_batch_embeddings(sequences)

    assert embeddings.shape == (5, 960)
    assert isinstance(embeddings, np.ndarray)


@pytest.mark.unit
def test_amplify_single_sequence_batch_returns_correct_shape(mock_transformers_model):
    """Verify single sequence in batch returns (1, 960)"""
    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M",
        device="cpu"
    )

    embeddings = extractor.extract_batch_embeddings(["QVQLVQSG"])

    assert embeddings.shape == (1, 960)


# ============================================================================
# 4. Device-Specific Attention Implementation Tests
# ============================================================================

@pytest.mark.unit
def test_amplify_uses_sdpa_for_mps(mock_transformers_model, caplog):
    """Verify MPS device triggers SDPA attention (not Flash Attention)"""
    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M",
        device="mps"
    )

    assert "Using SDPA attention for MPS" in caplog.text


@pytest.mark.unit
def test_amplify_uses_eager_for_cpu(mock_transformers_model, caplog):
    """Verify CPU device triggers eager attention"""
    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M",
        device="cpu"
    )

    assert "Using eager attention for CPU" in caplog.text


@pytest.mark.unit
def test_amplify_allows_auto_for_cuda(mock_transformers_model):
    """Verify CUDA device uses auto-detection (Flash Attention if available)"""
    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M",
        device="cuda"
    )

    # Should initialize without error (attention auto-selected)
    assert extractor.device == "cuda"


# ============================================================================
# 5. Sequence Validation Tests
# ============================================================================

@pytest.mark.unit
def test_amplify_rejects_invalid_amino_acids(mock_transformers_model):
    """Verify AMPLIFY rejects sequences with invalid characters"""
    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M",
        device="cpu"
    )

    with pytest.raises(ValueError, match="Invalid amino acid"):
        extractor.embed_sequence("QVQLBZX")  # B and Z are invalid


@pytest.mark.unit
def test_amplify_rejects_empty_sequence(mock_transformers_model):
    """Verify AMPLIFY rejects empty sequences"""
    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M",
        device="cpu"
    )

    with pytest.raises(ValueError, match="too short"):
        extractor.embed_sequence("")


@pytest.mark.unit
def test_amplify_accepts_valid_amino_acids(mock_transformers_model):
    """Verify AMPLIFY accepts all 20 standard amino acids + X"""
    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M",
        device="cpu"
    )

    # All 20 standard + X (unknown)
    valid_seq = "ACDEFGHIKLMNPQRSTVWYX"
    embedding = extractor.embed_sequence(valid_seq)

    assert embedding.shape == (960,)


@pytest.mark.unit
def test_amplify_handles_lowercase_sequences(mock_transformers_model):
    """Verify AMPLIFY converts lowercase to uppercase"""
    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M",
        device="cpu"
    )

    embedding_upper = extractor.embed_sequence("QVQLVQSG")
    embedding_lower = extractor.embed_sequence("qvqlvqsg")

    # Should produce identical embeddings
    np.testing.assert_array_almost_equal(embedding_upper, embedding_lower)


# ============================================================================
# 6. Edge Cases & Error Handling
# ============================================================================

@pytest.mark.unit
def test_amplify_handles_very_long_sequence(mock_transformers_model):
    """Verify AMPLIFY truncates sequences longer than max_length"""
    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M",
        device="cpu",
        max_length=512
    )

    # Create sequence longer than max_length
    long_seq = "A" * 1000

    # Should truncate without error
    embedding = extractor.embed_sequence(long_seq)
    assert embedding.shape == (960,)


@pytest.mark.unit
def test_amplify_batch_processes_one_at_a_time(mock_transformers_model, caplog):
    """Verify batch extraction processes sequences individually (batch_size=1)"""
    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M",
        device="cpu"
    )

    sequences = ["QVQL", "ACDG", "EFGH"]
    embeddings = extractor.extract_batch_embeddings(sequences)

    # Should log that it's processing with batch_size=1
    assert "batch_size=1 due to padding bug" in caplog.text
    assert embeddings.shape == (3, 960)


@pytest.mark.unit
def test_amplify_batch_fails_gracefully_on_invalid_sequence(mock_transformers_model):
    """Verify batch extraction fails with clear error on invalid sequence"""
    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M",
        device="cpu"
    )

    sequences = ["QVQL", "INVALID123", "ACDG"]

    with pytest.raises(RuntimeError, match="Embedding extraction failed at sequence 1"):
        extractor.extract_batch_embeddings(sequences)


# ============================================================================
# 7. Progress Logging Tests
# ============================================================================

@pytest.mark.unit
def test_amplify_logs_progress_every_100_sequences(mock_transformers_model, caplog):
    """Verify progress logging for large batches"""
    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M",
        device="cpu"
    )

    # Create 150 sequences
    sequences = ["QVQL"] * 150
    embeddings = extractor.extract_batch_embeddings(sequences)

    # Should log at 100 and 200
    assert "Processed 100/150" in caplog.text


# ============================================================================
# Total Tests: 20+ tests covering all requirements
# Expected Coverage: ≥90%
# ============================================================================
```

### 4.2 Mock Updates (Support 960d)

**File**: `tests/fixtures/mock_models.py` (Add to existing file)

```python
class MockAMPLIFYModel(MockESMModel):
    """
    Mock AMPLIFY model (960-d instead of 1280-d)

    Usage:
        Mock transformers.AutoModel.from_pretrained to return this
        when model_name contains "AMPLIFY"
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override hidden_size for AMPLIFY
        self.config.hidden_size = 960  # AMPLIFY dimension (not 1280)
        self.training = False  # eval mode


# Update mock_transformers_model fixture to detect AMPLIFY
def mock_automodel(*args, **kwargs):
    model_name = args[0] if args else kwargs.get("model_name", "")

    if "AMPLIFY" in model_name.upper():
        return MockAMPLIFYModel(*args, **kwargs)
    else:
        return MockESMModel(*args, **kwargs)  # ESM-1v/ESM-2
```

### 4.3 Implementation File (Write This SECOND, After Tests Pass)

**File**: `src/antibody_training_esm/core/embeddings_amplify.py`

```python
"""
AMPLIFY Embedding Module

Professional module for AMPLIFY 350M protein sequence embedding extraction.
Handles AMPLIFY-specific requirements: batch_size=1, trust_remote_code, attention workarounds.

CRITICAL WARNING:
    AMPLIFY has a padding/batching bug that causes non-reproducible embeddings
    when batch_size > 1. This module enforces batch_size=1.

    Source: https://www.nature.com/articles/s41598-025-05674-x

    "When processing a batch of multiple sequences with different lengths,
    shorter sequences need to be padded to the maximum length, and this
    padding should not affect computed embeddings, but if a transformer
    model does not properly mask padded sites when calculating attention
    then the padding can influence output embeddings, which will result
    in poor reproducibility."

Date: 2025-11-23
Author: Claude Code (Sonnet 4.5)
"""

import logging
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .config import (
    DEFAULT_MAX_SEQ_LENGTH,
    SEQUENCE_PREVIEW_LENGTH,
)

logger = logging.getLogger(__name__)


class AMPLIFYEmbeddingExtractor:
    """
    Extract AMPLIFY 350M embeddings for protein sequences.

    CRITICAL WARNING: AMPLIFY has a padding/batching reproducibility issue.
    This class enforces batch_size=1 for consistent results.

    Key Differences from ESMEmbeddingExtractor:
        1. Requires trust_remote_code=True (AMPLIFY uses custom modeling code)
        2. Requires attn_implementation workaround for MPS (Flash Attention is CUDA-only)
        3. Forces batch_size=1 (padding bug causes non-reproducible embeddings)
        4. Returns 960-d embeddings (vs ESM's 1280-d)

    Example:
        >>> extractor = AMPLIFYEmbeddingExtractor(
        ...     model_name="chandar-lab/AMPLIFY_350M",
        ...     device="mps"
        ... )
        >>> embedding = extractor.embed_sequence("QVQLVQSG")
        >>> embedding.shape
        (960,)
    """

    def __init__(
        self,
        model_name: str,
        device: str,
        batch_size: int = 1,
        max_length: int = DEFAULT_MAX_SEQ_LENGTH,
        revision: str = "main",
    ):
        """
        Initialize AMPLIFY embedding extractor

        Args:
            model_name: HuggingFace model identifier (e.g., 'chandar-lab/AMPLIFY_350M')
            device: Device to run model on ('cpu', 'cuda', or 'mps')
            batch_size: MUST be 1 due to AMPLIFY padding bug (forced if > 1)
            max_length: Maximum sequence length for tokenizer truncation
            revision: HuggingFace model revision (commit SHA or branch name)

        Raises:
            ImportError: If transformers library not installed
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.revision = revision

        # CRITICAL: Force batch_size=1 due to padding bug
        if batch_size != 1:
            logger.warning(
                f"⚠️  AMPLIFY PADDING BUG: batch_size={batch_size} requested but AMPLIFY has "
                f"non-reproducible embeddings with batching (Nature Sci Rep 2025). "
                f"Forcing batch_size=1. See: https://www.nature.com/articles/s41598-025-05674-x"
            )
            batch_size = 1
        self.batch_size = batch_size

        # Determine attention implementation based on device
        # Flash Attention is CUDA-only; MPS/CPU require workarounds
        attn_impl: str | None = None  # Auto-detect for CUDA
        if device == "mps":
            attn_impl = "sdpa"  # Scaled Dot-Product Attention (MPS-compatible)
            logger.info("Using SDPA attention for MPS (Flash Attention not supported)")
        elif device == "cpu":
            attn_impl = "eager"  # Standard attention for CPU
            logger.info("Using eager attention for CPU")

        # Load model with AMPLIFY-specific flags
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,  # REQUIRED for AMPLIFY
            attn_implementation=attn_impl,
            output_hidden_states=True,
            revision=revision,
        )
        self.model.to(device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,  # REQUIRED for AMPLIFY
            revision=revision,
        )

        logger.info(
            f"AMPLIFY model {model_name} (revision={revision}) loaded on {device} "
            f"with batch_size=1 (padding bug), max_length={max_length}, "
            f"attn_implementation={attn_impl or 'auto'}"
        )

    def embed_sequence(self, sequence: str) -> np.ndarray:
        """
        Extract AMPLIFY embedding for a single protein sequence

        Args:
            sequence: Amino acid sequence string (case-insensitive)

        Returns:
            Embedding vector as numpy array (960-d)

        Raises:
            ValueError: If sequence contains invalid amino acids or is too short

        Example:
            >>> embedding = extractor.embed_sequence("QVQLVQSG")
            >>> embedding.shape
            (960,)
        """
        # Validation (same logic as ESM)
        valid_aas = set("ACDEFGHIKLMNPQRSTVWYX")
        sequence = sequence.upper().strip()

        if not all(aa in valid_aas for aa in sequence):
            raise ValueError("Invalid amino acid characters in sequence")

        if len(sequence) < 1:
            raise ValueError("Sequence too short")

        # Tokenize
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            embeddings = outputs.hidden_states[-1]  # (1, seq_len, 960)

            # Masked mean pooling (exclude CLS/SEP special tokens)
            attention_mask = inputs["attention_mask"].unsqueeze(-1)  # (1, seq_len, 1)
            attention_mask[:, 0, :] = 0  # Mask CLS token
            attention_mask[:, -1, :] = 0  # Mask SEP token

            masked_embeddings = embeddings * attention_mask
            sum_embeddings = masked_embeddings.sum(dim=1)  # (1, 960)
            sum_mask = attention_mask.sum(dim=1)  # (1, 1)

            # Prevent division by zero
            if sum_mask.item() == 0:
                raise ValueError(
                    f"Attention mask is all zeros for sequence (length: {len(sequence)}). "
                    f"Sequence preview: '{sequence[:SEQUENCE_PREVIEW_LENGTH]}...'"
                )

            mean_embeddings = sum_embeddings / sum_mask  # (1, 960)
            result: np.ndarray = mean_embeddings.squeeze(0).cpu().numpy()
            return result

    def extract_batch_embeddings(self, sequences: list[str]) -> np.ndarray:
        """
        Extract embeddings for multiple sequences.

        CRITICAL: Due to AMPLIFY padding bug, this processes sequences one at a time
        (batch_size=1) to ensure reproducibility. This is ~8× slower than batched ESM.

        Args:
            sequences: List of amino acid sequence strings

        Returns:
            Array of embeddings with shape (n_sequences, 960)

        Raises:
            RuntimeError: If any sequence fails to embed

        Example:
            >>> sequences = ["QVQL", "ACDG", "EFGH"]
            >>> embeddings = extractor.extract_batch_embeddings(sequences)
            >>> embeddings.shape
            (3, 960)
        """
        embeddings_list: list[np.ndarray] = []

        logger.info(
            f"Extracting AMPLIFY embeddings for {len(sequences)} sequences "
            f"(batch_size=1 due to padding bug, this will be slow)..."
        )

        # Process one at a time (batch_size=1)
        for idx, seq in enumerate(sequences):
            try:
                emb = self.embed_sequence(seq)
                embeddings_list.append(emb)

                # Progress logging every 100 sequences
                if (idx + 1) % 100 == 0:
                    logger.info(f"Processed {idx + 1}/{len(sequences)} sequences...")

            except Exception as e:
                logger.error(
                    f"Failed to process sequence {idx}: {seq[:50]}... - {e}"
                )
                raise RuntimeError(
                    f"Embedding extraction failed at sequence {idx}. Cannot continue."
                ) from e

        return np.array(embeddings_list)

    def _clear_gpu_cache(self) -> None:
        """Clear GPU cache for CUDA or MPS devices to prevent memory leaks"""
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()
        elif str(self.device).startswith("mps"):
            torch.mps.empty_cache()
```

---

## 5. TDD Workflow (Step-by-Step)

### Step 1: Write All Tests (30 minutes)
```bash
# Create test file with all 20+ tests
vim tests/unit/core/test_embeddings_amplify.py

# Verify tests FAIL (red phase)
uv run pytest tests/unit/core/test_embeddings_amplify.py -v
# Expected: All tests fail with "ModuleNotFoundError: No module named 'embeddings_amplify'"
```

### Step 2: Implement Minimal Class (60 minutes)
```bash
# Create implementation file
vim src/antibody_training_esm/core/embeddings_amplify.py

# Implement just enough to make tests pass (green phase)
# Focus on one test at a time:
# 1. Initialization tests
# 2. Batch size enforcement
# 3. Embedding dimensions
# 4. Device-specific attention
# 5. Validation
```

### Step 3: Refactor (30 minutes)
```bash
# Clean up code, extract common logic, add docstrings
# Run tests after each refactor to ensure they still pass

uv run pytest tests/unit/core/test_embeddings_amplify.py -v
# Expected: All tests pass (green)
```

### Step 4: Type Check & Lint
```bash
# Type check with mypy
uv run mypy src/antibody_training_esm/core/embeddings_amplify.py --strict

# Format & lint with ruff
uv run ruff format src/antibody_training_esm/core/embeddings_amplify.py
uv run ruff check src/antibody_training_esm/core/embeddings_amplify.py
```

### Step 5: Coverage Check
```bash
# Measure test coverage
uv run pytest tests/unit/core/test_embeddings_amplify.py \
    --cov=src/antibody_training_esm/core/embeddings_amplify \
    --cov-report=term-missing \
    --cov-fail-under=90

# Expected: ≥90% coverage
```

---

## 6. Acceptance Criteria (Definition of Done)

- [ ] All 20+ unit tests written (TDD: tests first!)
- [ ] `AMPLIFYEmbeddingExtractor` class implemented
- [ ] All tests pass (`pytest` green)
- [ ] Test coverage ≥ 90% (`--cov-fail-under=90`)
- [ ] Type annotations 100% complete
- [ ] `mypy --strict` passes with zero errors
- [ ] `ruff format` clean (no changes needed)
- [ ] `ruff check` clean (no warnings)
- [ ] Batch size forced to 1 with warning logged
- [ ] Returns 960-d embeddings (not 1280-d)
- [ ] MPS uses SDPA attention (logged)
- [ ] CPU uses eager attention (logged)
- [ ] Invalid sequences rejected with clear errors
- [ ] No changes to existing codebase (fully independent)

---

## 7. Verification Commands

```bash
# Run this slice's tests only
uv run pytest tests/unit/core/test_embeddings_amplify.py -v

# Verify no regressions in existing tests
uv run pytest tests/unit/core/test_embeddings.py -v

# Full quality gate
make test      # All tests
make typecheck # Mypy strict
make lint      # Ruff
make format    # Ruff format
```

---

## 8. Git Commit Strategy (After Completion)

```bash
# Single atomic commit for this vertical slice
git add src/antibody_training_esm/core/embeddings_amplify.py
git add tests/unit/core/test_embeddings_amplify.py
git add tests/fixtures/mock_models.py  # MockAMPLIFYModel

git commit -m "feat(core): add AMPLIFYEmbeddingExtractor with batch_size=1 enforcement

- Implement AMPLIFY 350M embedding extractor with padding bug mitigation
- Enforce batch_size=1 to prevent non-reproducible embeddings
- Support MPS (SDPA), CPU (eager), CUDA (auto) attention modes
- Return 960-d embeddings (vs ESM's 1280-d)
- Add 20+ unit tests with ≥90% coverage
- Type-safe with mypy strict mode

BREAKING: None (fully additive, no existing code modified)
TESTED: 20+ unit tests, all passing
COVERAGE: 92%

Refs: Nature Sci Rep 2025 (padding bug), HuggingFace chandar-lab/AMPLIFY_350M"
```

---

## 9. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Tests fail to mock AMPLIFY correctly** | HIGH | Use `MockAMPLIFYModel` with 960-d output |
| **Type annotations incomplete** | MEDIUM | Run `mypy --strict` continuously |
| **Coverage below 90%** | MEDIUM | Add edge case tests (long sequences, errors) |
| **Breaking existing ESM tests** | CRITICAL | Run `pytest tests/unit/core/test_embeddings.py` after changes |

---

## 10. Success Metrics

- ✅ **20+ tests passing** (TDD discipline)
- ✅ **≥90% code coverage** (comprehensive testing)
- ✅ **0 mypy errors** (type safety)
- ✅ **0 ruff warnings** (code quality)
- ✅ **Independent deployment** (no dependencies on Phases B/C)

---

**STATUS**: 🔴 **READY FOR IMPLEMENTATION** (pending senior approval)

**NEXT PHASE**: Phase B (Hydra Config Integration) - only starts after Phase A complete
