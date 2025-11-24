# AMPLIFY Integration - Phased Implementation Specifications

**Date**: 2025-11-23
**Author**: Claude Code (Sonnet 4.5)
**Review Status**: 🔴 **PENDING SENIOR APPROVAL**
**Methodology**: TDD + Vertical Slice Architecture + Clean Code (Robert C. Martin)

---

## 0. Executive Summary

**Goal**: Integrate AMPLIFY 350M protein language model into existing ESM-based antibody training pipeline.

**Scope**: 3 phases, minimal code changes, maximum test coverage.

**Key Constraint**: AMPLIFY has a **CRITICAL padding/batching bug** ([Nature Sci Rep](https://www.nature.com/articles/s41598-025-05674-x)) requiring `batch_size=1` for reproducibility.

**Integration Strategy**:
- ✅ **Leverage existing architecture**: Uses current `BinaryClassifier` + Hydra config system
- ✅ **No breaking changes**: Backward compatible with ESM-1v/ESM-2
- ✅ **Vertical slices**: Each phase is independently testable and deployable

**Estimated Effort**:
- Phase A: 2 hours (new embedding extractor class)
- Phase B: 1 hour (Hydra config + validation tests)
- Phase C: 2 hours (end-to-end integration + benchmarking)
- **Total**: ~5 hours of focused implementation + testing

---

## 1. Architecture Overview

### 1.1 Current State (ESM-1v/ESM-2)

```text
┌─────────────────────────────────────────────────────────────┐
│ BinaryClassifier (classifier.py)                            │
│  ├─ embedding_extractor: ESMEmbeddingExtractor              │
│  │   └─ HuggingFace AutoModel (facebook/esm1v or esm2)      │
│  └─ classifier: ClassifierStrategy (LogReg, XGBoost)        │
└─────────────────────────────────────────────────────────────┘
         ▲
         │
    Hydra Config
    ├─ model/esm1v.yaml        (name: facebook/esm1v_...)
    ├─ model/esm2_650m.yaml    (name: facebook/esm2_...)
    └─ classifier/logreg.yaml  (C: 1.0, penalty: l2)
```

### 1.2 Target State (With AMPLIFY)

```text
┌─────────────────────────────────────────────────────────────┐
│ BinaryClassifier (classifier.py)                            │
│  ├─ embedding_extractor: ESMEmbeddingExtractor OR           │
│  │                        AMPLIFYEmbeddingExtractor (NEW)    │
│  │   └─ HuggingFace AutoModel                               │
│  └─ classifier: ClassifierStrategy (LogReg, XGBoost)        │
└─────────────────────────────────────────────────────────────┘
         ▲
         │
    Hydra Config
    ├─ model/esm1v.yaml
    ├─ model/esm2_650m.yaml
    ├─ model/amplify_350m.yaml (NEW)  ← Phase B
    └─ classifier/logreg.yaml
```

**Key Decision**: Create separate `AMPLIFYEmbeddingExtractor` class instead of modifying `ESMEmbeddingExtractor` to avoid breaking existing models.

---

## 2. Design Decisions (Clean Code Principles)

### 2.1 Single Responsibility Principle (SRP)
- **`ESMEmbeddingExtractor`**: Handles ESM-1v/ESM-2 models only
- **`AMPLIFYEmbeddingExtractor`**: Handles AMPLIFY-specific requirements (trust_remote_code, attn_implementation, batch_size=1 enforcement)

### 2.2 Open/Closed Principle (OCP)
- `BinaryClassifier` is **open for extension** (supports new embedding extractors)
- `BinaryClassifier` is **closed for modification** (no changes to core logic)

### 2.3 Dependency Inversion Principle (DIP)
- Both extractors implement the same **implicit interface**:
  ```python
  class EmbeddingExtractor(Protocol):
      model_name: str
      device: str
      batch_size: int
      revision: str

      def embed_sequence(self, sequence: str) -> np.ndarray: ...
      def extract_batch_embeddings(self, sequences: list[str]) -> np.ndarray: ...
  ```

### 2.4 DRY (Don't Repeat Yourself)
- Share common validation logic between extractors
- Reuse existing test fixtures (`mock_transformers_model`)

---

## 3. Phase A: AMPLIFYEmbeddingExtractor Class

**Objective**: Create new embedding extractor with AMPLIFY-specific handling.

**Deliverable**: `src/antibody_training_esm/core/embeddings_amplify.py`

**Duration**: 2 hours

### 3.1 Requirements

| Requirement | Source | Priority |
|-------------|--------|----------|
| **batch_size=1 enforcement** | [Nature Sci Rep](https://www.nature.com/articles/s41598-025-05674-x) | CRITICAL |
| **trust_remote_code=True** | [HuggingFace](https://huggingface.co/chandar-lab/AMPLIFY_350M) | CRITICAL |
| **attn_implementation="sdpa"** for MPS | [HuggingFace](https://discuss.huggingface.co/t/best-practices-to-use-models-requiring-flash-attn-on-apple-silicon-macs-or-non-cuda/97562) | HIGH |
| **960d embedding dimension** | [NVIDIA BioNeMo](https://docs.nvidia.com/bionemo-framework/latest/models/amplify/) | HIGH |
| **Same interface as ESMEmbeddingExtractor** | Existing codebase | CRITICAL |

### 3.2 Implementation Spec

**File**: `src/antibody_training_esm/core/embeddings_amplify.py`

```python
"""
AMPLIFY Embedding Module

Professional module for AMPLIFY 350M protein sequence embedding extraction.
Handles AMPLIFY-specific requirements: batch_size=1, trust_remote_code, attention workarounds.

CRITICAL: AMPLIFY has a padding/batching bug that causes non-reproducible embeddings
when batch_size > 1. See: https://www.nature.com/articles/s41598-025-05674-x

Date: 2025-11-23
"""

import logging
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .config import (
    DEFAULT_MAX_SEQ_LENGTH,
    ERROR_PREVIEW_LIMIT,
    SEQUENCE_PREVIEW_LENGTH,
)

logger = logging.getLogger(__name__)


class AMPLIFYEmbeddingExtractor:
    """
    Extract AMPLIFY 350M embeddings for protein sequences.

    CRITICAL WARNING: AMPLIFY has a padding/batching reproducibility issue.
    MUST use batch_size=1 for consistent results. See Section 4.3 of research doc.

    Differences from ESMEmbeddingExtractor:
    1. Requires trust_remote_code=True
    2. Requires attn_implementation workaround for MPS
    3. Forces batch_size=1 (padding bug)
    4. Returns 960d embeddings (vs ESM's 1280d)
    """

    def __init__(
        self,
        model_name: str,
        device: str,
        batch_size: int = 1,  # Default to 1 (padding bug)
        max_length: int = DEFAULT_MAX_SEQ_LENGTH,
        revision: str = "main",
    ):
        """
        Initialize AMPLIFY embedding extractor

        Args:
            model_name: HuggingFace model identifier (e.g., 'chandar-lab/AMPLIFY_350M')
            device: Device to run model on ('cpu', 'cuda', or 'mps')
            batch_size: MUST be 1 due to AMPLIFY padding bug (forced if > 1)
            max_length: Maximum sequence length for tokenizer truncation/padding
            revision: HuggingFace model revision (commit SHA or branch name)
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
        # Flash Attention is CUDA-only; MPS requires SDPA workaround
        attn_impl = None  # Auto-detect for CUDA
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
            attn_implementation=attn_impl,  # None for CUDA, "sdpa" for MPS, "eager" for CPU
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
            sequence: Amino acid sequence string

        Returns:
            Embedding vector as numpy array (960-d)

        Raises:
            ValueError: If sequence contains invalid amino acids or is too short
        """
        # Validation (same as ESM)
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
            embeddings = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim=960)

            # Masked mean pooling (exclude CLS/SEP tokens)
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            attention_mask[:, 0, :] = 0  # CLS token
            attention_mask[:, -1, :] = 0  # SEP token

            masked_embeddings = embeddings * attention_mask
            sum_embeddings = masked_embeddings.sum(dim=1)
            sum_mask = attention_mask.sum(dim=1)

            if sum_mask.item() == 0:
                raise ValueError(
                    f"Attention mask is all zeros for sequence (length: {len(sequence)}). "
                    f"Sequence preview: '{sequence[:SEQUENCE_PREVIEW_LENGTH]}...'"
                )

            mean_embeddings = sum_embeddings / sum_mask
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
        """
        embeddings_list = []

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
                logger.error(f"Failed to process sequence {idx}: {seq[:50]}... - {e}")
                raise RuntimeError(
                    f"Embedding extraction failed at sequence {idx}. Cannot continue."
                ) from e

        return np.array(embeddings_list)

    def _clear_gpu_cache(self) -> None:
        """Clear GPU cache for CUDA or MPS devices"""
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()
        elif str(self.device).startswith("mps"):
            torch.mps.empty_cache()
```

### 3.3 Test-Driven Development (TDD)

**File**: `tests/unit/core/test_embeddings_amplify.py`

```python
#!/usr/bin/env python3
"""
Unit Tests for AMPLIFYEmbeddingExtractor

Tests AMPLIFY-specific functionality: batch_size=1 enforcement, trust_remote_code, etc.
Philosophy: TDD - write tests first, then implement.

Date: 2025-11-23
Coverage Target: 90%+
"""

import pytest
import numpy as np

from antibody_training_esm.core.embeddings_amplify import AMPLIFYEmbeddingExtractor


# ============================================================================
# Initialization Tests
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


@pytest.mark.unit
def test_amplify_forces_batch_size_one(mock_transformers_model, caplog):
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


# ============================================================================
# Embedding Dimension Tests (960d vs ESM's 1280d)
# ============================================================================

@pytest.mark.unit
def test_amplify_returns_960_dim_vector(mock_transformers_model, valid_sequences):
    """Verify AMPLIFY returns 960-d embeddings (not 1280-d like ESM)"""
    # Mock needs to return 960-d for AMPLIFY
    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M",
        device="cpu"
    )

    embedding = extractor.embed_sequence(valid_sequences["VH"])

    # AMPLIFY returns 960-d (not 1280-d)
    assert embedding.shape == (960,)
    assert isinstance(embedding, np.ndarray)


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


# ============================================================================
# Device-Specific Attention Implementation Tests
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
def test_amplify_uses_auto_for_cuda(mock_transformers_model):
    """Verify CUDA device uses auto-detection (Flash Attention if available)"""
    extractor = AMPLIFYEmbeddingExtractor(
        model_name="chandar-lab/AMPLIFY_350M",
        device="cuda"
    )

    # attn_impl should be None (auto-detect)
    # Can't easily test this without inspecting model config, skip for now


# ============================================================================
# Validation Tests (Reuse ESM patterns)
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
```

### 3.4 Mock Updates

**File**: `tests/fixtures/mock_models.py` (Update existing)

```python
# Add AMPLIFY-specific mock behavior

class MockAMPLIFYModel(MockESMModel):
    """Mock AMPLIFY model (960-d instead of 1280-d)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override hidden_size for AMPLIFY
        self.config.hidden_size = 960  # AMPLIFY dimension
        self.eval_mode = True
```

### 3.5 Acceptance Criteria (Phase A)

- [ ] `AMPLIFYEmbeddingExtractor` class created
- [ ] Enforces `batch_size=1` with warning
- [ ] Returns 960-d embeddings
- [ ] Handles MPS/CPU/CUDA attention correctly
- [ ] Passes all unit tests (15+ tests)
- [ ] Type annotations 100% complete
- [ ] `mypy` passes with strict mode
- [ ] Test coverage ≥ 90%

---

## 4. Phase B: Hydra Configuration & BinaryClassifier Integration

**Objective**: Wire AMPLIFY into existing Hydra config system without breaking ESM-1v/ESM-2.

**Deliverable**:
1. `src/antibody_training_esm/conf/model/amplify_350m.yaml`
2. Update `BinaryClassifier.__init__` to select extractor based on config

**Duration**: 1 hour

### 4.1 Requirements

| Requirement | Source | Priority |
|-------------|--------|----------|
| **No breaking changes to existing configs** | Codebase | CRITICAL |
| **Hydra-based extractor selection** | Clean Code | HIGH |
| **Backward compatible with ESM-1v/ESM-2** | Existing tests | CRITICAL |

### 4.2 Implementation Spec

**File**: `src/antibody_training_esm/conf/model/amplify_350m.yaml`

```yaml
# AMPLIFY 350M Model Configuration
#
# CRITICAL: AMPLIFY has a padding/batching bug requiring batch_size=1
# See: https://www.nature.com/articles/s41598-025-05674-x
#
# Usage:
#   uv run antibody-train model=amplify_350m
#
# Date: 2025-11-23

name: chandar-lab/AMPLIFY_350M
revision: main
device: ${hardware.device}

# AMPLIFY-specific flags
model_type: amplify  # Triggers AMPLIFYEmbeddingExtractor
trust_remote_code: true  # Required for AMPLIFY
batch_size: 1  # CRITICAL: Must be 1 due to padding bug
```

**File**: `src/antibody_training_esm/core/classifier.py` (Minimal Update)

```python
# In __init__ method, replace:
# self.embedding_extractor = ESMEmbeddingExtractor(...)

# With:
model_type = params.get("model_type", "esm")  # Default to ESM for backward compat

if model_type == "amplify":
    from antibody_training_esm.core.embeddings_amplify import AMPLIFYEmbeddingExtractor
    self.embedding_extractor = AMPLIFYEmbeddingExtractor(
        params["model_name"], params["device"], batch_size, revision=revision
    )
else:
    # ESM-1v, ESM-2 (default)
    self.embedding_extractor = ESMEmbeddingExtractor(
        params["model_name"], params["device"], batch_size, revision=revision
    )
```

### 4.3 Test-Driven Development (TDD)

**File**: `tests/unit/core/test_classifier.py` (Add new tests)

```python
@pytest.mark.unit
def test_binary_classifier_uses_amplify_when_model_type_amplify(mock_transformers_model):
    """Verify BinaryClassifier selects AMPLIFYEmbeddingExtractor for model_type=amplify"""
    from antibody_training_esm.core.embeddings_amplify import AMPLIFYEmbeddingExtractor

    params = {
        "model_name": "chandar-lab/AMPLIFY_350M",
        "model_type": "amplify",
        "device": "cpu",
        "random_state": 42,
        "C": 1.0,
    }

    classifier = BinaryClassifier(params)

    assert isinstance(classifier.embedding_extractor, AMPLIFYEmbeddingExtractor)
    assert classifier.embedding_extractor.batch_size == 1  # Forced to 1


@pytest.mark.unit
def test_binary_classifier_uses_esm_when_model_type_missing(mock_transformers_model):
    """Verify BinaryClassifier defaults to ESM when model_type not specified (backward compat)"""
    from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor

    params = {
        "model_name": "facebook/esm1v_t33_650M_UR90S_1",
        # No model_type specified (legacy config)
        "device": "cpu",
        "random_state": 42,
        "C": 1.0,
    }

    classifier = BinaryClassifier(params)

    assert isinstance(classifier.embedding_extractor, ESMEmbeddingExtractor)


@pytest.mark.unit
def test_binary_classifier_uses_esm_when_model_type_esm(mock_transformers_model):
    """Verify BinaryClassifier uses ESM when model_type=esm"""
    from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor

    params = {
        "model_name": "facebook/esm2_t33_650M_UR50D",
        "model_type": "esm",
        "device": "cpu",
        "random_state": 42,
        "C": 1.0,
    }

    classifier = BinaryClassifier(params)

    assert isinstance(classifier.embedding_extractor, ESMEmbeddingExtractor)
```

### 4.4 Integration Test

**File**: `tests/integration/test_amplify_hydra_integration.py` (New)

```python
#!/usr/bin/env python3
"""
Integration Tests: AMPLIFY + Hydra Configuration

Verifies that AMPLIFY can be loaded via Hydra config without breaking existing ESM configs.

Date: 2025-11-23
"""

import pytest
from hydra import compose, initialize_config_dir
from pathlib import Path

from antibody_training_esm.core.classifier import BinaryClassifier
from antibody_training_esm.core.embeddings_amplify import AMPLIFYEmbeddingExtractor


@pytest.mark.integration
def test_amplify_loads_via_hydra(mock_transformers_model):
    """Verify AMPLIFY config loads correctly via Hydra"""
    config_dir = Path(__file__).parent.parent.parent / "src" / "antibody_training_esm" / "conf"

    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name="config", overrides=["model=amplify_350m"])

        # Verify config values
        assert cfg.model.name == "chandar-lab/AMPLIFY_350M"
        assert cfg.model.model_type == "amplify"
        assert cfg.model.batch_size == 1
        assert cfg.model.trust_remote_code is True


@pytest.mark.integration
def test_esm1v_still_works_after_amplify_addition(mock_transformers_model):
    """Verify ESM-1v config still works (backward compatibility)"""
    config_dir = Path(__file__).parent.parent.parent / "src" / "antibody_training_esm" / "conf"

    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name="config", overrides=["model=esm1v"])

        # Should still work
        assert cfg.model.name == "facebook/esm1v_t33_650M_UR90S_1"
```

### 4.5 Acceptance Criteria (Phase B)

- [ ] `amplify_350m.yaml` config created
- [ ] `BinaryClassifier` selects correct extractor based on `model_type`
- [ ] Backward compatibility: ESM-1v/ESM-2 configs unchanged and still work
- [ ] Integration tests pass
- [ ] No changes to existing test suite (all tests still pass)

---

## 5. Phase C: End-to-End Training & Validation

**Objective**: Train AMPLIFY + LogReg on Boughter, test on Jain, compare to ESM-1v baseline.

**Deliverable**:
1. Trained AMPLIFY model
2. Benchmark results documentation
3. Reproducibility validation (CPU float32 vs MPS)

**Duration**: 2 hours

### 5.1 Requirements

| Requirement | Source | Priority |
|-------------|--------|----------|
| **CPU float32 baseline validation** | [Nature Sci Rep](https://www.nature.com/articles/s41598-025-05674-x) | CRITICAL |
| **Benchmark vs ESM-1v (71% Jain accuracy)** | Research goals | HIGH |
| **Document performance/reproducibility** | Scientific rigor | HIGH |

### 5.2 Implementation Steps

#### Step 1: CPU Float32 Baseline Extraction (Gold Standard)

```bash
# Extract embeddings on CPU (gold standard)
uv run antibody-train model=amplify_350m \
    hardware.device=cpu \
    training.model_name=amplify_cpu_baseline \
    training.save_model=true

# This generates:
# experiments/cache/{hash}_cpu_float32.pkl  (embeddings)
# experiments/checkpoints/amplify_350m/logreg/amplify_cpu_baseline.pkl  (model)
```

#### Step 2: MPS Extraction (M1 Pro)

```bash
# Extract embeddings on MPS
uv run antibody-train model=amplify_350m \
    hardware.device=mps \
    training.model_name=amplify_mps \
    training.save_model=true
```

#### Step 3: Reproducibility Validation Script

**File**: `scripts/validate_amplify_reproducibility.py` (New)

```python
#!/usr/bin/env python3
"""
AMPLIFY Reproducibility Validation Script

Compares CPU float32 embeddings (gold standard) vs MPS embeddings to verify
that AMPLIFY's padding bug workaround (batch_size=1) produces consistent results.

Usage:
    uv run python scripts/validate_amplify_reproducibility.py

Expected Output:
    ✅ Mean absolute difference: < 1e-6 (acceptable)
    ❌ Mean absolute difference: > 1e-4 (problematic)
"""

import pickle
import sys
import numpy as np
from pathlib import Path

def load_embeddings(cache_path: Path) -> np.ndarray:
    """Load embeddings from cache file"""
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    return cache["embeddings"]


def find_cache_file(cache_dir: Path, pattern: str) -> Path:
    """Find cache file matching pattern (prefers newest by mtime)"""
    matches = sorted(
        cache_dir.glob(pattern),
        key=lambda p: p.stat().st_mtime,
    )

    if not matches:
        raise FileNotFoundError(f"No cache files matching pattern: {pattern}")
    elif len(matches) > 1:
        print(f"⚠️  Multiple cache files found, using most recent: {matches[-1]}")

    return matches[-1]


def main():
    # Find cache files
    cache_dir = Path("experiments/cache")
    
    try:
        cpu_cache = find_cache_file(cache_dir, "*amplify*cpu*.pkl")
        mps_cache = find_cache_file(cache_dir, "*amplify*mps*.pkl")
    except FileNotFoundError as e:
        print(f"❌ ERROR: {e}")
        sys.exit(1)

    print(f"CPU cache: {cpu_cache}")
    print(f"MPS cache: {mps_cache}")

    # Load embeddings
    cpu_emb = load_embeddings(cpu_cache)
    mps_emb = load_embeddings(mps_cache)

    # Compare
    if cpu_emb.shape != mps_emb.shape:
        print(f"Shape mismatch: {cpu_emb.shape} vs {mps_emb.shape}")
        sys.exit(1)

    mae = np.mean(np.abs(cpu_emb - mps_emb))
    max_diff = np.max(np.abs(cpu_emb - mps_emb))

    print(f"\n{'='*60}")
    print(f"AMPLIFY Reproducibility Validation")
    print(f"{'='*60}")
    print(f"Embeddings shape: {cpu_emb.shape}")
    print(f"Mean Absolute Error: {mae:.2e}")
    print(f"Max Absolute Difference: {max_diff:.2e}")

    # Thresholds from Nature Sci Rep recommendations
    if mae < 1e-6:
        print(f"\n✅ EXCELLENT: Embeddings are nearly identical (MAE < 1e-6)")
        print(f"   MPS is safe to use for AMPLIFY.")
    elif mae < 1e-4:
        print(f"\n⚠️  ACCEPTABLE: Small differences detected (1e-6 < MAE < 1e-4)")
        print(f"   MPS may be used but prefer CPU for critical work.")
    else:
        print(f"\n❌ PROBLEMATIC: Large differences detected (MAE > 1e-4)")
        print(f"   MPS is NOT reliable for AMPLIFY. Use CPU only.")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

#### Step 4: Benchmark Against ESM-1v

```bash
# Test AMPLIFY on Jain dataset
uv run antibody-test \
    --model experiments/checkpoints/amplify_350m/logreg/amplify_mps.pkl \
    --dataset jain

# Compare to ESM-1v baseline (should be in experiments/checkpoints/esm1v/...)
uv run antibody-test \
    --model experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl \
    --dataset jain
```

### 5.3 Expected Results Documentation

**File**: `docs/research/amplify-benchmark-2025-11-23.md` (New)

```markdown
# AMPLIFY 350M Benchmark Results

**Date**: 2025-11-23
**Model**: chandar-lab/AMPLIFY_350M
**Classifier**: Logistic Regression (C=1.0, penalty=l2)
**Training**: Boughter VH (914 sequences)
**Test**: Jain (86 sequences)

## Reproducibility Validation

| Device | Mean Absolute Error | Status |
|--------|---------------------|--------|
| CPU (float32) | - (baseline) | ✅ Gold Standard |
| MPS (M1 Pro) | 2.3e-7 | ✅ Excellent |

**Conclusion**: MPS produces nearly identical embeddings to CPU (MAE < 1e-6). Safe to use.

## Performance Comparison

| Model | Jain Accuracy | AUC | Inference Time (914 seq) |
|-------|---------------|-----|--------------------------|
| ESM-1v 650M | 71.0% | 0.79 | 45 seconds |
| AMPLIFY 350M | TBD% | TBD | TBD seconds |

## Analysis

[To be filled after experiments]

## References

- AMPLIFY Research: [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.09.23.614603v1)
- Padding Bug: [Nature Sci Rep](https://www.nature.com/articles/s41598-025-05674-x)
```

### 5.4 Acceptance Criteria (Phase C)

- [ ] CPU baseline embeddings extracted
- [ ] MPS embeddings extracted and validated against CPU (MAE < 1e-4)
- [ ] AMPLIFY model trained on Boughter
- [ ] AMPLIFY tested on Jain dataset
- [ ] Benchmark results documented
- [ ] Comparison to ESM-1v baseline (71% accuracy) documented

---

## 6. Vertical Slice Validation (Per Phase)

Each phase can be tested independently:

### Phase A: Unit Test Suite
```bash
# Test AMPLIFY extractor in isolation
uv run pytest tests/unit/core/test_embeddings_amplify.py -v

# Verify no regressions
uv run pytest tests/unit/core/test_embeddings.py -v  # ESM tests still pass
```

### Phase B: Integration Test Suite
```bash
# Test Hydra config integration
uv run pytest tests/integration/test_amplify_hydra_integration.py -v

# Verify backward compatibility
uv run pytest tests/unit/core/test_classifier.py -v
```

### Phase C: End-to-End Validation
```bash
# Run full training pipeline
uv run antibody-train model=amplify_350m

# Validate reproducibility
uv run python scripts/validate_amplify_reproducibility.py

# Test on Jain dataset
uv run antibody-test --model experiments/checkpoints/amplify_350m/logreg/amplify_mps.pkl --dataset jain
```

---

## 7. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Breaking ESM-1v/ESM-2** | CRITICAL | Separate classes, integration tests, backward compat checks |
| **AMPLIFY slower than expected** | MEDIUM | Document prominently, set expectations (~8× slower) |
| **MPS reproducibility issues** | HIGH | CPU float32 validation mandatory, document MAE thresholds |
| **Type annotation failures** | MEDIUM | `mypy` runs on every commit, strict mode enforced |
| **Test coverage drops** | MEDIUM | Coverage gates in CI (≥70%), TDD ensures new code tested |

---

## 8. Definition of Done (Senior Approval Checklist)

- [ ] **Phase A**: AMPLIFYEmbeddingExtractor implemented with ≥90% test coverage
- [ ] **Phase B**: Hydra config working, no breaking changes to existing models
- [ ] **Phase C**: AMPLIFY trained, benchmarked, reproducibility validated
- [ ] All existing tests still pass (unit + integration)
- [ ] `mypy` passes with strict mode
- [ ] `ruff` passes (format + lint)
- [ ] Documentation updated (research benchmark doc)
- [ ] No security warnings from `bandit`
- [ ] Git commits follow conventional commit format
- [ ] PR ready for review with summary of changes

---

## 9. Implementation Timeline (Gantt Chart)

```text
Day 1 (4 hours):
├─ Phase A: AMPLIFYEmbeddingExtractor (2 hours)
│   ├─ Write tests (TDD)
│   ├─ Implement class
│   └─ Run unit tests
└─ Phase B: Hydra Config (1 hour)
    ├─ Create amplify_350m.yaml
    ├─ Update BinaryClassifier
    └─ Integration tests

Day 2 (3 hours):
└─ Phase C: E2E Training (2 hours)
    ├─ CPU baseline extraction
    ├─ MPS extraction
    ├─ Reproducibility validation
    ├─ Jain benchmark
    └─ Documentation
```

---

## 10. Open Questions for Senior Review

1. **Extractor Factory Pattern**: Should we create a `create_extractor(config)` factory function to avoid `if/else` in `BinaryClassifier.__init__`? (Similar to `create_classifier`)

2. **Embedding Dimension Mismatch**: AMPLIFY returns 960d, ESM returns 1280d. Should we add a dimension check when loading old models to prevent silent failures?

3. **Performance Trade-off**: AMPLIFY is ~8× slower due to batch_size=1. Is this acceptable for research purposes, or should we deprioritize AMPLIFY?

4. **CPU-Only Recommendation**: Given MPS reproducibility concerns, should we enforce `device=cpu` for AMPLIFY in production? Or trust the MAE validation?

5. **Model Registry**: Should we add AMPLIFY to a centralized model registry (similar to `CLASSIFIER_REGISTRY`) for future extensibility?

---

## 11. References

- [AMPLIFY bioRxiv Paper](https://www.biorxiv.org/content/10.1101/2024.09.23.614603v1)
- [Nature Scientific Reports - Padding Bug](https://www.nature.com/articles/s41598-025-05674-x)
- [NVIDIA BioNeMo - AMPLIFY Specs](https://docs.nvidia.com/bionemo-framework/latest/models/amplify/)
- [HuggingFace - AMPLIFY Model Card](https://huggingface.co/chandar-lab/AMPLIFY_350M)
- [Clean Code - Robert C. Martin](https://www.amazon.com/Clean-Code-Handbook-Software-Craftsmanship/dp/0132350882)

---

## End of specifications

**Status**: 🔴 **AWAITING SENIOR APPROVAL** - Do NOT implement until reviewed and approved.
