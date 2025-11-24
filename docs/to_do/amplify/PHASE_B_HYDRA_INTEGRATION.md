# Phase B: Hydra Config Integration - Vertical Slice Specification

**Date**: 2025-11-23
**Author**: Claude Code (Sonnet 4.5)
**Status**: 🔴 **PENDING SENIOR APPROVAL**
**Methodology**: TDD + Open/Closed Principle
**Duration**: 1 hour
**Dependencies**: Phase A (AMPLIFYEmbeddingExtractor) must be complete

---

## 1. Objective

Wire `AMPLIFYEmbeddingExtractor` into existing Hydra config system and `BinaryClassifier` without breaking ESM-1v/ESM-2 workflows.

**Success Criteria**: AMPLIFY can be selected via `model=amplify_350m` CLI override, all existing ESM configs still work.

---

## 2. Requirements

| Requirement | Source | Priority | Acceptance Test |
|-------------|--------|----------|-----------------|
| **No breaking changes to ESM configs** | Existing tests | CRITICAL | All existing tests still pass |
| **Hydra-based model selection** | Clean Architecture | HIGH | `test_amplify_loads_via_hydra()` |
| **Backward compatible with legacy configs** | Existing codebase | CRITICAL | `test_binary_classifier_defaults_to_esm()` |
| **Type-based extractor selection** | Open/Closed Principle | HIGH | `test_binary_classifier_uses_amplify_when_model_type_amplify()` |

---

## 3. Design

### 3.1 Current Architecture (Before)

```python
# BinaryClassifier.__init__ (current)
self.embedding_extractor = ESMEmbeddingExtractor(
    params["model_name"],
    params["device"],
    batch_size,
    revision=revision
)
```

**Problem**: Hardcoded to `ESMEmbeddingExtractor`, cannot support AMPLIFY.

### 3.2 Target Architecture (After)

```python
# BinaryClassifier.__init__ (new)
model_type = params.get("model_type", "esm")  # Default for backward compat

if model_type == "amplify":
    from antibody_training_esm.core.embeddings_amplify import AMPLIFYEmbeddingExtractor
    self.embedding_extractor = AMPLIFYEmbeddingExtractor(...)
else:
    self.embedding_extractor = ESMEmbeddingExtractor(...)
```

**Rationale**: Open for extension (new model types), closed for modification (no changes to ESM logic).

---

## 4. Implementation (TDD: Write Tests First!)

### 4.1 Test File (Write This FIRST)

**File**: `tests/unit/core/test_classifier.py` (Add to existing file)

```python
# ============================================================================
# AMPLIFY Integration Tests (Phase B)
# ============================================================================

@pytest.mark.unit
def test_binary_classifier_uses_amplify_when_model_type_amplify(mock_transformers_model):
    """Verify BinaryClassifier selects AMPLIFYEmbeddingExtractor for model_type=amplify"""
    from antibody_training_esm.core.embeddings_amplify import AMPLIFYEmbeddingExtractor

    params = {
        "model_name": "chandar-lab/AMPLIFY_350M",
        "model_type": "amplify",  # NEW: triggers AMPLIFY extractor
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
    assert classifier.embedding_extractor.batch_size == 8  # Default ESM batch size


@pytest.mark.unit
def test_binary_classifier_uses_esm_when_model_type_esm(mock_transformers_model):
    """Verify BinaryClassifier uses ESM when model_type=esm (explicit)"""
    from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor

    params = {
        "model_name": "facebook/esm2_t33_650M_UR50D",
        "model_type": "esm",  # Explicit ESM
        "device": "cpu",
        "random_state": 42,
        "C": 1.0,
    }

    classifier = BinaryClassifier(params)

    assert isinstance(classifier.embedding_extractor, ESMEmbeddingExtractor)


@pytest.mark.unit
def test_binary_classifier_raises_on_invalid_model_type(mock_transformers_model):
    """Verify BinaryClassifier raises ValueError for unknown model_type"""
    params = {
        "model_name": "some/model",
        "model_type": "invalid_type",
        "device": "cpu",
        "random_state": 42,
        "C": 1.0,
    }

    with pytest.raises(ValueError, match="Unknown model_type"):
        BinaryClassifier(params)


@pytest.mark.unit
def test_binary_classifier_get_params_includes_model_type(mock_transformers_model):
    """Verify get_params() includes model_type for sklearn compatibility"""
    params = {
        "model_name": "chandar-lab/AMPLIFY_350M",
        "model_type": "amplify",
        "device": "cpu",
        "random_state": 42,
        "C": 1.0,
    }

    classifier = BinaryClassifier(params)
    retrieved_params = classifier.get_params()

    assert "model_type" in retrieved_params
    assert retrieved_params["model_type"] == "amplify"


@pytest.mark.unit
def test_binary_classifier_set_params_can_change_model_type(mock_transformers_model):
    """Verify set_params() can switch between ESM and AMPLIFY"""
    from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor
    from antibody_training_esm.core.embeddings_amplify import AMPLIFYEmbeddingExtractor

    # Start with ESM
    params = {
        "model_name": "facebook/esm1v_t33_650M_UR90S_1",
        "model_type": "esm",
        "device": "cpu",
        "random_state": 42,
        "C": 1.0,
    }
    classifier = BinaryClassifier(params)
    assert isinstance(classifier.embedding_extractor, ESMEmbeddingExtractor)

    # Switch to AMPLIFY
    classifier.set_params(
        model_type="amplify",
        model_name="chandar-lab/AMPLIFY_350M"
    )
    assert isinstance(classifier.embedding_extractor, AMPLIFYEmbeddingExtractor)
```

### 4.2 Integration Test File (Write This FIRST)

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
from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor


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
        assert cfg.model.revision == "main"
        assert cfg.model.device == cfg.hardware.device  # Interpolated


@pytest.mark.integration
def test_esm1v_still_works_after_amplify_addition(mock_transformers_model):
    """Verify ESM-1v config still works (backward compatibility)"""
    config_dir = Path(__file__).parent.parent.parent / "src" / "antibody_training_esm" / "conf"

    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name="config", overrides=["model=esm1v"])

        # Should still work (no model_type field in old configs)
        assert cfg.model.name == "facebook/esm1v_t33_650M_UR90S_1"
        assert cfg.model.revision == "main"
        # model_type should be missing (backward compat)
        assert not hasattr(cfg.model, "model_type") or cfg.model.get("model_type") is None


@pytest.mark.integration
def test_esm2_still_works_after_amplify_addition(mock_transformers_model):
    """Verify ESM-2 config still works (backward compatibility)"""
    config_dir = Path(__file__).parent.parent.parent / "src" / "antibody_training_esm" / "conf"

    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name="config", overrides=["model=esm2_650m"])

        assert cfg.model.name == "facebook/esm2_t33_650M_UR50D"
        assert cfg.model.revision == "main"


@pytest.mark.integration
def test_binary_classifier_with_amplify_hydra_config(mock_transformers_model):
    """Verify BinaryClassifier can be initialized from AMPLIFY Hydra config"""
    config_dir = Path(__file__).parent.parent.parent / "src" / "antibody_training_esm" / "conf"

    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=["model=amplify_350m", "hardware.device=cpu"]
        )

        # Convert Hydra config to dict for BinaryClassifier
        params = {
            "model_name": cfg.model.name,
            "model_type": cfg.model.model_type,
            "device": cfg.model.device,
            "batch_size": cfg.model.batch_size,
            "revision": cfg.model.revision,
            "random_state": cfg.training.random_state,
            "C": cfg.classifier.C,
        }

        classifier = BinaryClassifier(params)

        assert isinstance(classifier.embedding_extractor, AMPLIFYEmbeddingExtractor)
        assert classifier.embedding_extractor.batch_size == 1


@pytest.mark.integration
def test_binary_classifier_with_esm_hydra_config(mock_transformers_model):
    """Verify BinaryClassifier still works with ESM Hydra config (backward compat)"""
    config_dir = Path(__file__).parent.parent.parent / "src" / "antibody_training_esm" / "conf"

    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=["model=esm1v", "hardware.device=cpu"]
        )

        params = {
            "model_name": cfg.model.name,
            # No model_type (legacy)
            "device": cfg.model.device,
            "batch_size": cfg.training.batch_size,  # From training config
            "revision": cfg.model.revision,
            "random_state": cfg.training.random_state,
            "C": cfg.classifier.C,
        }

        classifier = BinaryClassifier(params)

        assert isinstance(classifier.embedding_extractor, ESMEmbeddingExtractor)
```

---

## 5. Implementation Files

### 5.1 Hydra Config File (Create This)

**File**: `src/antibody_training_esm/conf/model/amplify_350m.yaml`

```yaml
# AMPLIFY 350M Model Configuration
#
# CRITICAL WARNING: AMPLIFY has a padding/batching bug requiring batch_size=1
# Source: https://www.nature.com/articles/s41598-025-05674-x
#
# Usage:
#   uv run antibody-train model=amplify_350m
#   uv run antibody-train model=amplify_350m hardware.device=mps
#
# Date: 2025-11-23

name: chandar-lab/AMPLIFY_350M
revision: main
device: ${hardware.device}  # Inherits from hardware config

# AMPLIFY-specific configuration
model_type: amplify  # Triggers AMPLIFYEmbeddingExtractor in BinaryClassifier
trust_remote_code: true  # Required for AMPLIFY (custom modeling code)
batch_size: 1  # CRITICAL: Must be 1 due to padding bug (do not change!)

# Note: Unlike ESM configs, we do NOT inherit batch_size from training.batch_size
# because AMPLIFY MUST use batch_size=1 regardless of training config
```

### 5.2 BinaryClassifier Update (Modify Existing)

**File**: `src/antibody_training_esm/core/classifier.py`

```python
# In __init__ method, replace this section:
# ----------------------------------------
# OLD CODE (remove):
# self.embedding_extractor = ESMEmbeddingExtractor(
#     params["model_name"], params["device"], batch_size, revision=revision
# )

# NEW CODE (add):
model_type = params.get("model_type", "esm")  # Default to ESM for backward compat

if model_type == "amplify":
    from antibody_training_esm.core.embeddings_amplify import AMPLIFYEmbeddingExtractor

    self.embedding_extractor = AMPLIFYEmbeddingExtractor(
        params["model_name"],
        params["device"],
        batch_size,  # Will be forced to 1 by AMPLIFYEmbeddingExtractor
        revision=revision,
    )
elif model_type == "esm":
    self.embedding_extractor = ESMEmbeddingExtractor(
        params["model_name"],
        params["device"],
        batch_size,
        revision=revision,
    )
else:
    raise ValueError(
        f"Unknown model_type: '{model_type}'. "
        f"Supported types: 'esm', 'amplify'"
    )

# Also update get_params() to include model_type
# In get_params() method, add:
params = {
    "random_state": self.random_state,
    "model_name": self.model_name,
    "model_type": self._params.get("model_type", "esm"),  # NEW: include model_type
    "device": self.device,
    "batch_size": self.batch_size,
    "revision": self.revision,
}

# Also update set_params() to handle model_type changes
# In set_params() method, update needs_extractor_reload check:
embedding_params = {"model_name", "device", "batch_size", "revision", "model_type"}  # Add model_type
if any(key in params for key in embedding_params):
    needs_extractor_reload = True
    # ... rest of existing logic
```

---

## 6. TDD Workflow (Step-by-Step)

### Step 1: Write All Tests (20 minutes)
```bash
# Add tests to existing test_classifier.py
vim tests/unit/core/test_classifier.py

# Create new integration test file
vim tests/integration/test_amplify_hydra_integration.py

# Verify tests FAIL (red phase)
uv run pytest tests/unit/core/test_classifier.py::test_binary_classifier_uses_amplify_when_model_type_amplify -v
# Expected: FAIL (model_type not yet implemented)
```

### Step 2: Create Hydra Config (5 minutes)
```bash
# Create AMPLIFY config file
vim src/antibody_training_esm/conf/model/amplify_350m.yaml

# Verify Hydra can load it
uv run python -c "
from hydra import compose, initialize_config_dir
from pathlib import Path
config_dir = Path('src/antibody_training_esm/conf')
with initialize_config_dir(config_dir=str(config_dir), version_base='1.3'):
    cfg = compose(config_name='config', overrides=['model=amplify_350m'])
    print(cfg.model)
"
```

### Step 3: Update BinaryClassifier (20 minutes)
```bash
# Modify classifier.py
vim src/antibody_training_esm/core/classifier.py

# Run tests incrementally
uv run pytest tests/unit/core/test_classifier.py -k "amplify" -v
```

### Step 4: Integration Tests (15 minutes)
```bash
# Run Hydra integration tests
uv run pytest tests/integration/test_amplify_hydra_integration.py -v
```

### Step 5: Backward Compatibility Check (CRITICAL)
```bash
# Verify ALL existing tests still pass
uv run pytest tests/unit/core/test_classifier.py -v
uv run pytest tests/unit/core/test_embeddings.py -v

# If any fail, we broke backward compatibility (fix before proceeding!)
```

---

## 7. Acceptance Criteria (Definition of Done)

- [ ] `amplify_350m.yaml` config created
- [ ] `BinaryClassifier` updated with `model_type` selection
- [ ] All 6 new unit tests passing
- [ ] All 5 integration tests passing
- [ ] **CRITICAL**: All existing tests still pass (no regressions)
- [ ] `mypy --strict` passes with zero errors
- [ ] `ruff` passes (format + lint)
- [ ] Backward compatibility validated:
  - [ ] ESM-1v config still works
  - [ ] ESM-2 config still works
  - [ ] Legacy configs without `model_type` still work
- [ ] Hydra config loads without errors
- [ ] `get_params()` includes `model_type`
- [ ] `set_params()` can switch between ESM/AMPLIFY

---

## 8. Verification Commands

```bash
# Run this slice's tests
uv run pytest tests/unit/core/test_classifier.py -k "amplify" -v
uv run pytest tests/integration/test_amplify_hydra_integration.py -v

# CRITICAL: Verify backward compatibility
uv run pytest tests/unit/core/test_classifier.py -v  # All ESM tests must pass
uv run pytest tests/unit/core/test_embeddings.py -v

# Integration smoke test
uv run antibody-train model=amplify_350m --help  # Should not error
uv run antibody-train model=esm1v --help  # Should still work
```

---

## 9. Git Commit Strategy (After Completion)

```bash
# Single atomic commit for this vertical slice
git add src/antibody_training_esm/conf/model/amplify_350m.yaml
git add src/antibody_training_esm/core/classifier.py
git add tests/unit/core/test_classifier.py
git add tests/integration/test_amplify_hydra_integration.py

git commit -m "feat(config): add AMPLIFY Hydra config with model_type selection

- Add conf/model/amplify_350m.yaml for AMPLIFY 350M model
- Update BinaryClassifier to support model_type parameter
- Add model_type='esm'|'amplify' selection in __init__
- Maintain backward compatibility (model_type defaults to 'esm')
- Add 6 unit tests for model_type selection logic
- Add 5 integration tests for Hydra config loading

BREAKING: None (fully backward compatible)
TESTED: 11 new tests, all existing tests still pass
BACKWARD COMPAT: ESM-1v/ESM-2 configs unchanged and verified

Usage:
  uv run antibody-train model=amplify_350m  # AMPLIFY
  uv run antibody-train model=esm1v         # ESM-1v (unchanged)
  uv run antibody-train model=esm2_650m     # ESM-2 (unchanged)

Refs: Phase B of AMPLIFY integration"
```

---

## 10. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Breaking ESM-1v/ESM-2 workflows** | CRITICAL | Run full test suite, verify backward compat |
| **Hydra config syntax errors** | HIGH | Test config loading in isolation first |
| **get_params/set_params inconsistency** | MEDIUM | Add tests for sklearn compatibility |
| **Missing model_type in legacy models** | MEDIUM | Default to "esm", add tests |

---

## 11. Success Metrics

- ✅ **11 new tests passing** (6 unit + 5 integration)
- ✅ **0 regressions** (all existing tests still pass)
- ✅ **Backward compatible** (ESM configs unchanged)
- ✅ **Hydra integration working** (CLI overrides functional)
- ✅ **Type-safe** (mypy strict clean)

---

**STATUS**: 🔴 **BLOCKED** - Depends on Phase A completion

**NEXT PHASE**: Phase C (E2E Training & Validation) - only starts after Phase B complete
