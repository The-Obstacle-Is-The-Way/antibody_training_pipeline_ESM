# Pydantic Phase 1: Prediction Hardening

**Status:** Not Started
**Priority:** HIGH (User-facing validation)
**Risk:** LOW (Isolated to prediction surfaces)
**Dependencies:** None

---

## Overview

Harden prediction surfaces (CLI, Gradio Web App, Predictor class) with Pydantic v2 request/response models. Replace ad-hoc validation logic with declarative schemas that enforce correctness at the boundary.

**Key Benefits:**
- **User Safety:** Fail fast with clear error messages before expensive ESM computation
- **Code Deduplication:** Remove duplicated `validate_input` logic in `app.py`
- **Type Safety:** End-to-end type checking from Gradio → Predictor → Results
- **API Readiness:** Models are reusable for future FastAPI endpoints

---

## Dependencies

### Required Packages (Add to `pyproject.toml`)

```toml
[project.optional-dependencies]
validation = [
    "pydantic>=2.10.0",           # Stable v2 release
    "pydantic-settings>=2.6.0",   # For future config management
]
```

**Installation:**
```bash
uv sync --extra validation
```

**Version Rationale:**
- Pydantic 2.10.0: Latest stable (Nov 2024), mature v2 API
- No breaking changes expected (v2 API finalized)
- ~5-10x faster than v1 (Rust core)

---

## Implementation Scope

### Files to Modify

1. **Create:** `src/antibody_training_esm/models/__init__.py`
   - New module for Pydantic models

2. **Create:** `src/antibody_training_esm/models/prediction.py`
   - `PredictionRequest` model (single sequence)
   - `BatchPredictionRequest` model (list/file)
   - `PredictionResult` model (response)

3. **Modify:** `src/antibody_training_esm/core/prediction.py`
   - Update `Predictor.predict()` to accept `PredictionRequest`
   - Update `Predictor.predict_single()` to return `PredictionResult`
   - Maintain backward compatibility with raw strings

4. **Modify:** `src/antibody_training_esm/cli/app.py`
   - Replace `validate_input()` with `PredictionRequest` validation
   - Use `PredictionResult` for response formatting

5. **Modify:** `src/antibody_training_esm/cli/predict.py`
   - Integrate `PredictionRequest` for CLI input validation

---

## Model Specifications

### 1. `PredictionRequest` (Single Sequence)

**Location:** `src/antibody_training_esm/models/prediction.py`

```python
from pydantic import BaseModel, Field, field_validator
from typing import Literal

class PredictionRequest(BaseModel):
    """
    Single sequence prediction request.

    Validates amino acid sequence and optional parameters.
    """
    sequence: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Antibody amino acid sequence (VH or VL)",
        examples=["QVQLVQSGAEVKKPGASVKVSCKASGYTFT..."],
    )

    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Classification threshold (0-1)",
    )

    assay_type: Literal["ELISA", "PSR"] | None = Field(
        default=None,
        description="Assay type for calibrated thresholds",
    )

    @field_validator("sequence")
    @classmethod
    def validate_amino_acids(cls, v: str) -> str:
        """Validate sequence contains only valid amino acids."""
        # Clean whitespace
        cleaned = v.strip().upper()

        if not cleaned:
            raise ValueError("Sequence cannot be empty after cleaning")

        # Standard 20 amino acids + X (unknown)
        valid_chars = set("ACDEFGHIKLMNPQRSTVWYX")
        invalid_chars = set(cleaned) - valid_chars

        if invalid_chars:
            raise ValueError(
                f"Invalid characters found: {', '.join(sorted(invalid_chars))}. "
                f"Only standard amino acids (ACDEFGHIKLMNPQRSTVWY) and X are allowed."
            )

        return cleaned

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "sequence": "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYNMHWVRQAPGQGLEWMG",
                    "threshold": 0.5,
                    "assay_type": "ELISA",
                }
            ]
        }
    }
```

### 2. `BatchPredictionRequest` (Multiple Sequences)

```python
class BatchPredictionRequest(BaseModel):
    """
    Batch prediction request for multiple sequences.

    Supports both inline lists and file uploads (future).
    """
    sequences: list[str] = Field(
        ...,
        min_length=1,
        max_length=1000,  # Batch size limit
        description="List of antibody sequences",
    )

    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    assay_type: Literal["ELISA", "PSR"] | None = None

    @field_validator("sequences")
    @classmethod
    def validate_all_sequences(cls, v: list[str]) -> list[str]:
        """Validate each sequence in batch."""
        cleaned = []
        errors = []

        for i, seq in enumerate(v):
            try:
                # Reuse PredictionRequest validator
                request = PredictionRequest(sequence=seq)
                cleaned.append(request.sequence)
            except ValueError as e:
                errors.append(f"Sequence {i+1}: {e}")

        if errors:
            raise ValueError(
                f"Batch validation failed:\n" + "\n".join(errors)
            )

        return cleaned
```

### 3. `PredictionResult` (Response)

```python
class PredictionResult(BaseModel):
    """
    Prediction result for a single sequence.

    Standardizes output format across CLI, Gradio, and future APIs.
    """
    sequence: str = Field(..., description="Input sequence (cleaned)")

    prediction: Literal["specific", "non-specific"] = Field(
        ...,
        description="Classification result",
    )

    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of non-specificity (class 1)",
    )

    threshold: float = Field(
        ...,
        description="Threshold used for classification",
    )

    assay_type: str | None = Field(
        default=None,
        description="Assay type if calibrated threshold was used",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "sequence": "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYNMH...",
                    "prediction": "specific",
                    "probability": 0.23,
                    "threshold": 0.5,
                    "assay_type": "ELISA",
                }
            ]
        }
    }
```

---

## Integration Steps (TDD)

### Step 1: Create Models Package

```bash
mkdir -p src/antibody_training_esm/models
touch src/antibody_training_esm/models/__init__.py
```

**`__init__.py` contents:**
```python
"""
Pydantic models for runtime validation.

This package contains schema definitions for:
- Prediction requests/responses
- Configuration validation (Phase 2)
- Dataset schemas (Phase 3)
- Model artifacts (Phase 4)
"""

from antibody_training_esm.models.prediction import (
    PredictionRequest,
    BatchPredictionRequest,
    PredictionResult,
)

__all__ = [
    "PredictionRequest",
    "BatchPredictionRequest",
    "PredictionResult",
]
```

### Step 2: Write Tests FIRST (TDD)

**Create:** `tests/unit/models/test_prediction.py`

```python
"""Unit tests for prediction models."""

import pytest
from pydantic import ValidationError

from antibody_training_esm.models.prediction import (
    PredictionRequest,
    PredictionResult,
    BatchPredictionRequest,
)


class TestPredictionRequest:
    """Test suite for PredictionRequest validation."""

    def test_valid_sequence(self):
        """Valid sequence passes validation."""
        request = PredictionRequest(
            sequence="QVQLVQSGAEVKKPGASVKVSCKASGYTFT"
        )
        assert request.sequence == "QVQLVQSGAEVKKPGASVKVSCKASGYTFT"
        assert request.threshold == 0.5  # default
        assert request.assay_type is None  # default

    def test_sequence_cleaned(self):
        """Whitespace is stripped and uppercased."""
        request = PredictionRequest(sequence="  qvql  ")
        assert request.sequence == "QVQL"

    def test_empty_sequence_rejected(self):
        """Empty sequence raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PredictionRequest(sequence="")

        errors = exc_info.value.errors()
        assert any("empty" in str(e).lower() for e in errors)

    def test_invalid_amino_acids_rejected(self):
        """Non-amino acid characters raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PredictionRequest(sequence="QVQL123")

        errors = exc_info.value.errors()
        assert any("invalid characters" in str(e).lower() for e in errors)

    def test_gap_characters_rejected(self):
        """Gap characters (-, *, .) are rejected."""
        for gap_char in ["-", "*", "."]:
            with pytest.raises(ValidationError):
                PredictionRequest(sequence=f"QVQL{gap_char}VQS")

    def test_threshold_out_of_range_rejected(self):
        """Threshold must be 0-1."""
        with pytest.raises(ValidationError):
            PredictionRequest(sequence="QVQL", threshold=1.5)

        with pytest.raises(ValidationError):
            PredictionRequest(sequence="QVQL", threshold=-0.1)

    def test_invalid_assay_type_rejected(self):
        """Only ELISA and PSR are valid."""
        with pytest.raises(ValidationError):
            PredictionRequest(
                sequence="QVQL",
                assay_type="INVALID",  # type: ignore
            )

    def test_sequence_length_limits(self):
        """Sequences longer than 2000 are rejected."""
        long_seq = "A" * 2001
        with pytest.raises(ValidationError) as exc_info:
            PredictionRequest(sequence=long_seq)

        errors = exc_info.value.errors()
        assert any("max_length" in str(e).lower() for e in errors)


class TestBatchPredictionRequest:
    """Test suite for BatchPredictionRequest validation."""

    def test_valid_batch(self):
        """Valid batch passes validation."""
        request = BatchPredictionRequest(
            sequences=["QVQL", "EVQL", "DIQM"]
        )
        assert len(request.sequences) == 3

    def test_empty_batch_rejected(self):
        """Empty batch list is rejected."""
        with pytest.raises(ValidationError):
            BatchPredictionRequest(sequences=[])

    def test_batch_size_limit(self):
        """Batches >1000 are rejected."""
        large_batch = ["QVQL"] * 1001
        with pytest.raises(ValidationError):
            BatchPredictionRequest(sequences=large_batch)

    def test_invalid_sequence_in_batch_rejected(self):
        """One invalid sequence fails entire batch."""
        with pytest.raises(ValidationError) as exc_info:
            BatchPredictionRequest(
                sequences=["QVQL", "INVALID123", "EVQL"]
            )

        error_msg = str(exc_info.value)
        assert "Sequence 2" in error_msg  # 1-indexed


class TestPredictionResult:
    """Test suite for PredictionResult output."""

    def test_valid_result(self):
        """Valid result constructs correctly."""
        result = PredictionResult(
            sequence="QVQL",
            prediction="specific",
            probability=0.23,
            threshold=0.5,
        )
        assert result.prediction == "specific"
        assert result.probability == 0.23

    def test_invalid_prediction_rejected(self):
        """Only 'specific' or 'non-specific' allowed."""
        with pytest.raises(ValidationError):
            PredictionResult(
                sequence="QVQL",
                prediction="unknown",  # type: ignore
                probability=0.5,
                threshold=0.5,
            )

    def test_probability_out_of_range_rejected(self):
        """Probability must be 0-1."""
        with pytest.raises(ValidationError):
            PredictionResult(
                sequence="QVQL",
                prediction="specific",
                probability=1.5,
                threshold=0.5,
            )
```

**Run tests (should FAIL initially):**
```bash
uv run pytest tests/unit/models/test_prediction.py -v
```

### Step 3: Implement Models

Create `src/antibody_training_esm/models/prediction.py` with the model specifications above.

**Run tests again (should PASS):**
```bash
uv run pytest tests/unit/models/test_prediction.py -v
```

### Step 4: Integrate into Predictor

**Modify:** `src/antibody_training_esm/core/prediction.py`

```python
from antibody_training_esm.models.prediction import (
    PredictionRequest,
    PredictionResult,
)

class Predictor:
    # ... existing code ...

    def predict_single(
        self,
        sequence: str | PredictionRequest,
        threshold: float = 0.5,
        assay_type: str | None = None,
    ) -> PredictionResult:
        """
        Predict single sequence with Pydantic validation.

        Args:
            sequence: Raw string OR PredictionRequest model
            threshold: Decision threshold (ignored if PredictionRequest passed)
            assay_type: Assay type (ignored if PredictionRequest passed)

        Returns:
            PredictionResult model
        """
        # Normalize input to PredictionRequest
        if isinstance(sequence, str):
            request = PredictionRequest(
                sequence=sequence,
                threshold=threshold,
                assay_type=assay_type,
            )
        else:
            request = sequence

        # Extract validated sequence
        cleaned_seq = request.sequence

        # Run prediction (existing logic)
        results_df = self.predict(
            [cleaned_seq],
            threshold=request.threshold,
            assay_type=request.assay_type,
        )

        # Convert to PredictionResult
        return PredictionResult(
            sequence=cleaned_seq,
            prediction=results_df["prediction"].iloc[0],
            probability=float(results_df["probability"].iloc[0]),
            threshold=request.threshold,
            assay_type=request.assay_type,
        )
```

### Step 5: Update Gradio App

**Modify:** `src/antibody_training_esm/cli/app.py`

```python
from antibody_training_esm.models.prediction import PredictionRequest
from pydantic import ValidationError

def predict_sequence(sequence: str) -> tuple[str, str]:
    """Gradio prediction handler with Pydantic validation."""
    try:
        # Validate with Pydantic (replaces old validate_input)
        request = PredictionRequest(sequence=sequence)

        # Log request
        logger.info(f"Processing: length={len(request.sequence)}")

        # Predict (returns PydanticResult)
        result = predictor.predict_single(request)

        # Format response
        prob_percent = f"{result.probability:.1%}"
        return result.prediction, prob_percent

    except ValidationError as e:
        # Extract first error message for user-friendly display
        error_msg = e.errors()[0]["msg"]
        raise gr.Error(error_msg) from e
    except Exception as e:
        logger.exception("Prediction failed")
        raise gr.Error(f"Prediction failed: {str(e)}") from e
```

**Delete old `validate_input` function** (lines 88-102 in current `app.py`)

### Step 6: Update CLI Predict

**Modify:** `src/antibody_training_esm/cli/predict.py`

```python
from antibody_training_esm.models.prediction import PredictionRequest
from pydantic import ValidationError

def predict_sequence_cli(sequence: str, threshold: float, assay_type: str | None):
    """CLI prediction with Pydantic validation."""
    try:
        request = PredictionRequest(
            sequence=sequence,
            threshold=threshold,
            assay_type=assay_type,
        )
        result = predictor.predict_single(request)

        # Print formatted output
        print(f"Sequence: {result.sequence[:50]}...")
        print(f"Prediction: {result.prediction}")
        print(f"Probability: {result.probability:.2%}")

    except ValidationError as e:
        print(f"❌ Validation Error:")
        for error in e.errors():
            print(f"  - {error['loc'][0]}: {error['msg']}")
        sys.exit(1)
```

---

## Testing Strategy

### Unit Tests

**Coverage targets:**
- ✅ `PredictionRequest` validation (10 tests)
- ✅ `BatchPredictionRequest` validation (5 tests)
- ✅ `PredictionResult` construction (3 tests)

**Run:**
```bash
uv run pytest tests/unit/models/ -v --cov=src/antibody_training_esm/models
```

### Integration Tests

**Create:** `tests/integration/test_prediction_integration.py`

```python
"""Integration tests for Pydantic + Predictor."""

import pytest
from pydantic import ValidationError

from antibody_training_esm.core.prediction import Predictor
from antibody_training_esm.models.prediction import (
    PredictionRequest,
    PredictionResult,
)


@pytest.fixture
def mock_predictor():
    """Mock predictor for integration testing."""
    # Use mock ESM model to avoid HF downloads
    return Predictor(
        model_name="facebook/esm1v_t33_650M_UR90S_1",
        classifier_path="tests/fixtures/mock_classifier.pkl",
        device="cpu",
    )


def test_predictor_accepts_raw_string(mock_predictor):
    """Predictor maintains backward compatibility with raw strings."""
    result = mock_predictor.predict_single("QVQLVQSGAEVK")

    assert isinstance(result, PredictionResult)
    assert result.prediction in ["specific", "non-specific"]
    assert 0 <= result.probability <= 1


def test_predictor_accepts_pydantic_model(mock_predictor):
    """Predictor accepts PredictionRequest directly."""
    request = PredictionRequest(
        sequence="QVQLVQSGAEVK",
        threshold=0.6,
        assay_type="ELISA",
    )
    result = mock_predictor.predict_single(request)

    assert isinstance(result, PredictionResult)
    assert result.threshold == 0.6
    assert result.assay_type == "ELISA"


def test_predictor_rejects_invalid_sequence(mock_predictor):
    """Invalid sequence raises ValidationError before ESM computation."""
    with pytest.raises(ValidationError):
        mock_predictor.predict_single("INVALID123")
```

**Run:**
```bash
uv run pytest tests/integration/test_prediction_integration.py -v
```

### E2E Tests

**Extend:** `tests/e2e/test_gradio_app.py`

```python
"""E2E test for Gradio app with Pydantic validation."""

def test_gradio_validates_input():
    """Gradio app rejects invalid sequences with Pydantic."""
    # Launch app in test mode
    # ... test code ...

    response = client.predict("INVALID123")
    assert "Invalid characters" in response.error
```

---

## Success Criteria

### Functional Requirements

- [ ] `PredictionRequest` validates amino acids (20 + X)
- [ ] `PredictionRequest` rejects gaps (-, *, .)
- [ ] `PredictionRequest` enforces length limits (1-2000)
- [ ] `PredictionRequest` validates threshold (0-1)
- [ ] `PredictionRequest` validates assay_type enum
- [ ] `BatchPredictionRequest` validates all sequences
- [ ] `PredictionResult` structures output consistently
- [ ] Gradio app uses `PredictionRequest` (no `validate_input`)
- [ ] CLI uses `PredictionRequest`
- [ ] Predictor maintains backward compatibility

### Quality Gates

- [ ] All unit tests pass (≥18 tests)
- [ ] Integration tests pass
- [ ] `make test` passes
- [ ] `make lint` passes (ruff)
- [ ] `make typecheck` passes (mypy)
- [ ] Code coverage ≥70%
- [ ] No Pydantic ValidationErrors leak to user (wrapped in gr.Error)

---

## Rollout Plan

1. **PR 1: Models Only** (Low Risk)
   - Add `models/prediction.py`
   - Add tests
   - No integration yet

2. **PR 2: Predictor Integration** (Medium Risk)
   - Update `core/prediction.py`
   - Maintain backward compat with raw strings

3. **PR 3: UI Integration** (User-Facing)
   - Update `cli/app.py`
   - Update `cli/predict.py`
   - Delete old `validate_input`

---

## Backward Compatibility

**Critical:** Predictor must accept BOTH raw strings AND PydanticRequest to avoid breaking existing code.

**Implementation:**
```python
def predict_single(
    self,
    sequence: str | PredictionRequest,  # Union type
    threshold: float = 0.5,
    assay_type: str | None = None,
) -> PredictionResult:
    # Normalize to PredanticRequest internally
    if isinstance(sequence, str):
        request = PredictionRequest(sequence=sequence, ...)
    else:
        request = sequence
    # ... rest of logic ...
```

---

## Non-Goals (Out of Scope)

- ❌ Hydra config validation (Phase 2)
- ❌ DataFrame schemas (Phase 3)
- ❌ Model artifact validation (Phase 4)
- ❌ FastAPI endpoint creation (future work)

---

## Rollback Plan

If Phase 1 fails, rollback is trivial:
1. Delete `src/antibody_training_esm/models/`
2. Revert changes to `prediction.py`, `app.py`, `predict.py`
3. Remove `pydantic` from dependencies

---

**Last Updated:** 2025-11-20
**Next Phase:** [Phase 2: Configuration Safety](PYDANTIC_PHASE_2_CONFIGURATION_SAFETY.md)
