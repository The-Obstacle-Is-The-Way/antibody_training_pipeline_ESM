"""
Pydantic models for runtime validation.

This package contains schema definitions for:
- Prediction requests/responses
- Configuration validation (Phase 2)
- Dataset schemas (Phase 3)
- Model artifacts (Phase 4)
"""

from antibody_training_esm.models.prediction import (
    AssayType,
    BatchPredictionRequest,
    PredictionRequest,
    PredictionResult,
)

__all__ = [
    "AssayType",
    "PredictionRequest",
    "BatchPredictionRequest",
    "PredictionResult",
]
