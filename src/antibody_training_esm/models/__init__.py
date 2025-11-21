"""
Pydantic models for runtime validation.

This package contains schema definitions for:
- Prediction requests/responses
- Configuration validation (Phase 2)
- Dataset schemas (Phase 3)
- Model artifacts (Phase 4)
"""

from antibody_training_esm.models.config import (
    ClassifierConfig,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    TrainingPipelineConfig,
)
from antibody_training_esm.models.prediction import (
    BatchPredictionRequest,
    PredictionRequest,
    PredictionResult,
)

__all__ = [
    # Prediction models
    "PredictionRequest",
    "BatchPredictionRequest",
    "PredictionResult",
    # Config models
    "ModelConfig",
    "DataConfig",
    "ClassifierConfig",
    "TrainingConfig",
    "ExperimentConfig",
    "TrainingPipelineConfig",
]
