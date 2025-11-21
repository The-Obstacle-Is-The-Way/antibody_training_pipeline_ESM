"""Integration tests for Pydantic + Predictor."""

from collections.abc import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError

from antibody_training_esm.core.prediction import Predictor
from antibody_training_esm.models.prediction import (
    PredictionRequest,
    PredictionResult,
)


@pytest.fixture
def mock_predictor() -> Generator[Predictor, None, None]:
    """Mock predictor for integration testing."""
    with (
        patch("antibody_training_esm.core.prediction.load_model_from_npz"),
        patch("antibody_training_esm.core.prediction.joblib.load") as mock_load,
        patch(
            "antibody_training_esm.core.prediction.ESMEmbeddingExtractor"
        ) as mock_embedder_cls,
    ):
        # Setup mock classifier
        mock_classifier = MagicMock()
        # Class 0: Specific, Class 1: Non-specific
        mock_classifier.predict_proba.return_value = np.array([[0.2, 0.8]])

        # Mock predict method with signature for inspection
        def dummy_predict(
            embeddings: np.ndarray,
            threshold: float = 0.5,
            assay_type: str | None = None,
        ) -> np.ndarray:
            return np.array([1])  # Return 1 (non-specific)

        mock_classifier.predict = MagicMock(side_effect=dummy_predict)
        mock_classifier.predict.__code__ = dummy_predict.__code__

        mock_load.return_value = mock_classifier

        # Setup mock embedder
        mock_embedder = mock_embedder_cls.return_value
        mock_embedder.extract_batch_embeddings.return_value = np.zeros((1, 1280))

        predictor = Predictor(
            model_name="dummy_model",
            classifier_path="dummy_path.pkl",
            device="cpu",
        )

        yield predictor


def test_predictor_accepts_raw_string(mock_predictor: Predictor) -> None:
    """Predictor maintains backward compatibility with raw strings."""
    result = mock_predictor.predict_single("QVQLVQSGAEVK")

    assert isinstance(result, PredictionResult)
    assert result.prediction in ["specific", "non-specific"]
    assert 0 <= result.probability <= 1


def test_predictor_accepts_pydantic_model(mock_predictor: Predictor) -> None:
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


def test_predictor_rejects_invalid_sequence(mock_predictor: Predictor) -> None:
    """Invalid sequence raises ValidationError before ESM computation."""
    with pytest.raises(ValidationError):
        mock_predictor.predict_single("INVALID123")
