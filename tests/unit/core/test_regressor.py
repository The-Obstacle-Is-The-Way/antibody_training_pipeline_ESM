"""Unit tests for AntibodyRegressor class."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from antibody_training_esm.core.regressor import AntibodyRegressor


@pytest.mark.unit
class TestAntibodyRegressor:
    """Test suite for AntibodyRegressor class."""

    @pytest.fixture
    def mock_sequences(self) -> list[str]:
        """Create mock antibody sequences for testing."""
        return [
            "QVQLQQPGAELVKPGASVKMSCKASGYTFT",  # Mock VH sequence 1
            "EVQLVESGGGLVQPGGSLRLSCAASGFTFS",  # Mock VH sequence 2
            "QVKLQESGAELARPGASVKLSCKASGYTFT",  # Mock VH sequence 3
        ]

    @pytest.fixture
    def mock_labels(self) -> np.ndarray:
        """Create mock continuous labels for testing."""
        return np.array([0.3, 0.7, 0.5])

    def test_regressor_initialization(self) -> None:
        """Test that regressor can be initialized with default parameters."""
        regressor = AntibodyRegressor(
            model_name="facebook/esm2_t33_650M_UR50D",
            device="cpu",
            batch_size=8,
            alpha=1.0,
        )

        assert regressor.model_name == "facebook/esm2_t33_650M_UR50D"
        assert regressor.device == "cpu"
        assert regressor.batch_size == 8
        assert regressor.alpha == 1.0
        assert not regressor._is_fitted

    def test_regressor_initialization_with_kwargs(self) -> None:
        """Test that regressor can be initialized with additional Ridge kwargs."""
        regressor = AntibodyRegressor(
            alpha=0.5,
            fit_intercept=False,
            max_iter=500,
        )

        assert regressor.alpha == 0.5
        assert regressor.regressor_kwargs == {"fit_intercept": False, "max_iter": 500}

    @patch("antibody_training_esm.core.regressor.ESMEmbeddingExtractor")
    def test_fit_trains_regressor(
        self,
        mock_extractor_class: Mock,
        mock_sequences: list[str],
        mock_labels: np.ndarray,
    ) -> None:
        """Test that fit() trains the Ridge regressor on embeddings."""
        # Mock embedding extraction
        mock_embeddings = np.random.randn(3, 1280)  # 3 sequences, 1280-dim embeddings
        mock_extractor = MagicMock()
        mock_extractor.extract_batch_embeddings.return_value = mock_embeddings
        mock_extractor_class.return_value = mock_extractor

        # Create and fit regressor
        regressor = AntibodyRegressor(alpha=1.0)
        regressor.fit(mock_sequences, mock_labels)

        # Check that embeddings were extracted
        mock_extractor.extract_batch_embeddings.assert_called_once()

        # Check that regressor was fitted
        assert regressor._is_fitted

    @patch("antibody_training_esm.core.regressor.ESMEmbeddingExtractor")
    def test_predict_returns_continuous_values(
        self,
        mock_extractor_class: Mock,
        mock_sequences: list[str],
        mock_labels: np.ndarray,
    ) -> None:
        """Test that predict() returns continuous float values."""
        # Mock embedding extraction
        mock_embeddings = np.random.randn(3, 1280)
        mock_extractor = MagicMock()
        mock_extractor.extract_batch_embeddings.return_value = mock_embeddings
        mock_extractor_class.return_value = mock_extractor

        # Create, fit, and predict
        regressor = AntibodyRegressor(alpha=1.0)
        regressor.fit(mock_sequences, mock_labels)
        predictions = regressor.predict(mock_sequences)

        # Check output type and shape
        assert isinstance(predictions, np.ndarray)
        assert predictions.dtype in [np.float64, np.float32]
        assert predictions.shape == (3,)

    @patch("antibody_training_esm.core.regressor.ESMEmbeddingExtractor")
    def test_predict_raises_if_not_fitted(
        self,
        mock_extractor_class: Mock,
        mock_sequences: list[str],
    ) -> None:
        """Test that predict() raises error if model not fitted."""
        mock_extractor = MagicMock()
        mock_extractor_class.return_value = mock_extractor

        regressor = AntibodyRegressor(alpha=1.0)

        with pytest.raises(ValueError, match="Model must be fitted before predict"):
            regressor.predict(mock_sequences)

    @patch("antibody_training_esm.core.regressor.ESMEmbeddingExtractor")
    def test_save_and_load_model(
        self,
        mock_extractor_class: Mock,
        mock_sequences: list[str],
        mock_labels: np.ndarray,
    ) -> None:
        """Test that model can be saved and loaded."""
        # Mock embedding extraction
        mock_embeddings = np.random.randn(3, 1280)
        mock_extractor = MagicMock()
        mock_extractor.extract_batch_embeddings.return_value = mock_embeddings
        mock_extractor_class.return_value = mock_extractor

        # Create and fit regressor
        regressor = AntibodyRegressor(alpha=1.0, device="cpu", batch_size=8)
        regressor.fit(mock_sequences, mock_labels)

        # Save model
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            model_path = f.name

        try:
            regressor.save(model_path)

            # Load model
            loaded_regressor = AntibodyRegressor.load(model_path)

            # Check that loaded model has same parameters
            assert loaded_regressor.model_name == regressor.model_name
            assert loaded_regressor.device == regressor.device
            assert loaded_regressor.batch_size == regressor.batch_size
            assert loaded_regressor.alpha == regressor.alpha
            assert loaded_regressor._is_fitted

        finally:
            Path(model_path).unlink(missing_ok=True)

    @patch("antibody_training_esm.core.regressor.ESMEmbeddingExtractor")
    def test_save_raises_if_not_fitted(
        self,
        mock_extractor_class: Mock,
    ) -> None:
        """Test that save() raises error if model not fitted."""
        mock_extractor = MagicMock()
        mock_extractor_class.return_value = mock_extractor

        regressor = AntibodyRegressor(alpha=1.0)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            model_path = f.name

        try:
            with pytest.raises(ValueError, match="Cannot save unfitted model"):
                regressor.save(model_path)
        finally:
            Path(model_path).unlink(missing_ok=True)

    @patch("antibody_training_esm.core.regressor.ESMEmbeddingExtractor")
    def test_fit_accepts_cache_path_for_compatibility(
        self,
        mock_extractor_class: Mock,
        mock_sequences: list[str],
        mock_labels: np.ndarray,
    ) -> None:
        """Test that fit() accepts _cache_path for API compatibility (but ignores it)."""
        mock_embeddings = np.random.randn(3, 1280)
        mock_extractor = MagicMock()
        mock_extractor.extract_batch_embeddings.return_value = mock_embeddings
        mock_extractor_class.return_value = mock_extractor

        regressor = AntibodyRegressor(alpha=1.0)
        cache_path = "embeddings_cache/test.npy"
        regressor.fit(mock_sequences, mock_labels, _cache_path=cache_path)

        # Check that extract_batch_embeddings was called (cache_path is ignored)
        mock_extractor.extract_batch_embeddings.assert_called_once_with(mock_sequences)

    @patch("antibody_training_esm.core.regressor.ESMEmbeddingExtractor")
    def test_predict_accepts_cache_path_for_compatibility(
        self,
        mock_extractor_class: Mock,
        mock_sequences: list[str],
        mock_labels: np.ndarray,
    ) -> None:
        """Test that predict() accepts _cache_path for API compatibility (but ignores it)."""
        mock_embeddings = np.random.randn(3, 1280)
        mock_extractor = MagicMock()
        mock_extractor.extract_batch_embeddings.return_value = mock_embeddings
        mock_extractor_class.return_value = mock_extractor

        regressor = AntibodyRegressor(alpha=1.0)
        regressor.fit(mock_sequences, mock_labels)

        cache_path = "embeddings_cache/test_predict.npy"
        regressor.predict(mock_sequences, _cache_path=cache_path)

        # Check that extract_batch_embeddings was called twice (fit + predict)
        assert mock_extractor.extract_batch_embeddings.call_count == 2
