"""Unit tests for predefined fold cross-validation in trainer.py."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from antibody_training_esm.core.trainer import train_model_with_predefined_folds


@pytest.mark.unit
class TestPredefinedFoldCV:
    """Test suite for train_model_with_predefined_folds function."""

    @pytest.fixture
    def mock_sequences(self) -> list[str]:
        """Create mock antibody sequences for testing."""
        return [
            "QVQLQQPGAELVKPGASVKMSCK" * 5,  # Fold 0
            "EVQLVESGGGLVQPGGSLRLSCA" * 5,  # Fold 0
            "QVKLQESGAELARPGASVKLSCK" * 5,  # Fold 1
            "EVQLVESGGGLVKPGGSLRLSCA" * 5,  # Fold 1
            "QVQLQQSGPELVKPGASVKMSCG" * 5,  # Fold 2
        ]

    @pytest.fixture
    def mock_labels(self) -> np.ndarray:
        """Create mock continuous labels for testing."""
        return np.array([0.2, 0.3, 0.4, 0.5, 0.6])

    @pytest.fixture
    def mock_folds(self) -> np.ndarray:
        """Create mock fold assignments."""
        return np.array([0, 0, 1, 1, 2])

    @pytest.fixture
    def mock_config(self, tmp_path: Path) -> dict:
        """Create mock config dictionary."""
        return {
            "model": {
                "name": "facebook/esm2_t33_650M_UR50D",
            },
            "regressor": {
                "alpha": 1.0,
            },
            "training": {
                "batch_size": 2,
            },
            "hardware": {
                "device": "cpu",
            },
            "data": {
                "embeddings_cache_dir": "./embeddings_cache",
            },
        }

    @patch("antibody_training_esm.core.trainer.get_or_create_embeddings")
    @patch("antibody_training_esm.core.trainer.AntibodyRegressor")
    def test_function_exists(
        self,
        mock_regressor_class: Mock,
        mock_get_embeddings: Mock,
    ) -> None:
        """Test that train_model_with_predefined_folds function exists."""
        assert callable(train_model_with_predefined_folds)

    @patch("antibody_training_esm.core.trainer.get_or_create_embeddings")
    @patch("antibody_training_esm.core.trainer.AntibodyRegressor")
    def test_returns_tuple_of_three_elements(
        self,
        mock_regressor_class: Mock,
        mock_get_embeddings: Mock,
        mock_sequences: list[str],
        mock_labels: np.ndarray,
        mock_folds: np.ndarray,
        mock_config: dict,
    ) -> None:
        """Test that function returns (model, cv_results, oof_predictions)."""
        # Mock embedding extraction
        mock_embeddings = np.random.randn(len(mock_sequences), 1280)
        mock_get_embeddings.return_value = mock_embeddings

        # Mock regressor behavior
        mock_regressor = MagicMock()
        mock_regressor.fit.return_value = mock_regressor
        # Return predictions matching input size (now embeddings, not sequences)
        mock_regressor.predict.side_effect = lambda emb, **kwargs: np.random.randn(
            len(emb)
        )
        mock_regressor_class.return_value = mock_regressor

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            result = train_model_with_predefined_folds(
                sequences=mock_sequences,
                labels=mock_labels,
                fold_assignments=mock_folds,
                config=mock_config,
                output_dir=output_dir,
            )

            # Check return type
            assert isinstance(result, tuple)
            assert len(result) == 3

            model, cv_results, oof_predictions = result
            assert isinstance(cv_results, dict)
            assert isinstance(oof_predictions, np.ndarray)

    @patch("antibody_training_esm.core.trainer.get_or_create_embeddings")
    @patch("antibody_training_esm.core.trainer.AntibodyRegressor")
    def test_cv_results_contains_spearman_metrics(
        self,
        mock_regressor_class: Mock,
        mock_get_embeddings: Mock,
        mock_sequences: list[str],
        mock_labels: np.ndarray,
        mock_folds: np.ndarray,
        mock_config: dict,
    ) -> None:
        """Test that cv_results contains Spearman correlation metrics."""
        # Mock embedding extraction
        mock_embeddings = np.random.randn(len(mock_sequences), 1280)
        mock_get_embeddings.return_value = mock_embeddings

        mock_regressor = MagicMock()
        mock_regressor.fit.return_value = mock_regressor
        mock_regressor.predict.side_effect = lambda emb, **kwargs: np.random.randn(
            len(emb)
        )
        mock_regressor_class.return_value = mock_regressor

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            _, cv_results, _ = train_model_with_predefined_folds(
                sequences=mock_sequences,
                labels=mock_labels,
                fold_assignments=mock_folds,
                config=mock_config,
                output_dir=output_dir,
            )

            # Check that Spearman metrics are present
            assert "cv_spearman_mean" in cv_results
            assert "cv_spearman_std" in cv_results
            assert "overall_spearman" in cv_results
            assert "overall_spearman_pval" in cv_results
            assert "fold_metrics" in cv_results

    @patch("antibody_training_esm.core.trainer.get_or_create_embeddings")
    @patch("antibody_training_esm.core.trainer.AntibodyRegressor")
    def test_fold_metrics_list_has_correct_length(
        self,
        mock_regressor_class: Mock,
        mock_get_embeddings: Mock,
        mock_sequences: list[str],
        mock_labels: np.ndarray,
        mock_folds: np.ndarray,
        mock_config: dict,
    ) -> None:
        """Test that fold_metrics has one entry per unique fold."""
        # Mock embedding extraction
        mock_embeddings = np.random.randn(len(mock_sequences), 1280)
        mock_get_embeddings.return_value = mock_embeddings

        mock_regressor = MagicMock()
        mock_regressor.fit.return_value = mock_regressor
        mock_regressor.predict.side_effect = lambda emb, **kwargs: np.random.randn(
            len(emb)
        )
        mock_regressor_class.return_value = mock_regressor

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            _, cv_results, _ = train_model_with_predefined_folds(
                sequences=mock_sequences,
                labels=mock_labels,
                fold_assignments=mock_folds,
                config=mock_config,
                output_dir=output_dir,
            )

            fold_metrics = cv_results["fold_metrics"]

            # 3 unique folds: 0, 1, 2
            assert len(fold_metrics) == 3

            # Each fold should have required metrics
            for fold_metric in fold_metrics:
                assert "fold" in fold_metric
                assert "spearman" in fold_metric
                assert "spearman_pval" in fold_metric
                assert "n_samples" in fold_metric

    @patch("antibody_training_esm.core.trainer.get_or_create_embeddings")
    @patch("antibody_training_esm.core.trainer.AntibodyRegressor")
    def test_oof_predictions_same_length_as_input(
        self,
        mock_regressor_class: Mock,
        mock_get_embeddings: Mock,
        mock_sequences: list[str],
        mock_labels: np.ndarray,
        mock_folds: np.ndarray,
        mock_config: dict,
    ) -> None:
        """Test that out-of-fold predictions have same length as input."""
        # Mock embedding extraction
        mock_embeddings = np.random.randn(len(mock_sequences), 1280)
        mock_get_embeddings.return_value = mock_embeddings

        mock_regressor = MagicMock()
        mock_regressor.fit.return_value = mock_regressor
        mock_regressor.predict.side_effect = lambda emb, **kwargs: np.random.randn(
            len(emb)
        )
        mock_regressor_class.return_value = mock_regressor

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            _, _, oof_predictions = train_model_with_predefined_folds(
                sequences=mock_sequences,
                labels=mock_labels,
                fold_assignments=mock_folds,
                config=mock_config,
                output_dir=output_dir,
            )

            # OOF predictions should match input length
            assert len(oof_predictions) == len(mock_sequences)
            assert oof_predictions.dtype in [np.float64, np.float32]

    @patch("antibody_training_esm.core.trainer.get_or_create_embeddings")
    @patch("antibody_training_esm.core.trainer.AntibodyRegressor")
    def test_final_model_trained_on_full_data(
        self,
        mock_regressor_class: Mock,
        mock_get_embeddings: Mock,
        mock_sequences: list[str],
        mock_labels: np.ndarray,
        mock_folds: np.ndarray,
        mock_config: dict,
    ) -> None:
        """Test that final model is trained on full dataset."""
        # Mock embedding extraction
        mock_embeddings = np.random.randn(len(mock_sequences), 1280)
        mock_get_embeddings.return_value = mock_embeddings

        mock_regressor = MagicMock()
        mock_regressor.fit.return_value = mock_regressor
        mock_regressor.predict.side_effect = lambda emb, **kwargs: np.random.randn(
            len(emb)
        )
        mock_regressor_class.return_value = mock_regressor

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            model, _, _ = train_model_with_predefined_folds(
                sequences=mock_sequences,
                labels=mock_labels,
                fold_assignments=mock_folds,
                config=mock_config,
                output_dir=output_dir,
            )

            # Check that fit was called multiple times (once per fold + final)
            # 3 folds + 1 final = 4 total calls
            assert mock_regressor.fit.call_count == 4

            # Check that final fit was on all embeddings (not sequences)
            final_fit_call = mock_regressor.fit.call_args_list[-1]
            final_embeddings = final_fit_call[0][0]  # First positional arg
            assert len(final_embeddings) == len(mock_sequences)

    @patch("antibody_training_esm.core.trainer.get_or_create_embeddings")
    @patch("antibody_training_esm.core.trainer.AntibodyRegressor")
    def test_model_saved_to_output_dir(
        self,
        mock_regressor_class: Mock,
        mock_get_embeddings: Mock,
        mock_sequences: list[str],
        mock_labels: np.ndarray,
        mock_folds: np.ndarray,
        mock_config: dict,
    ) -> None:
        """Test that trained model is saved to output directory."""
        # Mock embedding extraction
        mock_embeddings = np.random.randn(len(mock_sequences), 1280)
        mock_get_embeddings.return_value = mock_embeddings

        mock_regressor = MagicMock()
        mock_regressor.fit.return_value = mock_regressor
        mock_regressor.predict.side_effect = lambda emb, **kwargs: np.random.randn(
            len(emb)
        )
        mock_regressor_class.return_value = mock_regressor

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            train_model_with_predefined_folds(
                sequences=mock_sequences,
                labels=mock_labels,
                fold_assignments=mock_folds,
                config=mock_config,
                output_dir=output_dir,
            )

            # Check that save was called
            mock_regressor.save.assert_called_once()

            # Check that save path is in output_dir
            save_call = mock_regressor.save.call_args[0][0]
            assert str(output_dir) in save_call
            assert "ginkgo_regressor.pkl" in save_call
