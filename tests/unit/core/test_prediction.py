from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from antibody_training_esm.core.prediction import Predictor, run_prediction


@pytest.fixture
def sample_input_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence": [
                "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAKDRLGRYFDYWGQGTLVTVSS",
                "QVQLQESGPGLVKPSETLSLTCTVSGGSISSSYYWGWIRQPPGKGLEWIGSIYYSGSTYYNPSLKSRVTISVDTSKNQFSLKLSSVTAADTAVYYCARDRLGRYFDYWGQGTLVTVSS",
            ]
        }
    )


def test_predictor_creates_embedder_when_missing(sample_input_df: pd.DataFrame) -> None:
    """Test that Predictor creates an embedder if the classifier doesn't have one."""
    with (
        patch("joblib.load") as mock_joblib_load,
        patch(
            "antibody_training_esm.core.prediction.ESMEmbeddingExtractor"
        ) as mock_embedder_cls,
    ):
        # Setup mock classifier WITHOUT an embedding_extractor
        mock_classifier = MagicMock()
        del (
            mock_classifier.embedding_extractor
        )  # Explicitly remove it to trigger fallback

        # Configure mock to support introspection (needed for predict method check)
        mock_code = MagicMock()
        mock_code.co_varnames = ("X",)
        mock_classifier.predict.__code__ = mock_code

        mock_classifier.predict.return_value = np.array([1, 0])
        mock_classifier.predict_proba.return_value = np.array(
            [[0.15, 0.85], [0.77, 0.23]]
        )
        mock_joblib_load.return_value = mock_classifier

        # Mock embedder instance
        mock_embedder_instance = mock_embedder_cls.return_value
        mock_embedder_instance.extract_batch_embeddings.return_value = np.zeros(
            (2, 1280)
        )

        # Initialize
        predictor = Predictor("model", "path")

        # Run
        output_df = predictor.predict_dataframe(sample_input_df)

        # Assertions
        mock_embedder_cls.assert_called_once()  # Should be called!
        assert "prediction" in output_df.columns


def test_predictor_reuses_embedder(sample_input_df: pd.DataFrame) -> None:
    """Test that Predictor reuses the classifier's embedder if available."""
    with (
        patch("joblib.load") as mock_joblib_load,
        patch(
            "antibody_training_esm.core.prediction.ESMEmbeddingExtractor"
        ) as mock_embedder_cls,
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=True),
    ):
        # Setup mock classifier WITH an embedding_extractor
        mock_classifier = MagicMock()
        # It has .embedding_extractor by default (MagicMock behavior), so we keep it.

        # Configure mock to support introspection
        mock_code = MagicMock()
        mock_code.co_varnames = ("X",)
        mock_classifier.predict.__code__ = mock_code

        mock_classifier.predict.return_value = np.array([1, 0])
        mock_classifier.predict_proba.return_value = np.array(
            [[0.15, 0.85], [0.77, 0.23]]
        )
        mock_joblib_load.return_value = mock_classifier

        # The classifier's embedded extractor
        existing_embedder = mock_classifier.embedding_extractor
        existing_embedder.extract_batch_embeddings.return_value = np.zeros((2, 1280))
        # CRITICAL: Match the device so Predictor doesn't recreate it
        existing_embedder.device = "mps"

        # Initialize (will auto-detect "mps" because we mocked is_available=True)
        predictor = Predictor("model", "path")

        # Run
        output_df = predictor.predict_dataframe(sample_input_df)

        # Assertions
        mock_embedder_cls.assert_not_called()  # Optimization: Should NOT be called!
        existing_embedder.extract_batch_embeddings.assert_called_once()  # The existing one should be used
        assert "prediction" in output_df.columns


def test_predictor_missing_column(sample_input_df: pd.DataFrame) -> None:
    """Test error handling for missing column."""
    predictor = Predictor("model", "path")
    bad_df = pd.DataFrame({"wrong_col": ["SEQ"]})

    with pytest.raises(ValueError, match="Input DataFrame must contain"):
        predictor.predict_dataframe(bad_df)


def test_run_prediction_wrapper(sample_input_df: pd.DataFrame) -> None:
    """Test the backward-compatible wrapper function."""
    with patch("antibody_training_esm.core.prediction.Predictor") as mock_predictor_cls:
        mock_instance = MagicMock()
        mock_predictor_cls.return_value = mock_instance

        cfg = OmegaConf.create(
            {
                "model": {"name": "test_model"},
                "classifier": {"path": "test_path"},
                "sequence_column": "sequence",
                "threshold": 0.5,
                "assay_type": None,
            }
        )

        run_prediction(sample_input_df, cfg)

        # Argument check needs to be loose since we added config_path=None by default in wrapper
        # or strict if we know the defaults.
        # The wrapper passes cfg.classifier.path, so we check calls.
        # However, mock_predictor_cls call args will now include config_path=None.
        # Let's update the assertion to reflect reality or just check kwargs.
        call_args = mock_predictor_cls.call_args
        assert call_args.kwargs["model_name"] == "test_model"
        assert call_args.kwargs["classifier_path"] == "test_path"

        mock_instance.predict_dataframe.assert_called_with(
            sample_input_df, sequence_col="sequence", threshold=0.5, assay_type=None
        )


def test_predictor_loads_from_npz_with_implicit_config(tmp_path: Path) -> None:
    """Test loading from .npz infers the json config path."""
    npz_path = tmp_path / "model.npz"
    json_path = tmp_path / "model_config.json"
    npz_path.touch()
    json_path.touch()

    with patch(
        "antibody_training_esm.core.prediction.load_model_from_npz"
    ) as mock_load:
        predictor = Predictor(model_name="model", classifier_path=str(npz_path))
        # Trigger lazy load
        _ = predictor.classifier

        mock_load.assert_called_once_with(str(npz_path), str(json_path))


def test_predictor_loads_from_npz_with_explicit_config(tmp_path: Path) -> None:
    """Test loading from .npz uses the provided config path."""
    npz_path = tmp_path / "model.npz"
    custom_json = tmp_path / "custom.json"
    npz_path.touch()
    custom_json.touch()

    with patch(
        "antibody_training_esm.core.prediction.load_model_from_npz"
    ) as mock_load:
        predictor = Predictor(
            model_name="model",
            classifier_path=str(npz_path),
            config_path=str(custom_json),
        )
        # Trigger lazy load
        _ = predictor.classifier

        mock_load.assert_called_once_with(str(npz_path), str(custom_json))


def test_predictor_raises_error_if_json_missing_for_npz(tmp_path: Path) -> None:
    """Test FileNotFoundError if the JSON config is missing for .npz."""
    npz_path = tmp_path / "model.npz"
    npz_path.touch()
    # json config NOT created

    predictor = Predictor(model_name="model", classifier_path=str(npz_path))

    with pytest.raises(FileNotFoundError, match="JSON config not found"):
        _ = predictor.classifier
