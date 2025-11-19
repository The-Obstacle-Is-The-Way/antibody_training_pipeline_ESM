"""
Unit Tests for Prediction CLI

This module contains unit tests for the antibody-predict CLI command.
It tests the command-line interface validation, error handling, and main execution flow,
ensuring that the CLI correctly processes arguments and handles missing files or configurations.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import DictConfig, OmegaConf

from antibody_training_esm.cli.predict import main


@pytest.fixture
def mock_predict_cfg(tmp_path: Path) -> DictConfig:
    """Create a mock Hydra configuration for prediction."""
    input_file = tmp_path / "input.csv"
    input_file.touch()

    output_file = tmp_path / "output.csv"

    classifier_path = tmp_path / "model.pkl"
    classifier_path.touch()

    cfg = OmegaConf.create(
        {
            "input_file": str(input_file),
            "output_file": str(output_file),
            "classifier": {"path": str(classifier_path)},
        }
    )
    # Cast to DictConfig for type checking
    assert isinstance(cfg, DictConfig)
    return cfg


def test_predict_cli_success(mock_predict_cfg: DictConfig) -> None:
    """Test successful execution of the predict CLI."""
    with (
        patch("antibody_training_esm.cli.predict.pd.read_csv") as mock_read_csv,
        patch(
            "antibody_training_esm.cli.predict.run_prediction"
        ) as mock_run_prediction,
    ):
        # Setup mocks
        mock_df = MagicMock()
        mock_read_csv.return_value = mock_df
        mock_run_prediction.return_value = mock_df

        # Run main
        main(mock_predict_cfg)

        # Verify calls
        mock_read_csv.assert_called_once_with(mock_predict_cfg.input_file)
        mock_run_prediction.assert_called_once_with(mock_df, mock_predict_cfg)
        mock_df.to_csv.assert_called_once_with(
            mock_predict_cfg.output_file, index=False
        )


def test_predict_cli_missing_input_file() -> None:
    """Test CLI fails when input_file is missing from config."""
    cfg = OmegaConf.create({"input_file": None, "classifier": {"path": "model.pkl"}})
    assert isinstance(cfg, DictConfig)

    with pytest.raises(ValueError, match="Input file must be specified"):
        main(cfg)


def test_predict_cli_missing_classifier_path(tmp_path: Path) -> None:
    """Test CLI fails when classifier.path is missing from config."""
    cfg = OmegaConf.create({"input_file": "input.csv", "classifier": {"path": None}})
    assert isinstance(cfg, DictConfig)

    with pytest.raises(ValueError, match="Classifier path must be specified"):
        main(cfg)


def test_predict_cli_classifier_file_not_found(tmp_path: Path) -> None:
    """Test CLI fails when the classifier file does not exist."""
    # Path that definitely doesn't exist
    non_existent_model = tmp_path / "ghost_model.pkl"

    cfg = OmegaConf.create(
        {"input_file": "input.csv", "classifier": {"path": str(non_existent_model)}}
    )
    assert isinstance(cfg, DictConfig)

    with pytest.raises(FileNotFoundError, match="Classifier file not found"):
        main(cfg)


def test_predict_cli_input_file_not_found(mock_predict_cfg: DictConfig) -> None:
    """Test CLI handles FileNotFoundError for input file gracefully."""
    # Force read_csv to raise FileNotFoundError
    with patch(
        "antibody_training_esm.cli.predict.pd.read_csv", side_effect=FileNotFoundError
    ):
        with pytest.raises(SystemExit) as exc_info:
            main(mock_predict_cfg)
        assert exc_info.value.code == 1


def test_predict_cli_generic_exception(mock_predict_cfg: DictConfig) -> None:
    """Test CLI handles generic exceptions gracefully."""
    # Force run_prediction to raise a generic Exception
    with (
        patch("antibody_training_esm.cli.predict.pd.read_csv") as mock_read_csv,
        patch(
            "antibody_training_esm.cli.predict.run_prediction",
            side_effect=Exception("Boom"),
        ),
    ):
        mock_read_csv.return_value = MagicMock()

        with pytest.raises(SystemExit) as exc_info:
            main(mock_predict_cfg)
        assert exc_info.value.code == 1
