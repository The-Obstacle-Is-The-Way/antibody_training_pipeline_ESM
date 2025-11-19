from pathlib import Path
from unittest.mock import MagicMock, patch

import gradio as gr
import pytest
from hydra import compose, initialize

from antibody_training_esm.cli.app import launch_gradio_app, main


@patch("gradio.Interface")
@patch("antibody_training_esm.cli.app.Predictor")
def test_launch_gradio_app(
    mock_predictor_cls: MagicMock, mock_interface: MagicMock, tmp_path: Path
) -> None:
    """
    Tests that the Gradio app launches with the correct parameters and logic.
    """
    # Setup mock predictor
    mock_predictor = mock_predictor_cls.return_value
    mock_predictor.predict_single.return_value = {
        "prediction": "non-specific",
        "probability": 0.875,
    }

    # Create a dummy classifier file
    classifier_path = tmp_path / "model.pkl"
    classifier_path.touch()

    with initialize(config_path="../../../src/antibody_training_esm/conf"):
        cfg = compose(
            config_name="predict",
            overrides=[f"classifier.path={classifier_path}"],
        )
        launch_gradio_app(cfg)

    # Assert that Predictor was initialized
    mock_predictor_cls.assert_called_once()

    # Assert that the Gradio interface was created
    mock_interface.assert_called_once()
    _, kwargs = mock_interface.call_args
    assert "fn" in kwargs
    assert "examples" in kwargs

    # Extract the prediction function
    predict_fn = kwargs["fn"]

    # --- Test Valid Prediction ---
    prediction, probability = predict_fn("QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYNMHWVR")

    # Verify predictor call
    mock_predictor.predict_single.assert_called_with(
        "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYNMHWVR"
    )

    # Verify output formatting
    assert prediction == "non-specific"
    assert probability == "87.5%"

    # --- Test Input Validation (Invalid Chars) ---
    with pytest.raises(gr.Error) as excinfo:
        predict_fn("QVQL...123")
    assert "Invalid characters found" in str(excinfo.value)

    # --- Test Input Validation (Empty) ---
    with pytest.raises(gr.Error) as excinfo:
        predict_fn("")
    assert "Input sequence cannot be empty" in str(excinfo.value)


@patch("antibody_training_esm.cli.app.launch_gradio_app")
def test_main(mock_launch_gradio_app: MagicMock, tmp_path: Path) -> None:
    """
    Tests the main function of the Gradio app.
    """
    # Create a dummy classifier file
    classifier_path = tmp_path / "model.pkl"
    classifier_path.touch()

    with initialize(config_path="../../../src/antibody_training_esm/conf"):
        cfg = compose(
            config_name="predict",
            overrides=[f"classifier.path={classifier_path}"],
        )
        main(cfg)

    mock_launch_gradio_app.assert_called_once_with(cfg)


def test_launch_gradio_app_no_classifier_path() -> None:
    """
    Tests that the Gradio app raises a ValueError when no classifier path is provided.
    """
    with initialize(config_path="../../../src/antibody_training_esm/conf"):
        cfg = compose(config_name="predict")
        with pytest.raises(ValueError):
            launch_gradio_app(cfg)


def test_launch_gradio_app_classifier_not_found(tmp_path: Path) -> None:
    """
    Tests that the Gradio app raises a FileNotFoundError when the classifier is not found.
    """
    with initialize(config_path="../../../src/antibody_training_esm/conf"):
        cfg = compose(
            config_name="predict",
            overrides=[f"classifier.path={tmp_path / 'non_existent_model.pkl'}"],
        )
        with pytest.raises(FileNotFoundError):
            launch_gradio_app(cfg)
