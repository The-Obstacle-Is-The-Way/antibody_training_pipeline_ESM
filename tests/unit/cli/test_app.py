from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from hydra import compose, initialize

from antibody_training_esm.cli.app import launch_gradio_app, main


# Mock the Gradio interface to avoid launching the web app during tests
@patch("gradio.Interface")
def test_launch_gradio_app(mock_interface: MagicMock, tmp_path: Path) -> None:
    """
    Tests that the Gradio app launches with the correct parameters.
    """
    # Create a dummy classifier file
    classifier_path = tmp_path / "model.pkl"
    classifier_path.touch()

    with initialize(config_path="../../../src/antibody_training_esm/conf"):
        cfg = compose(
            config_name="predict",
            overrides=[f"classifier.path={classifier_path}"],
        )
        launch_gradio_app(cfg)

    # Assert that the Gradio interface was created and launched
    mock_interface.assert_called_once()
    mock_interface.return_value.launch.assert_called_once()


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
