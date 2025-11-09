"""
Unit tests for training CLI.

Tests cover:
- CLI argument parsing (--config flag)
- Default config path
- train_model() invocation
- Exit codes (success/failure)
- Error handling and reporting

Testing philosophy:
- Test behaviors, not implementation
- Mock train_model() (testing CLI, not trainer)
- Mock stdout/stderr capture
- Use real argparse (no mocking CLI logic)
- Follow AAA pattern (Arrange-Act-Assert)

Date: 2025-11-07
Phase: 4 (CLI & E2E Tests)
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Generator
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from antibody_training_esm.cli.train import main

# ==================== Fixtures ====================


@pytest.fixture
def mock_train_model() -> Generator[MagicMock, None, None]:
    """Mock train_model function"""
    with patch("antibody_training_esm.cli.train.train_model") as mock:
        yield mock


# ==================== CLI Argument Parsing Tests ====================


@pytest.mark.unit
def test_train_cli_uses_default_config_path(mock_train_model: MagicMock) -> None:
    """Verify CLI uses default config path when not specified"""
    # Arrange
    with patch("sys.argv", ["antibody-train"]):
        # Act
        exit_code = main()

        # Assert
        assert exit_code == 0
        mock_train_model.assert_called_once_with("configs/config.yaml")


@pytest.mark.unit
def test_train_cli_accepts_custom_config_path(mock_train_model: MagicMock) -> None:
    """Verify CLI accepts custom config path via --config"""
    # Arrange
    custom_config = "custom/path/config.yaml"
    with patch("sys.argv", ["antibody-train", "--config", custom_config]):
        # Act
        exit_code = main()

        # Assert
        assert exit_code == 0
        mock_train_model.assert_called_once_with(custom_config)


@pytest.mark.unit
def test_train_cli_accepts_short_flag(mock_train_model: MagicMock) -> None:
    """Verify CLI accepts -c short flag"""
    # Arrange
    custom_config = "test/config.yaml"
    with patch("sys.argv", ["antibody-train", "-c", custom_config]):
        # Act
        exit_code = main()

        # Assert
        assert exit_code == 0
        mock_train_model.assert_called_once_with(custom_config)


# ==================== Success Path Tests ====================


@pytest.mark.unit
def test_train_cli_returns_zero_on_success(mock_train_model: MagicMock) -> None:
    """Verify CLI returns 0 exit code when training succeeds"""
    # Arrange
    mock_train_model.return_value = None  # Success (no exception)
    with patch("sys.argv", ["antibody-train"]):
        # Act
        exit_code = main()

        # Assert
        assert exit_code == 0


@pytest.mark.unit
def test_train_cli_prints_success_message(mock_train_model: MagicMock) -> None:
    """Verify CLI prints success message after training"""
    # Arrange
    mock_train_model.return_value = None
    with patch("sys.argv", ["antibody-train"]):
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        # Act
        main()
        output = captured_output.getvalue()
        sys.stdout = old_stdout

        # Assert
        assert "Training completed successfully" in output
        assert "✅" in output


@pytest.mark.unit
def test_train_cli_prints_config_path_on_start(mock_train_model: MagicMock) -> None:
    """Verify CLI prints config path when starting"""
    # Arrange
    custom_config = "my_config.yaml"
    mock_train_model.return_value = None
    with patch("sys.argv", ["antibody-train", "--config", custom_config]):
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        # Act
        main()
        output = captured_output.getvalue()
        sys.stdout = old_stdout

        # Assert
        assert "Starting training" in output
        assert custom_config in output


# ==================== Error Handling Tests ====================


@pytest.mark.unit
def test_train_cli_returns_one_on_failure(mock_train_model: MagicMock) -> None:
    """Verify CLI returns 1 exit code when training fails"""
    # Arrange
    mock_train_model.side_effect = RuntimeError("Training failed")
    with patch("sys.argv", ["antibody-train"]):
        # Act
        exit_code = main()

        # Assert
        assert exit_code == 1


@pytest.mark.unit
def test_train_cli_prints_error_message_on_failure(mock_train_model: MagicMock) -> None:
    """Verify CLI prints error message to stderr when training fails"""
    # Arrange
    error_msg = "Config file not found"
    mock_train_model.side_effect = FileNotFoundError(error_msg)
    with patch("sys.argv", ["antibody-train"]):
        old_stderr = sys.stderr
        sys.stderr = captured_error = StringIO()

        # Act
        main()
        error = captured_error.getvalue()
        sys.stderr = old_stderr

        # Assert
        assert "Training failed" in error
        assert error_msg in error
        assert "❌" in error


@pytest.mark.unit
def test_train_cli_handles_value_error(mock_train_model: MagicMock) -> None:
    """Verify CLI handles ValueError from train_model"""
    # Arrange
    mock_train_model.side_effect = ValueError("Invalid configuration")
    with patch("sys.argv", ["antibody-train"]):
        # Act
        exit_code = main()

        # Assert
        assert exit_code == 1


@pytest.mark.unit
def test_train_cli_handles_file_not_found_error(mock_train_model: MagicMock) -> None:
    """Verify CLI handles FileNotFoundError from train_model"""
    # Arrange
    mock_train_model.side_effect = FileNotFoundError("Config not found")
    with patch("sys.argv", ["antibody-train"]):
        # Act
        exit_code = main()

        # Assert
        assert exit_code == 1


@pytest.mark.unit
def test_train_cli_handles_runtime_error(mock_train_model: MagicMock) -> None:
    """Verify CLI handles RuntimeError from train_model"""
    # Arrange
    mock_train_model.side_effect = RuntimeError("Training error")
    with patch("sys.argv", ["antibody-train"]):
        # Act
        exit_code = main()

        # Assert
        assert exit_code == 1


@pytest.mark.unit
def test_train_cli_handles_keyboard_interrupt(mock_train_model: MagicMock) -> None:
    """Verify CLI handles KeyboardInterrupt gracefully"""
    # Arrange
    mock_train_model.side_effect = KeyboardInterrupt()
    with patch("sys.argv", ["antibody-train"]):
        old_stderr = sys.stderr
        sys.stderr = captured_error = StringIO()

        # Act
        exit_code = main()
        error = captured_error.getvalue()
        sys.stderr = old_stderr

        # Assert
        assert exit_code == 1
        assert "Training failed" in error


# ==================== Help Message Tests ====================


@pytest.mark.unit
def test_train_cli_shows_help_message() -> None:
    """Verify CLI shows help message with --help"""
    # Arrange
    with patch("sys.argv", ["antibody-train", "--help"]):
        # Act & Assert
        with pytest.raises(SystemExit) as exc_info:
            main()

        # --help exits with code 0
        assert exc_info.value.code == 0


@pytest.mark.unit
def test_train_cli_help_describes_config_option() -> None:
    """Verify CLI help describes --config option"""
    # Arrange
    with patch("sys.argv", ["antibody-train", "--help"]):
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        # Act
        with contextlib.suppress(SystemExit):
            main()

        output = captured_output.getvalue()
        sys.stdout = old_stdout

        # Assert
        assert "--config" in output or "-c" in output
        assert "config" in output.lower()


# ==================== Integration Tests ====================


@pytest.mark.unit
def test_train_cli_invokes_train_model_exactly_once(
    mock_train_model: MagicMock,
) -> None:
    """Verify CLI invokes train_model exactly once"""
    # Arrange
    with patch("sys.argv", ["antibody-train"]):
        # Act
        main()

        # Assert
        assert mock_train_model.call_count == 1


@pytest.mark.unit
def test_train_cli_passes_config_path_to_train_model(
    mock_train_model: MagicMock,
) -> None:
    """Verify CLI passes correct config path to train_model"""
    # Arrange
    test_configs = [
        "config1.yaml",
        "path/to/config2.yaml",
        "../configs/test.yaml",
    ]

    for config_path in test_configs:
        mock_train_model.reset_mock()
        with patch("sys.argv", ["antibody-train", "--config", config_path]):
            # Act
            main()

            # Assert
            mock_train_model.assert_called_once_with(config_path)


# ==================== Edge Case Tests ====================


@pytest.mark.unit
def test_train_cli_handles_empty_string_config() -> None:
    """Verify CLI handles empty string config path"""
    # Arrange
    with patch("antibody_training_esm.cli.train.train_model") as mock_train:
        mock_train.side_effect = FileNotFoundError("Empty config")
        with patch("sys.argv", ["antibody-train", "--config", ""]):
            # Act
            exit_code = main()

            # Assert
            assert exit_code == 1


@pytest.mark.unit
def test_train_cli_handles_exception_with_no_message(
    mock_train_model: MagicMock,
) -> None:
    """Verify CLI handles exception with no message"""
    # Arrange
    mock_train_model.side_effect = Exception()
    with patch("sys.argv", ["antibody-train"]):
        old_stderr = sys.stderr
        sys.stderr = captured_error = StringIO()

        # Act
        exit_code = main()
        error = captured_error.getvalue()
        sys.stderr = old_stderr

        # Assert
        assert exit_code == 1
        assert "Training failed" in error
