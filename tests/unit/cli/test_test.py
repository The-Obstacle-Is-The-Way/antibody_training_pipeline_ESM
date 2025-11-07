"""
Unit tests for test CLI.

Tests cover:
- CLI argument parsing (--model, --data, --config flags)
- Config loading from YAML
- Model loading from pickle
- Dataset loading from CSV
- Error handling (missing files, invalid inputs)
- Exit codes

Testing philosophy:
- Test behaviors, not implementation
- Mock I/O boundaries (file system, plotting)
- Use real argument parsing
- Follow AAA pattern (Arrange-Act-Assert)

NOTE: This tests the CLI layer, not ModelTester internals.
      ModelTester integration is covered in E2E tests.

Date: 2025-11-07
Phase: 4 (CLI & E2E Tests)
"""

import contextlib
import sys
from io import StringIO
from unittest.mock import Mock, patch

import pytest
import yaml

from antibody_training_esm.cli.test import TestConfig, create_sample_test_config, main

# ==================== Fixtures ====================


@pytest.fixture
def mock_model_tester():
    """Mock ModelTester class"""
    with patch("antibody_training_esm.cli.test.ModelTester") as mock:
        instance = Mock()
        instance.run_comprehensive_test.return_value = {
            "test_dataset": {
                "test_model": {
                    "test_scores": {"accuracy": 0.85, "f1": 0.82},
                    "confusion_matrix": [[10, 2], [3, 15]],
                }
            }
        }
        mock.return_value = instance
        yield mock


# ==================== CLI Argument Parsing Tests ====================


@pytest.mark.unit
def test_test_cli_requires_model_and_data_or_config():
    """Verify CLI requires either --config or both --model and --data"""
    # Arrange
    with patch("sys.argv", ["antibody-test"]):
        # Act & Assert
        with pytest.raises(SystemExit) as exc_info:
            main()

        # argparse error exits with code 2
        assert exc_info.value.code == 2


@pytest.mark.unit
def test_test_cli_accepts_model_and_data_arguments(mock_model_tester):
    """Verify CLI accepts --model and --data arguments"""
    # Arrange
    with patch(
        "sys.argv",
        [
            "antibody-test",
            "--model",
            "model.pkl",
            "--data",
            "data.csv",
        ],
    ):
        # Act
        exit_code = main()

        # Assert
        assert exit_code == 0
        mock_model_tester.assert_called_once()


@pytest.mark.unit
def test_test_cli_accepts_multiple_models_and_datasets(mock_model_tester):
    """Verify CLI accepts multiple --model and --data paths"""
    # Arrange
    with patch(
        "sys.argv",
        [
            "antibody-test",
            "--model",
            "model1.pkl",
            "model2.pkl",
            "--data",
            "data1.csv",
            "data2.csv",
        ],
    ):
        # Act
        exit_code = main()

        # Assert
        assert exit_code == 0
        # Verify ModelTester received both models and datasets
        call_args = mock_model_tester.call_args[0][0]
        assert len(call_args.model_paths) == 2
        assert len(call_args.data_paths) == 2


@pytest.mark.unit
def test_test_cli_accepts_config_file(mock_model_tester, tmp_path):
    """Verify CLI accepts --config argument"""
    # Arrange
    config_file = tmp_path / "test_config.yaml"
    config_data = {
        "model_paths": ["model.pkl"],
        "data_paths": ["data.csv"],
    }
    config_file.write_text(yaml.dump(config_data))

    with patch("sys.argv", ["antibody-test", "--config", str(config_file)]):
        # Act
        exit_code = main()

        # Assert
        assert exit_code == 0


# ==================== Config Loading Tests ====================


@pytest.mark.unit
def test_test_cli_loads_config_from_yaml(mock_model_tester, tmp_path):
    """Verify CLI correctly loads configuration from YAML file"""
    # Arrange
    config_file = tmp_path / "test_config.yaml"
    config_data = {
        "model_paths": ["model1.pkl", "model2.pkl"],
        "data_paths": ["data.csv"],
        "output_dir": "./results",
        "device": "cpu",
        "batch_size": 16,
    }
    config_file.write_text(yaml.dump(config_data))

    with patch("sys.argv", ["antibody-test", "--config", str(config_file)]):
        # Act
        main()

        # Assert
        call_args = mock_model_tester.call_args[0][0]
        assert call_args.model_paths == ["model1.pkl", "model2.pkl"]
        assert call_args.data_paths == ["data.csv"]
        assert call_args.device == "cpu"
        assert call_args.batch_size == 16


# ==================== Command Line Override Tests ====================


@pytest.mark.unit
def test_test_cli_overrides_device_from_config(mock_model_tester, tmp_path):
    """Verify CLI --device overrides config file device"""
    # Arrange
    config_file = tmp_path / "test_config.yaml"
    config_data = {
        "model_paths": ["model.pkl"],
        "data_paths": ["data.csv"],
        "device": "cpu",
    }
    config_file.write_text(yaml.dump(config_data))

    with patch(
        "sys.argv",
        ["antibody-test", "--config", str(config_file), "--device", "cuda"],
    ):
        # Act
        main()

        # Assert
        call_args = mock_model_tester.call_args[0][0]
        assert call_args.device == "cuda"


@pytest.mark.unit
def test_test_cli_overrides_batch_size_from_config(mock_model_tester, tmp_path):
    """Verify CLI --batch-size overrides config file batch_size"""
    # Arrange
    config_file = tmp_path / "test_config.yaml"
    config_data = {
        "model_paths": ["model.pkl"],
        "data_paths": ["data.csv"],
        "batch_size": 32,
    }
    config_file.write_text(yaml.dump(config_data))

    with patch(
        "sys.argv",
        ["antibody-test", "--config", str(config_file), "--batch-size", "64"],
    ):
        # Act
        main()

        # Assert
        call_args = mock_model_tester.call_args[0][0]
        assert call_args.batch_size == 64


# ==================== Success Path Tests ====================


@pytest.mark.unit
def test_test_cli_returns_zero_on_success(mock_model_tester):
    """Verify CLI returns 0 exit code when testing succeeds"""
    # Arrange
    with patch(
        "sys.argv",
        ["antibody-test", "--model", "model.pkl", "--data", "data.csv"],
    ):
        # Act
        exit_code = main()

        # Assert
        assert exit_code == 0


@pytest.mark.unit
def test_test_cli_prints_success_message(mock_model_tester):
    """Verify CLI prints success message after testing"""
    # Arrange
    with patch(
        "sys.argv",
        ["antibody-test", "--model", "model.pkl", "--data", "data.csv"],
    ):
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        # Act
        main()
        output = captured_output.getvalue()
        sys.stdout = old_stdout

        # Assert
        assert "TESTING COMPLETED SUCCESSFULLY" in output


@pytest.mark.unit
def test_test_cli_prints_results_summary(mock_model_tester):
    """Verify CLI prints results summary after testing"""
    # Arrange
    with patch(
        "sys.argv",
        ["antibody-test", "--model", "model.pkl", "--data", "data.csv"],
    ):
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        # Act
        main()
        output = captured_output.getvalue()
        sys.stdout = old_stdout

        # Assert
        assert "accuracy" in output.lower()


# ==================== Error Handling Tests ====================


@pytest.mark.unit
def test_test_cli_returns_one_on_failure(mock_model_tester):
    """Verify CLI returns 1 exit code when testing fails"""
    # Arrange
    mock_model_tester.return_value.run_comprehensive_test.side_effect = RuntimeError(
        "Testing failed"
    )
    with patch(
        "sys.argv",
        ["antibody-test", "--model", "model.pkl", "--data", "data.csv"],
    ):
        # Act
        exit_code = main()

        # Assert
        assert exit_code == 1


@pytest.mark.unit
def test_test_cli_prints_error_message_on_failure(mock_model_tester):
    """Verify CLI prints error message to stderr when testing fails"""
    # Arrange
    error_msg = "Model file not found"
    mock_model_tester.return_value.run_comprehensive_test.side_effect = (
        FileNotFoundError(error_msg)
    )
    with patch(
        "sys.argv",
        ["antibody-test", "--model", "model.pkl", "--data", "data.csv"],
    ):
        old_stderr = sys.stderr
        sys.stderr = captured_error = StringIO()

        # Act
        main()
        error = captured_error.getvalue()
        sys.stderr = old_stderr

        # Assert
        assert "Error during testing" in error
        assert error_msg in error


# ==================== Create Config Tests ====================


@pytest.mark.unit
def test_test_cli_creates_sample_config_file(tmp_path):
    """Verify --create-config creates sample config file"""
    # Arrange
    import os

    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        with patch("sys.argv", ["antibody-test", "--create-config"]):
            # Act
            exit_code = main()

            # Assert
            assert exit_code == 0
            assert (tmp_path / "test_config.yaml").exists()
    finally:
        os.chdir(original_cwd)


@pytest.mark.unit
def test_create_sample_test_config_writes_valid_yaml(tmp_path):
    """Verify create_sample_test_config writes valid YAML"""
    # Arrange
    import os

    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        # Act
        create_sample_test_config()

        # Assert
        config_file = tmp_path / "test_config.yaml"
        assert config_file.exists()

        # Verify it's valid YAML
        with open(config_file) as f:
            config = yaml.safe_load(f)
        assert "model_paths" in config
        assert "data_paths" in config
    finally:
        os.chdir(original_cwd)


# ==================== TestConfig Tests ====================


@pytest.mark.unit
def test_test_config_has_default_metrics():
    """Verify TestConfig sets default metrics"""
    # Arrange & Act
    config = TestConfig(
        model_paths=["model.pkl"],
        data_paths=["data.csv"],
    )

    # Assert
    assert config.metrics is not None
    assert "accuracy" in config.metrics
    assert "f1" in config.metrics
    assert "roc_auc" in config.metrics


@pytest.mark.unit
def test_test_config_accepts_custom_sequence_column():
    """Verify TestConfig accepts custom sequence_column"""
    # Arrange & Act
    config = TestConfig(
        model_paths=["model.pkl"],
        data_paths=["data.csv"],
        sequence_column="custom_seq",
    )

    # Assert
    assert config.sequence_column == "custom_seq"


@pytest.mark.unit
def test_test_config_accepts_custom_label_column():
    """Verify TestConfig accepts custom label_column"""
    # Arrange & Act
    config = TestConfig(
        model_paths=["model.pkl"],
        data_paths=["data.csv"],
        label_column="custom_label",
    )

    # Assert
    assert config.label_column == "custom_label"


# ==================== Help Message Tests ====================


@pytest.mark.unit
def test_test_cli_shows_help_message():
    """Verify CLI shows help message with --help"""
    # Arrange
    with patch("sys.argv", ["antibody-test", "--help"]):
        # Act & Assert
        with pytest.raises(SystemExit) as exc_info:
            main()

        # --help exits with code 0
        assert exc_info.value.code == 0


@pytest.mark.unit
def test_test_cli_help_includes_examples():
    """Verify CLI help includes usage examples"""
    # Arrange
    with patch("sys.argv", ["antibody-test", "--help"]):
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        # Act
        with contextlib.suppress(SystemExit):
            main()

        output = captured_output.getvalue()
        sys.stdout = old_stdout

        # Assert
        assert "Examples:" in output
        assert "--model" in output
        assert "--data" in output


# ==================== Device Choice Tests ====================


@pytest.mark.unit
def test_test_cli_accepts_cpu_device(mock_model_tester):
    """Verify CLI accepts cpu device"""
    # Arrange
    with patch(
        "sys.argv",
        [
            "antibody-test",
            "--model",
            "model.pkl",
            "--data",
            "data.csv",
            "--device",
            "cpu",
        ],
    ):
        # Act
        exit_code = main()

        # Assert
        assert exit_code == 0
        call_args = mock_model_tester.call_args[0][0]
        assert call_args.device == "cpu"


@pytest.mark.unit
def test_test_cli_accepts_cuda_device(mock_model_tester):
    """Verify CLI accepts cuda device"""
    # Arrange
    with patch(
        "sys.argv",
        [
            "antibody-test",
            "--model",
            "model.pkl",
            "--data",
            "data.csv",
            "--device",
            "cuda",
        ],
    ):
        # Act
        exit_code = main()

        # Assert
        assert exit_code == 0
        call_args = mock_model_tester.call_args[0][0]
        assert call_args.device == "cuda"


@pytest.mark.unit
def test_test_cli_accepts_mps_device(mock_model_tester):
    """Verify CLI accepts mps device"""
    # Arrange
    with patch(
        "sys.argv",
        [
            "antibody-test",
            "--model",
            "model.pkl",
            "--data",
            "data.csv",
            "--device",
            "mps",
        ],
    ):
        # Act
        exit_code = main()

        # Assert
        assert exit_code == 0
        call_args = mock_model_tester.call_args[0][0]
        assert call_args.device == "mps"


# ==================== Output Directory Tests ====================


@pytest.mark.unit
def test_test_cli_uses_default_output_dir(mock_model_tester):
    """Verify CLI uses default output directory"""
    # Arrange
    with patch(
        "sys.argv",
        ["antibody-test", "--model", "model.pkl", "--data", "data.csv"],
    ):
        # Act
        main()

        # Assert
        call_args = mock_model_tester.call_args[0][0]
        assert call_args.output_dir == "./test_results"


@pytest.mark.unit
def test_test_cli_accepts_custom_output_dir(mock_model_tester):
    """Verify CLI accepts custom output directory"""
    # Arrange
    with patch(
        "sys.argv",
        [
            "antibody-test",
            "--model",
            "model.pkl",
            "--data",
            "data.csv",
            "--output-dir",
            "./custom_results",
        ],
    ):
        # Act
        main()

        # Assert
        call_args = mock_model_tester.call_args[0][0]
        assert call_args.output_dir == "./custom_results"
