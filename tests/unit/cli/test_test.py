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

from __future__ import annotations

import contextlib
import logging
import sys
from collections.abc import Generator
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import yaml

from antibody_training_esm.cli.test import TestConfig, create_sample_test_config, main

# ==================== Fixtures ====================


@pytest.fixture
def mock_model_tester() -> Generator[MagicMock, None, None]:
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
def test_test_cli_requires_model_and_data_or_config() -> None:
    """Verify CLI requires either --config or both --model and --data"""
    # Arrange
    with patch("sys.argv", ["antibody-test"]):
        # Act & Assert
        with pytest.raises(SystemExit) as exc_info:
            main()

        # argparse error exits with code 2
        assert exc_info.value.code == 2


@pytest.mark.unit
def test_test_cli_accepts_model_and_data_arguments(
    mock_model_tester: MagicMock,
) -> None:
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
def test_test_cli_accepts_multiple_models_and_datasets(
    mock_model_tester: MagicMock,
) -> None:
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
def test_test_cli_accepts_config_file(
    mock_model_tester: MagicMock, tmp_path: Path
) -> None:
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
def test_test_cli_loads_config_from_yaml(
    mock_model_tester: MagicMock, tmp_path: Path
) -> None:
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
def test_test_cli_overrides_device_from_config(
    mock_model_tester: MagicMock, tmp_path: Path
) -> None:
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
def test_test_cli_overrides_batch_size_from_config(
    mock_model_tester: MagicMock, tmp_path: Path
) -> None:
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
def test_test_cli_returns_zero_on_success(mock_model_tester: MagicMock) -> None:
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
def test_test_cli_prints_success_message(mock_model_tester: MagicMock) -> None:
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
def test_test_cli_prints_results_summary(mock_model_tester: MagicMock) -> None:
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
def test_test_cli_returns_one_on_failure(mock_model_tester: MagicMock) -> None:
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
def test_test_cli_prints_error_message_on_failure(mock_model_tester: MagicMock) -> None:
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
def test_test_cli_creates_sample_config_file(tmp_path: Path) -> None:
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
def test_create_sample_test_config_writes_valid_yaml(tmp_path: Path) -> None:
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
def test_test_config_has_default_metrics() -> None:
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
def test_test_config_accepts_custom_sequence_column() -> None:
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
def test_test_config_accepts_custom_label_column() -> None:
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
def test_test_cli_shows_help_message() -> None:
    """Verify CLI shows help message with --help"""
    # Arrange
    with patch("sys.argv", ["antibody-test", "--help"]):
        # Act & Assert
        with pytest.raises(SystemExit) as exc_info:
            main()

        # --help exits with code 0
        assert exc_info.value.code == 0


@pytest.mark.unit
def test_test_cli_help_includes_examples() -> None:
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
def test_test_cli_accepts_cpu_device(mock_model_tester: MagicMock) -> None:
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
def test_test_cli_accepts_cuda_device(mock_model_tester: MagicMock) -> None:
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
def test_test_cli_accepts_mps_device(mock_model_tester: MagicMock) -> None:
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
def test_test_cli_uses_default_output_dir(mock_model_tester: MagicMock) -> None:
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
def test_test_cli_accepts_custom_output_dir(mock_model_tester: MagicMock) -> None:
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


# ==================== ModelTester Error Handling Tests ====================
# Tests for uncovered error handling paths (Issue #14)
# Target lines: 141-171 (device mismatch), 223-225 (Jain validation), 481-520 (config errors)


@pytest.fixture
def mock_transformers_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock transformers model loading to avoid HuggingFace downloads"""
    from tests.fixtures.mock_models import MockESMModel, MockTokenizer

    monkeypatch.setattr(
        "antibody_training_esm.core.embeddings.AutoModel.from_pretrained",
        MockESMModel,
    )
    monkeypatch.setattr(
        "antibody_training_esm.core.embeddings.AutoTokenizer.from_pretrained",
        MockTokenizer,
    )


@pytest.mark.unit
def test_device_mismatch_recreates_extractor(
    tmp_path: Path,
    mock_transformers_model: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test device mismatch triggers extractor recreation with proper cleanup (lines 141-171).

    CRITICAL: This tests the P0 semaphore leak fix.
    DO NOT mock the device mismatch logic - let it execute REAL cleanup code.
    """
    import pickle

    from antibody_training_esm.cli.test import ModelTester, TestConfig
    from antibody_training_esm.core.classifier import BinaryClassifier

    # Arrange - Create model trained on CPU
    model_path = tmp_path / "model.pkl"
    config = {
        "model_name": "facebook/esm1v_t33_650M_UR90S_1",
        "device": "cpu",
        "batch_size": 8,
        "random_state": 42,
        "max_iter": 1000,
    }
    classifier = BinaryClassifier(params=config)

    with open(model_path, "wb") as f:
        pickle.dump(classifier, f)

    # Create test config specifying CUDA device
    test_config = TestConfig(
        model_paths=[str(model_path)],
        data_paths=["path/to/test.csv"],
        output_dir=str(tmp_path / "output"),
        device="cuda",  # Different from model's CPU
    )

    tester = ModelTester(test_config)

    # Act - Load model (triggers device mismatch check)
    caplog.set_level(logging.INFO)
    model = tester.load_model(str(model_path))

    # Assert - REAL cleanup executed (lines 146-171)
    assert "Device mismatch" in caplog.text
    assert "Recreating extractor" in caplog.text
    assert "Cleaned up old extractor on cpu" in caplog.text
    assert "Created new extractor on cuda" in caplog.text

    # Assert - Extractor actually recreated on new device
    assert model.device == "cuda"
    assert model.embedding_extractor.device == "cuda"


@pytest.mark.unit
def test_jain_test_set_size_validation_fails_on_invalid_size(
    tmp_path: Path,
) -> None:
    """Test Jain test set validation raises ValueError for wrong sizes (lines 223-225).

    DO NOT mock the validation logic - create REAL CSV with wrong size
    and verify REAL exception is raised.
    """
    import pandas as pd

    from antibody_training_esm.cli.test import ModelTester, TestConfig

    # Arrange - Create Jain test CSV with WRONG size (not 86 or 94)
    jain_test_path = tmp_path / "jain_test_WRONG.csv"
    wrong_size_df = pd.DataFrame(
        {
            "id": [f"AB{i:03d}" for i in range(50)],  # WRONG: 50 antibodies
            "sequence": ["EVQLVESGGGLVQPGGSLRLSCAASGFTFS" for _ in range(50)],
            "label": [0] * 50,
        }
    )
    wrong_size_df.to_csv(jain_test_path, index=False)

    config = TestConfig(
        model_paths=["dummy.pkl"],
        data_paths=[str(jain_test_path)],  # Contains "jain" and "test"
        output_dir=str(tmp_path / "output"),
    )
    tester = ModelTester(config)

    # Act & Assert - REAL validation executes (lines 223-225)
    with pytest.raises(
        ValueError,
        match=r"Jain test set has 50 antibodies but expected one of",
    ):
        tester.load_dataset(str(jain_test_path))


@pytest.mark.unit
def test_jain_test_set_size_validation_passes_canonical_86(
    tmp_path: Path,
) -> None:
    """Test Jain validation accepts canonical 86-antibody set (lines 223-225)."""
    import pandas as pd

    from antibody_training_esm.cli.test import ModelTester, TestConfig

    # Arrange - Create Jain test CSV with CORRECT size (86)
    jain_test_path = tmp_path / "VH_only_jain_test_PARITY_86.csv"
    correct_size_df = pd.DataFrame(
        {
            "id": [f"AB{i:03d}" for i in range(86)],  # CORRECT: 86
            "sequence": ["EVQLVESGGGLVQPGGSLRLSCAASGFTFS" for _ in range(86)],
            "label": [0] * 86,
        }
    )
    correct_size_df.to_csv(jain_test_path, index=False)

    config = TestConfig(
        model_paths=["dummy.pkl"],
        data_paths=[str(jain_test_path)],
        output_dir=str(tmp_path / "output"),
    )
    tester = ModelTester(config)

    # Act - NO exception raised
    sequences, labels = tester.load_dataset(str(jain_test_path))

    # Assert
    assert len(sequences) == 86
    assert len(labels) == 86


@pytest.mark.unit
def test_jain_test_set_size_validation_passes_legacy_94(
    tmp_path: Path,
) -> None:
    """Test Jain validation accepts legacy 94-antibody set (lines 223-225)."""
    import pandas as pd

    from antibody_training_esm.cli.test import ModelTester, TestConfig

    # Arrange - Create Jain test CSV with legacy size (94)
    jain_test_path = tmp_path / "VH_only_jain_test_legacy_94.csv"
    legacy_size_df = pd.DataFrame(
        {
            "id": [f"AB{i:03d}" for i in range(94)],  # LEGACY: 94
            "sequence": ["EVQLVESGGGLVQPGGSLRLSCAASGFTFS" for _ in range(94)],
            "label": [0] * 94,
        }
    )
    legacy_size_df.to_csv(jain_test_path, index=False)

    config = TestConfig(
        model_paths=["dummy.pkl"],
        data_paths=[str(jain_test_path)],
        output_dir=str(tmp_path / "output"),
    )
    tester = ModelTester(config)

    # Act - NO exception raised
    sequences, labels = tester.load_dataset(str(jain_test_path))

    # Assert
    assert len(sequences) == 94
    assert len(labels) == 94


@pytest.mark.unit
def test_determine_output_dir_falls_back_when_config_missing(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test fallback to flat structure when model config file missing (lines 481-484).

    DO NOT mock the file existence check - use REAL tmp_path without config.
    """
    from antibody_training_esm.cli.test import ModelTester, TestConfig

    # Arrange - Model path without config file
    model_path = tmp_path / "model.pkl"
    model_path.write_text("dummy")  # Create model file
    # NOTE: model_config.json does NOT exist

    config = TestConfig(
        model_paths=[str(model_path)],
        data_paths=["test.csv"],
        output_dir=str(tmp_path / "output"),
    )
    tester = ModelTester(config)

    # Act
    caplog.set_level(logging.INFO)
    output_dir = tester._compute_output_directory(str(model_path), "jain")

    # Assert - REAL fallback logic executed (lines 481-484)
    assert "Model config not found" in caplog.text
    assert "using flat output structure" in caplog.text
    assert output_dir == str(tmp_path / "output")  # Flat structure


@pytest.mark.unit
def test_determine_output_dir_handles_corrupt_json(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test fallback when model config has corrupt JSON (lines 518-520)."""
    from antibody_training_esm.cli.test import ModelTester, TestConfig

    # Arrange - Create model with CORRUPT config
    model_path = tmp_path / "model.pkl"
    config_path = tmp_path / "model_config.json"

    model_path.write_text("dummy")
    config_path.write_text("{bad json syntax")  # CORRUPT

    config = TestConfig(
        model_paths=[str(model_path)],
        data_paths=["test.csv"],
        output_dir=str(tmp_path / "output"),
    )
    tester = ModelTester(config)

    # Act
    caplog.set_level(logging.WARNING)
    output_dir = tester._compute_output_directory(str(model_path), "jain")

    # Assert - REAL exception handling executed (line 518)
    assert "Could not determine hierarchical path" in caplog.text
    assert "Using flat structure" in caplog.text
    assert output_dir == str(tmp_path / "output")


@pytest.mark.unit
def test_determine_output_dir_handles_missing_model_name(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test fallback when model config missing 'model_name' key (lines 495-496, 518-520)."""
    import json

    from antibody_training_esm.cli.test import ModelTester, TestConfig

    # Arrange - Create model with config missing model_name
    model_path = tmp_path / "model.pkl"
    config_path = tmp_path / "model_config.json"

    model_path.write_text("dummy")
    config_path.write_text(json.dumps({"classifier": {"C": 1.0}}))  # Missing model_name

    config = TestConfig(
        model_paths=[str(model_path)],
        data_paths=["test.csv"],
        output_dir=str(tmp_path / "output"),
    )
    tester = ModelTester(config)

    # Act
    caplog.set_level(logging.WARNING)
    output_dir = tester._compute_output_directory(str(model_path), "jain")

    # Assert - REAL ValueError caught (line 496, 518)
    assert "Could not determine hierarchical path" in caplog.text
    assert output_dir == str(tmp_path / "output")


@pytest.mark.unit
def test_determine_output_dir_handles_missing_classifier_key(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test fallback when model config missing 'classifier' key (Issue #14)."""
    import json

    from antibody_training_esm.cli.test import ModelTester, TestConfig

    # Arrange - Create model with config missing classifier
    model_path = tmp_path / "model.pkl"
    config_path = tmp_path / "model_config.json"

    model_path.write_text("dummy")
    config_path.write_text(
        json.dumps(
            {
                "model_name": "facebook/esm1v_t33_650M_UR90S_1",
                # "classifier": {...}  ← MISSING!
            }
        )
    )

    config = TestConfig(
        model_paths=[str(model_path)],
        data_paths=["test.csv"],
        output_dir=str(tmp_path / "output"),
    )
    tester = ModelTester(config)

    # Act
    caplog.set_level(logging.INFO)
    output_dir = tester._compute_output_directory(str(model_path), "jain")

    # Assert - Should use hierarchical structure with "unknown" classifier
    assert "Using hierarchical output" in caplog.text
    assert "unknown" in output_dir
    assert "jain" in output_dir


@pytest.mark.unit
def test_compute_embeddings_handles_corrupt_cache(
    tmp_path: Path,
    mock_transformers_model: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test recomputation when embedding cache is corrupt (Issue #14)."""
    import pickle

    from antibody_training_esm.cli.test import ModelTester, TestConfig
    from antibody_training_esm.core.classifier import BinaryClassifier

    # Arrange - Create valid model
    model_path = tmp_path / "model.pkl"
    config = {
        "model_name": "facebook/esm1v_t33_650M_UR90S_1",
        "device": "cpu",
        "batch_size": 8,
        "random_state": 42,
        "max_iter": 1000,
    }
    classifier = BinaryClassifier(params=config)

    with open(model_path, "wb") as f:
        pickle.dump(classifier, f)

    # Create CORRUPT cache file
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    cache_file = output_dir / "test_dataset_test_embeddings.pkl"
    cache_file.write_text("CORRUPT PICKLE DATA NOT VALID")  # ← Corrupt cache

    test_config = TestConfig(
        model_paths=[str(model_path)],
        data_paths=["test.csv"],
        output_dir=str(output_dir),
        device="cpu",  # Avoid MPS/CUDA device mismatch in test environment
    )

    tester = ModelTester(test_config)
    model = tester.load_model(str(model_path))

    # Create test sequences
    sequences = ["EVQLVESGGGLVQPGG", "QVQLQQWGAGLLKPSE"]

    # Act - Should handle corrupt cache gracefully
    caplog.set_level(logging.WARNING)

    embeddings = tester.embed_sequences(
        sequences, model, "test_dataset", str(output_dir)
    )

    # Assert - CRITICAL: corrupt cache was detected and logged
    assert "Failed to load cached embeddings" in caplog.text
    assert "Recomputing embeddings" in caplog.text

    # Assert - Embeddings were successfully recomputed
    assert embeddings is not None
    assert embeddings.shape == (2, 1280)  # Valid embeddings computed


@pytest.mark.unit
def test_determine_output_dir_uses_hierarchical_structure_with_valid_config(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test hierarchical path generation with valid config (lines 487-516)."""
    import json

    from antibody_training_esm.cli.test import ModelTester, TestConfig

    # Arrange - Create model with VALID config
    model_path = tmp_path / "model.pkl"
    config_path = tmp_path / "model_config.json"

    model_path.write_text("dummy")
    valid_config = {
        "model_name": "facebook/esm1v_t33_650M_UR90S_1",
        "classifier": {
            "type": "logistic_regression",
            "C": 1.0,
            "penalty": "l2",
        },
    }
    config_path.write_text(json.dumps(valid_config))

    config = TestConfig(
        model_paths=[str(model_path)],
        data_paths=["test.csv"],
        output_dir=str(tmp_path / "output"),
    )
    tester = ModelTester(config)

    # Act
    caplog.set_level(logging.INFO)
    output_dir = tester._compute_output_directory(str(model_path), "jain")

    # Assert - REAL hierarchical path logic (lines 501-516)
    assert "Using hierarchical output" in caplog.text
    assert "esm1v" in output_dir  # Model shortname
    assert "logreg" in output_dir  # Classifier shortname
    assert "jain" in output_dir  # Dataset name
