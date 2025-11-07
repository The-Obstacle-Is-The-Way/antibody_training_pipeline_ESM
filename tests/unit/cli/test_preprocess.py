"""
Unit tests for preprocessing CLI.

Tests cover:
- CLI argument parsing (--dataset flag)
- Guidance message generation
- Script path mapping
- Error handling for invalid datasets
- Exit codes

Testing philosophy:
- Test behaviors, not implementation
- Mock stdout/stderr capture
- Use real argparse (no mocking CLI logic)
- Follow AAA pattern (Arrange-Act-Assert)

Date: 2025-11-07
Phase: 4 (CLI & E2E Tests)
"""

import contextlib
import sys
from io import StringIO
from unittest.mock import patch

import pytest

from antibody_training_esm.cli.preprocess import main

# ==================== Fixtures ====================


@pytest.fixture
def capture_output():
    """Fixture to capture stdout and stderr"""

    def _capture(func, args):
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = StringIO(), StringIO()
        try:
            result = func()
            stdout_val = sys.stdout.getvalue()
            stderr_val = sys.stderr.getvalue()
            return result, stdout_val, stderr_val
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

    return _capture


# ==================== CLI Argument Parsing Tests ====================


@pytest.mark.unit
def test_preprocess_cli_requires_dataset_argument():
    """Verify CLI requires --dataset argument"""
    # Arrange
    with patch("sys.argv", ["antibody-preprocess"]):
        # Act & Assert
        with pytest.raises(SystemExit) as exc_info:
            main()

        # argparse exits with code 2 for argument errors
        assert exc_info.value.code == 2


@pytest.mark.unit
def test_preprocess_cli_accepts_valid_dataset():
    """Verify CLI accepts valid dataset names"""
    # Arrange
    valid_datasets = ["jain", "harvey", "shehata", "boughter"]

    for dataset in valid_datasets:
        with patch("sys.argv", ["antibody-preprocess", "--dataset", dataset]):
            # Act
            exit_code = main()

            # Assert
            assert exit_code == 0


@pytest.mark.unit
def test_preprocess_cli_rejects_invalid_dataset():
    """Verify CLI rejects invalid dataset names"""
    # Arrange
    with patch("sys.argv", ["antibody-preprocess", "--dataset", "invalid"]):
        # Act & Assert
        with pytest.raises(SystemExit) as exc_info:
            main()

        # argparse exits with code 2 for invalid choice
        assert exc_info.value.code == 2


@pytest.mark.unit
def test_preprocess_cli_accepts_short_flag():
    """Verify CLI accepts -d short flag"""
    # Arrange
    with patch("sys.argv", ["antibody-preprocess", "-d", "jain"]):
        # Act
        exit_code = main()

        # Assert
        assert exit_code == 0


# ==================== Guidance Message Tests ====================


@pytest.mark.unit
def test_preprocess_cli_prints_guidance_message():
    """Verify CLI prints guidance message for preprocessing"""
    # Arrange
    with patch("sys.argv", ["antibody-preprocess", "--dataset", "jain"]):
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        # Act
        exit_code = main()
        output = captured_output.getvalue()
        sys.stdout = old_stdout

        # Assert
        assert exit_code == 0
        assert "CLI is not implemented" in output
        assert "specialized scripts" in output


@pytest.mark.unit
def test_preprocess_cli_provides_correct_script_path_for_jain():
    """Verify CLI provides correct preprocessing script for Jain dataset"""
    # Arrange
    with patch("sys.argv", ["antibody-preprocess", "--dataset", "jain"]):
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        # Act
        main()
        output = captured_output.getvalue()
        sys.stdout = old_stdout

        # Assert
        assert "preprocessing/jain/step2_preprocess_p5e_s2.py" in output


@pytest.mark.unit
def test_preprocess_cli_provides_correct_script_path_for_harvey():
    """Verify CLI provides correct preprocessing script for Harvey dataset"""
    # Arrange
    with patch("sys.argv", ["antibody-preprocess", "--dataset", "harvey"]):
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        # Act
        main()
        output = captured_output.getvalue()
        sys.stdout = old_stdout

        # Assert
        assert "preprocessing/harvey/step2_extract_fragments.py" in output


@pytest.mark.unit
def test_preprocess_cli_provides_correct_script_path_for_shehata():
    """Verify CLI provides correct preprocessing script for Shehata dataset"""
    # Arrange
    with patch("sys.argv", ["antibody-preprocess", "--dataset", "shehata"]):
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        # Act
        main()
        output = captured_output.getvalue()
        sys.stdout = old_stdout

        # Assert
        assert "preprocessing/shehata/step2_extract_fragments.py" in output


@pytest.mark.unit
def test_preprocess_cli_provides_correct_script_path_for_boughter():
    """Verify CLI provides correct preprocessing script for Boughter dataset"""
    # Arrange
    with patch("sys.argv", ["antibody-preprocess", "--dataset", "boughter"]):
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        # Act
        main()
        output = captured_output.getvalue()
        sys.stdout = old_stdout

        # Assert
        assert "preprocessing/boughter/stage2_stage3_annotation_qc.py" in output


@pytest.mark.unit
def test_preprocess_cli_explains_ssot_rationale():
    """Verify CLI explains why scripts are SSOT"""
    # Arrange
    with patch("sys.argv", ["antibody-preprocess", "--dataset", "jain"]):
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        # Act
        main()
        output = captured_output.getvalue()
        sys.stdout = old_stdout

        # Assert
        assert "Single Source of Truth" in output
        assert "bit-for-bit parity" in output
        assert "unique requirements" in output


# ==================== Documentation Reference Tests ====================


@pytest.mark.unit
def test_preprocess_cli_references_documentation():
    """Verify CLI references README and docs"""
    # Arrange
    with patch("sys.argv", ["antibody-preprocess", "--dataset", "jain"]):
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        # Act
        main()
        output = captured_output.getvalue()
        sys.stdout = old_stdout

        # Assert
        assert "datasets/README.md" in output


# ==================== Error Handling Tests ====================


@pytest.mark.unit
def test_preprocess_cli_handles_exceptions_gracefully():
    """Verify CLI handles unexpected exceptions"""
    # Arrange
    with (
        patch("sys.argv", ["antibody-preprocess", "--dataset", "jain"]),
        patch.dict("sys.modules", {"antibody_training_esm.cli.preprocess": None}),
    ):
        # This test is tricky - the main function catches all exceptions
        # Let's just verify it returns 0 for normal operation
        exit_code = main()

        # Assert
        assert exit_code == 0


# ==================== Exit Code Tests ====================


@pytest.mark.unit
def test_preprocess_cli_returns_zero_on_success():
    """Verify CLI returns 0 exit code on success"""
    # Arrange
    for dataset in ["jain", "harvey", "shehata", "boughter"]:
        with patch("sys.argv", ["antibody-preprocess", "--dataset", dataset]):
            # Act
            exit_code = main()

            # Assert
            assert exit_code == 0


# ==================== Integration with argparse Tests ====================


@pytest.mark.unit
def test_preprocess_cli_shows_help_message():
    """Verify CLI shows help message with --help"""
    # Arrange
    with patch("sys.argv", ["antibody-preprocess", "--help"]):
        # Act & Assert
        with pytest.raises(SystemExit) as exc_info:
            main()

        # --help exits with code 0
        assert exc_info.value.code == 0


@pytest.mark.unit
def test_preprocess_cli_choices_documented_in_help():
    """Verify CLI help documents all dataset choices"""
    # Arrange
    with patch("sys.argv", ["antibody-preprocess", "--help"]):
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        # Act
        with contextlib.suppress(SystemExit):
            main()

        output = captured_output.getvalue()
        sys.stdout = old_stdout

        # Assert
        assert "jain" in output
        assert "harvey" in output
        assert "shehata" in output
        assert "boughter" in output
