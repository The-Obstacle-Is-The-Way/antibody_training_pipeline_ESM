"""
Unit tests for training CLI (Hydra-based).

Tests cover:
- CLI invokes Hydra entry point
- Hydra main() is called correctly
- CLI delegates to trainer.main()

HYDRA CLI TESTS - Testing new Hydra-based CLI

Testing philosophy:
- Test that CLI delegates to Hydra correctly
- Don't test Hydra internals (Hydra's responsibility)
- Don't test train_pipeline logic (tested in test_trainer_hydra.py)
- Test minimal integration: CLI → Hydra → trainer.main()

Date: 2025-11-11
Phase: Hydra Integration Complete
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from antibody_training_esm.cli.train import main

# ==================== CLI Delegation Tests ====================


@pytest.mark.unit
def test_train_cli_delegates_to_hydra_main() -> None:
    """Verify CLI delegates to Hydra-decorated main() from trainer"""
    # Arrange
    with patch("antibody_training_esm.cli.train.hydra_main") as mock_hydra:
        # Act
        main()

        # Assert
        mock_hydra.assert_called_once()


@pytest.mark.unit
def test_train_cli_passes_through_to_hydra() -> None:
    """Verify CLI is pure passthrough to Hydra (no argument parsing)"""
    # Arrange
    with patch("antibody_training_esm.cli.train.hydra_main") as mock_hydra:
        # Act
        main()

        # Assert
        # CLI should call hydra_main with no arguments (Hydra reads sys.argv)
        mock_hydra.assert_called_once_with()


@pytest.mark.unit
def test_train_cli_does_not_catch_hydra_exceptions() -> None:
    """Verify CLI doesn't catch exceptions (Hydra handles them)"""
    # Arrange
    with patch("antibody_training_esm.cli.train.hydra_main") as mock_hydra:
        mock_hydra.side_effect = RuntimeError("Hydra error")

        # Act & Assert
        with pytest.raises(RuntimeError, match="Hydra error"):
            main()


# ==================== Integration Notes ====================

# NOTE: Full CLI testing happens in integration tests where we actually
# invoke `uv run antibody-train` with various Hydra overrides.
#
# Unit tests here just verify the CLI correctly delegates to Hydra.
# All actual training logic is tested in:
# - tests/unit/core/test_trainer_hydra.py (train_pipeline with Hydra configs)
# - tests/integration/test_cli_hydra.py (full CLI invocation with subprocess)
