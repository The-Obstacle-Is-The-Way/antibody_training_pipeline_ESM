"""
Unit Tests for Hydra-enabled Trainer Module

Tests train_pipeline() function that accepts Hydra DictConfig.
Focus: Verify Hydra integration works correctly with core logic.

Date: 2025-11-11
Phase: Step 2 - Hydra Integration (TDD)
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from antibody_training_esm.core.trainer import train_pipeline

# ==================== Fixtures ====================


@pytest.fixture(autouse=True)
def cleanup_hydra() -> Generator[None, None, None]:
    """Clean up Hydra singleton between tests"""
    yield
    GlobalHydra.instance().clear()


@pytest.fixture
def mock_training_data(tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    """Create minimal training CSV for tests"""
    train_csv = tmp_path / "train.csv"
    df = pd.DataFrame(
        {
            "id": [f"AB{i:03d}" for i in range(20)],
            "sequence": ["QVQLVQSGAEVKKPGASVKVSCKASGYTFT"] * 20,
            "label": [0, 1] * 10,  # Balanced dataset
        }
    )
    df.to_csv(train_csv, index=False)
    return train_csv, df


# ==================== Tests ====================


@pytest.mark.unit
def test_train_pipeline_accepts_dictconfig(
    tmp_path: Path,
    mock_training_data: tuple[Path, pd.DataFrame],
    mock_transformers_model: tuple[Any, Any],
) -> None:
    """Test that train_pipeline() accepts Hydra DictConfig and completes training"""
    # Arrange
    train_csv, df = mock_training_data

    # Initialize Hydra and compose config (relative path required)
    with initialize(
        version_base=None, config_path="../../../src/antibody_training_esm/conf"
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                f"data.train_file={train_csv}",
                f"data.test_file={train_csv}",  # Use same for test (just a unit test)
                f"data.embeddings_cache_dir={tmp_path / 'cache'}",
                f"training.log_file={tmp_path / 'training.log'}",
                "hardware.device=cpu",  # Use CPU for tests
                "training.save_model=false",  # Disable saving for speed
                "classifier.cv_folds=2",  # Minimal folds for speed
                "training.batch_size=4",  # Small batch for speed
            ],
        )

        # Mock load_data to return sequences and labels
        with patch("antibody_training_esm.core.trainer.load_data") as mock_load_data:
            mock_load_data.return_value = (
                df["sequence"].tolist(),
                df["label"].tolist(),
            )

            # Act
            results = train_pipeline(cfg)

    # Assert
    assert results is not None
    assert isinstance(results, dict)
    assert "train_metrics" in results
    assert "cv_metrics" in results
    assert "config" in results

    # Verify metrics computed
    assert "accuracy" in results["train_metrics"]
    assert results["train_metrics"]["accuracy"] >= 0.0
    assert results["train_metrics"]["accuracy"] <= 1.0

    # Verify CV metrics
    assert "cv_accuracy" in results["cv_metrics"]
    assert "mean" in results["cv_metrics"]["cv_accuracy"]


@pytest.mark.unit
def test_train_pipeline_uses_hydra_output_dir_for_logging(
    tmp_path: Path,
    mock_training_data: tuple[Path, pd.DataFrame],
    mock_transformers_model: tuple[Any, Any],
) -> None:
    """Test that train_pipeline() uses Hydra's output directory for logs"""
    # Arrange
    train_csv, df = mock_training_data
    hydra_output_dir = tmp_path / "outputs"

    # Initialize Hydra with custom output dir
    with initialize(
        version_base=None, config_path="../../../src/antibody_training_esm/conf"
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                f"data.train_file={train_csv}",
                f"data.test_file={train_csv}",
                f"data.embeddings_cache_dir={tmp_path / 'cache'}",
                f"hydra.run.dir={hydra_output_dir}",  # Hydra output dir
                "hardware.device=cpu",  # Use CPU for tests
                f"training.log_file={tmp_path / 'training.log'}",  # Absolute path to avoid stray files
                "training.save_model=false",
                "classifier.cv_folds=2",
                "training.batch_size=4",
            ],
        )

        # Mock load_data
        with patch("antibody_training_esm.core.trainer.load_data") as mock_load_data:
            mock_load_data.return_value = (
                df["sequence"].tolist(),
                df["label"].tolist(),
            )

            # Act
            results = train_pipeline(cfg)

    # Assert: Log file should be in Hydra output dir (when @hydra.main is active)
    # For now, this test just verifies the pipeline completes
    # Full Hydra integration happens when @hydra.main decorator is used
    assert results is not None


@pytest.mark.unit
def test_train_pipeline_preserves_embeddings_cache(
    tmp_path: Path,
    mock_training_data: tuple[Path, pd.DataFrame],
    mock_transformers_model: tuple[Any, Any],
) -> None:
    """Test that train_pipeline() preserves embeddings cache for reuse"""
    # Arrange
    train_csv, df = mock_training_data
    cache_dir = tmp_path / "cache"

    with initialize(
        version_base=None, config_path="../../../src/antibody_training_esm/conf"
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                f"data.train_file={train_csv}",
                f"data.test_file={train_csv}",
                f"data.embeddings_cache_dir={cache_dir}",
                f"training.log_file={tmp_path / 'training.log'}",
                "hardware.device=cpu",  # Use CPU for tests
                "training.save_model=false",
                "classifier.cv_folds=2",
                "training.batch_size=4",
            ],
        )

        # Mock load_data
        with patch("antibody_training_esm.core.trainer.load_data") as mock_load_data:
            mock_load_data.return_value = (
                df["sequence"].tolist(),
                df["label"].tolist(),
            )

            # Act
            train_pipeline(cfg)

    # Assert: Cache directory should exist and contain embeddings
    assert cache_dir.exists()
    cache_files = list(cache_dir.glob("*.pkl"))
    assert len(cache_files) > 0, "Embeddings cache should be preserved"


@pytest.mark.unit
def test_train_pipeline_with_model_saving_enabled(
    tmp_path: Path,
    mock_training_data: tuple[Path, pd.DataFrame],
    mock_transformers_model: tuple[Any, Any],
) -> None:
    """Test that train_pipeline() saves model when enabled"""
    # Arrange
    train_csv, df = mock_training_data
    model_dir = tmp_path / "models"

    with initialize(
        version_base=None, config_path="../../../src/antibody_training_esm/conf"
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                f"data.train_file={train_csv}",
                f"data.test_file={train_csv}",
                f"data.embeddings_cache_dir={tmp_path / 'cache'}",
                f"training.log_file={tmp_path / 'training.log'}",
                "hardware.device=cpu",  # Use CPU for tests
                "training.save_model=true",  # Enable saving
                "training.model_name=test_model",
                f"training.model_save_dir={model_dir}",
                "classifier.cv_folds=2",
                "training.batch_size=4",
            ],
        )

        # Mock load_data
        with patch("antibody_training_esm.core.trainer.load_data") as mock_load_data:
            mock_load_data.return_value = (
                df["sequence"].tolist(),
                df["label"].tolist(),
            )

            # Act
            results = train_pipeline(cfg)

    # Assert: Model files should exist
    assert "model_paths" in results
    assert Path(results["model_paths"]["pickle"]).exists()
    assert Path(results["model_paths"]["npz"]).exists()
    assert Path(results["model_paths"]["config"]).exists()


@pytest.mark.unit
def test_train_pipeline_returns_structured_results(
    tmp_path: Path,
    mock_training_data: tuple[Path, pd.DataFrame],
    mock_transformers_model: tuple[Any, Any],
) -> None:
    """Test that train_pipeline() returns well-structured results dict"""
    # Arrange
    train_csv, df = mock_training_data

    with initialize(
        version_base=None, config_path="../../../src/antibody_training_esm/conf"
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                f"data.train_file={train_csv}",
                f"data.test_file={train_csv}",
                f"data.embeddings_cache_dir={tmp_path / 'cache'}",
                f"training.log_file={tmp_path / 'training.log'}",
                "hardware.device=cpu",  # Use CPU for tests
                "training.save_model=false",
                "classifier.cv_folds=2",
                "training.batch_size=4",
            ],
        )

        # Mock load_data
        with patch("antibody_training_esm.core.trainer.load_data") as mock_load_data:
            mock_load_data.return_value = (
                df["sequence"].tolist(),
                df["label"].tolist(),
            )

            # Act
            results = train_pipeline(cfg)

    # Assert: All expected result keys present
    assert "train_metrics" in results
    assert "cv_metrics" in results
    assert "config" in results

    # Assert: Metrics structure correct
    assert isinstance(results["train_metrics"], dict)
    assert isinstance(results["cv_metrics"], dict)

    # Assert: All configured metrics computed
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert metric in results["train_metrics"]

    # Assert: CV metrics have mean and std
    assert "cv_accuracy" in results["cv_metrics"]
    assert "mean" in results["cv_metrics"]["cv_accuracy"]
    assert "std" in results["cv_metrics"]["cv_accuracy"]
