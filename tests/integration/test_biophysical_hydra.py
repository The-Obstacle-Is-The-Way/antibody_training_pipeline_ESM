"""
Integration tests for Biophysical Track (Track B) using Hydra.

Verifies that the biophysical pipeline can be run end-to-end using Hydra configuration,
producing valid models and metrics consistent with Phase B baselines.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from omegaconf import OmegaConf

from antibody_training_esm.core.trainer import train_pipeline


@pytest.mark.integration
def test_biophysical_training_run(tmp_path: Path) -> None:
    """
    Test full training run for biophysical model using train_pipeline directly.
    """
    # Create dummy data files to satisfy Pydantic validation
    # Content doesn't matter because load_data is mocked, but files must exist
    train_file = tmp_path / "dummy_train.csv"
    test_file = tmp_path / "dummy_test.csv"
    train_file.touch()
    test_file.touch()

    # Create a valid config for biophysical
    cfg = OmegaConf.create(
        {
            "model": {
                "name": "biophysical",
                "device": "cpu",
                "batch_size": 1,
                "model_type": "biophysical",
                "revision": "1.0.0",
                "trust_remote_code": False,
            },
            "data": {
                "train_file": str(train_file),
                "test_file": str(test_file),
                "embeddings_cache_dir": str(tmp_path / "cache"),
            },
            "classifier": {
                "strategy": "logistic_regression",
                "C": 1.0,
                "penalty": "l2",
                "solver": "lbfgs",
                "max_iter": 100,
                "random_state": 42,
            },
            "training": {
                "n_splits": 2,
                "random_state": 42,
                "stratify": True,
                "metrics": ["accuracy"],
                "save_model": True,
                "model_save_dir": str(tmp_path / "experiments" / "checkpoints"),
                "model_name": "test_biophysical_model",
                "log_level": "INFO",
                "log_file": "training.log",
                "batch_size": 1,
                "num_workers": 0,
            },
            "experiment": {
                "name": "integration_test",
                "tags": [],
                "description": "Integration test",
            },
            "hardware": {"device": "cpu"},
        }
    )

    # Mock data
    # Need enough samples for 2-fold CV (min 2 per class)
    mock_X = ["ACDEF", "GHIKL", "MNPQR", "STVWY"] * 2  # 8 sequences
    mock_y = [0, 1, 0, 1] * 2

    with patch(
        "antibody_training_esm.core.trainer.load_data", return_value=(mock_X, mock_y)
    ):
        results = train_pipeline(cfg)

    # Verify results
    assert results["train_metrics"] is not None

    # Check model files were created
    model_paths = results["model_paths"]
    assert model_paths is not None

    # Verify file existence
    assert Path(model_paths["pickle"]).exists()
    assert Path(model_paths["npz"]).exists()
    assert Path(model_paths["config"]).exists()

    # Verify hierarchical structure (biophysical/logreg) is used
    assert "biophysical/logreg" in str(model_paths["pickle"])


@pytest.mark.integration
def test_biophysical_filtering_logic(tmp_path: Path) -> None:
    """
    Test that biophysical pipeline correctly filters 'X' and '*' sequences.
    """
    # Create dummy data files to satisfy Pydantic validation
    train_file = tmp_path / "dummy_train.csv"
    test_file = tmp_path / "dummy_test.csv"
    train_file.touch()
    test_file.touch()

    cfg = OmegaConf.create(
        {
            "model": {
                "name": "biophysical",
                "device": "cpu",
                "batch_size": 1,
                "model_type": "biophysical",
                "revision": "1.0.0",
                "trust_remote_code": False,
            },
            "data": {
                "train_file": str(train_file),
                "test_file": str(test_file),
                "embeddings_cache_dir": str(tmp_path / "cache"),
            },
            "classifier": {
                "strategy": "logistic_regression",
            },
            "training": {
                "n_splits": 2,
                "random_state": 42,
                "stratify": True,
                "metrics": ["accuracy"],
                "save_model": False,
                "model_save_dir": str(tmp_path),
                "model_name": "test",
                "log_level": "INFO",
                "log_file": "training.log",
            },
            "experiment": {"name": "filtering_test", "tags": []},
            "hardware": {"device": "cpu"},
        }
    )

    # 6 sequences: 4 valid, 2 invalid
    # Classes: 0, 0, 0 (invalid), 1, 1, 1 (invalid) -> After filter: 0,0, 1,1 (Valid for 2-fold)
    mock_X = [
        "ACDEF",
        "ACDEF",  # Valid Class 0
        "GHIXK",  # Invalid
        "STVWY",
        "STVWY",  # Valid Class 1
        "MN*QR",  # Invalid
    ]
    mock_y = [0, 0, 0, 1, 1, 1]

    with (
        patch(
            "antibody_training_esm.core.trainer.load_data",
            return_value=(mock_X, mock_y),
        ),
        patch(
            "antibody_training_esm.core.trainer.get_or_create_embeddings"
        ) as mock_embed,
    ):
        # Return random embeddings for the 4 valid sequences
        mock_embed.return_value = np.random.rand(4, 3).astype(np.float32)

        train_pipeline(cfg)

        # Check what was passed to get_or_create_embeddings
        # call_args[0] is (sequences, extractor, ...)
        passed_sequences = mock_embed.call_args[0][0]

        assert len(passed_sequences) == 4
        assert "GHIXK" not in passed_sequences
        assert "MN*QR" not in passed_sequences
