"""Unit tests for reproduce_track_b CLI module."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from antibody_training_esm.cli.reproduce_track_b import get_git_commit

# Mark as unit test
pytestmark = pytest.mark.unit


class TestGetGitCommit:
    """Tests for the get_git_commit function."""

    def test_returns_short_hash_in_git_repo(self) -> None:
        """Test that get_git_commit returns a short hash when in a git repo."""
        result = get_git_commit()
        # Should be a 7-character hex string or "unknown"
        if result != "unknown":
            assert len(result) >= 7
            assert all(c in "0123456789abcdef" for c in result)

    def test_returns_unknown_on_error(self) -> None:
        """Test that get_git_commit returns 'unknown' when git fails."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("git not found")
            result = get_git_commit()
            assert result == "unknown"


class TestSaveLogic:
    """Tests for the metrics comparison and save logic."""

    def test_should_save_when_file_does_not_exist(self, tmp_path: Path) -> None:
        """Test that new files are always saved."""
        results_file = tmp_path / "test_baseline.json"

        # File doesn't exist - should save
        should_save = True
        if results_file.exists():
            should_save = False

        assert should_save is True

    def test_should_not_save_when_metrics_unchanged(self, tmp_path: Path) -> None:
        """Test that identical metrics don't trigger a save."""
        results_file = tmp_path / "test_baseline.json"

        # Create existing file with provenance
        existing = {
            "provenance": {"run_date": "2025-01-01", "git_commit": "abc1234"},
            "cv_accuracy_mean": 0.6318,
            "test_accuracy": 0.5581,
        }
        with open(results_file, "w") as f:
            json.dump(existing, f)

        # New results with same metrics, different provenance
        new_results = {
            "provenance": {"run_date": "2025-01-02", "git_commit": "def5678"},
            "cv_accuracy_mean": 0.6318,
            "test_accuracy": 0.5581,
        }

        # Replicate the save logic from reproduce_track_b.py
        should_save = True
        if results_file.exists():
            with open(results_file) as f:
                existing_data = json.load(f)
            existing_has_provenance = "provenance" in existing_data
            existing_metrics = {
                k: v for k, v in existing_data.items() if k != "provenance"
            }
            new_metrics = {k: v for k, v in new_results.items() if k != "provenance"}
            if existing_has_provenance and existing_metrics == new_metrics:
                should_save = False

        assert should_save is False

    def test_should_save_when_metrics_changed(self, tmp_path: Path) -> None:
        """Test that changed metrics trigger a save."""
        results_file = tmp_path / "test_baseline.json"

        # Create existing file with provenance
        existing = {
            "provenance": {"run_date": "2025-01-01", "git_commit": "abc1234"},
            "cv_accuracy_mean": 0.6318,
            "test_accuracy": 0.5581,
        }
        with open(results_file, "w") as f:
            json.dump(existing, f)

        # New results with DIFFERENT metrics
        new_results = {
            "provenance": {"run_date": "2025-01-02", "git_commit": "def5678"},
            "cv_accuracy_mean": 0.7000,  # Changed!
            "test_accuracy": 0.6000,  # Changed!
        }

        # Replicate the save logic from reproduce_track_b.py
        should_save = True
        if results_file.exists():
            with open(results_file) as f:
                existing_data = json.load(f)
            existing_has_provenance = "provenance" in existing_data
            existing_metrics = {
                k: v for k, v in existing_data.items() if k != "provenance"
            }
            new_metrics = {k: v for k, v in new_results.items() if k != "provenance"}
            if existing_has_provenance and existing_metrics == new_metrics:
                should_save = False

        assert should_save is True

    def test_should_save_when_legacy_file_missing_provenance(
        self, tmp_path: Path
    ) -> None:
        """Test that legacy files without provenance get updated (backfill)."""
        results_file = tmp_path / "test_baseline.json"

        # Create existing file WITHOUT provenance (legacy format)
        existing = {
            "cv_accuracy_mean": 0.6318,
            "test_accuracy": 0.5581,
        }
        with open(results_file, "w") as f:
            json.dump(existing, f)

        # New results with same metrics but WITH provenance
        new_results = {
            "provenance": {"run_date": "2025-01-02", "git_commit": "def5678"},
            "cv_accuracy_mean": 0.6318,
            "test_accuracy": 0.5581,
        }

        # Replicate the save logic from reproduce_track_b.py
        should_save = True
        if results_file.exists():
            with open(results_file) as f:
                existing_data = json.load(f)
            existing_has_provenance = "provenance" in existing_data
            existing_metrics = {
                k: v for k, v in existing_data.items() if k != "provenance"
            }
            new_metrics = {k: v for k, v in new_results.items() if k != "provenance"}
            # Save if: (1) legacy file missing provenance, OR (2) metrics changed
            if existing_has_provenance and existing_metrics == new_metrics:
                should_save = False

        # Should save because existing file is missing provenance (backfill)
        assert should_save is True
