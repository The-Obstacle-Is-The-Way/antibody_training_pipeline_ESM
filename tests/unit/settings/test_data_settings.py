"""Unit tests for DataSettings configuration."""

import pytest

from antibody_training_esm.settings import DataSettings


class TestDataSettings:
    """Test DataSettings path resolution and environment overrides."""

    def test_default_paths_resolve_correctly(self) -> None:
        """Default paths resolve to absolute paths."""
        settings = DataSettings()

        # Base paths should be absolute
        assert settings.PROJECT_ROOT.is_absolute()
        assert settings.DATA_DIR.is_absolute()
        assert settings.EXPERIMENTS_DIR.is_absolute()

        # Derived paths should also be absolute
        assert settings.DATA_TRAIN_DIR.is_absolute()
        assert settings.DATA_TEST_DIR.is_absolute()
        assert settings.BOUGHTER_DIR.is_absolute()
        assert settings.JAIN_DIR.is_absolute()

        # Paths should be under PROJECT_ROOT
        assert str(settings.DATA_DIR).startswith(str(settings.PROJECT_ROOT))
        assert str(settings.EXPERIMENTS_DIR).startswith(str(settings.PROJECT_ROOT))

    def test_env_var_overrides_work(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Environment variables override default paths."""
        # Override DATA_DIR via env var
        monkeypatch.setenv("ANTIBODY_DATA_DIR", "/tmp/test_data")
        monkeypatch.setenv("ANTIBODY_EXPERIMENTS_DIR", "/tmp/test_experiments")

        settings = DataSettings()

        # Overrides should take effect
        assert str(settings.DATA_DIR) == "/tmp/test_data"
        assert str(settings.EXPERIMENTS_DIR) == "/tmp/test_experiments"

        # Derived paths should use overridden base
        assert str(settings.DATA_TRAIN_DIR) == "/tmp/test_data/train"
        assert str(settings.DATA_TEST_DIR) == "/tmp/test_data/test"

    def test_computed_fields_resolve_correctly(self) -> None:
        """Computed fields resolve paths correctly."""
        settings = DataSettings()

        # Dataset directories
        assert settings.BOUGHTER_DIR == settings.DATA_TRAIN_DIR / "boughter"
        assert settings.JAIN_DIR == settings.DATA_TEST_DIR / "jain"
        assert settings.HARVEY_DIR == settings.DATA_TEST_DIR / "harvey"
        assert settings.SHEHATA_DIR == settings.DATA_TEST_DIR / "shehata"

        # Nested directories
        assert settings.BOUGHTER_RAW_DIR == settings.DATA_TRAIN_DIR / "boughter" / "raw"
        assert (
            settings.JAIN_PROCESSED_DIR == settings.DATA_TEST_DIR / "jain" / "processed"
        )

        # Specific files
        assert (
            settings.BOUGHTER_PROCESSED_CSV
            == settings.BOUGHTER_PROCESSED_DIR / "boughter.csv"
        )
        assert (
            settings.JAIN_FULL_CSV
            == settings.JAIN_PROCESSED_DIR / "jain_with_private_elisa_FULL.csv"
        )

    def test_relative_paths_resolve_against_project_root(self) -> None:
        """Relative paths are resolved against PROJECT_ROOT."""
        settings = DataSettings()

        # If DATA_DIR is relative, it should resolve to PROJECT_ROOT / DATA_DIR
        # (Default is Path("data"), which is relative)
        expected_data_dir = settings.PROJECT_ROOT / "data"
        assert expected_data_dir == settings.DATA_DIR

        expected_experiments_dir = settings.PROJECT_ROOT / "experiments"
        assert expected_experiments_dir == settings.EXPERIMENTS_DIR
