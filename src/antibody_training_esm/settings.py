from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataSettings(BaseSettings):
    """
    Centralized data path configuration.

    Replaces hardcoded paths in preprocessing/paths.py.
    Allows override via env vars (e.g. ANTIBODY_DATA_DIR=/tmp/data).
    """

    model_config = SettingsConfigDict(
        env_prefix="ANTIBODY_", env_file=".env", extra="ignore"
    )

    PROJECT_ROOT: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
    )
    DATA_DIR: Path = Field(default_factory=lambda: Path("data"))
    EXPERIMENTS_DIR: Path = Field(default_factory=lambda: Path("experiments"))

    def _resolve(self, path: Path) -> Path:
        """Resolve relative paths against PROJECT_ROOT."""
        return path if path.is_absolute() else self.PROJECT_ROOT / path

    @model_validator(mode="after")
    def _normalize_base_dirs(self) -> "DataSettings":
        object.__setattr__(self, "DATA_DIR", self._resolve(self.DATA_DIR))
        object.__setattr__(self, "EXPERIMENTS_DIR", self._resolve(self.EXPERIMENTS_DIR))
        return self

    # ============================================================================
    # Base data directories
    # ============================================================================
    @property
    def DATA_TRAIN_DIR(self) -> Path:
        return self._resolve(self.DATA_DIR) / "train"

    @property
    def DATA_TEST_DIR(self) -> Path:
        return self._resolve(self.DATA_DIR) / "test"

    # ============================================================================
    # Boughter (training set)
    # ============================================================================
    @property
    def BOUGHTER_DIR(self) -> Path:
        return self.DATA_TRAIN_DIR / "boughter"

    @property
    def BOUGHTER_RAW_DIR(self) -> Path:
        return self.BOUGHTER_DIR / "raw"

    @property
    def BOUGHTER_PROCESSED_DIR(self) -> Path:
        return self.BOUGHTER_DIR / "processed"

    @property
    def BOUGHTER_ANNOTATED_DIR(self) -> Path:
        return self.BOUGHTER_DIR / "annotated"

    @property
    def BOUGHTER_CANONICAL_DIR(self) -> Path:
        return self.BOUGHTER_DIR / "canonical"

    @property
    def BOUGHTER_PROCESSED_CSV(self) -> Path:
        return self.BOUGHTER_PROCESSED_DIR / "boughter.csv"

    @property
    def BOUGHTER_CANONICAL_CSV(self) -> Path:
        return self.BOUGHTER_CANONICAL_DIR / "boughter_vh_914.csv"

    # ============================================================================
    # Jain (test set)
    # ============================================================================
    @property
    def JAIN_DIR(self) -> Path:
        return self.DATA_TEST_DIR / "jain"

    @property
    def JAIN_RAW_DIR(self) -> Path:
        return self.JAIN_DIR / "raw"

    @property
    def JAIN_PROCESSED_DIR(self) -> Path:
        return self.JAIN_DIR / "processed"

    @property
    def JAIN_FRAGMENTS_DIR(self) -> Path:
        return self.JAIN_DIR / "fragments"

    @property
    def JAIN_CANONICAL_DIR(self) -> Path:
        return self.JAIN_DIR / "canonical"

    @property
    def JAIN_FULL_CSV(self) -> Path:
        return self.JAIN_PROCESSED_DIR / "jain_with_private_elisa_FULL.csv"

    @property
    def JAIN_SD03_CSV(self) -> Path:
        return self.JAIN_PROCESSED_DIR / "jain_sd03.csv"

    @property
    def JAIN_OUTPUT_DIR(self) -> Path:
        return self.JAIN_FRAGMENTS_DIR

    # ============================================================================
    # Harvey (test set)
    # ============================================================================
    @property
    def HARVEY_DIR(self) -> Path:
        return self.DATA_TEST_DIR / "harvey"

    @property
    def HARVEY_RAW_DIR(self) -> Path:
        return self.HARVEY_DIR / "raw"

    @property
    def HARVEY_PROCESSED_DIR(self) -> Path:
        return self.HARVEY_DIR / "processed"

    @property
    def HARVEY_FRAGMENTS_DIR(self) -> Path:
        return self.HARVEY_DIR / "fragments"

    @property
    def HARVEY_HIGH_POLY_CSV(self) -> Path:
        return self.HARVEY_RAW_DIR / "high_polyreactivity_high_throughput.csv"

    @property
    def HARVEY_LOW_POLY_CSV(self) -> Path:
        return self.HARVEY_RAW_DIR / "low_polyreactivity_high_throughput.csv"

    @property
    def HARVEY_OUTPUT_DIR(self) -> Path:
        return self.HARVEY_FRAGMENTS_DIR

    # ============================================================================
    # Shehata (test set)
    # ============================================================================
    @property
    def SHEHATA_DIR(self) -> Path:
        return self.DATA_TEST_DIR / "shehata"

    @property
    def SHEHATA_RAW_DIR(self) -> Path:
        return self.SHEHATA_DIR / "raw"

    @property
    def SHEHATA_PROCESSED_DIR(self) -> Path:
        return self.SHEHATA_DIR / "processed"

    @property
    def SHEHATA_FRAGMENTS_DIR(self) -> Path:
        return self.SHEHATA_DIR / "fragments"

    @property
    def SHEHATA_CANONICAL_DIR(self) -> Path:
        return self.SHEHATA_DIR / "canonical"

    @property
    def SHEHATA_EXCEL_PATH(self) -> Path:
        return self.SHEHATA_RAW_DIR / "shehata-mmc2.xlsx"

    @property
    def SHEHATA_OUTPUT_DIR(self) -> Path:
        return self.SHEHATA_FRAGMENTS_DIR


# Global instance
settings = DataSettings()
