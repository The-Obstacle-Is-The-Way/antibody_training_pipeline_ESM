"""
Abstract Base Class for Dataset Preprocessing

Defines the common interface and shared logic for all antibody datasets.
Follows Open/Closed Principle - datasets extend this without modifying it.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, cast

import pandas as pd

from antibody_training_esm.datasets.mixins.annotation_mixin import AnnotationMixin
from antibody_training_esm.datasets.mixins.fragment_mixin import FragmentMixin


class AntibodyDataset(ABC, AnnotationMixin, FragmentMixin):
    """
    Abstract base class for antibody dataset preprocessing.

    This class defines the common interface that all dataset preprocessors must implement
    and provides shared utility methods for common operations like sequence validation,
    ANARCI annotation, and fragment generation.

    Design Principles:
    - Single Responsibility: Each concrete class handles ONE dataset
    - Open/Closed: New datasets extend this class without modifying it
    - Dependency Inversion: High-level preprocessing depends on this abstraction
    """

    # Standard fragment types for full antibodies (VH + VL)
    FULL_ANTIBODY_FRAGMENTS = [
        "VH_only",
        "VL_only",
        "VH+VL",
        "H-CDR1",
        "H-CDR2",
        "H-CDR3",
        "L-CDR1",
        "L-CDR2",
        "L-CDR3",
        "H-CDRs",
        "L-CDRs",
        "All-CDRs",
        "H-FWRs",
        "L-FWRs",
        "All-FWRs",
        "Full",
    ]

    # Standard fragment types for nanobodies (VHH only)
    NANOBODY_FRAGMENTS = [
        "VHH_only",
        "H-CDR1",
        "H-CDR2",
        "H-CDR3",
        "H-CDRs",
        "H-FWRs",
    ]

    # Valid amino acid characters (20 standard + X for unknown/ambiguous)
    # X is included for compatibility with ESM models which support ambiguous residues
    VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYX")

    def __init__(
        self,
        dataset_name: str,
        output_dir: Path | None = None,
        logger: logging.Logger | None = None,
    ):
        """
        Initialize dataset preprocessor.

        Args:
            dataset_name: Name of the dataset (e.g., "jain", "harvey")
            output_dir: Directory to write processed outputs
            logger: Logger instance (creates default if None)
        """
        self.dataset_name = dataset_name
        self.output_dir = (
            Path(output_dir) if output_dir else Path(f"experiments/runs/{dataset_name}")
        )
        self.logger = logger or self._create_default_logger()

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _create_default_logger(self) -> logging.Logger:
        """Create a default logger if none provided"""
        logger = logging.getLogger(
            f"antibody_training_esm.datasets.{self.dataset_name}"
        )
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    # ========== ABSTRACT METHODS (MUST BE IMPLEMENTED) ==========

    @abstractmethod
    def load_data(self, **kwargs: Any) -> pd.DataFrame:
        """
        Load raw dataset from source files.

        This method must be implemented by each dataset since data loading
        is dataset-specific (Excel, CSV, FASTA, etc.).

        Returns:
            DataFrame with columns: id, VH_sequence, VL_sequence (optional), label
        """
        pass

    @abstractmethod
    def get_fragment_types(self) -> list[str]:
        """
        Return the list of fragment types for this dataset.

        Most datasets use FULL_ANTIBODY_FRAGMENTS (16 types).
        Nanobody datasets (like Harvey) use NANOBODY_FRAGMENTS (6 types).

        Returns:
            List of fragment type names
        """
        pass

    # ========== COMMON UTILITY METHODS ==========

    def sanitize_sequence(self, sequence: str) -> str:
        """
        Clean and validate a protein sequence.

        Operations:
        - Remove gap characters (-)
        - Remove whitespace
        - Convert to uppercase
        - Validate amino acids

        Args:
            sequence: Raw protein sequence

        Returns:
            Cleaned sequence

        Raises:
            ValueError: If sequence contains invalid characters
        """
        if not sequence or not isinstance(sequence, str):
            raise ValueError("Sequence must be a non-empty string")

        # Remove gaps and whitespace
        sequence = sequence.replace("-", "").replace(" ", "").upper()

        # Validate amino acids
        invalid_chars = set(sequence) - self.VALID_AMINO_ACIDS
        if invalid_chars:
            raise ValueError(
                f"Sequence contains invalid amino acids: {invalid_chars}. "
                f"Valid amino acids: {self.VALID_AMINO_ACIDS}"
            )

        return sequence

    def validate_sequences(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Validate all sequences in a DataFrame.

        Checks:
        - Valid amino acids
        - Sequence lengths
        - Missing sequences

        Args:
            df: DataFrame with VH_sequence and optionally VL_sequence columns

        Returns:
            Dictionary with validation statistics
        """
        stats = {
            "total_sequences": len(df),
            "valid_sequences": 0,
            "invalid_sequences": 0,
            "missing_vh": 0,
            "missing_vl": 0,
            "length_stats": {},
        }

        # Check VH sequences
        if "VH_sequence" in df.columns:
            missing_vh = int(df["VH_sequence"].isna().sum())
            stats["missing_vh"] = missing_vh
            valid_vh = df["VH_sequence"].notna()

            if valid_vh.any():
                vh_lengths = df.loc[valid_vh, "VH_sequence"].str.len()
                # Cast to help mypy understand the type
                length_stats = cast(
                    dict[str, dict[str, int | float]], stats["length_stats"]
                )
                length_stats["VH"] = {
                    "min": int(vh_lengths.min()),
                    "max": int(vh_lengths.max()),
                    "mean": float(vh_lengths.mean()),
                }

        # Check VL sequences (if present)
        if "VL_sequence" in df.columns:
            missing_vl = int(df["VL_sequence"].isna().sum())
            stats["missing_vl"] = missing_vl
            valid_vl = df["VL_sequence"].notna()

            if valid_vl.any():
                vl_lengths = df.loc[valid_vl, "VL_sequence"].str.len()
                # Cast to help mypy understand the type
                length_stats = cast(
                    dict[str, dict[str, int | float]], stats["length_stats"]
                )
                length_stats["VL"] = {
                    "min": int(vl_lengths.min()),
                    "max": int(vl_lengths.max()),
                    "mean": float(vl_lengths.mean()),
                }

        # Use explicit variable for type safety
        missing_vh_count = cast(int, stats["missing_vh"])
        stats["valid_sequences"] = len(df) - missing_vh_count
        stats["invalid_sequences"] = missing_vh_count

        return stats

    def print_statistics(self, df: pd.DataFrame, stage: str = "Final") -> None:
        """
        Print dataset statistics to logger.

        Args:
            df: DataFrame with processed data
            stage: Stage name for logging (e.g., "Raw", "Filtered", "Final")
        """
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"{stage} Dataset Statistics - {self.dataset_name}")
        self.logger.info(f"{'=' * 60}")

        # Basic counts
        self.logger.info(f"Total sequences: {len(df)}")

        # Label distribution
        if "label" in df.columns:
            label_counts = df["label"].value_counts()
            self.logger.info("\nLabel distribution:")
            for label, count in label_counts.items():
                percentage = (count / len(df)) * 100
                label_name = "Non-specific" if label == 1 else "Specific"
                self.logger.info(
                    f"  {label_name} (label={label}): {count} ({percentage:.1f}%)"
                )

        # Sequence validation stats
        val_stats = self.validate_sequences(df)
        self.logger.info("\nSequence validation:")
        self.logger.info(f"  Valid sequences: {val_stats['valid_sequences']}")
        self.logger.info(f"  Invalid sequences: {val_stats['invalid_sequences']}")

        if val_stats["length_stats"]:
            self.logger.info("\nSequence length statistics:")
            for chain, stats in val_stats["length_stats"].items():
                self.logger.info(
                    f"  {chain}: min={stats['min']}, max={stats['max']}, mean={stats['mean']:.1f}"
                )

        self.logger.info(f"{'=' * 60}\n")
