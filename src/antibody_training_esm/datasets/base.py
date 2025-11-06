"""
Abstract Base Class for Dataset Preprocessing

Defines the common interface and shared logic for all antibody datasets.
Follows Open/Closed Principle - datasets extend this without modifying it.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


class AntibodyDataset(ABC):
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

    # Valid amino acid characters
    VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

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
            Path(output_dir) if output_dir else Path(f"outputs/{dataset_name}")
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
    def load_data(self, **kwargs) -> pd.DataFrame:
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
            stats["missing_vh"] = df["VH_sequence"].isna().sum()
            valid_vh = df["VH_sequence"].notna()

            if valid_vh.any():
                vh_lengths = df.loc[valid_vh, "VH_sequence"].str.len()
                stats["length_stats"]["VH"] = {
                    "min": int(vh_lengths.min()),
                    "max": int(vh_lengths.max()),
                    "mean": float(vh_lengths.mean()),
                }

        # Check VL sequences (if present)
        if "VL_sequence" in df.columns:
            stats["missing_vl"] = df["VL_sequence"].isna().sum()
            valid_vl = df["VL_sequence"].notna()

            if valid_vl.any():
                vl_lengths = df.loc[valid_vl, "VL_sequence"].str.len()
                stats["length_stats"]["VL"] = {
                    "min": int(vl_lengths.min()),
                    "max": int(vl_lengths.max()),
                    "mean": float(vl_lengths.mean()),
                }

        stats["valid_sequences"] = len(df) - stats["missing_vh"]
        stats["invalid_sequences"] = stats["missing_vh"]

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

    def annotate_sequence(
        self, sequence_id: str, sequence: str, chain: str
    ) -> dict[str, str] | None:
        """
        Annotate a single sequence using ANARCI (IMGT numbering).

        This method wraps riot_na.create_riot_aa() to extract CDR/FWR regions.

        Args:
            sequence_id: Unique identifier for the sequence
            sequence: Protein sequence to annotate
            chain: Chain type ("H" for heavy, "L" for light)

        Returns:
            Dictionary with keys: FWR1, CDR1, FWR2, CDR2, FWR3, CDR3, FWR4
            Returns None if annotation fails
        """
        try:
            # Import riot_na here to avoid dependency issues
            from riot_na import create_riot_aa

            # Run ANARCI annotation
            result = create_riot_aa(sequence_id, sequence, chain=chain)

            if result is None:
                self.logger.warning(
                    f"ANARCI annotation failed for {sequence_id} ({chain} chain)"
                )
                return None

            # Extract regions
            annotations = {
                "FWR1": result.get("FWR1", ""),
                "CDR1": result.get("CDR1", ""),
                "FWR2": result.get("FWR2", ""),
                "CDR2": result.get("CDR2", ""),
                "FWR3": result.get("FWR3", ""),
                "CDR3": result.get("CDR3", ""),
                "FWR4": result.get("FWR4", ""),
            }

            # Validate annotations (should not be empty)
            if not any(annotations.values()):
                self.logger.warning(
                    f"All annotations empty for {sequence_id} ({chain} chain)"
                )
                return None

            return annotations

        except Exception as e:
            self.logger.error(f"Error annotating {sequence_id} ({chain} chain): {e}")
            return None

    def annotate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Annotate all sequences in a DataFrame.

        Adds annotation columns for heavy and light chains.

        Args:
            df: DataFrame with VH_sequence and optionally VL_sequence columns

        Returns:
            DataFrame with annotation columns added
        """
        self.logger.info(f"Annotating {len(df)} sequences...")

        # Annotate heavy chains
        if "VH_sequence" in df.columns:
            self.logger.info("Annotating VH sequences...")
            vh_annotations = df.apply(
                lambda row: self.annotate_sequence(
                    row.get("id", f"seq_{row.name}"), row["VH_sequence"], "H"
                )
                if pd.notna(row["VH_sequence"])
                else None,
                axis=1,
            )

            # Extract annotation fields
            for field in ["FWR1", "CDR1", "FWR2", "CDR2", "FWR3", "CDR3", "FWR4"]:
                df[f"VH_{field}"] = vh_annotations.apply(
                    lambda x, f=field: x.get(f, "") if x else ""
                )

        # Annotate light chains (if present)
        if "VL_sequence" in df.columns:
            self.logger.info("Annotating VL sequences...")
            vl_annotations = df.apply(
                lambda row: self.annotate_sequence(
                    row.get("id", f"seq_{row.name}"), row["VL_sequence"], "L"
                )
                if pd.notna(row["VL_sequence"])
                else None,
                axis=1,
            )

            # Extract annotation fields
            for field in ["FWR1", "CDR1", "FWR2", "CDR2", "FWR3", "CDR3", "FWR4"]:
                df[f"VL_{field}"] = vl_annotations.apply(
                    lambda x, f=field: x.get(f, "") if x else ""
                )

        self.logger.info("Annotation complete")
        return df

    def create_fragments(self, row: pd.Series) -> dict[str, tuple[str, int, str]]:
        """
        Create all fragment types from an annotated sequence row.

        Args:
            row: DataFrame row with annotation columns

        Returns:
            Dictionary mapping fragment_type -> (sequence, label, source)
        """
        fragments = {}
        sequence_id = row.get("id", f"seq_{row.name}")
        label = row.get("label", 0)

        fragment_types = self.get_fragment_types()

        # Helper to concatenate regions
        def concat(*regions):
            return "".join(str(r) for r in regions if pd.notna(r) and r != "")

        # Full antibody fragments
        if "VH_only" in fragment_types:
            fragments["VH_only"] = (row.get("VH_sequence", ""), label, sequence_id)

        if "VL_only" in fragment_types:
            fragments["VL_only"] = (row.get("VL_sequence", ""), label, sequence_id)

        if "VH+VL" in fragment_types:
            vh = row.get("VH_sequence", "")
            vl = row.get("VL_sequence", "")
            fragments["VH+VL"] = (concat(vh, vl), label, sequence_id)

        # Heavy chain fragments
        if "H-CDR1" in fragment_types:
            fragments["H-CDR1"] = (row.get("VH_CDR1", ""), label, sequence_id)
        if "H-CDR2" in fragment_types:
            fragments["H-CDR2"] = (row.get("VH_CDR2", ""), label, sequence_id)
        if "H-CDR3" in fragment_types:
            fragments["H-CDR3"] = (row.get("VH_CDR3", ""), label, sequence_id)

        if "H-CDRs" in fragment_types:
            h_cdrs = concat(
                row.get("VH_CDR1", ""),
                row.get("VH_CDR2", ""),
                row.get("VH_CDR3", ""),
            )
            fragments["H-CDRs"] = (h_cdrs, label, sequence_id)

        if "H-FWRs" in fragment_types:
            h_fwrs = concat(
                row.get("VH_FWR1", ""),
                row.get("VH_FWR2", ""),
                row.get("VH_FWR3", ""),
                row.get("VH_FWR4", ""),
            )
            fragments["H-FWRs"] = (h_fwrs, label, sequence_id)

        # Light chain fragments
        if "L-CDR1" in fragment_types:
            fragments["L-CDR1"] = (row.get("VL_CDR1", ""), label, sequence_id)
        if "L-CDR2" in fragment_types:
            fragments["L-CDR2"] = (row.get("VL_CDR2", ""), label, sequence_id)
        if "L-CDR3" in fragment_types:
            fragments["L-CDR3"] = (row.get("VL_CDR3", ""), label, sequence_id)

        if "L-CDRs" in fragment_types:
            l_cdrs = concat(
                row.get("VL_CDR1", ""),
                row.get("VL_CDR2", ""),
                row.get("VL_CDR3", ""),
            )
            fragments["L-CDRs"] = (l_cdrs, label, sequence_id)

        if "L-FWRs" in fragment_types:
            l_fwrs = concat(
                row.get("VL_FWR1", ""),
                row.get("VL_FWR2", ""),
                row.get("VL_FWR3", ""),
                row.get("VL_FWR4", ""),
            )
            fragments["L-FWRs"] = (l_fwrs, label, sequence_id)

        # Combined fragments
        if "All-CDRs" in fragment_types:
            all_cdrs = concat(
                row.get("VH_CDR1", ""),
                row.get("VH_CDR2", ""),
                row.get("VH_CDR3", ""),
                row.get("VL_CDR1", ""),
                row.get("VL_CDR2", ""),
                row.get("VL_CDR3", ""),
            )
            fragments["All-CDRs"] = (all_cdrs, label, sequence_id)

        if "All-FWRs" in fragment_types:
            all_fwrs = concat(
                row.get("VH_FWR1", ""),
                row.get("VH_FWR2", ""),
                row.get("VH_FWR3", ""),
                row.get("VH_FWR4", ""),
                row.get("VL_FWR1", ""),
                row.get("VL_FWR2", ""),
                row.get("VL_FWR3", ""),
                row.get("VL_FWR4", ""),
            )
            fragments["All-FWRs"] = (all_fwrs, label, sequence_id)

        if "Full" in fragment_types:
            full = concat(
                row.get("VH_sequence", ""),
                row.get("VL_sequence", ""),
            )
            fragments["Full"] = (full, label, sequence_id)

        # Nanobody-specific (VHH)
        if "VHH_only" in fragment_types:
            fragments["VHH_only"] = (row.get("VH_sequence", ""), label, sequence_id)

        return fragments

    def create_fragment_csvs(self, df: pd.DataFrame, suffix: str = "") -> None:
        """
        Generate CSV files for all fragment types.

        Creates one CSV file per fragment type with columns:
        - id: sequence identifier
        - sequence: fragment sequence
        - label: binary label (0=specific, 1=non-specific)
        - source: original sequence ID

        Args:
            df: Annotated DataFrame
            suffix: Optional suffix for output filenames (e.g., "_filtered")
        """
        self.logger.info("Generating fragment CSVs...")

        fragment_types = self.get_fragment_types()

        # Collect fragments for each type
        fragment_data = {ftype: [] for ftype in fragment_types}

        for _, row in df.iterrows():
            fragments = self.create_fragments(row)
            for ftype, (seq, label, source) in fragments.items():
                if seq:  # Skip empty sequences
                    fragment_data[ftype].append(
                        {
                            "id": f"{source}_{ftype}",
                            "sequence": seq,
                            "label": label,
                            "source": source,
                        }
                    )

        # Write CSV files
        for ftype, data in fragment_data.items():
            if not data:
                self.logger.warning(f"No data for fragment type: {ftype}")
                continue

            output_file = self.output_dir / f"{ftype}_{self.dataset_name}{suffix}.csv"
            fragment_df = pd.DataFrame(data)

            # Write with metadata header
            with open(output_file, "w") as f:
                f.write(f"# Dataset: {self.dataset_name}\n")
                f.write(f"# Fragment type: {ftype}\n")
                f.write(f"# Total sequences: {len(fragment_df)}\n")
                f.write(
                    f"# Label distribution: "
                    f"{(fragment_df['label'] == 0).sum()} specific, "
                    f"{(fragment_df['label'] == 1).sum()} non-specific\n"
                )
                fragment_df.to_csv(f, index=False)

            self.logger.info(
                f"  {ftype}: {len(fragment_df)} sequences → {output_file.name}"
            )

        self.logger.info(f"Fragment CSVs written to {self.output_dir}")

    # ========== MAIN PROCESSING PIPELINE ==========

    def process(self, **kwargs) -> pd.DataFrame:
        """
        Main entry point for preprocessing pipeline.

        This method orchestrates the complete preprocessing workflow:
        1. Load raw data
        2. Validate sequences
        3. Annotate with ANARCI
        4. Generate fragments
        5. Save outputs

        Concrete classes can override this method to add dataset-specific steps.

        Args:
            **kwargs: Dataset-specific arguments passed to load_data()

        Returns:
            Processed DataFrame
        """
        self.logger.info(f"Starting preprocessing for {self.dataset_name} dataset")

        # Step 1: Load data
        self.logger.info("Step 1: Loading data...")
        df = self.load_data(**kwargs)
        self.print_statistics(df, stage="Raw")

        # Step 2: Annotate sequences
        self.logger.info("Step 2: Annotating sequences...")
        df = self.annotate_all(df)
        self.print_statistics(df, stage="Annotated")

        # Step 3: Generate fragment CSVs
        self.logger.info("Step 3: Generating fragment CSVs...")
        self.create_fragment_csvs(df)

        self.logger.info(f"Preprocessing complete for {self.dataset_name}")
        return df
