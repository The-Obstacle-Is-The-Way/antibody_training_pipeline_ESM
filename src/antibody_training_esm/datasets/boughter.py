"""
Boughter Dataset Loader

Loads preprocessed Boughter mouse antibody dataset.

IMPORTANT: This module is for LOADING preprocessed data, not for running
the preprocessing pipeline. The preprocessing scripts that CREATE the data
are in: preprocessing/boughter/stage2_stage3_annotation_qc.py

Dataset characteristics:
- Full antibodies (VH + VL)
- Mouse antibodies from 6 subsets (flu, hiv, gut, mouse IgA)
- DNA sequences requiring translation to protein
- Novo flagging strategy (0/1-3/4+ flags)
- 3-stage quality control pipeline
- 16 fragment types (full antibody)

Processing Pipeline:
  Stage 1: DNA translation (FASTA → protein sequences)
  Stage 2: ANARCI annotation (riot_na)
  Stage 3: Post-annotation QC (filter X in CDRs, empty CDRs)

Source:
- train_datasets/boughter/raw/ (multiple subsets)
- Sequences in DNA format requiring translation

Reference:
- Boughter et al., "Biochemical patterns of antibody polyreactivity revealed through a bioinformatics-based analysis of CDR loops"
"""

from pathlib import Path
from typing import NoReturn

import pandas as pd

from .base import AntibodyDataset


class BoughterDataset(AntibodyDataset):
    """
    Loader for Boughter mouse antibody dataset.

    This class provides an interface to LOAD preprocessed Boughter dataset files.
    It does NOT run the preprocessing pipeline - use preprocessing/boughter/stage2_stage3_annotation_qc.py for that.

    The Boughter dataset originally requires DNA translation before standard preprocessing.
    Sequences are provided as DNA in FASTA format and must be translated
    to protein sequences using a hybrid translation strategy (done by preprocessing scripts).
    """

    # Novo flagging strategy
    FLAG_SPECIFIC = 0  # 0 flags = specific (include in training)
    FLAG_MILD = [1, 2, 3]  # 1-3 flags = mild (EXCLUDE from training)
    FLAG_NONSPECIFIC = [4, 5, 6, 7]  # 4+ flags = non-specific (include in training)

    # Dataset subsets
    SUBSETS = ["flu", "hiv_nat", "hiv_cntrl", "hiv_plos", "gut_hiv", "mouse_iga"]

    def __init__(self, output_dir: Path | None = None, logger=None):
        """
        Initialize Boughter dataset loader.

        Args:
            output_dir: Directory containing preprocessed fragment files
            logger: Logger instance
        """
        super().__init__(
            dataset_name="boughter",
            output_dir=output_dir or Path("train_datasets/boughter/annotated"),
            logger=logger,
        )

    def get_fragment_types(self) -> list[str]:
        """
        Return full antibody fragment types.

        Boughter contains VH + VL sequences, so we generate all 16 fragment types.

        Returns:
            List of 16 full antibody fragment types
        """
        return self.FULL_ANTIBODY_FRAGMENTS

    def load_data(  # type: ignore[override]
        self,
        processed_csv: str | None = None,
        subset: str | None = None,
        include_mild: bool = False,
    ) -> pd.DataFrame:
        """
        Load Boughter dataset from processed CSV.

        Note: This assumes DNA translation has already been performed.
        For DNA translation from FASTA files, use the preprocessing scripts
        in preprocessing/boughter/

        Args:
            processed_csv: Path to processed CSV with protein sequences
            subset: Specific subset to load (flu, hiv_nat, etc.) or None for all
            include_mild: If True, include mild (1-3 flags). Default False.

        Returns:
            DataFrame with columns: id, VH_sequence, VL_sequence, label, flags, include_in_training

        Raises:
            FileNotFoundError: If processed CSV not found
        """
        # Default path
        if processed_csv is None:
            processed_csv = "train_datasets/boughter/boughter_translated.csv"

        csv_file = Path(processed_csv)
        if not csv_file.exists():
            raise FileNotFoundError(
                f"Boughter processed CSV not found: {csv_file}\n"
                f"Please run DNA translation preprocessing first:\n"
                f"  python preprocessing/boughter/stage1_dna_translation.py"
            )

        # Load data
        self.logger.info(f"Reading Boughter dataset from {csv_file}...")
        df = pd.read_csv(csv_file)
        self.logger.info(f"  Loaded {len(df)} sequences")

        # Filter by subset if specified
        if subset is not None:
            if subset not in self.SUBSETS:
                raise ValueError(f"Unknown subset: {subset}. Valid: {self.SUBSETS}")
            df = df[df["subset"] == subset].copy()
            self.logger.info(f"  Filtered to subset '{subset}': {len(df)} sequences")

        # Apply Novo flagging strategy
        if not include_mild:
            # Exclude mild (1-3 flags) per Novo Nordisk methodology
            df["include_in_training"] = ~df["flags"].isin(self.FLAG_MILD)
            df_training = df[df["include_in_training"]].copy()

            excluded = len(df) - len(df_training)
            self.logger.info("\nNovo flagging strategy:")
            self.logger.info(f"  Excluded {excluded} sequences with mild flags (1-3)")
            self.logger.info(f"  Training set: {len(df_training)} sequences")

            df = df_training

        # Standardize column names
        column_mapping = {
            "heavy_seq": "VH_sequence",
            "light_seq": "VL_sequence",
        }
        if "heavy_seq" in df.columns:
            df = df.rename(columns=column_mapping)

        # Create binary labels from flags
        # 0 flags → specific (label=0)
        # 4+ flags → non-specific (label=1)
        if "flags" in df.columns:
            df["label"] = (df["flags"] >= 4).astype(int)

        self.logger.info("\nLabel distribution:")
        label_counts = df["label"].value_counts().sort_index()
        for label, count in label_counts.items():
            label_name = "Specific" if label == 0 else "Non-specific"
            percentage = (count / len(df)) * 100
            self.logger.info(
                f"  {label_name} (label={label}): {count} ({percentage:.1f}%)"
            )

        return df

    def translate_dna_to_protein(self, dna_sequence: str) -> NoReturn:  # noqa: ARG002
        """
        This method is NOT IMPLEMENTED and will always raise an error.

        DNA translation logic belongs in the preprocessing scripts, not in
        dataset loader classes. Loaders are for LOADING preprocessed data,
        not for creating it.

        For DNA translation, use:
            preprocessing/boughter/stage1_dna_translation.py

        Args:
            dna_sequence: DNA sequence string (unused - always raises)

        Raises:
            NotImplementedError: Always - this method intentionally does nothing
        """
        raise NotImplementedError(
            "DNA translation is not implemented in dataset loader classes.\n"
            "Dataset loaders are for LOADING preprocessed data, not creating it.\n"
            "\n"
            "For DNA translation, use the preprocessing script:\n"
            "  python preprocessing/boughter/stage1_dna_translation.py\n"
            "\n"
            "This script implements the full hybrid translation strategy:\n"
            "  1. Direct V-domain translation (pre-trimmed sequences)\n"
            "  2. ATG-based translation (full-length with signal peptide)\n"
            "  3. V-domain motif detection (EVQL, QVQL, etc.)"
        )

    def filter_quality_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stage 3 QC: Filter sequences with quality issues.

        Removes:
        - Sequences with X in CDRs (ambiguous amino acids)
        - Sequences with empty CDRs
        - Invalid annotations

        Args:
            df: Annotated DataFrame

        Returns:
            Filtered DataFrame
        """
        initial_count = len(df)

        # Filter X in CDRs
        cdr_cols = [
            col for col in df.columns if "CDR" in col and ("VH_" in col or "VL_" in col)
        ]

        if cdr_cols:
            for col in cdr_cols:
                if col in df.columns:
                    df = df[~df[col].str.contains("X", na=False)].copy()

        # Filter empty CDRs
        for col in cdr_cols:
            if col in df.columns:
                df = df[df[col].str.len() > 0].copy()

        filtered_count = initial_count - len(df)

        if filtered_count > 0:
            self.logger.info(f"\nStage 3 QC filtered {filtered_count} sequences:")
            self.logger.info(f"  Remaining: {len(df)} sequences")

        return df


# ========== CONVENIENCE FUNCTIONS FOR LOADING DATA ==========


def load_boughter_data(
    processed_csv: str | None = None,
    subset: str | None = None,
    include_mild: bool = False,
) -> pd.DataFrame:
    """
    Convenience function to load preprocessed Boughter dataset.

    IMPORTANT: This loads PREPROCESSED data. To preprocess raw data, use:
    preprocessing/boughter/stage2_stage3_annotation_qc.py

    Args:
        processed_csv: Path to processed CSV with protein sequences
        subset: Specific subset to load or None for all
        include_mild: If True, include mild (1-3 flags)

    Returns:
        DataFrame with preprocessed data

    Example:
        >>> from antibody_training_esm.datasets.boughter import load_boughter_data
        >>> df = load_boughter_data(include_mild=False)  # Novo flagging
        >>> print(f"Loaded {len(df)} sequences")
    """
    dataset = BoughterDataset()
    return dataset.load_data(
        processed_csv=processed_csv, subset=subset, include_mild=include_mild
    )
