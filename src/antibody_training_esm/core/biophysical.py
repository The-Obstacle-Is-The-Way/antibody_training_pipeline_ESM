"""
Biophysical Descriptor Module

Implements Track B (biophysical descriptors) from Sakhnini et al. 2025 (Novo Nordisk).
Phase A: Biopython-only implementation (3 descriptors) - no Schrödinger dependency.

Source: Table S1 from Sakhnini et al. 2025 "Prediction of Antibody Non-Specificity
using Protein Language Models and Biophysical Parameters"

Descriptors implemented (marked with * in Table S1 = Biopython):
- #21: Charge at pH 6.0
- #22: Charge at pH 7.4
- #66: Theoretical pI (isoelectric point)

Key finding from Table S2: Theoretical pI alone achieves 65.2% accuracy,
making it the most predictive single descriptor.

GitHub Issue: https://github.com/The-Obstacle-Is-The-Way/antibody_training_pipeline_ESM/issues/4

Date: 2025-11-27
"""

import logging

import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis

logger = logging.getLogger(__name__)


class BiophysicalExtractor:
    """
    Extract biophysical descriptors for protein sequences.

    Implements the 3 Biopython-calculable descriptors from Novo Nordisk 2025 paper
    (Sakhnini et al., Table S1). These are the ONLY descriptors not requiring
    Schrödinger BioLuminate (~$5-20K/year license).

    Descriptors (in order):
    1. Charge at pH 6.0 (Table S1 #21) - Endosomal compartment
    2. Charge at pH 7.4 (Table S1 #22) - Blood/plasma
    3. Theoretical pI (Table S1 #66) - Isoelectric point

    Note: The paper's Table S2 shows these charge descriptors were EXCLUDED from
    the "all descriptors" model due to correlation with pI. However, for our
    Phase A implementation, we include all 3 as they are the only free options.

    Reference:
        Sakhnini et al. 2025, "Prediction of Antibody Non-Specificity using
        Protein Language Models and Biophysical Parameters", Table S1.
    """

    # Descriptor names matching paper Table S1 numbering
    DESCRIPTOR_NAMES: list[str] = [
        "Charge_pH6.0",  # Table S1 #21
        "Charge_pH7.4",  # Table S1 #22
        "Theoretical_pI",  # Table S1 #66
    ]

    # Valid amino acids for Biopython ProteinAnalysis
    # NOTE: Unlike ESM, Biopython does NOT support 'X' (ambiguous)
    # Standard 20 amino acids only
    VALID_AMINO_ACIDS: set[str] = set("ACDEFGHIKLMNPQRSTVWY")

    def __init__(self) -> None:
        """
        Initialize BiophysicalExtractor.

        No model loading required - uses Biopython's ProteinAnalysis which
        performs calculations from amino acid property tables.
        """
        logger.info(
            f"BiophysicalExtractor initialized with {len(self.DESCRIPTOR_NAMES)} "
            f"Biopython descriptors (Table S1 #21, #22, #66): {self.DESCRIPTOR_NAMES}"
        )

    def extract_features(self, sequence: str) -> np.ndarray:
        """
        Extract biophysical features for a single protein sequence.

        Args:
            sequence: Amino acid sequence string (standard 20 AAs only)

        Returns:
            Feature vector as numpy array with shape (3,):
                [charge_pH6, charge_pH7.4, theoretical_pI]

        Raises:
            ValueError: If sequence is empty or contains invalid amino acids

        Note:
            Unlike ESMEmbeddingExtractor.embed_sequence, this does NOT support
            'X' (ambiguous) amino acids because Biopython's ProteinAnalysis
            requires exact amino acid identities for charge/pI calculations.
        """
        # Clean sequence (case and whitespace only - no silent mutation)
        seq = sequence.upper().strip()

        # Validate sequence length
        if len(seq) < 1:
            raise ValueError("Sequence is empty after stripping whitespace")

        # Validate amino acids (strict - no 'X' or '*' allowed)
        # Fail fast on invalid characters instead of silently stripping them
        invalid_chars = set(seq) - self.VALID_AMINO_ACIDS
        if invalid_chars:
            raise ValueError(
                f"Invalid amino acid characters: {invalid_chars}. "
                f"Biopython ProteinAnalysis requires standard 20 amino acids only. "
                f"'X' (ambiguous) and '*' (stop codon) are NOT supported. "
                f"Filter sequences at the dataset level before calling this extractor."
            )

        # Compute descriptors using Biopython
        # Note: Biopython lacks type stubs, hence type: ignore comments
        try:
            analysis = ProteinAnalysis(seq)  # type: ignore[no-untyped-call]

            charge_ph6: float = analysis.charge_at_pH(6.0)  # type: ignore[no-untyped-call]
            charge_ph74: float = analysis.charge_at_pH(7.4)  # type: ignore[no-untyped-call]
            theoretical_pi: float = analysis.isoelectric_point()  # type: ignore[no-untyped-call]

            features = np.array(
                [charge_ph6, charge_ph74, theoretical_pi],  # #21, #22, #66
                dtype=np.float32,
            )

            return features

        except Exception as e:
            logger.error(f"Biopython analysis failed for sequence: {seq[:50]}...")
            raise RuntimeError(f"Failed to compute biophysical features: {e}") from e

    def extract_batch_features(self, sequences: list[str]) -> np.ndarray:
        """
        Extract biophysical features for multiple sequences.

        Args:
            sequences: List of amino acid sequence strings

        Returns:
            Array of features with shape (n_sequences, 3)

        Raises:
            ValueError: If any sequence contains invalid amino acids
            RuntimeError: If feature extraction fails for any sequence
        """
        if not sequences:
            raise ValueError("Empty sequence list provided")

        logger.info(
            f"Extracting biophysical features for {len(sequences)} sequences..."
        )

        features_list: list[np.ndarray] = []

        for idx, seq in enumerate(sequences):
            try:
                features = self.extract_features(seq)
                features_list.append(features)

                # Progress logging every 100 sequences
                if (idx + 1) % 100 == 0:
                    logger.info(f"Processed {idx + 1}/{len(sequences)} sequences...")

            except ValueError as e:
                # Re-raise with sequence index context
                raise ValueError(f"Invalid sequence at index {idx}: {e}") from e
            except RuntimeError as e:
                # Re-raise with sequence index context
                raise RuntimeError(
                    f"Feature extraction failed at index {idx}: {e}"
                ) from e

        logger.info(f"Completed feature extraction for {len(sequences)} sequences")
        return np.array(features_list, dtype=np.float32)

    @property
    def n_features(self) -> int:
        """Number of features returned by this extractor."""
        return len(self.DESCRIPTOR_NAMES)

    @property
    def feature_names(self) -> list[str]:
        """
        Names of features for interpretability.

        Returns a copy to prevent external modification.
        """
        return self.DESCRIPTOR_NAMES.copy()
