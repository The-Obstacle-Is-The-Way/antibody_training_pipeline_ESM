#!/usr/bin/env python3
"""
Mock Antibody Sequences for Testing

Provides realistic and edge-case antibody sequences for unit tests.
All sequences are for testing only and should NOT be used for production.

Usage:
    from tests.fixtures.mock_sequences import VALID_VH, VALID_VL, SEQUENCE_WITH_GAP

Date: 2025-11-07
Philosophy: Real sequences for realistic tests, edge cases for robustness
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

# Valid antibody sequences (real-world examples)

# Heavy chain variable region (VH) - 117 amino acids
VALID_VH = (
    "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYNMHWVRQAPGQGLEWMG"
    "GIYPGDSDTRYSPSFQGQVTISADKSISTAYLQWSSLKASDTAMYYCAR"
    "STYYGGDWYFNVWGQGTLVTVSS"
)

# Light chain variable region (VL) - 107 amino acids
VALID_VL = (
    "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIY"
    "AASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPLTF"
    "GGGTKVEIK"
)

# Short valid sequence (minimal test case)
SHORT_VH = "QVQLVQSGAEVKKPGA"

# Long valid sequence (stress test - ~250 aa)
LONG_VH = (
    "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYNMHWVRQAPGQGLEWMG"
    "GIYPGDSDTRYSPSFQGQVTISADKSISTAYLQWSSLKASDTAMYYCAR"
    "STYYGGDWYFNVWGQGTLVTVSSASTTKGPSVFPLAPSSKSTSGGTAAL"
    "GCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSS"
    "SLGTQTYICNVNHKPSNTKVDKKVEPKSCDKTHTCPPCPAPELLGGPSV"
)

# Edge case: Single amino acid
SINGLE_AA = "M"

# Edge case: All same amino acid (homopolymer)
HOMOPOLYMER = "AAAAAAAAAA"

# Invalid sequences (for error testing)

# Gap character (ANARCI alignment output - should fail)
SEQUENCE_WITH_GAP = "QVQL-VQSGAEVKKPGA"

# Multiple gaps
SEQUENCE_WITH_MULTIPLE_GAPS = "QVQL--VQ-SGAEVKKPGA"

# Invalid amino acid characters (not in standard 20 + X)
SEQUENCE_WITH_INVALID_AA = "QVQLVQSGAEVKKPGABBB"  # 'B' is ambiguous

# Non-amino acid characters
SEQUENCE_WITH_NUMBERS = "QVQLVQ123SGAEVKKPGA"
SEQUENCE_WITH_SPECIAL_CHARS = "QVQLVQ!@#SGAEVKKPGA"

# Empty sequence
EMPTY_SEQUENCE = ""

# Whitespace only
WHITESPACE_SEQUENCE = "   "

# Mixed case (should be uppercase)
MIXED_CASE_SEQUENCE = "qvQLvqSGAEvkkPGA"

# Valid amino acids for ESM-1v (from model.py:86)
VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYX")

# CDR sequences (for fragment testing)
H_CDR1 = "GYTFTSYN"
H_CDR2 = "IYPGDSDT"
H_CDR3 = "STYYGGDWYFNV"

L_CDR1 = "RASQSISSYLN"
L_CDR2 = "AASSLQS"
L_CDR3 = "QQSYSTPLT"

# Full sequences for paired testing
PAIRED_SEQUENCES = [
    {
        "id": "TEST001",
        "VH_sequence": VALID_VH,
        "VL_sequence": VALID_VL,
        "label": 0,  # Specific
    },
    {
        "id": "TEST002",
        "VH_sequence": SHORT_VH,
        "VL_sequence": VALID_VL,
        "label": 1,  # Non-specific
    },
]

# Batch of valid sequences (for batch embedding tests)
VALID_SEQUENCE_BATCH = [
    "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYNMH",
    "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLN",
    "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMH",
    "QSALTQPASVSGSPGQSITISCTGTSSDVGGYNYV",
    "QVQLQQWGAGLLKPSETLSLTCAVYGGSFSGYYWN",
]

# Mixed batch (valid + invalid - for error handling tests)
MIXED_SEQUENCE_BATCH = [
    "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYNMH",  # Valid
    "QVQL-VQSGAEVKKPGA",  # Gap
    "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLN",  # Valid
    "QVQLVQ123SGAEVKKPGA",  # Invalid chars
]


def validate_sequence(sequence: str) -> bool:
    """
    Validate antibody sequence (mirrors ESMEmbeddingExtractor logic).

    Args:
        sequence: Amino acid sequence to validate

    Returns:
        True if valid, False otherwise
    """
    if not sequence:
        return False
    return all(aa in VALID_AMINO_ACIDS for aa in sequence.upper())


def create_mock_dataframe(n_samples: int = 10, balanced: bool = True) -> pd.DataFrame:
    """
    Create mock pandas DataFrame for testing datasets.

    Args:
        n_samples: Number of samples to generate
        balanced: If True, creates 50/50 class distribution

    Returns:
        pandas.DataFrame with columns: id, VH_sequence, VL_sequence, label
    """
    import pandas as pd

    data = []
    for i in range(n_samples):
        label = i % 2 if balanced else 0
        data.append(
            {
                "id": f"MOCK{i:03d}",
                "VH_sequence": VALID_VH,
                "VL_sequence": VALID_VL,
                "label": label,
            }
        )

    return pd.DataFrame(data)
