"""
Unit Tests for BiophysicalExtractor

Tests Biopython-based descriptor extraction (Phase A of Track B).
Uses known sequences with verified pI/charge values from literature.

Reference: Sakhnini et al. 2025, Table S1 (#21, #22, #66)

Date: 2025-11-27
"""

import numpy as np
import pytest

from antibody_training_esm.core.biophysical import BiophysicalExtractor

# ============================================================================
# Test Fixtures - Known Sequences with Expected Properties
# ============================================================================

# Trastuzumab VH (Herceptin) - well-characterized therapeutic antibody
TRASTUZUMAB_VH = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAED"
    "TAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
)

# Simple test sequences with predictable properties
ACIDIC_SEQUENCE = "EEEEEEEEEE"  # All glutamic acid - low pI (~3.2)
BASIC_SEQUENCE = "KKKKKKKKKK"  # All lysine - high pI (~9.7)
NEUTRAL_SEQUENCE = "GGGGGGGGGG"  # All glycine - neutral pI (~6.0)
MIXED_SEQUENCE = "ACDEFGHIKLMNPQRSTVWY"  # All 20 standard amino acids


@pytest.fixture
def extractor() -> BiophysicalExtractor:
    """Create BiophysicalExtractor instance."""
    return BiophysicalExtractor()


# ============================================================================
# Initialization Tests
# ============================================================================


@pytest.mark.unit
def test_extractor_initializes(extractor: BiophysicalExtractor) -> None:
    """Verify extractor initializes without errors."""
    assert extractor is not None


@pytest.mark.unit
def test_extractor_has_correct_feature_count(extractor: BiophysicalExtractor) -> None:
    """Verify extractor reports 3 features (Phase A)."""
    assert extractor.n_features == 3


@pytest.mark.unit
def test_feature_names_match_paper_table_s1(extractor: BiophysicalExtractor) -> None:
    """Verify feature names align with Novo paper Table S1."""
    names = extractor.feature_names

    assert len(names) == 3
    assert "Charge_pH6.0" in names  # Table S1 #21
    assert "Charge_pH7.4" in names  # Table S1 #22
    assert "Theoretical_pI" in names  # Table S1 #66


@pytest.mark.unit
def test_feature_names_returns_copy(extractor: BiophysicalExtractor) -> None:
    """Verify feature_names returns a copy (immutable)."""
    names1 = extractor.feature_names
    names2 = extractor.feature_names

    assert names1 is not names2  # Different objects
    assert names1 == names2  # Same content


# ============================================================================
# Single Sequence Extraction Tests
# ============================================================================


@pytest.mark.unit
def test_extract_returns_correct_shape(extractor: BiophysicalExtractor) -> None:
    """Verify extraction returns (3,) array."""
    features = extractor.extract_features(TRASTUZUMAB_VH)

    assert features.shape == (3,)


@pytest.mark.unit
def test_extract_returns_float32(extractor: BiophysicalExtractor) -> None:
    """Verify extraction returns float32 dtype."""
    features = extractor.extract_features(TRASTUZUMAB_VH)

    assert features.dtype == np.float32


@pytest.mark.unit
def test_extract_returns_numpy_array(extractor: BiophysicalExtractor) -> None:
    """Verify extraction returns numpy array."""
    features = extractor.extract_features(TRASTUZUMAB_VH)

    assert isinstance(features, np.ndarray)


# ============================================================================
# Scientific Validation Tests - pI
# ============================================================================


@pytest.mark.unit
def test_acidic_sequence_has_low_pi(extractor: BiophysicalExtractor) -> None:
    """Verify acidic sequence (all Glu) has low pI (<5.0)."""
    features = extractor.extract_features(ACIDIC_SEQUENCE)
    pi = features[2]  # Theoretical_pI is index 2

    assert pi < 5.0, f"Expected pI < 5.0 for all-Glu sequence, got {pi:.2f}"


@pytest.mark.unit
def test_basic_sequence_has_high_pi(extractor: BiophysicalExtractor) -> None:
    """Verify basic sequence (all Lys) has high pI (>9.0)."""
    features = extractor.extract_features(BASIC_SEQUENCE)
    pi = features[2]

    assert pi > 9.0, f"Expected pI > 9.0 for all-Lys sequence, got {pi:.2f}"


@pytest.mark.unit
def test_neutral_sequence_has_neutral_pi(extractor: BiophysicalExtractor) -> None:
    """Verify neutral sequence (all Gly) has near-neutral pI (~6.0)."""
    features = extractor.extract_features(NEUTRAL_SEQUENCE)
    pi = features[2]

    assert 5.5 < pi < 6.5, f"Expected pI ~6.0 for all-Gly sequence, got {pi:.2f}"


# ============================================================================
# Scientific Validation Tests - Charge
# ============================================================================


@pytest.mark.unit
def test_charge_higher_at_lower_ph(extractor: BiophysicalExtractor) -> None:
    """
    Verify charge at pH 6 > charge at pH 7.4.

    Scientific basis: At lower pH, proteins are more protonated,
    resulting in more positive (less negative) charge.
    """
    features = extractor.extract_features(TRASTUZUMAB_VH)
    charge_ph6 = features[0]
    charge_ph74 = features[1]

    assert charge_ph6 > charge_ph74, (
        f"Charge at pH 6 ({charge_ph6:.2f}) should be higher than at pH 7.4 ({charge_ph74:.2f})"
    )


@pytest.mark.unit
def test_acidic_sequence_has_negative_charge(extractor: BiophysicalExtractor) -> None:
    """Verify acidic sequence has negative charge at physiological pH."""
    features = extractor.extract_features(ACIDIC_SEQUENCE)
    charge_ph74 = features[1]

    assert charge_ph74 < 0, (
        f"Expected negative charge for all-Glu, got {charge_ph74:.2f}"
    )


@pytest.mark.unit
def test_basic_sequence_has_positive_charge(extractor: BiophysicalExtractor) -> None:
    """Verify basic sequence has positive charge at physiological pH."""
    features = extractor.extract_features(BASIC_SEQUENCE)
    charge_ph74 = features[1]

    assert charge_ph74 > 0, (
        f"Expected positive charge for all-Lys, got {charge_ph74:.2f}"
    )


# ============================================================================
# Batch Extraction Tests
# ============================================================================


@pytest.mark.unit
def test_batch_returns_correct_shape(extractor: BiophysicalExtractor) -> None:
    """Verify batch extraction returns (n, 3) array."""
    sequences = [TRASTUZUMAB_VH, ACIDIC_SEQUENCE, BASIC_SEQUENCE]
    features = extractor.extract_batch_features(sequences)

    assert features.shape == (3, 3)


@pytest.mark.unit
def test_batch_returns_float32(extractor: BiophysicalExtractor) -> None:
    """Verify batch extraction returns float32 dtype."""
    sequences = [TRASTUZUMAB_VH, ACIDIC_SEQUENCE]
    features = extractor.extract_batch_features(sequences)

    assert features.dtype == np.float32


@pytest.mark.unit
def test_batch_single_sequence(extractor: BiophysicalExtractor) -> None:
    """Verify batch extraction works with single sequence."""
    features = extractor.extract_batch_features([TRASTUZUMAB_VH])

    assert features.shape == (1, 3)


@pytest.mark.unit
def test_batch_matches_single_extraction(extractor: BiophysicalExtractor) -> None:
    """Verify batch extraction matches individual extractions."""
    sequences = [ACIDIC_SEQUENCE, BASIC_SEQUENCE]

    # Individual extractions
    individual = [extractor.extract_features(seq) for seq in sequences]
    individual_array = np.array(individual)

    # Batch extraction
    batch = extractor.extract_batch_features(sequences)

    np.testing.assert_array_almost_equal(batch, individual_array)


# ============================================================================
# Input Validation Tests
# ============================================================================


@pytest.mark.unit
def test_rejects_empty_sequence(extractor: BiophysicalExtractor) -> None:
    """Verify extractor rejects empty sequences."""
    with pytest.raises(ValueError, match="empty"):
        extractor.extract_features("")


@pytest.mark.unit
def test_rejects_whitespace_only(extractor: BiophysicalExtractor) -> None:
    """Verify extractor rejects whitespace-only sequences."""
    with pytest.raises(ValueError, match="empty"):
        extractor.extract_features("   ")


@pytest.mark.unit
def test_rejects_invalid_amino_acids(extractor: BiophysicalExtractor) -> None:
    """Verify extractor rejects sequences with invalid characters."""
    with pytest.raises(ValueError, match="Invalid amino acid"):
        extractor.extract_features("QVQLBZJ")  # B, Z, J are invalid


@pytest.mark.unit
def test_rejects_numbers(extractor: BiophysicalExtractor) -> None:
    """Verify extractor rejects sequences with numbers."""
    with pytest.raises(ValueError, match="Invalid amino acid"):
        extractor.extract_features("QVQL123")


@pytest.mark.unit
def test_rejects_ambiguous_x(extractor: BiophysicalExtractor) -> None:
    """
    Verify extractor rejects 'X' (ambiguous amino acid).

    Note: Unlike ESM which supports 'X', Biopython's ProteinAnalysis
    requires exact amino acid identities for charge/pI calculations.
    """
    with pytest.raises(ValueError, match="Invalid amino acid"):
        extractor.extract_features("QVQLXESG")


@pytest.mark.unit
def test_rejects_empty_batch(extractor: BiophysicalExtractor) -> None:
    """Verify extractor rejects empty batch."""
    with pytest.raises(ValueError, match="Empty sequence list"):
        extractor.extract_batch_features([])


@pytest.mark.unit
def test_batch_rejects_invalid_sequence_with_index(
    extractor: BiophysicalExtractor,
) -> None:
    """Verify batch extraction reports index of invalid sequence."""
    sequences = [ACIDIC_SEQUENCE, "INVALIDXSEQ", BASIC_SEQUENCE]

    with pytest.raises(ValueError, match="Invalid sequence at index 1"):
        extractor.extract_batch_features(sequences)


@pytest.mark.unit
def test_batch_progress_logging(extractor: BiophysicalExtractor) -> None:
    """Verify batch extraction handles 100+ sequences with progress logging."""
    # Create 105 sequences to trigger progress logging at 100
    sequences = [ACIDIC_SEQUENCE] * 105

    features = extractor.extract_batch_features(sequences)

    assert features.shape == (105, 3)


# ============================================================================
# Input Normalization Tests
# ============================================================================


@pytest.mark.unit
def test_handles_lowercase_input(extractor: BiophysicalExtractor) -> None:
    """Verify extractor handles lowercase sequences."""
    features_upper = extractor.extract_features(ACIDIC_SEQUENCE)
    features_lower = extractor.extract_features(ACIDIC_SEQUENCE.lower())

    np.testing.assert_array_equal(features_upper, features_lower)


@pytest.mark.unit
def test_handles_mixed_case_input(extractor: BiophysicalExtractor) -> None:
    """Verify extractor handles mixed case sequences."""
    features = extractor.extract_features("EeEeEeEeEe")

    assert features.shape == (3,)


@pytest.mark.unit
def test_strips_whitespace(extractor: BiophysicalExtractor) -> None:
    """Verify extractor strips leading/trailing whitespace."""
    features_clean = extractor.extract_features(ACIDIC_SEQUENCE)
    features_whitespace = extractor.extract_features(f"  {ACIDIC_SEQUENCE}  ")

    np.testing.assert_array_equal(features_clean, features_whitespace)


@pytest.mark.unit
def test_rejects_stop_codon_asterisk(extractor: BiophysicalExtractor) -> None:
    """Verify extractor fails fast on stop codon asterisks (no silent stripping)."""
    # Stop codons should be filtered at the dataset level, not silently stripped
    with pytest.raises(ValueError, match=r"Invalid amino acid.*\*"):
        extractor.extract_features(f"{ACIDIC_SEQUENCE}*")


# ============================================================================
# Edge Cases
# ============================================================================


@pytest.mark.unit
def test_single_amino_acid(extractor: BiophysicalExtractor) -> None:
    """Verify extractor handles single amino acid sequences."""
    features = extractor.extract_features("K")

    assert features.shape == (3,)
    # Single lysine pI is ~8.75 (affected by terminal groups, not just side chain)
    assert features[2] > 8.0, f"Expected pI > 8.0 for single Lys, got {features[2]:.2f}"


@pytest.mark.unit
def test_all_standard_amino_acids(extractor: BiophysicalExtractor) -> None:
    """Verify extractor handles sequence with all 20 standard amino acids."""
    features = extractor.extract_features(MIXED_SEQUENCE)

    assert features.shape == (3,)
    assert not np.isnan(features).any(), "Features should not contain NaN"
    assert not np.isinf(features).any(), "Features should not contain Inf"


@pytest.mark.unit
def test_long_sequence(extractor: BiophysicalExtractor) -> None:
    """Verify extractor handles long sequences (>500 AA)."""
    long_sequence = TRASTUZUMAB_VH * 5  # ~600 AA

    features = extractor.extract_features(long_sequence)

    assert features.shape == (3,)
    assert not np.isnan(features).any(), "Features should not contain NaN"
    assert not np.isinf(features).any(), "Features should not contain Inf"
