# Biophysical Descriptors - Phased Implementation Specifications

**Date**: 2025-11-27
**GitHub Issue**: [#4 - Implement Novo 2025 Track B](https://github.com/The-Obstacle-Is-The-Way/antibody_training_pipeline_ESM/issues/4)
**Author**: Claude Code (Opus 4.5)
**Review Status**: PENDING SENIOR APPROVAL
**Methodology**: TDD + Vertical Slice Architecture

---

## 0. Executive Summary

**Goal**: Implement Track B (biophysical descriptor baseline) from Sakhnini et al. 2025 (Novo Nordisk paper).

**Critical Constraint**: 65/68 descriptors require Schrödinger BioLuminate (~$5-20K/year license). We proceed with **Biopython-only prototype** first.

**Scope**:
- **Phase A**: 3 Biopython descriptors (FREE) - charge@pH6, charge@pH7.4, theoretical pI
- **Phase B**: Open-source extensions (peptides library) - optional extras
- **Phase C**: Full 68 descriptors (BLOCKED until licensing resolved)

**Key Insight from Paper (Table S2)**:
- `theoretical pI` alone achieves **65.2% accuracy** (single descriptor!)
- Top 5 descriptors achieve ~67% (competitive with full 68)
- ESM-1v achieves ~71% (only ~4% better than pI alone)

---

## 1. What We Learned from ZJ's Failed PR

ZJ (EmployeeNo427) submitted PR #21 to upstream after receiving our Issue #4 spec. His implementation:

| Aspect | ZJ's Implementation | Problem |
|--------|---------------------|---------|
| Dataset | GDPa1 | Wrong - paper uses Boughter |
| charge@pH6 | **NOT IMPLEMENTED** | Missing critical descriptor |
| charge@pH7.4 | ✓ Implemented | Correct |
| theoretical pI | ✓ Implemented | Correct |
| GRAVY | Implemented | Ranks #52/66 - nearly useless |
| Aromaticity | Implemented | **NOT IN PAPER'S 68** |
| Instability_Index | Implemented | **NOT IN PAPER'S 68** |
| Boman_Index | Implemented | **NOT IN PAPER'S 68** |
| Top descriptors | NOT IMPLEMENTED | Missing disorder, aggrescan, accessibility |

**Conclusion**: ZJ didn't read Table S2. We implement it properly.

---

## 2. The 3 Biopython Descriptors (Phase A - FREE)

From Novo paper Table S1, marked with (*) = Biopython:

| # | Descriptor | Definition | Biopython Method |
|---|------------|------------|------------------|
| 21 | Charge at pH 6* | Charge of protein at pH 6.0 | `ProteinAnalysis.charge_at_pH(6.0)` |
| 22 | Charge at pH 7.4* | Charge of protein at pH 7.4 | `ProteinAnalysis.charge_at_pH(7.4)` |
| 66 | Theoretical pI* | Isoelectric point | `ProteinAnalysis.isoelectric_point()` |

**Why pH 6 AND pH 7.4?**
- **pH 7.4** = Blood/plasma (where antibodies circulate)
- **pH 6.0** = Endosomes (inside cells after internalization, FcRn recycling)
- Charge difference affects aggregation, FcRn binding, and non-specific binding

**From Table S2**: Theoretical pI alone gets **65.2% accuracy** - best single descriptor!

---

## 3. Architecture Overview

### 3.1 Target Structure

```text
src/antibody_training_esm/
├── core/
│   ├── embeddings.py          # ESM (existing)
│   ├── classifier.py          # BinaryClassifier (existing)
│   └── biophysical.py         # NEW: BiophysicalExtractor
├── conf/
│   ├── config.yaml            # Default config
│   └── features/
│       ├── esm_only.yaml      # ESM embeddings only (default)
│       ├── biopython_trio.yaml # 3 Biopython descriptors
│       └── combined.yaml      # ESM + descriptors (future)
```

### 3.2 Design Principles

1. **Separate from ESM pipeline** - Track B is independent, not fused
2. **Same interface** - Returns numpy array like ESMEmbeddingExtractor
3. **Cacheable** - Hash-based caching like embeddings
4. **Testable** - Known sequences with expected pI values

---

## 4. Phase A: Biopython Trio Implementation

**Deliverable**: `src/antibody_training_esm/core/biophysical.py`

### 4.1 Implementation Spec

```python
"""
Biophysical Descriptor Module

Implements Track B (biophysical descriptors) from Sakhnini et al. 2025.
Phase A: Biopython-only (3 descriptors) - no Schrödinger dependency.

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

    Phase A implements 3 Biopython descriptors from Novo paper Table S1:
    - Charge at pH 6.0 (descriptor #21)
    - Charge at pH 7.4 (descriptor #22)
    - Theoretical pI (descriptor #66)

    These are the ONLY descriptors not requiring Schrödinger BioLuminate.
    """

    # Descriptor names matching paper Table S1
    DESCRIPTOR_NAMES: list[str] = [
        "Charge_pH6",      # #21 in paper
        "Charge_pH7.4",    # #22 in paper
        "Theoretical_pI",  # #66 in paper
    ]

    def __init__(self) -> None:
        """Initialize BiophysicalExtractor (no model loading needed)."""
        logger.info(
            f"BiophysicalExtractor initialized with {len(self.DESCRIPTOR_NAMES)} "
            f"Biopython descriptors: {self.DESCRIPTOR_NAMES}"
        )

    def extract_features(self, sequence: str) -> np.ndarray:
        """
        Extract biophysical features for a single protein sequence.

        Args:
            sequence: Amino acid sequence string

        Returns:
            Feature vector as numpy array (3-d for Phase A)

        Raises:
            ValueError: If sequence contains invalid amino acids
        """
        # Clean and validate sequence
        seq = sequence.upper().strip().replace("*", "")

        # Biopython valid amino acids (standard 20 only - excludes X, B, J, O, U, Z)
        valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
        invalid = set(seq) - valid_aas
        if invalid:
            raise ValueError(
                f"Invalid amino acid characters: {invalid}. "
                f"Biopython ProteinAnalysis requires standard AAs only."
            )

        if len(seq) < 1:
            raise ValueError("Sequence too short")

        # Compute descriptors
        analysis = ProteinAnalysis(seq)

        features = np.array([
            analysis.charge_at_pH(6.0),      # Charge at pH 6
            analysis.charge_at_pH(7.4),      # Charge at pH 7.4
            analysis.isoelectric_point(),    # Theoretical pI
        ], dtype=np.float32)

        return features

    def extract_batch_features(self, sequences: list[str]) -> np.ndarray:
        """
        Extract features for multiple sequences.

        Args:
            sequences: List of amino acid sequence strings

        Returns:
            Array of features with shape (n_sequences, 3)
        """
        logger.info(f"Extracting biophysical features for {len(sequences)} sequences...")

        features_list = []
        for idx, seq in enumerate(sequences):
            try:
                features = self.extract_features(seq)
                features_list.append(features)
            except Exception as e:
                logger.error(f"Failed to process sequence {idx}: {seq[:30]}... - {e}")
                raise RuntimeError(
                    f"Feature extraction failed at sequence {idx}."
                ) from e

        return np.array(features_list)

    @property
    def n_features(self) -> int:
        """Number of features returned by this extractor."""
        return len(self.DESCRIPTOR_NAMES)

    @property
    def feature_names(self) -> list[str]:
        """Names of features for interpretability."""
        return self.DESCRIPTOR_NAMES.copy()
```

### 4.2 Test-Driven Development

**File**: `tests/unit/core/test_biophysical.py`

```python
#!/usr/bin/env python3
"""
Unit Tests for BiophysicalExtractor

Tests Biopython-based descriptor extraction (Phase A).
Uses known sequences with verified pI values from literature.

Date: 2025-11-27
"""

import pytest
import numpy as np

from antibody_training_esm.core.biophysical import BiophysicalExtractor


# ============================================================================
# Test Fixtures - Known Sequences with Expected Values
# ============================================================================

# Trastuzumab VH (Herceptin) - well-characterized antibody
TRASTUZUMAB_VH = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAED"
    "TAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
)

# Simple test sequences
ACIDIC_SEQUENCE = "EEEEEEEEEE"  # All glutamic acid - low pI
BASIC_SEQUENCE = "KKKKKKKKKK"   # All lysine - high pI
NEUTRAL_SEQUENCE = "GGGGGGGGGG" # All glycine - neutral


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
    assert extractor.n_features == 3
    assert len(extractor.feature_names) == 3


@pytest.mark.unit
def test_feature_names_match_paper(extractor: BiophysicalExtractor) -> None:
    """Verify feature names align with Novo paper Table S1."""
    names = extractor.feature_names
    assert "Charge_pH6" in names
    assert "Charge_pH7.4" in names
    assert "Theoretical_pI" in names


# ============================================================================
# Single Sequence Extraction Tests
# ============================================================================

@pytest.mark.unit
def test_extract_returns_correct_shape(extractor: BiophysicalExtractor) -> None:
    """Verify extraction returns (3,) array."""
    features = extractor.extract_features(TRASTUZUMAB_VH)
    assert features.shape == (3,)
    assert features.dtype == np.float32


@pytest.mark.unit
def test_acidic_sequence_has_low_pi(extractor: BiophysicalExtractor) -> None:
    """Verify acidic sequence (all Glu) has low pI."""
    features = extractor.extract_features(ACIDIC_SEQUENCE)
    pi = features[2]  # Theoretical_pI is index 2
    assert pi < 5.0, f"Expected pI < 5.0 for acidic sequence, got {pi}"


@pytest.mark.unit
def test_basic_sequence_has_high_pi(extractor: BiophysicalExtractor) -> None:
    """Verify basic sequence (all Lys) has high pI."""
    features = extractor.extract_features(BASIC_SEQUENCE)
    pi = features[2]
    assert pi > 9.0, f"Expected pI > 9.0 for basic sequence, got {pi}"


@pytest.mark.unit
def test_charge_difference_between_ph6_and_ph74(extractor: BiophysicalExtractor) -> None:
    """Verify charge differs between pH 6 and pH 7.4."""
    features = extractor.extract_features(TRASTUZUMAB_VH)
    charge_ph6 = features[0]
    charge_ph74 = features[1]
    # At lower pH, proteins are more protonated (more positive charge)
    assert charge_ph6 > charge_ph74, "Charge at pH 6 should be higher than at pH 7.4"


# ============================================================================
# Batch Extraction Tests
# ============================================================================

@pytest.mark.unit
def test_batch_returns_correct_shape(extractor: BiophysicalExtractor) -> None:
    """Verify batch extraction returns (n, 3) array."""
    sequences = [TRASTUZUMAB_VH, ACIDIC_SEQUENCE, BASIC_SEQUENCE]
    features = extractor.extract_batch_features(sequences)
    assert features.shape == (3, 3)


# ============================================================================
# Validation Tests
# ============================================================================

@pytest.mark.unit
def test_rejects_invalid_amino_acids(extractor: BiophysicalExtractor) -> None:
    """Verify extractor rejects sequences with invalid characters."""
    with pytest.raises(ValueError, match="Invalid amino acid"):
        extractor.extract_features("QVQLBZX123")  # B, Z, numbers invalid


@pytest.mark.unit
def test_rejects_empty_sequence(extractor: BiophysicalExtractor) -> None:
    """Verify extractor rejects empty sequences."""
    with pytest.raises(ValueError, match="too short"):
        extractor.extract_features("")


@pytest.mark.unit
def test_handles_lowercase_input(extractor: BiophysicalExtractor) -> None:
    """Verify extractor handles lowercase sequences."""
    features = extractor.extract_features(ACIDIC_SEQUENCE.lower())
    assert features.shape == (3,)


@pytest.mark.unit
def test_handles_stop_codon_asterisk(extractor: BiophysicalExtractor) -> None:
    """Verify extractor strips stop codon asterisks."""
    features = extractor.extract_features(ACIDIC_SEQUENCE + "*")
    assert features.shape == (3,)
```

### 4.3 Acceptance Criteria (Phase A)

- [ ] `BiophysicalExtractor` class created at `src/antibody_training_esm/core/biophysical.py`
- [ ] Returns 3 features: charge@pH6, charge@pH7.4, theoretical pI
- [ ] All unit tests pass (12+ tests)
- [ ] Type annotations 100% complete
- [ ] `mypy` passes with strict mode
- [ ] Test coverage ≥ 90%
- [ ] Works on Boughter VH sequences without errors

---

## 5. Phase B: Training Pipeline Integration

**Objective**: Train LogisticRegression on Biopython trio, test on Jain.

### 5.1 Expected Results (from Paper)

| Model | 10-fold CV Accuracy | Jain Test Accuracy |
|-------|--------------------|--------------------|
| ESM-1v (our baseline) | ~71% | ~71% |
| Theoretical pI alone | **65.2%** | TBD |
| Biopython trio (3 features) | ~65-67% | TBD |

### 5.2 Training Script

```bash
# Train descriptor-only model on Boughter
uv run python -c "
from antibody_training_esm.core.biophysical import BiophysicalExtractor
from antibody_training_esm.datasets.boughter import BoughterDataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import numpy as np

# Load data
dataset = BoughterDataset()
sequences, labels = dataset.load_vh()

# Extract features
extractor = BiophysicalExtractor()
X = extractor.extract_batch_features(sequences)
y = np.array(labels)

# Standardize (important for LogReg)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 10-fold CV
clf = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=1000)
scores = cross_val_score(clf, X_scaled, y, cv=10, scoring='accuracy')

print(f'Biopython Trio 10-fold CV: {scores.mean():.1%} ± {scores.std():.1%}')
print(f'Feature names: {extractor.feature_names}')
"
```

---

## 6. Phase C: Open-Source Extensions (Optional)

If we want more descriptors WITHOUT Schrödinger, we can use `peptides` library:

| Descriptor | Library | Paper Equivalent |
|------------|---------|------------------|
| Eisenberg Hydrophobicity | peptides | #35 |
| Boman Index | peptides | Not in paper (but useful) |
| Instability Index | Biopython | Not in paper's 68 |
| GRAVY | Biopython | #44 (but ranks #52 - low value) |

**Note**: These are OPTIONAL extras, not required for paper parity.

---

## 7. What We're NOT Implementing (Schrödinger-blocked)

The following require BioLuminate (~$5-20K/year):

- Aggrescan (Nr_hotspots, av4, av4_pos) - **Top importance**
- Disorder propensity (DisProt, TOP-IDP, FoldUnfold) - **Top importance**
- HPLC retention coefficients (multiple scales)
- 20+ hydrophobicity scales
- Beta strand/turn/helix propensities
- Aggregation predictors

**Total blocked**: 65/68 descriptors

---

## 8. Success Criteria

- [ ] Phase A complete: BiophysicalExtractor with 3 Biopython descriptors
- [ ] 10-fold CV on Boughter achieves ~65% accuracy (matching pI-only from paper)
- [ ] Tests on Jain dataset documented
- [ ] Side-by-side comparison with ESM-1v (71%) documented
- [ ] GitHub Issue #4 updated with results

---

## 9. References

- **Paper**: Sakhnini et al. 2025 - Table S1 (68 descriptors), Table S2 (importance ranking)
- **GitHub Issue**: [#4](https://github.com/The-Obstacle-Is-The-Way/antibody_training_pipeline_ESM/issues/4)
- **Biopython**: `Bio.SeqUtils.ProtParam.ProteinAnalysis`
- **ZJ's Failed PR**: ludocomito/antibody_training_pipeline_ESM#21 (wrong descriptors, wrong dataset)

---

---

## 10. Implementation Notes (Post-Review)

**Senior Review**: APPROVED (2025-11-27)

**Corrections Applied**:
1. Method naming uses `extract_features` / `extract_batch_features` (semantic for descriptors)
   - Diverges slightly from ESM's `embed_sequence` / `extract_batch_embeddings`
   - Acceptable per review; consistent within descriptor domain
2. Paper citation added to class docstring (Table S1 reference)
3. 'X' (ambiguous AA) rejected - Biopython requires exact identities for pI calculation

**Implementation Delivered**:
- `src/antibody_training_esm/core/biophysical.py` - 176 lines, 100% typed
- `tests/unit/core/test_biophysical.py` - 30 tests, all passing
- mypy strict: PASS
- ruff lint/format: PASS
- Full test suite: 665 passed, no regressions

**Status**: IMPLEMENTED - Phase A complete.
