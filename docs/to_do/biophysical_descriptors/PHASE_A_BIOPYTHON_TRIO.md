# Phase A: Biopython Trio Implementation

**Date**: 2025-11-27
**Parent**: [BIOPHYSICAL_IMPLEMENTATION_SPECS.md](./BIOPHYSICAL_IMPLEMENTATION_SPECS.md)
**GitHub Issue**: [#4](https://github.com/The-Obstacle-Is-The-Way/antibody_training_pipeline_ESM/issues/4)

---

## Overview

Implement the **3 FREE Biopython descriptors** from Novo paper Table S1:

| Paper # | Descriptor | Method | Expected Impact |
|---------|------------|--------|-----------------|
| #21 | Charge at pH 6.0 | `charge_at_pH(6.0)` | Endosomal behavior |
| #22 | Charge at pH 7.4 | `charge_at_pH(7.4)` | Blood/plasma behavior |
| #66 | Theoretical pI | `isoelectric_point()` | **65.2% accuracy alone** |

---

## TDD Implementation Order

### Step 1: Write Tests First

```bash
# Create test file
touch tests/unit/core/test_biophysical.py

# Run tests (should fail - no implementation yet)
uv run pytest tests/unit/core/test_biophysical.py -v
```

### Step 2: Implement Minimum to Pass

```bash
# Create implementation
touch src/antibody_training_esm/core/biophysical.py

# Run tests again (should pass)
uv run pytest tests/unit/core/test_biophysical.py -v
```

### Step 3: Refactor & Type Check

```bash
# Type check
uv run mypy src/antibody_training_esm/core/biophysical.py --strict

# Lint
uv run ruff check src/antibody_training_esm/core/biophysical.py
```

---

## Scientific Validation

### Known pI Values for Validation

| Sequence | Expected pI | Source |
|----------|-------------|--------|
| All Glutamic Acid (E×10) | ~3.2 | Amino acid tables |
| All Lysine (K×10) | ~9.7 | Amino acid tables |
| Trastuzumab VH | ~8.5-9.0 | Literature |
| Human serum albumin | ~5.7 | Literature |

### Why pH 6 vs pH 7.4 Matters

```text
Blood (pH 7.4)          Endosome (pH 6.0)
     │                        │
     ▼                        ▼
┌─────────┐              ┌─────────┐
│ Antibody│──────────────│ Antibody│
│ binds   │  internalize │ releases│
│ target  │              │ or recycles
└─────────┘              └─────────┘
     │                        │
     ▼                        ▼
  Charge X                 Charge Y
  (less +)                 (more +)
```

The **charge difference** between pH 6 and 7.4 affects:
- FcRn binding (pH-dependent recycling)
- Aggregation propensity in different compartments
- Non-specific binding behavior

---

## File Locations

| File | Purpose |
|------|---------|
| `src/antibody_training_esm/core/biophysical.py` | Implementation |
| `tests/unit/core/test_biophysical.py` | Unit tests + test sequences (TRASTUZUMAB_VH, ACIDIC_SEQUENCE, etc.) |

---

## Acceptance Criteria

- [x] `BiophysicalExtractor` class exists
- [x] `extract_features(sequence)` returns (3,) numpy array
- [x] `extract_batch_features(sequences)` returns (n, 3) numpy array
- [x] Feature order: [charge_pH6, charge_pH7.4, pI]
- [x] All 30 unit tests pass
- [x] mypy strict passes
- [x] ruff passes
- [ ] Coverage ≥ 90% (currently 81.63%)

---

## Next Phase

After Phase A passes, proceed to:
1. Integration with `BinaryClassifier` (optional `feature_extractor` param)
2. Training on Boughter VH sequences
3. 10-fold CV to verify ~65% accuracy
4. Testing on Jain dataset

---

## Dependencies

```toml
# Already in pyproject.toml
biopython = ">=1.80"
```

No new dependencies required.
