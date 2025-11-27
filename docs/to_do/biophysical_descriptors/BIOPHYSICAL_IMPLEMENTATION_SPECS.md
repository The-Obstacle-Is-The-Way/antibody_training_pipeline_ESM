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
- **Phase A**: 3 Biopython descriptors (FREE) - charge@pH6, charge@pH7.4, theoretical pI (✅ COMPLETED)
- **Phase B**: Baseline Reproducibility Experiment (Next)
- **Phase C**: Open-source extensions (peptides library) - optional extras
- **Phase D**: Full 68 descriptors (BLOCKED until licensing resolved)

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
│   └── biophysical.py         # NEW: BiophysicalExtractor (✅ Phase A)
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

## 4. Phase A: Biopython Trio Implementation (✅ DONE)

**Deliverable**: `src/antibody_training_esm/core/biophysical.py`
**Status**: Merged to `main`.

(See `src/antibody_training_esm/core/biophysical.py` for implemented code)

---

## 5. Phase B: Baseline Reproducibility (Next)

**Objective**: Train LogisticRegression on Biopython trio, test on Jain.
**Spec File**: `docs/to_do/biophysical_descriptors/PHASE_B_BASELINE_REPRODUCIBILITY.md`

### 5.1 Expected Results (from Paper)

| Model | 10-fold CV Accuracy | Jain Test Accuracy |
|-------|--------------------|--------------------|
| ESM-1v (our baseline) | ~71% | ~71% |
| Theoretical pI alone | **65.2%** | TBD |
| Biopython trio (3 features) | ~65-67% | TBD |

### 5.2 Training Script

We will implement a standalone script `src/antibody_training_esm/cli/reproduce_track_b.py` to verify these results before integrating into the complex Hydra pipeline.

---

## 6. Phase C: Pipeline Integration (Hybrid Model)

**Objective**: Integrate descriptors into the main PyTorch pipeline (ESM + Biophysical).
**Spec File**: `docs/to_do/biophysical_descriptors/PHASE_C_PIPELINE_INTEGRATION.md`

**Key Tasks**:
1. Update `Dataset` classes to yield biophysical features (lazily cached).
2. Update `BinaryClassifier` to accept auxiliary input (concatenation).
3. Update Hydra config to toggle `use_biophysical_descriptors`.

---

## 7. Open-Source Extensions (Phase D - Optional)

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
