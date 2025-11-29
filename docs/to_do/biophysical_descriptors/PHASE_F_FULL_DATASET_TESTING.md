# Phase F: Full Dataset Testing for Track B

**Date**: 2025-11-28
**Status**: PENDING
**Depends On**: Phase E (Hydra Integration)
**Blocked By**: Phase E must be completed first

---

## 1. Objective

Test Track B (biophysical descriptors) on ALL test datasets, not just Jain.
Document results in the same format as Track A for fair comparison.

---

## 2. Background

### Current State (Incomplete)
The standalone `reproduce_track_b.py` only tests on Jain dataset:
- Train: Boughter (914 VH sequences)
- Test: Jain only (86 antibodies)
- Missing: Harvey, Shehata

### Desired State
Track B tested on all three test datasets:
- Jain (ELISA assay, threshold 0.5)
- Harvey (PSR assay, threshold 0.5495)
- Shehata (PSR assay, threshold 0.5495)

---

## 3. Test Datasets

| Dataset | Assay Type | Threshold | Sequences | Notes |
|---------|------------|-----------|-----------|-------|
| Jain | ELISA | 0.5 | 86 | Novo parity benchmark |
| Harvey | PSR | 0.5495 | ~141k | Nanobodies (VHH) |
| Shehata | PSR | 0.5495 | 398 | Human antibodies |

---

## 4. Commands (After Phase E)

```bash
# Train on Boughter, test on Jain
uv run antibody-train model=biophysical data=boughter_jain

# Train on Boughter, test on Harvey
uv run antibody-train model=biophysical \
    data.test_file=data/test/harvey/fragments/VHH_only.csv

# Train on Boughter, test on Shehata
uv run antibody-train model=biophysical \
    data.test_file=data/test/shehata/fragments/VH_VL.csv
```

---

## 5. Expected Results

### 5.1 Track B (Biophysical) Predictions

Based on the Novo Nordisk paper and our Phase B results:

| Dataset | Metric | Expected Value | Notes |
|---------|--------|----------------|-------|
| **Boughter** | 10-fold CV Accuracy | ~63% | From Phase B baseline |
| **Jain** | Test Accuracy | ~56% | From Phase B baseline |
| **Jain** | ROC-AUC | ~0.67 | From Phase B baseline |
| **Harvey** | Test Accuracy | TBD | First measurement |
| **Harvey** | ROC-AUC | TBD | First measurement |
| **Shehata** | Test Accuracy | TBD | First measurement |
| **Shehata** | ROC-AUC | TBD | First measurement |

### 5.2 Comparison with Track A (ESM)

For reference, Track A (ESM-1v) results from Novo parity:

| Dataset | Track A (ESM) | Track B (Biophysical) | Delta |
|---------|---------------|----------------------|-------|
| Jain | ~71% | ~56% | -15% |
| Harvey | TBD | TBD | TBD |
| Shehata | TBD | TBD | TBD |

---

## 6. Output Structure

After Phase E + F, Track B outputs should mirror Track A:

```
experiments/
├── checkpoints/
│   ├── esm1v/                    # Track A
│   │   └── logreg/
│   │       └── boughter_vh_esm1v_logreg.pkl
│   └── biophysical/              # Track B (NEW)
│       └── logreg/
│           └── boughter_vh_biophysical_logreg.pkl
└── runs/
    ├── esm_training/             # Track A runs
    └── biophysical_training/     # Track B runs (NEW)
        └── {timestamp}/
            ├── .hydra/
            │   ├── config.yaml
            │   └── overrides.yaml
            └── train.log
```

---

## 7. Acceptance Criteria

### 7.1 Jain Testing
- [ ] Track B test accuracy on Jain documented
- [ ] Results match Phase B baseline (~56%)
- [ ] Model checkpoint saved correctly

### 7.2 Harvey Testing
- [ ] Track B test accuracy on Harvey documented
- [ ] PSR threshold (0.5495) applied correctly
- [ ] Handle VHH-only sequences (no VL)

### 7.3 Shehata Testing
- [ ] Track B test accuracy on Shehata documented
- [ ] PSR threshold (0.5495) applied correctly
- [ ] Results comparable to Track A delta

### 7.4 Documentation
- [ ] Benchmark results table in `docs/research/benchmark-results.md`
- [ ] Track A vs Track B comparison documented
- [ ] Conclusions about biophysical feature utility

---

## 8. Data Considerations

### 8.1 Harvey Dataset (Nanobodies)
- Contains VHH sequences only (no VL chain)
- Much larger than other datasets (~141k sequences)
- May need subsampling for practical testing
- PSR assay with 0.5495 threshold

### 8.2 Shehata Dataset
- Human antibodies with VH+VL
- PSR assay with 0.5495 threshold
- Moderate size (398 sequences)

### 8.3 Sequence Filtering
BiophysicalExtractor requires valid amino acid sequences:
- No 'X' (ambiguous residues)
- No '*' (stop codons) - already filtered by dataset loaders
- Length > 0

---

## 9. Implementation Steps

1. **Verify Phase E Complete**: Ensure `antibody-train model=biophysical` works
2. **Test on Jain**: Validate results match Phase B baseline
3. **Test on Harvey**: Document first biophysical results on nanobodies
4. **Test on Shehata**: Document first biophysical results on PSR assay
5. **Update Documentation**: Add results to benchmark tables
6. **Create Comparison Table**: Track A vs Track B across all datasets

---

## 10. Success Metrics

Phase F is successful when:
1. Track B runs on all three test datasets without errors
2. Results are documented in consistent format
3. Comparison with Track A is clear
4. Insights about biophysical feature utility are documented

---

## 11. Future Work

After Phase F, potential next steps:
- Feature importance analysis (which biophysical descriptor matters most?)
- Cross-validation on Harvey/Shehata (not just Jain)
- Investigate why Track B underperforms Track A
- Consider adding more Biopython-compatible descriptors
