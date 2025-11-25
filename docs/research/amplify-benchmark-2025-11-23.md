# AMPLIFY 350M Benchmark Results

**Date**: 2025-11-23
**Model**: chandar-lab/AMPLIFY_350M (960-d embeddings)
**Classifier**: Logistic Regression (C=1.0, penalty=l2)
**Training**: Boughter VH (914 sequences)
**Test**: Jain (86 sequences)
**Hardware**: M1 Pro (MPS) + CPU validation

---

## 1. Reproducibility Validation

### 1.1 CPU vs MPS Embeddings

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Mean Absolute Error | TBD | < 1e-4 | TBD |
| Max Absolute Difference | TBD | - | - |
| Mean Squared Error | TBD | - | - |

**Conclusion**: [FILL IN: MPS embeddings are/are not reproducible compared to CPU gold standard]

**Source**: [Nature Scientific Reports (2025)](https://www.nature.com/articles/s41598-025-05674-x)

---

## 2. Performance Comparison

### 2.1 Jain Dataset Benchmark

| Model | Accuracy | AUC | Precision | Recall | F1 | Params | Inference Time |
|-------|----------|-----|-----------|--------|----|----|----------------|
| **ESM-1v 650M** | **71.0%** | 0.79 | - | - | - | 650M | ~45s |
| **AMPLIFY 350M** | TBD% | TBD | TBD | TBD | TBD | 350M | ~TBDs |

**Delta from Baseline**: [+/-X.X%]

### 2.2 Cross-Validation Results (Boughter)

| Model | Mean CV Accuracy | Std Dev |
|-------|------------------|---------|
| ESM-1v 650M | ~0.XX ± 0.XX | - |
| AMPLIFY 350M | TBD ± TBD | - |

---

## 3. Analysis

### 3.1 Performance Analysis

[FILL IN AFTER EXPERIMENTS:]
- Did AMPLIFY beat ESM-1v (>71%)?
- If yes: Why? (OAS training, data quality)
- If no: Why? (evolutionary training superiority, padding bug impact)

### 3.2 Speed Analysis

**Expected Speed** (based on parameter count):
- AMPLIFY 350M should be ~1.86× faster than ESM-1v 650M (350M / 650M ≈ 0.54)

**Actual Speed**:
- ESM-1v: ~45 seconds for 914 sequences (batch_size=8)
- AMPLIFY: ~XXX seconds for 914 sequences (batch_size=1, forced)

**Conclusion**: [FILL IN: AMPLIFY is slower/faster than expected due to batch_size=1 constraint]

### 3.3 Embedding Quality

**Dimension**: 960d (vs ESM-1v's 1280d)
- Fewer dimensions = faster downstream classifier training
- Risk: Less information encoded?

---

## 4. Scientific Findings

### 4.1 Hypothesis Testing

**Hypothesis**: "OAS training + data quality beats evolutionary variant training"

**Result**: [ACCEPT/REJECT]

**Evidence**:
- AMPLIFY Jain accuracy: XX.X%
- ESM-1v Jain accuracy: 71.0%
- Delta: +/-X.X%

### 4.2 Padding Bug Impact

**Mitigation**: Enforced batch_size=1 throughout pipeline
**Validation**: CPU vs MPS MAE = X.XXe-X (< 1e-4 threshold)
**Conclusion**: Padding bug successfully mitigated

---

## 5. Recommendations

### 5.1 Production Use

- [ ] **Use AMPLIFY** if accuracy ≥ ESM-1v and reproducibility validated
- [ ] **Use ESM-1v** if AMPLIFY < 71% or reproducibility issues
- [ ] **Use CPU** for AMPLIFY if MAE > 1e-6 on MPS

### 5.2 Research Use

- [ ] AMPLIFY is valuable for model zoo regardless of performance
- [ ] Validates "data quality > scale" hypothesis
- [ ] Tests OAS-specific training effectiveness

---

## 6. References

- **AMPLIFY Paper**: [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.09.23.614603v1)
- **Padding Bug**: [Nature Sci Rep](https://www.nature.com/articles/s41598-025-05674-x)
- **ESM-1v Baseline**: Sakhnini et al. (2025) - 71% Jain accuracy
- **HuggingFace Model**: [chandar-lab/AMPLIFY_350M](https://huggingface.co/chandar-lab/AMPLIFY_350M)

---

## 7. Appendix: Reproducibility Details

### 7.1 Training Command
```bash
uv run antibody-train model=amplify_350m hardware.device=mps
```

### 7.2 Test Command
```bash
uv run antibody-test --model experiments/checkpoints/amplify_350m/logreg/amplify_mps.pkl --dataset jain
```

### 7.3 Environment
- Python: 3.11+
- PyTorch: 2.x with MPS support
- Transformers: 4.39.0+
- Device: Apple M1 Pro (MPS)
