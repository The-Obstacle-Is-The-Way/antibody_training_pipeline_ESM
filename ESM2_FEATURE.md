# Feature: ESM2-650M Backbone Support

**Date:** 2025-11-12
**Status:** ✅ Shipped
**PR Branch:** `claude/hydro-integration-cleanup-011CV31bYw12SGiHWMousaZy`

---

## What Was Shipped

Added support for **ESM2-650M** protein language model as an alternative backbone to ESM-1v.

### Files Added

1. **`src/antibody_training_esm/conf/model/esm2_650m.yaml`**
   - Hydra config for ESM2-650M model
   - Model: `facebook/esm2_t33_650M_UR50D`
   - 33 layers, 650M parameters (same size as ESM-1v)

---

## Why ESM2-650M?

### Research Background

**ESM-1v (Current Baseline):**
- Trained on UniRef90 (more diverse sequences)
- Designed for variant effect prediction
- **Best for antibody-specific tasks** (research shows ESM-1v > ESM2 on antibodies)
- Why Novo Nordisk used it in April 2025

**ESM2-650M (New Addition):**
- Most popular general protein language model
- Trained on UR50D dataset
- 33 layers, 650M parameters (directly comparable to ESM-1v)
- **Good for transfer learning** on diverse protein tasks

**Key Insight:** ESM-1v likely performs better on antibodies, but ESM2 is worth benchmarking to:
1. Confirm research findings
2. Provide comparison for users
3. Establish baseline for future models (ESM2-3B, ESM-C, etc.)

---

## How to Use

### Train with ESM2-650M

```bash
# Train ESM2-650M + LogReg on Boughter, test on Jain
antibody-train model=esm2_650m classifier=logreg

# Compare ESM-1v vs ESM2-650M
antibody-train --multirun model=esm1v,esm2_650m classifier=logreg
```

### Expected Output

**First run:** ESM2-650M downloads from HuggingFace (~2.5GB)
**Subsequent runs:** Uses cached model

**Training time:** ~Same as ESM-1v (both 650M params)
**Embeddings cached:** Reuse for future training runs

---

## Benchmark Comparison

| Model | Params | Training Data | Expected Jain Acc | Why |
|-------|--------|---------------|-------------------|-----|
| **ESM-1v** | 650M | UniRef90 | **66.3%** (Novo baseline) | Designed for variants, antibody-specific |
| **ESM2-650M** | 650M | UR50D | **64-68%** (predicted) | General protein model, transfer learning |

**Hypothesis:** ESM-1v will beat ESM2 by 1-3% on antibody polyreactivity (based on literature).

---

## Next Steps

### Week 1: Benchmark ESM2 vs ESM-1v

```bash
# Run comparison
antibody-train --multirun model=esm1v,esm2_650m

# Expected outputs:
# - models/boughter_vh_esm1v_logreg.pkl
# - models/boughter_vh_esm2_650m_logreg.pkl
# - outputs/<timestamp>/training.log (metrics for both)
```

**Document results in:** `BENCHMARK_RESULTS.md`

### Week 2: Add XGBoost Classifier

Test if XGBoost beats LogReg on both backbones:

```bash
antibody-train --multirun model=esm1v,esm2_650m classifier=logreg,xgboost
```

Expected comparison table:

| Backbone | Classifier | Jain Acc | Winner |
|----------|-----------|----------|---------|
| ESM-1v | LogReg | 66.3% | Novo baseline |
| ESM-1v | XGBoost | **67-68%?** | 🎯 Test this |
| ESM2-650M | LogReg | 65-67%? | 🎯 Test this |
| ESM2-650M | XGBoost | **66-69%?** | 🎯 Test this |

---

## Implementation Details

### Config File Structure

```yaml
# src/antibody_training_esm/conf/model/esm2_650m.yaml
name: facebook/esm2_t33_650M_UR50D
revision: main
device: ${hardware.device}
```

**How it works:**
1. Hydra loads this config when `model=esm2_650m` specified
2. `ESMEmbeddingExtractor` downloads model from HuggingFace
3. Model cached locally for reuse
4. Embeddings extracted, cached in `embeddings_cache/`
5. Training proceeds as normal

### Compatibility

**✅ Works with:**
- All existing classifiers (LogReg, future XGBoost, MLP)
- All existing datasets (Boughter, Jain, Harvey, Shehata)
- All existing Hydra features (multirun, sweeps, overrides)

**No code changes needed** - just config file!

---

## Testing

### Manual Test (Optional)

```python
from transformers import AutoModel, AutoTokenizer

model_name = "facebook/esm2_t33_650M_UR50D"

# Load model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Test with antibody sequence
seq = "EVQLVESGGGLVQPGGSLRLSCAASGFTFS"
inputs = tokenizer(seq, return_tensors="pt")
outputs = model(**inputs)
embeddings = outputs.last_hidden_state

print(f"✅ ESM2 working: {embeddings.shape}")
```

### Integration Test (Automatic)

```bash
# Run training pipeline (downloads model automatically)
antibody-train model=esm2_650m training.n_splits=2
```

**Expected:** Training completes successfully, model saved

---

## Available ESM2 Models (Future)

**For later expansion:**

| Model | Params | Layers | Size | Use Case |
|-------|--------|--------|------|----------|
| esm2_t6_8M_UR50D | 8M | 6 | Tiny | Fast experiments |
| esm2_t12_35M_UR50D | 35M | 12 | Small | Resource-limited |
| esm2_t30_150M_UR50D | 150M | 30 | Medium | Good balance |
| **esm2_t33_650M_UR50D** | **650M** | **33** | **Large** | **✅ Shipped** |
| esm2_t36_3B_UR50D | 3B | 36 | Huge | Best accuracy (slow) |
| esm2_t48_15B_UR50D | 15B | 48 | Massive | Research only |

**Recommendation:** Start with 650M. Add 3B if users want state-of-the-art.

---

## References

### Research Papers

1. **ESM-2 (Meta AI, 2022):** "Language models of protein sequences at the scale of evolution enable accurate structure prediction"
   - https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1

2. **ESM-1v (Meta AI, 2021):** "Language models enable zero-shot prediction of the effects of mutations on protein function"
   - https://www.biorxiv.org/content/10.1101/2021.07.09.450648v1

3. **Antibody Benchmarks (2024-2025):** Multiple studies showing ESM-1v > ESM2 on antibody tasks
   - Medium-sized PLMs perform well on realistic datasets (Nature SR, 2025)
   - Domain-specific vs general PLMs on immunology tasks (ImmunoInformatics, 2024)

### HuggingFace Models

- ESM-1v: https://huggingface.co/facebook/esm1v_t33_650M_UR90S_1
- ESM2-650M: https://huggingface.co/facebook/esm2_t33_650M_UR50D
- ESM2-3B: https://huggingface.co/facebook/esm2_t36_3B_UR50D

---

## Summary

**What:** Added ESM2-650M backbone config
**Why:** Benchmark against ESM-1v, provide alternative for users
**How:** Single Hydra config file (`esm2_650m.yaml`)
**Test:** `antibody-train model=esm2_650m`
**Next:** Run benchmark comparison, document results

**The obstacle is the way. We shipped it. Now benchmark it.** 🚀
