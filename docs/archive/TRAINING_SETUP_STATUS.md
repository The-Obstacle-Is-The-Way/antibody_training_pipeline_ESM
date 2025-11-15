# Boughter Training Setup - Ready to Launch

> **⚠️ LEGACY DOCUMENTATION (v1.x)**
>
> This document references the old root file structure (train.py, test.py, etc.) which was removed in v2.0.0.
>
> **For v2.0.0+, use:**
> - `antibody-train --config configs/config.yaml` instead of `python train.py`
> - `antibody-test --model X --data Y` instead of `python test.py`
> - `from antibody_training_esm.core.classifier import BinaryClassifier` instead of `from classifier import BinaryClassifier`
>
> See [IMPORT_AND_STRUCTURE_GUIDE.md](../../IMPORT_AND_STRUCTURE_GUIDE.md) for v2.0.0 usage.

**Date**: 2025-11-02
**Status**: ✅ **READY TO TRAIN** (updated for v2.0.0)

---

## Summary

The existing OSS repo is **fully wired and working**. We fixed one bug and created a Boughter-specific config. Everything is ready to launch training.

---

## What's Working

### 1. Model Architecture (from OSS repo)
- ✅ **ESM-1v pretrained model**: `facebook/esm1v_t33_650M_UR90S_1` (652M params)
  - Already cached on your machine (~2.5GB)
  - **Stays frozen** - we never train it
  - Used only to extract embeddings

- ✅ **Logistic Regression Classifier**: Simple head we train
  - Trains on top of ESM-1v embeddings
  - 10-fold cross-validation
  - Takes ~10-30 minutes

### 2. Training Pipeline (professional package structure)
- ✅ `src/antibody_training_esm/cli/train.py` - Training CLI
- ✅ `src/antibody_training_esm/data/loaders.py` - Data loading
- ✅ `src/antibody_training_esm/core/embeddings.py` - ESM-1v wrapper
- ✅ `src/antibody_training_esm/core/classifier.py` - Logistic regression wrapper
- ✅ Complete evaluation with CV metrics

### 3. Boughter Dataset (our processing)
- ✅ **Training**: 914 sequences (443 specific + 471 non-specific)
- ✅ **Test (Jain)**: 70 sequences (67 specific + 3 non-specific)
- ✅ All P0 blockers fixed (gap-free, ESM-compatible)
- ✅ Filtered training file created: `data/train/boughter/canonical/VH_only_boughter_training.csv`

### 4. Configuration
- ✅ `config_boughter.yaml` - Boughter-specific config
- ✅ Points to correct training/test files
- ✅ Uses MPS device (Apple Silicon GPU)
- ✅ Matches Novo methodology (10-fold CV, balanced classes)

---

## Bug Fixed

**Original Issue**: `data.py` line 142 used `data_config['file_path']` but `config.yaml` uses `train_file`

**Fix**: Changed to `data_config['train_file']`

**Files Modified**:
1. `data.py` - Fixed config key
2. `config_boughter.yaml` - Created Boughter config
3. `data/train/boughter/canonical/VH_only_boughter_training.csv` - Created filtered training file

---

## Launch Training (v2.0.0)

### Option 1: Direct Launch
```bash
antibody-train --config configs/config_boughter.yaml
```

### Option 2: Tmux (Recommended)
```bash
tmux new -s boughter_training
antibody-train --config configs/config_boughter.yaml 2>&1 | tee logs/boughter_$(date +%Y%m%d_%H%M%S).log
# Detach: Ctrl+B, then D
```

### Option 3: Background
```bash
nohup antibody-train --config configs/config_boughter.yaml > logs/boughter_training.log 2>&1 &
```

---

## Expected Results

### Training (10-fold CV on Boughter)
- **Novo**: 71% accuracy
- **Hybri**: 68.2% accuracy
- **Our target**: ~68-71%

### Testing (on Jain)
- **Novo**: 69% accuracy
- **Hybri**: 65.1% accuracy
- **Our target**: ~65-69%

**Note**: Slightly lower than Novo is expected due to:
- Random seed differences
- Minor preprocessing variations
- 70 test sequences vs Novo's 86 (different threshold)

---

## What Happens During Training

1. **Load ESM-1v** (~5-10 sec): Download/load pretrained model
2. **Extract embeddings** (~5-15 min): Convert 914 sequences to 1280-dim vectors
   - Cached to `embeddings_cache/` for future runs
3. **10-fold CV** (~2-5 min): Train logistic regression 10 times
4. **Final training** (~30 sec): Train on full dataset
5. **Save model** (~1 sec): Save to `models/boughter_vh_esm1v_logreg.pkl`

**Total time**: ~10-30 minutes depending on hardware

---

## Files Generated

After training:
```
models/boughter_vh_esm1v_logreg.pkl       # Trained classifier
embeddings_cache/train_<hash>_embeddings.pkl  # Cached embeddings
logs/boughter_training.log                # Training logs
```

---

## Verification

Ran end-to-end config test:
```python
✅ Config loaded successfully
✅ Training data loaded: 914 sequences (443 specific + 471 non-specific)
✅ Test data loaded: 70 test sequences (67 specific + 3 non-specific)
✅ ESM-1v model loaded: 652.4M parameters
```

---

## Next Steps

1. **Launch training** using one of the methods above
2. **Monitor progress** via logs
3. **Compare results** to Hybri (68.2%) and Novo (71%)
4. **Optional**: Update Jain threshold to >=3 flags for 86 test sequences (matches Novo better)

---

**Status**: Ready to launch - OSS repo works perfectly with Boughter data ✅
