# Pending Model Validation Roadmap

**Status**: Planning
**Created**: 2025-11-17
**Purpose**: Document validation roadmap for expanding model zoo beyond ESM1v/LogReg baseline

---

## ✅ Currently Validated (ESM1v + Logistic Regression)

The following pipeline has been **fully validated end-to-end**:

- **Model Backbone**: ESM1v (`facebook/esm1v_t33_650M_UR90S_1`)
- **Classifier Head**: Logistic Regression
- **Config**: `src/antibody_training_esm/conf/config.yaml`
- **Validation Scope**:
  - ✅ Training pipeline (10-fold CV)
  - ✅ Hyperparameter sweeps (Hydra multirun)
  - ✅ Testing on 3 datasets (Jain, Harvey, Shehata)
  - ✅ Fresh clone reproducibility
  - ✅ Directory routing (hierarchical outputs)
  - ✅ Checkpoint metadata (model_name + classifier block)
  - ✅ Unit tests (353 passed, 81.93% coverage)

**Evidence**: See `VALIDATION_ROADMAP.md` and `VALIDATION_REALITY.md`

---

## 🎯 Motivation: Why Multi-Model Validation Matters

Hydra's **entire value proposition** is enabling reproducible multi-model experimentation. Our current validation only covers **one model configuration** (ESM1v + LogReg). To build a robust **model zoo**, we need to validate:

1. **Alternative backbones** (ESM2, other PLMs)
2. **Alternative classifier heads** (SVM, MLP, ensemble methods)
3. **Cross-product combinations** (ESM1v + SVM, ESM2 + MLP, etc.)

This ensures that:
- Hydra configs work for all model types
- Directory routing generalizes beyond `esm1v/logreg/`
- Checkpoint metadata supports diverse architectures
- Training/testing pipelines handle different model APIs

---

## 📋 Pending Validation: Model Zoo Expansion

### Phase 1: ESM2 Backbone (Same Classifier)

**Model**: ESM2-650M (`facebook/esm2_t33_650M_UR50D`)
**Classifier**: Logistic Regression
**Config**: Modify `model.esm_model` in Hydra

**Validation Tasks**:
- [ ] Create `configs/model/esm2_650m.yaml` override
- [ ] Train ESM2 + LogReg on Boughter
- [ ] Verify checkpoint saves to `experiments/checkpoints/esm2_650m/logreg/`
- [ ] Test on Jain/Harvey/Shehata
- [ ] Verify directory routing creates `esm2_650m/logreg/` (not `unknown/`)
- [ ] Compare performance to ESM1v baseline

**Expected Outputs**:
```
experiments/
├── checkpoints/esm2_650m/logreg/
│   └── boughter_vh_esm2_650m_logreg.pkl
├── benchmarks/esm2_650m/logreg/
│   ├── VH_only_jain_86_p5e_s2/
│   ├── VH_only_shehata/
│   └── VHH_only_harvey/
```

**Time Estimate**: ~2-3 hours (training + testing on 3 datasets)

---

### Phase 2: Alternative Classifier Heads (Same Backbone)

**Model**: ESM1v (validated backbone)
**Classifiers**: SVM, MLP, Random Forest

**Validation Tasks**:
- [ ] Create `configs/classifier/svm.yaml`
- [ ] Create `configs/classifier/mlp.yaml`
- [ ] Create `configs/classifier/random_forest.yaml`
- [ ] Implement classifier classes in `core/classifier.py`
- [ ] Train each classifier on Boughter (ESM1v embeddings)
- [ ] Verify checkpoints save to `experiments/checkpoints/esm1v/{svm,mlp,rf}/`
- [ ] Test on all 3 datasets
- [ ] Compare performance vs LogReg baseline

**Expected Outputs**:
```
experiments/
├── checkpoints/esm1v/
│   ├── logreg/ ✅ (validated)
│   ├── svm/
│   ├── mlp/
│   └── random_forest/
├── benchmarks/esm1v/
│   ├── logreg/ ✅
│   ├── svm/
│   ├── mlp/
│   └── random_forest/
```

**Time Estimate**: ~1 day (implementation + validation for 3 classifiers)

---

### Phase 3: Cross-Product Validation (Model Zoo Matrix)

**Goal**: Validate **all combinations** of backbones × classifiers

**Matrix**:
| Backbone | Classifier | Status |
|----------|------------|--------|
| ESM1v    | LogReg     | ✅ Validated |
| ESM1v    | SVM        | ⏸️ Pending |
| ESM1v    | MLP        | ⏸️ Pending |
| ESM2-650M | LogReg    | ⏸️ Pending |
| ESM2-650M | SVM       | ⏸️ Pending |
| ESM2-650M | MLP       | ⏸️ Pending |

**Validation Tasks**:
- [ ] Hydra sweep across all combinations
- [ ] Verify directory routing: `{model}/{classifier}/`
- [ ] Verify checkpoint metadata distinguishes models
- [ ] Compare performance across model zoo
- [ ] Document best-performing combinations

**Command Example**:
```bash
uv run antibody-train --multirun \
  model=esm1v,esm2_650m \
  classifier=logreg,svm,mlp \
  training.model_name=model_zoo_sweep
```

**Expected**: 6 models trained, 18 test runs (6 models × 3 datasets)

**Time Estimate**: ~1 day (parallelizable with compute resources)

---

### Phase 4: Advanced Model Architectures

**Future Expansion** (post-baseline model zoo):

- [ ] **Ensemble methods** (stacking, bagging)
- [ ] **Fine-tuned ESM** (vs frozen embeddings)
- [ ] **Attention-based classifiers** (self-attention on embeddings)
- [ ] **Cross-architecture ensembles** (ESM1v + ESM2 fusion)
- [ ] **Domain-specific PLMs** (antibody-specific models like AbLang)

---

## 🛠️ Implementation Checklist

### Prerequisites
- [x] Hydra config system validated (ESM1v/LogReg)
- [x] Directory routing working (`directory_utils.py` handles new `{model}/{classifier}` paths)
- [x] Checkpoint metadata system (`model_name` + `classifier`)
- [x] Testing pipeline (`antibody-test`)
- [x] Fresh clone validation protocol

### New Components Needed
- [ ] `configs/model/esm2_650m.yaml`
- [ ] `configs/classifier/svm.yaml`
- [ ] `configs/classifier/mlp.yaml`
- [ ] `configs/classifier/random_forest.yaml`
- [ ] Classifier implementations in `core/classifier.py`
- [ ] Model zoo comparison utilities (plotting, tables)

### Validation Protocol (Per Model)
1. **Train** with `antibody-train`
2. **Verify** checkpoint in `experiments/checkpoints/{model}/{classifier}/`
3. **Verify** metadata has correct `model_name` + `classifier.type`
4. **Test** on Jain, Harvey, Shehata
5. **Verify** outputs in `experiments/benchmarks/{model}/{classifier}/`
6. **Compare** performance to baseline
7. **Document** in model zoo README

---

## 📊 Success Criteria

A model is considered **validated** when:
- ✅ Trains without errors
- ✅ Saves checkpoint to correct hierarchical directory
- ✅ Checkpoint JSON has correct metadata
- ✅ Tests successfully on all 3 datasets
- ✅ Outputs route to correct benchmark directories
- ✅ Performance is documented and compared to baseline
- ✅ Fresh clone can reproduce results

---

## 📝 Notes

- **Priority Order**: ESM2 → alternative classifiers → cross-product
- **Baseline Comparison**: Always compare to ESM1v/LogReg (66.28% Jain accuracy)
- **Documentation**: Update `docs/research/model-zoo.md` as we expand
- **Hydra Configs**: Keep configs modular and composable
- **Directory Routing**: Ensure `directory_utils.py` handles new {model}/{classifier} paths
- **CI/CD**: Eventually automate model zoo validation in GitHub Actions

---

## 🎯 Why This Matters

**For researchers**: Enables systematic comparison of model architectures
**For practitioners**: Provides validated, reproducible model zoo
**For Hydra**: Proves multi-model config system works end-to-end
**For science**: Documents performance across diverse architectures

**This is the whole point of Hydra** - enabling reproducible, configurable experiments across model families. Let's build it right! 🚀

---

## References

- **Current Validation**: `VALIDATION_ROADMAP.md`, `VALIDATION_REALITY.md`
- **Hydra Docs**: https://hydra.cc/docs/intro/
- **ESM Models**: https://github.com/facebookresearch/esm
- **Model Zoo Examples**: HuggingFace Model Hub, TorchHub
