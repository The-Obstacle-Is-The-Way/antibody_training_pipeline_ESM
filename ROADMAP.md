# Roadmap: Antibody ML Model Zoo

**Last Updated:** 2025-11-12
**Current Version:** v0.4.0
**Status:** Active Development

---

## Vision

Build the definitive model zoo for antibody polyreactivity prediction—comprehensive benchmarks, pretrained models, and easy-to-use inference tools for the antibody ML community.

---

## What We've Built (v0.4.0)

### Research Foundation

**Novo Nordisk Benchmark Reproduced ✅**
- Trained ESM-1v + Logistic Regression on Boughter dataset (914 VH sequences)
- Achieved **66.28% accuracy** on Jain test set (86 clinical antibodies)
- Exact parity with published Novo Nordisk results
- Published methodology: `docs/research/novo-parity.md`

**ESM2-650M Benchmarking Complete ✅**
- Trained ESM2-650M + Logistic Regression on Boughter
- Benchmarked on 3 test datasets:
  - **Jain** (86 sequences): 62.79% accuracy
  - **Shehata** (398 sequences): 62.05% accuracy
  - **Harvey** (141k nanobodies): 66.25% accuracy
- **Finding:** ESM-1v outperforms ESM2 on clinical antibodies, mixed results on nanobodies

**4 Preprocessed Datasets Ready ✅**
- **Boughter** (training): 914 VH sequences, ELISA polyreactivity assay
- **Jain** (test): 86 clinical mAbs, Novo Nordisk parity benchmark
- **Harvey** (test): 141,021 nanobodies, PSR assay
- **Shehata** (test): 398 antibodies, PSR assay cross-validation

All datasets in `train_datasets/` and `test_datasets/` with canonical formats.

### Production Infrastructure

**Hydra Configuration System ✅**
- Structured configs with composition: `model`, `data`, `classifier`, `training`, `hardware`
- CLI overrides for rapid iteration:
  ```bash
  antibody-train model.batch_size=16 classifier.C=0.5
  ```
- Multirun sweeps for hyperparameter search:
  ```bash
  antibody-train --multirun classifier.C=0.1,1.0,10.0
  ```
- Automatic experiment tracking and provenance

**Dual Model Serialization ✅**
- **NPZ + JSON** (production-ready, pickle-free)
- **Pickle** (research/debugging with full Python objects)
- HuggingFace-compatible format ready for Hub publishing

**Professional Development Workflow ✅**
- 100% type safety (mypy strict mode enforced in CI)
- CI/CD with GitHub Actions (quality, tests, security)
- Docker images (dev + prod environments)
- 90%+ test coverage with unit, integration, and e2e tests

**Commands Available:**
```bash
# Training
antibody-train                          # Default: ESM-1v + LogReg on Boughter
antibody-train model=esm2_650m          # Override model
antibody-train --multirun model=esm1v,esm2_650m  # Compare backbones

# Testing
antibody-test --model models/model.pkl --data test_datasets/jain/canonical/jain.csv
```

### Current Models

**Trained and benchmarked:**
- `models/boughter_vh_esm1v_logreg.{pkl,npz,json}` (Novo baseline)
- `models/boughter_vh_esm2_650m_logreg.{pkl,npz,json}` (ESM2 comparison)

**Test results organized hierarchically:**
```
test_results/
├── esm1v/logreg/
│   ├── harvey/
│   ├── jain/
│   └── shehata/
└── esm2_650m/logreg/
    ├── VHH_only_harvey/
    ├── VH_only_jain_test_PARITY_86/
    └── VH_only_shehata/
```

---

## What's Next (Priority Order)

### Phase 1: Complete Model Zoo (4-6 weeks)

**Goal:** Train 9 models total (3 backbones × 3 classifiers)

#### Week 1-2: XGBoost Classifier

**Why:** XGBoost typically outperforms Logistic Regression on tabulated embeddings. Expected +1-3% accuracy improvement.

**Tasks:**
1. Add `xgboost` to `pyproject.toml` dependencies
2. Create XGBoost classifier wrapper with sklearn API
3. Add Hydra config: `conf/classifier/xgboost.yaml`
4. Train on both backbones (ESM-1v, ESM2-650M)
5. Benchmark on Jain/Harvey/Shehata

**Commands:**
```bash
# Train all combinations
antibody-train --multirun model=esm1v,esm2_650m classifier=logreg,xgboost
```

**Expected Results:**
| Backbone | Classifier | Jain Acc | Hypothesis |
|----------|-----------|----------|------------|
| ESM-1v | LogReg | 66.3% | Baseline |
| ESM-1v | XGBoost | **67-68%?** | Test if XGBoost wins |
| ESM2-650M | LogReg | 62.8% | Current |
| ESM2-650M | XGBoost | **64-66%?** | Improvement expected |

**Blockers:**
- Dependency: `xgboost>=2.0.0`
- Compute: ~12 hours GPU time (or ~36 hours CPU)

#### Week 3-4: AntiBERTa Backbone

**Why:** AntiBERTa is antibody-specific and may capture domain-specific patterns better than general protein LMs.

**Tasks:**
1. Verify HuggingFace model path: `alchemab/antiberta2` or `jeffreyruffolo/AntiBERTa`
2. Create Hydra config: `conf/model/antiberta.yaml`
3. Test embedding extraction works
4. Train with both LogReg and XGBoost
5. Benchmark on all test sets

**Commands:**
```bash
# Train AntiBERTa with both classifiers
antibody-train --multirun model=antiberta classifier=logreg,xgboost
```

**Expected Results:**
- AntiBERTa may beat ESM-1v on antibody-specific tasks
- Faster inference (smaller model than ESM-2)
- ~65-67% Jain accuracy predicted

**Blockers:**
- Verify correct HF model path exists and is accessible
- Embedding dimension may differ (requires testing)

#### Week 5-6: MLP Classifier + Benchmark Analysis

**Why:** Neural network classifier may capture non-linear patterns. Completes the 3×3 model grid.

**Tasks:**
1. Implement 2-layer MLP classifier (sklearn MLPClassifier)
2. Add Hydra config: `conf/classifier/mlp.yaml`
3. Train all 9 combinations (3 backbones × 3 classifiers)
4. Generate comprehensive comparison table
5. Create visualizations (bar charts, ROC curves)
6. Write analysis: which combination works best?

**Commands:**
```bash
# Train everything (9 models total)
antibody-train --multirun \
  model=esm1v,esm2_650m,antiberta \
  classifier=logreg,xgboost,mlp
```

**Deliverables:**
- 9 trained models with full benchmarks
- Comprehensive comparison in `docs/research/benchmark-results.md`
- Publishable blog post: "Which PLM is Best for Antibody Polyreactivity?"

**Timeline:** 4-6 weeks total (depending on GPU availability)

---

### Phase 2: Publish to HuggingFace Model Hub (1-2 weeks)

**Goal:** Make all trained weights publicly accessible

#### Create HuggingFace Organization

**Proposed:** `huggingface.co/antibody-esm/`

**Structure:**
```
antibody-esm/
├── esm1v-logreg-boughter/       # Novo baseline (66.3% Jain)
├── esm1v-xgboost-boughter/      # XGBoost upgrade
├── esm2-logreg-boughter/        # ESM2 baseline
├── esm2-xgboost-boughter/       # Predicted best overall
├── antiberta-logreg-boughter/   # Antibody-specific model
├── antiberta-xgboost-boughter/  # Fast + accurate
└── ... (all 9 model combinations)
```

#### Each Model Includes

**Files:**
- Model weights (NPZ format, no pickle)
- Config file (JSON)
- Model card (README.md with metrics, usage, citation)

**Model Card Template:**
```markdown
# ESM2-XGBoost Antibody Polyreactivity Classifier

Trained on Boughter dataset (914 VH sequences, ELISA assay)

## Performance

| Dataset | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---------|----------|-----------|--------|----|----|--------|
| Jain (86 clinical mAbs) | 66.5% | 0.67 | 0.65 | 0.66 | 0.71 |
| Harvey (141k nanobodies) | 66.8% | 0.67 | 0.66 | 0.66 | 0.72 |
| Shehata (398 PSR assay) | 63.2% | 0.64 | 0.62 | 0.63 | 0.68 |

## Usage

```python
from antibody_predictor import AntibodyClassifier
model = AntibodyClassifier.from_pretrained("antibody-esm/esm2-xgboost-boughter")
result = model.predict("EVQLVESGGGLVQPGGSLRLSCAASGFTFS...")
```

## Citation
[Paper/repo citation]
```

**Blockers:**
- Create HuggingFace organization (or use personal account)
- Convert all models to NPZ+JSON (pickle-free)
- Write 9 model cards

**Timeline:** 1-2 weeks

---

### Phase 3: Inference API Library (2-3 weeks)

**Goal:** `pip install antibody-predictor` for users to run predictions on THEIR sequences

#### Package Design

**Name:** `antibody-predictor` (PyPI)

**Installation:**
```bash
pip install antibody-predictor
```

**Core API:**
```python
from antibody_predictor import AntibodyClassifier

# Load best pretrained model from HF Hub
model = AntibodyClassifier.from_pretrained("antibody-esm/esm2-xgboost-boughter")

# Predict single sequence
seq = "EVQLVESGGGLVQPGGSLRLSCAASGFTFS..."
result = model.predict(seq)
# Returns: {'polyreactive': True, 'score': 0.78, 'confidence': 0.92}

# Predict batch from CSV
results = model.predict_csv("my_antibodies.csv", output="predictions.csv")

# Compare multiple models
comparison = model.compare_models(
    seq,
    models=["esm1v-logreg", "esm2-xgboost", "antiberta-xgboost"]
)
```

#### Features (Phase 1 Scope)

**In Scope:**
- ✅ Load pretrained models from HF Hub
- ✅ Predict on single sequences
- ✅ Batch prediction from CSV
- ✅ Handle VH-only, VL-only sequences
- ✅ Basic preprocessing (strip whitespace, validate amino acids)
- ✅ Model comparison tool

**Out of Scope (Phase 2):**
- ❌ Training on custom datasets
- ❌ Complex CSV format handling (messy headers, multiple sequence columns)
- ❌ ANARCI annotation (use sequences as-is)
- ❌ VH+VL paired sequences

#### Implementation Plan

1. Create package structure: `src/antibody_predictor/`
2. Implement model loading from HF Hub
3. Implement prediction API
4. Add CSV input/output utilities
5. Write comprehensive documentation
6. Publish to PyPI

**Blockers:**
- PyPI package name availability (`antibody-predictor`)
- Documentation site (ReadTheDocs or GitHub Pages)

**Timeline:** 2-3 weeks

---

### Phase 4: Web Demo (1 week)

**Goal:** Non-coders can use models via browser

#### Streamlit App on HuggingFace Spaces

**URL:** `huggingface.co/spaces/antibody-esm/polyreactivity-predictor`

**Features:**

1. **Single Sequence Prediction**
   - Text box: paste VH sequence
   - Dropdown: select model
   - Output: polyreactivity score + confidence

2. **Batch Prediction**
   - Upload CSV with sequences
   - Select model
   - Download results CSV

3. **Model Comparison**
   - Run all 9 models on same sequence
   - Side-by-side comparison table
   - Benchmark results visualization

**UI Mockup:**
```
┌─────────────────────────────────────────┐
│  Antibody Polyreactivity Predictor      │
├─────────────────────────────────────────┤
│  Paste VH sequence:                     │
│  ┌───────────────────────────────────┐  │
│  │ EVQLVESGGGLVQPGGSLRLSCAASGFTFS... │  │
│  └───────────────────────────────────┘  │
│                                         │
│  Model: [ESM2-XGBoost (Best) ▼]        │
│                                         │
│  [Predict Polyreactivity]              │
│                                         │
│  Result: 78% likely POLYREACTIVE       │
│  Confidence: High (0.92)               │
└─────────────────────────────────────────┘
```

**Deliverables:**
- Public Streamlit app on HF Spaces
- Easy-to-share link for non-technical users
- Viral potential (Twitter, Reddit)

**Timeline:** 1 week

---

## How to Contribute

### We Need Your Input!

**Join the discussion:** [GitHub Discussions - Roadmap 2025](https://github.com/The-Obstacle-Is-The-Way/antibody_training_pipeline_ESM/discussions/9)

**Help us prioritize:**
1. Should we finish XGBoost first, or publish current models to HF Hub?
2. Which additional backbones are most valuable (ProtBERT, AbLang, ESM2-3B)?
3. What inference features are critical for your workflow?

### Ways to Help

**Testing & Validation:**
- Test models on your antibody sequences
- Report benchmark results on new datasets
- Compare predictions to experimental assays

**Code Contributions:**
- Add new classifiers (SVM, Random Forest, Gradient Boosting)
- Implement additional backbones (ProtBERT, AbLang)
- Improve preprocessing pipelines
- Enhance documentation

**Community:**
- Star the repo if this is useful
- Share with colleagues working on antibody design
- Cite in publications using these models
- Report bugs/issues via GitHub Issues

**Open Issues for Contributors:**
- Issue #8: Standardize output directory hierarchy
- See `good-first-issue` label for beginner-friendly tasks

---

## Future Directions (Post v1.0)

### Option A: Benchmark Submission Platform

**When:** After Model Zoo is established and 10+ users request it

**Vision:** Community-driven leaderboard like Papers with Code
- Submit Docker images with custom models
- Automatic evaluation on standardized test sets
- Public rankings and model comparisons

**Why defer:** Need baseline models first; infrastructure is complex

### Option B: Custom Training on User Data

**When:** After 50+ users of inference API

**Vision:**
```python
from antibody_predictor import train_custom_model

model = train_custom_model(
    train_csv="my_proprietary_antibodies.csv",
    backbone="esm2",
    classifier="xgboost"
)
```

**Why defer:** User CSVs are messy; need robust preprocessing first

### Option C: Advanced Features (6-12 months)

**Drug Discovery Integration:**
- Multi-chain scoring (VH+VL pairs)
- Mutagenesis suggestions
- Multi-criteria ranking (polyreactivity + stability + expressibility)

**Structure-Aware Models:**
- AlphaFold/ESMFold integration
- Graph Neural Networks (GNNs)
- 3D structure predictions

**Interpretability Suite:**
- Per-residue contribution heatmaps
- Cross-dataset diagnostics
- Interactive antibody inspector

---

## Success Metrics

### After Phase 1 (Model Zoo Complete)
- ✅ 9 trained models (3 backbones × 3 classifiers)
- ✅ Comprehensive benchmark comparison
- ✅ Published blog post/paper comparing all combinations
- 🎯 Identify best model (e.g., "ESM2-XGBoost achieves 68% Jain accuracy")

### After Phase 4 (Full Release)
- ✅ All 9 models on HuggingFace Hub with model cards
- ✅ PyPI package published: `pip install antibody-predictor`
- ✅ Web demo live with 100+ users in first month
- ✅ Comprehensive documentation (API docs, tutorials, examples)

### After 6 Months (Community Adoption)
- 🎯 500+ PyPI downloads
- 🎯 1,000+ HF Space demo users
- 🎯 50+ GitHub stars
- 🎯 5+ external citations in papers
- 🎯 10+ community contributions (PRs, issues, discussions)

---

## Dependencies & Requirements

### Software (Phase 1)

**Required:**
- Python 3.12
- PyTorch 2.x
- transformers (HuggingFace)
- scikit-learn
- hydra-core (installed)

**To Add:**
- `xgboost>=2.0.0` (Week 1-2)
- Verify AntiBERTa model availability (Week 3-4)

### Compute Resources

**Minimum (CPU-only):**
- 16GB RAM
- 50GB disk space
- ~72 hours total for 9 models

**Recommended (GPU):**
- NVIDIA GPU with 16GB+ VRAM (T4, V100, A10, RTX 4090)
- 32GB RAM
- 50GB disk space
- ~24 hours total for 9 models

**For ESM2-3B (future):**
- NVIDIA A100 40GB+ VRAM
- 64GB RAM
- Additional ~24 hours per model

### Open Questions

1. **GPU Access:** Do we have cloud credits or local GPU? Affects timeline.
2. **HF Organization:** Create `antibody-esm` or use personal account?
3. **PyPI Package Name:** Is `antibody-predictor` available?
4. **AntiBERTa Model:** Verify exact HuggingFace model path.

---

## Get Involved

**Start here:**
- ⭐ Star the repo: [antibody_training_pipeline_ESM](https://github.com/The-Obstacle-Is-The-Way/antibody_training_pipeline_ESM)
- 💬 Join the discussion: [Roadmap 2025](https://github.com/The-Obstacle-Is-The-Way/antibody_training_pipeline_ESM/discussions/9)
- 🐛 Report issues: [GitHub Issues](../../issues)
- 📚 Read the docs: `docs/`

**Contact:**
- GitHub: [@The-Obstacle-Is-The-Way](https://github.com/The-Obstacle-Is-The-Way)
- Issues: [Report bugs or request features](../../issues)
- Discussions: [Ask questions or share ideas](../../discussions)

---

**The obstacle is the way. Let's build the definitive antibody ML model zoo together.** 🚀
