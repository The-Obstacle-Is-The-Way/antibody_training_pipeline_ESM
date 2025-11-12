# Strategic Roadmap: Antibody ML Model Zoo

**Date:** 2025-11-12
**Status:** Active - Post Hydra v0.4.0
**Vision:** Build the definitive model zoo for antibody polyreactivity prediction

---

## Current State: What We Have (v0.4.0)

✅ **Research Foundation**
- Novo Nordisk paper reproduced (66.28% accuracy on Jain benchmark)
- 4 preprocessed datasets: Boughter (train), Jain/Harvey/Shehata (test)
- Working training pipeline (ESM-1v + Logistic Regression)

✅ **Production Infrastructure**
- Dual-format model serialization (NPZ+JSON + pickle)
- Professional CI/CD, Docker images, comprehensive tests
- Public GitHub repo with full documentation

✅ **Experiment Infrastructure**
- Hydra configuration system with structured configs
- CLI overrides for rapid iteration: `antibody-train model=esm2_650m classifier.C=0.5`
- Multirun sweeps for hyperparameter search
- Automatic experiment tracking and provenance

✅ **ESM2 Support** (v0.4.1-dev)
- ESM2-650M backbone config ready
- Ready to benchmark against ESM-1v baseline

---

## The Vision: Model Zoo + Inference Platform

**Core Insight:** Researchers need pretrained models, not training infrastructure.

**The AlphaFold Model:**
- AlphaFold team trained models on public datasets
- Published weights and benchmark results
- Users use pretrained weights for structure prediction
- **Users DON'T retrain AlphaFold themselves**

**Our Approach:**
- We train on Boughter (public training set)
- Publish weights and benchmark results
- Users use OUR weights for polyreactivity prediction
- Users DON'T retrain on their own datasets (Phase 1)

---

## Strategic Phases

### Phase 1: Build Model Zoo (4-6 weeks) ✅ **IN PROGRESS**

**Goal:** Train every reasonable PLM×classifier combination

**Backbones:**
- ✅ ESM-1v (baseline - 66.28% Jain accuracy)
- 🔄 ESM2-650M (config shipped, ready to benchmark)
- ⏳ AntiBERTa (antibody-specific)
- ⏳ ESM2-3B (optional - if compute available)

**Classifiers:**
- ✅ LogisticRegression (baseline)
- ⏳ XGBoost (likely beats LogReg)
- ⏳ MLP (2-layer neural network)
- ⏳ SVM (optional)

**Training Strategy:**
```bash
# Train all combinations with Hydra multirun
antibody-train --multirun \
  model=esm1v,esm2_650m,antiberta \
  classifier=logreg,xgboost,mlp

# Result: 9 trained models (3 backbones × 3 classifiers)
```

**Deliverables:**
- 9-12 trained model weights
- Comprehensive benchmark comparison table
- Model cards with metrics
- Publishable comparison: "Which PLM is Best for Antibody Polyreactivity?"

**Timeline:** Week 1-4 (ESM2 → XGBoost → AntiBERTa → Benchmark)

---

### Phase 2: Publish to HuggingFace Model Hub (1-2 weeks)

**Goal:** Make trained weights publicly accessible

**Create HF Organization:**
```
huggingface.co/antibody-esm/
├── esm1v-logreg-boughter/       # Novo baseline
├── esm2-xgboost-boughter/       # Best overall (predicted)
├── antiberta-xgboost-boughter/  # Fast + accurate
└── ... (all other combinations)
```

**Each Model Includes:**
- Model weights (NPZ + JSON, no pickle!)
- Config file
- README with:
  - Benchmark metrics (Jain/Harvey/Shehata accuracy)
  - Training details
  - Usage example
  - Citation info

**Deliverables:**
- All models on HuggingFace Hub
- Model cards with comprehensive metrics
- Public model zoo accessible via `from_pretrained()`

**Timeline:** Week 5

---

### Phase 3: Inference API + Library (2-3 weeks)

**Goal:** Let users run predictions on THEIR sequences using YOUR weights

**Create Package:** `antibody-predictor`

**Installation:**
```bash
pip install antibody-predictor
```

**Usage:**
```python
from antibody_predictor import AntibodyClassifier

# Load best model from HF Hub
model = AntibodyClassifier.from_pretrained("antibody-esm/esm2-xgboost-boughter")

# Predict single sequence
seq = "EVQLVESGGGLVQPGGSLRLSCAASGFTFS..."
result = model.predict(seq)
# Output: {'polyreactive': True, 'score': 0.78, 'confidence': 0.92}

# Predict batch from CSV
results = model.predict_csv("my_antibodies.csv", output="predictions.csv")
```

**Features:**
- ✅ Load pretrained models from HF Hub
- ✅ Predict on single sequences or batches
- ✅ Handle VH-only, VL-only, or VH+VL pairs
- ✅ Basic preprocessing (chain detection, cleaning)
- ❌ NOT handling messy CSVs (Phase 2)
- ❌ NOT training on custom data (Phase 2)

**Deliverables:**
- `pip install antibody-predictor` working
- PyPI package published
- API documentation
- Usage tutorials

**Timeline:** Week 6-7

---

### Phase 4: Web Demo (1 week)

**Goal:** Non-coders can use your models

**Build:** Streamlit/Gradio app on HuggingFace Spaces

**Features:**
1. **Single Sequence Prediction**
   - Text box: Paste VH sequence
   - Dropdown: Select model (ESM2-XGBoost, ESM-1v-LogReg, etc.)
   - Button: "Predict"
   - Output: Polyreactivity score + confidence

2. **Batch Prediction**
   - Upload CSV (columns: `sequence_id`, `sequence`)
   - Select model
   - Download results CSV

3. **Model Comparison**
   - Show benchmark results table
   - Interactive plots
   - Model selection guide

**Deliverables:**
- Public demo at `huggingface.co/spaces/username/antibody-predictor`
- Easy-to-share link for non-technical users
- Viral potential (Twitter, Reddit)

**Timeline:** Week 8

---

## Phase 1 Detailed Breakdown (Next 4 Weeks)

### Week 1: ESM2-650M Benchmark ✅ **READY TO START**

**Goal:** Compare ESM2 vs ESM-1v

**Tasks:**
1. ✅ ESM2 config already exists (`conf/model/esm2_650m.yaml`)
2. Train ESM2-650M + LogReg on Boughter
3. Test on Jain/Harvey/Shehata
4. Compare to ESM-1v baseline

**Commands:**
```bash
# Compare both backbones
antibody-train --multirun model=esm1v,esm2_650m classifier=logreg
```

**Expected Result:**
- ESM-1v: 66.3% Jain accuracy (Novo baseline)
- ESM2-650M: 64-68% predicted (literature suggests ESM-1v wins on antibodies)

**Deliverable:** Benchmark comparison table in `BENCHMARK_RESULTS.md`

---

### Week 2: XGBoost Classifier

**Goal:** Test if XGBoost beats LogReg

**Tasks:**
1. Add XGBoost classifier wrapper (sklearn API)
2. Add Hydra config for XGBoost
3. Train on both backbones

**Commands:**
```bash
# Train all combinations
antibody-train --multirun \
  model=esm1v,esm2_650m \
  classifier=logreg,xgboost
```

**Expected Result:**
| Backbone | Classifier | Jain Acc | Hypothesis |
|----------|-----------|----------|------------|
| ESM-1v | LogReg | 66.3% | Novo baseline |
| ESM-1v | XGBoost | **67-68%?** | Test if XGBoost wins |
| ESM2-650M | LogReg | 65-67%? | ESM2 comparison |
| ESM2-650M | XGBoost | **66-69%?** | Best combo? |

**Deliverable:** 4 trained models, updated benchmark table

---

### Week 3: AntiBERTa + MLP

**Goal:** Complete the model zoo

**Tasks:**
1. Add AntiBERTa backbone config
2. Add MLP classifier (2-layer neural network)
3. Train all combinations

**Commands:**
```bash
# Train everything
antibody-train --multirun \
  model=esm1v,esm2_650m,antiberta \
  classifier=logreg,xgboost,mlp
```

**Deliverable:** 12 trained models (3 backbones × 4 classifiers)

---

### Week 4: Benchmark Analysis

**Goal:** Publish comprehensive comparison

**Tasks:**
1. Create comparison table (all models × all datasets)
2. Generate visualizations (bar charts, ROC curves)
3. Write `BENCHMARK_RESULTS.md`
4. Analyze insights (which works best?)

**Deliverable:** `BENCHMARK_RESULTS.md` with full comparison, publishable blog post

---

## Success Metrics

### After Week 4 (Model Zoo Complete):
- ✅ 12+ trained models
- ✅ Comprehensive benchmark comparison
- ✅ Clear winner identified (e.g., "ESM2-XGBoost is best")

### After Week 8 (Full Release):
- ✅ Models on HuggingFace Hub
- ✅ PyPI package published
- ✅ Web demo live
- ✅ Documentation complete

### After 3 Months (Adoption):
- 🎯 100+ PyPI downloads
- 🎯 500+ HF Space users
- 🎯 10+ GitHub stars
- 🎯 1+ external citations

---

## Future Directions (Post Phase 1-4)

### Option A: Benchmark Submission Platform (Community-Driven)

**When:** After Model Zoo is established and users request it

**What it is:**
- Public leaderboard (like Papers with Code)
- Users submit Docker images with their models
- Automatic evaluation on standardized test sets
- Public rankings and model comparisons

**Why defer to Phase 2:**
- Need to establish baseline first (your Model Zoo)
- Leaderboard is empty without your models
- Infrastructure is complex (evaluation compute, submissions)
- Better to validate demand before building

**Trigger:** If 10+ researchers ask "Can I submit my model to compare?"

---

### Option B: Custom Training on User Data (Advanced)

**When:** After 50+ users of inference tool

**What it is:**
```python
# Phase 2 feature (not building yet)
from antibody_predictor import train_custom_model

model = train_custom_model(
    train_csv="my_proprietary_antibodies.csv",
    backbone="esm2",
    classifier="xgboost"
)
```

**Why defer:**
- User CSVs are messy (inconsistent formats)
- Need robust preprocessing (ANARCI might fail)
- Complex error handling
- Time better spent on inference API first

**Trigger:** Multiple users request training on their data

---

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

## Why This Path?

### Model Zoo First Because:
1. **Immediate Value**: Users get predictions on their sequences TODAY
2. **Clear Scope**: You control everything (training data, models, benchmarks)
3. **Proven Model**: AlphaFold, ESMFold, BioBERT all follow this pattern
4. **Foundation**: Everything else (leaderboard, custom training) builds on this

### NOT Building (Yet):
- ❌ Benchmark submission platform (wait for demand)
- ❌ Custom training on user data (too complex for Phase 1)
- ❌ Advanced features (after core is stable)

### The Key Insight:
Small labs need **predictions**, not **training infrastructure**. Give them pretrained models that work, then expand based on feedback.

---

## Technical Foundation (Already Shipped)

### Hydra Configuration System ✅ v0.4.0
- CLI overrides: `antibody-train model.batch_size=16 classifier.C=0.5`
- Multirun sweeps: `antibody-train --multirun classifier.C=0.1,1.0,10.0`
- Automatic experiment tracking
- Config composition

### Production Serialization
- NPZ+JSON format (pickle-free for production)
- Model cards with provenance
- HuggingFace-ready format

### Professional Infrastructure
- 100% type safety (mypy strict mode)
- CI/CD with GitHub Actions
- Docker images (dev + prod)
- Comprehensive test suite (70%+ coverage)

---

## Open Questions

1. **ESM2 Performance:** Will ESM2 beat ESM-1v? (Benchmark in Week 1)
2. **XGBoost vs LogReg:** How much improvement? (Week 2)
3. **HF Hub Organization:** Create `antibody-esm` or use personal account?
4. **PyPI Package Name:** `antibody-predictor` or `antibody-esm`?
5. **Compute Resources:** Do we have GPU access for ESM2-3B?

---

## Next Action: This Week

**Start Week 1: ESM2-650M Benchmark**

**Steps:**
1. Verify ESM2 config works:
   ```bash
   antibody-train model=esm2_650m training.n_splits=2
   ```

2. Run full benchmark:
   ```bash
   antibody-train --multirun model=esm1v,esm2_650m classifier=logreg
   ```

3. Compare results in `BENCHMARK_RESULTS.md`

4. Tweet/blog: "ESM2 vs ESM-1v for Antibody Prediction"

---

## The Bottom Line

**Simple Plan:**
1. **Model Zoo:** Train ESM-1v/ESM2/AntiBERTa × LogReg/XGBoost/MLP (4 weeks)
2. **Publish:** Upload weights to HuggingFace Hub (1 week)
3. **Inference:** Let users predict on their sequences (2 weeks)
4. **Demo:** Web UI for non-coders (1 week)
5. **Done:** That's it. 8 weeks to full platform.

**NOT building (yet):**
- Benchmark submission platform (defer until demand proven)
- Custom data training (defer until inference API validated)
- Advanced features (defer until core is stable)

**Start this week: Benchmark ESM2. The rest follows from there.**

**The obstacle is the way. Stop planning. Start benchmarking.** 🚀
