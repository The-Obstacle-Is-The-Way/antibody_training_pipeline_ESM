# The Plan: Antibody ML Model Zoo + Inference Platform

**Date:** 2025-11-12
**Status:** Post-Hydra integration - Ready to build
**Goal:** Train every reasonable PLM×classifier combo, publish weights, let people use them

---

## TL;DR - What We're Building

**Simple version:**
1. You train models on Boughter (all combinations of backbones + classifiers)
2. You benchmark on Jain/Harvey/Shehata
3. You publish trained weights to HuggingFace
4. Users load your weights and run predictions on THEIR antibody sequences
5. Done.

**NOT building yet:** User training on custom datasets (too complex, Phase 2)

---

## The Core Insight: Two Separate Workflows

### Workflow A: YOU Train Models (Research Pipeline)

```
You: Download ESM2 → Extract embeddings on Boughter → Train XGBoost → Test on Jain
You: Upload weights to HuggingFace → Publish benchmark results
```

**What you control:**
- Training data: Boughter (FIXED)
- Test data: Jain, Harvey, Shehata (FIXED)
- Models: You decide which PLMs and classifiers to try
- Compute: You run the training (GPU overnight jobs)

**Output:**
- Trained model weights on HuggingFace
- Benchmark comparison table
- "We tested 12 combinations. Here's what works best."

---

### Workflow B: USERS Run Inference (Production Pipeline)

```
User: pip install antibody-predictor
User: Load pretrained model from HF Hub
User: Predict polyreactivity on THEIR antibody sequences
User: Get results
```

**What users control:**
- Their antibody sequences (their CSV)
- Which of YOUR models to use
- Nothing else - they don't train anything

**Output:**
- Predictions on user's sequences
- Confidence scores
- Model comparison (if they run multiple models)

---

## Why This Makes Sense

**For Research (AlphaFold Model):**
- AlphaFold team trained models on public datasets
- Published weights and benchmark results
- Users use pretrained weights for structure prediction
- Users DON'T retrain AlphaFold themselves

**For You:**
- You train on Boughter (public training set)
- Publish weights and benchmark results
- Users use YOUR weights for polyreactivity prediction
- Users DON'T retrain on their own datasets (at least not in Phase 1)

**The value:**
- Small labs: "Use SOTA models without ML expertise"
- Researchers: "Reproduce published baselines easily"
- Your contribution: "Comprehensive comparison of PLM×classifier combos"

---

## What You're Building (Concrete Plan)

### Component 1: Model Zoo (Weeks 1-4)

**Goal:** Train every reasonable combination

**Backbones to add:**
- ✅ ESM-1v (you have this - Novo baseline)
- ⏳ ESM2-650M (most popular, better than ESM-1v)
- ⏳ AntiBERTa (antibody-specific model)
- ⏳ ESM2-3B (optional - if you have compute)
- ⏳ AbLang (optional - antibody-focused)

**Classifiers to add:**
- ✅ LogisticRegression (you have this - Novo baseline)
- ⏳ XGBoost (popular, often beats LogReg)
- ⏳ MLP (2-layer neural network)
- ⏳ SVM (optional)

**Training strategy:**
- Train on Boughter (914 sequences, ELISA assay)
- Test on:
  - Jain (86 sequences, clinical antibodies, Novo benchmark)
  - Harvey (141k nanobodies, PSR assay)
  - Shehata (398 antibodies, PSR assay)

**Hydra makes this EASY:**
```bash
# Train all combinations overnight
antibody-train --multirun \
  model=esm1v,esm2-650m,antiberta \
  classifier=logreg,xgboost,mlp

# Result: 9 trained models (3 backbones × 3 classifiers)
```

**Output:**
- 9-12 trained model weights
- Benchmark comparison table
- Model cards with metrics

---

### Component 2: Benchmark Results (Week 4)

**Goal:** Show which models work best

**Create:**
- `BENCHMARK_RESULTS.md` with comparison table
- Visualizations (bar charts, ROC curves)
- Analysis of which works best on which dataset

**Example table:**
```markdown
| Backbone      | Classifier | Jain Acc | Harvey Acc | Shehata Acc | Speed  |
|---------------|------------|----------|------------|-------------|--------|
| ESM2-650M     | XGBoost    | 68.5%    | 63.8%      | 64.9%       | 2.1s   |
| ESM2-650M     | MLP        | 67.9%    | 63.2%      | 64.3%       | 2.0s   |
| AntiBERTa     | XGBoost    | 67.8%    | 63.1%      | 64.1%       | 1.8s   |
| ESM-1v (Novo) | LogReg     | 66.3%    | 61.5%      | 62.8%       | 2.0s   |
```

**Insights you can publish:**
- "ESM2 beats ESM-1v by 2-3% accuracy"
- "XGBoost consistently beats LogReg"
- "AntiBERTa is fastest while maintaining good accuracy"
- "MLP shows no significant benefit over XGBoost"

**This is publishable:** "Comprehensive Benchmark of Protein Language Models for Antibody Polyreactivity Prediction"

---

### Component 3: HuggingFace Model Hub (Week 5)

**Goal:** Publish trained weights publicly

**Create HF organization:**
```
huggingface.co/antibody-esm/
├── esm1v-logreg-boughter/       # Novo baseline
├── esm2-xgboost-boughter/       # Best overall
├── antiberta-xgboost-boughter/  # Fast + accurate
├── esm2-mlp-boughter/           # Neural network head
└── ... (all other combinations)
```

**Each model includes:**
- Model weights (NPZ + JSON, no pickle!)
- Config file
- README with:
  - Benchmark metrics
  - Training details
  - Usage example
  - Citation info

**Users can then:**
```python
from transformers import AutoModel
# Or your custom loader
model = load_antibody_model("antibody-esm/esm2-xgboost-boughter")
```

---

### Component 4: Inference Library (Weeks 6-7)

**Goal:** Easy API for using your pretrained models

**Create package:** `antibody-predictor`

**Installation:**
```bash
pip install antibody-predictor
```

**Usage:**
```python
from antibody_predictor import AntibodyClassifier

# Load best model from your HF Hub
model = AntibodyClassifier.from_pretrained("antibody-esm/esm2-xgboost-boughter")

# Predict single sequence
seq = "EVQLVESGGGLVQPGGSLRLSCAASGFTFS..."
result = model.predict(seq)
# Output: {'polyreactive': True, 'score': 0.78, 'confidence': 0.92}

# Predict batch from CSV
results = model.predict_csv("my_antibodies.csv", output="predictions.csv")
```

**Key features:**
- ✅ Load pretrained models from HF Hub
- ✅ Predict on single sequences or batches
- ✅ Handle VH-only, VL-only, or VH+VL pairs
- ✅ Basic preprocessing (chain detection, cleaning)
- ❌ NOT handling messy CSVs (Phase 2)
- ❌ NOT training on custom data (Phase 2)

**What it does:**
- Loads YOUR pretrained weights
- Runs inference on user sequences
- Returns predictions

**What it doesn't do:**
- Train new models
- Handle complex preprocessing (yet)
- Accept arbitrary CSV formats (yet)

---

### Component 5: Web Demo (Week 8)

**Goal:** Non-coders can use your models

**Build:** Streamlit/Gradio app on HuggingFace Spaces

**Features:**
1. **Single Sequence Prediction:**
   - Text box: Paste VH sequence
   - Dropdown: Select model (ESM2-XGBoost, ESM-1v-LogReg, etc.)
   - Button: "Predict"
   - Output: Polyreactivity score + confidence

2. **Batch Prediction:**
   - Upload CSV (columns: `sequence_id`, `sequence`)
   - Select model
   - Download results CSV

3. **Model Comparison:**
   - Show benchmark results table
   - Interactive plots
   - Model selection guide

**Example UI:**
```
┌─────────────────────────────────────────┐
│  Antibody Polyreactivity Predictor      │
├─────────────────────────────────────────┤
│  Paste antibody sequence (VH):          │
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

**Value:** Anyone can screen antibodies without coding

---

## Phase 2: Custom Data Pipelines (LATER)

**NOT building now. Deal with this after Phase 1 works.**

**Why it's hard:**
- User CSVs are messy (inconsistent formats)
- Need robust preprocessing (ANARCI might fail)
- Need to handle VH-only, VL-only, paired, nanobodies, etc.
- Need error handling for invalid sequences

**When to build:**
- After you have 50+ users of inference tool
- After you understand common pain points
- After users request it

**What it would look like:**
```python
# Phase 2 feature (not building yet)
from antibody_predictor import train_custom_model

model = train_custom_model(
    train_csv="my_proprietary_antibodies.csv",
    backbone="esm2",
    classifier="xgboost"
)
```

**But for now:** Users just use your pretrained weights.

---

## Weekly Plan (Next 8 Weeks)

### Week 1: Add ESM2 Support
**Goal:** Train ESM2 models

**Tasks:**
1. Add ESM2 config to Hydra (copy ESM-1v config, change model name)
2. Test embedding extraction works
3. Train ESM2-650M + LogReg on Boughter
4. Benchmark on Jain/Harvey/Shehata
5. Compare to ESM-1v baseline

**Deliverable:** ESM2 model trained, metrics in table

**Commands:**
```bash
# Add config file: conf/model/esm2-650m.yaml
# Run training
antibody-train model=esm2-650m classifier=logreg

# Compare to baseline
antibody-train --multirun model=esm1v,esm2-650m classifier=logreg
```

---

### Week 2: Add XGBoost Classifier
**Goal:** Test if XGBoost beats LogReg

**Tasks:**
1. Add XGBoost classifier wrapper (sklearn API)
2. Add Hydra config for XGBoost
3. Train ESM-1v + XGBoost
4. Train ESM2 + XGBoost
5. Compare all combinations

**Deliverable:** XGBoost classifier working, 4 models total

**Commands:**
```bash
# Train all combinations
antibody-train --multirun \
  model=esm1v,esm2-650m \
  classifier=logreg,xgboost
```

---

### Week 3: Add AntiBERTa + MLP
**Goal:** Complete the model zoo

**Tasks:**
1. Add AntiBERTa backbone
2. Add MLP classifier (2-layer neural network)
3. Train all combinations (Hydra multirun)
4. Collect all benchmark results

**Deliverable:** 12 models trained (3 backbones × 4 classifiers)

**Commands:**
```bash
# Train everything
antibody-train --multirun \
  model=esm1v,esm2-650m,antiberta \
  classifier=logreg,xgboost,mlp
```

---

### Week 4: Benchmark Analysis
**Goal:** Publish comprehensive comparison

**Tasks:**
1. Create comparison table (all models × all datasets)
2. Generate visualizations (bar charts, ROC curves)
3. Write `BENCHMARK_RESULTS.md`
4. Analyze insights (which works best?)

**Deliverable:** `BENCHMARK_RESULTS.md` with full comparison

---

### Week 5: HuggingFace Model Hub
**Goal:** Publish trained weights

**Tasks:**
1. Convert models to NPZ+JSON format (no pickle)
2. Create HF organization: `antibody-esm`
3. Upload all model weights
4. Write model cards (README with metrics)
5. Test downloading works

**Deliverable:** All models on HF Hub, publicly accessible

---

### Week 6: Inference Library (Core)
**Goal:** Build `antibody-predictor` package

**Tasks:**
1. Create package structure
2. Implement model loading from HF Hub
3. Implement prediction API
4. Add preprocessing (basic cleaning)
5. Write tests

**Deliverable:** Working package, can install locally

---

### Week 7: Inference Library (Polish)
**Goal:** Publish to PyPI

**Tasks:**
1. Add batch prediction
2. Add CSV input/output
3. Write documentation
4. Publish to PyPI
5. Test `pip install antibody-predictor` works

**Deliverable:** `pip install antibody-predictor` live

---

### Week 8: Web Demo
**Goal:** Public HuggingFace Space

**Tasks:**
1. Build Streamlit app
2. Add single sequence prediction
3. Add batch CSV upload
4. Deploy to HF Spaces
5. Polish UI

**Deliverable:** Public demo at `huggingface.co/spaces/username/antibody-predictor`

---

## Success Metrics

**After Week 4 (Benchmark):**
- ✅ 12+ trained models
- ✅ Comprehensive comparison table
- ✅ Clear winner identified (e.g., "ESM2-XGBoost is best")

**After Week 8 (Full Release):**
- ✅ Models on HuggingFace Hub
- ✅ PyPI package published
- ✅ Web demo live
- ✅ Documentation complete

**After 3 Months (Adoption):**
- 🎯 100+ PyPI downloads
- 🎯 500+ HF Space users
- 🎯 10+ GitHub stars
- 🎯 1+ external citations

---

## What About Custom Data Training? (FAQ)

**Q: What if users want to train on their own datasets?**

**A: Phase 2.** For now, they use your pretrained weights. If enough users ask, you build it later.

**Q: What if their sequences don't match Boughter format?**

**A: Basic preprocessing only.** Inference library handles simple cases (strip whitespace, validate amino acids). Complex preprocessing is Phase 2.

**Q: Can they fine-tune your models?**

**A: Eventually.** Not in Phase 1. Advanced users can clone your repo and modify configs.

**Q: What if they have paired VH+VL sequences?**

**A: Start with VH-only.** Add VH+VL support in Phase 2 if users request it.

---

## How This Compares to Original Plan

**Original plan (WRONG):**
- ❌ Build benchmark submission platform (like SzCORE)
- ❌ Accept Docker images from users
- ❌ Automate evaluation with GitHub Actions
- ❌ Too complex, unclear use case

**New plan (RIGHT):**
- ✅ Train all model combinations yourself
- ✅ Publish weights + benchmark results
- ✅ Let users use your pretrained models
- ✅ Simple, clear value proposition

**Why the new plan is better:**
- Easier to build (you control everything)
- Clearer value (pretrained models anyone can use)
- Proven model (AlphaFold, ESMFold, etc.)
- Addresses real need (small labs want predictions, not training)

---

## Addressing Your Concerns

### "Will HuggingFace even like this?"

**Yes.** They love:
- Pretrained models for specific domains
- Comprehensive benchmarks
- Easy-to-use inference demos
- Scientific applications

**Examples they promote:**
- AlphaFold/ESMFold (protein structure)
- AntiBERTa (antibody-specific PLM)
- BioBERT (biomedical NLP)
- Your project fits perfectly

### "Is this a shit project?"

**No.** You're solving a real problem:
- Small labs need polyreactivity predictions
- Current tools are scattered/hard to use
- No comprehensive PLM comparison exists
- You're providing both research (benchmark) and utility (inference)

### "Too many permutations, I'm confused"

**Fixed.** The plan is now:
1. You train on Boughter (FIXED training set)
2. You test on Jain/Harvey/Shehata (FIXED test sets)
3. Users use your weights for predictions (THEIR sequences)
4. No custom training (Phase 2)

**That's it. Three clear steps.**

### "Should we even build this?"

**YES.** Start with Week 1: Add ESM2 support. Build incrementally. Learn by doing. If it's useful, people will use it. If not, you learned ML engineering skills.

**But I think it WILL be useful.** Small labs need this.

---

## Next Action: This Week

**Task:** Add ESM2-650M support

**Steps:**
1. Create `conf/model/esm2-650m.yaml`:
   ```yaml
   name: facebook/esm2_t33_650M_UR90S_1
   revision: main
   batch_size: 16
   max_length: 1024
   ```

2. Test it works:
   ```bash
   antibody-train model=esm2-650m classifier=logreg
   ```

3. Compare to ESM-1v:
   ```bash
   antibody-train --multirun model=esm1v,esm2-650m
   ```

4. Document results in `BENCHMARK_RESULTS.md`

**Want me to help you do this RIGHT NOW?** We can knock out ESM2 support in 30 minutes.

---

## Conclusion: The Simple Plan

**Stop overthinking. Here's what you're building:**

1. **Model Zoo:** Train ESM-1v/ESM2/AntiBERTa × LogReg/XGBoost/MLP
2. **Benchmark:** Test all on Jain/Harvey/Shehata, publish comparison
3. **Publish:** Upload weights to HuggingFace Hub
4. **Inference:** Let users predict on their sequences using your weights
5. **Done:** That's it. Simple. Useful. Doable.

**NOT building (yet):**
- Custom data training
- Complex preprocessing pipelines
- Benchmark submission platform
- Anything else that's confusing

**Start this week: Add ESM2. Then build from there.**

**The obstacle is the way. Stop planning. Start building.** 🚀
