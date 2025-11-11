# Strategic Roadmap: Antibody Training Pipeline

**Date:** 2025-11-10 (Updated: 2025-11-11 for v0.4.0)
**Status:** Active - Post Hydra integration (v0.4.0)
**Context:** We've successfully replicated the Novo Nordisk paper with benchmark parity and shipped Hydra. What's next?

---

## Current State: What We Have

✅ **Replicated Research**
- Novo Nordisk paper reproduced (Jain benchmark: 66.28% accuracy)
- 4 preprocessed datasets: Boughter (train), Jain/Harvey/Shehata (test)
- Working training pipeline (ESM-1v + Logistic Regression)

✅ **Production Infrastructure**
- Dual-format model serialization (NPZ+JSON + pickle)
- Professional CI/CD, Docker images, comprehensive tests
- Public GitHub repo with full documentation

✅ **Technical Foundation**
- ESM-1v embedding extraction with caching
- 10-fold cross-validation training
- Comprehensive evaluation metrics

✅ **Experiment Infrastructure (v0.4.0)**
- Hydra configuration system with structured configs
- CLI overrides for rapid iteration
- Multirun sweeps for hyperparameter search
- Automatic experiment tracking and provenance

---

## The Core Question

**"Who gives a fuck about replicating a paper?"**

We need to turn this from a reproduction into a **platform** that serves the antibody ML community.

---

## Strategic Options (Prioritized)

### Option 1: 🏆 Benchmark Platform + Leaderboard (HIGHEST IMPACT)

**Vision:** Create `antibodybenchmark.org` - the GLUE benchmark for antibody prediction

**What it is:**
- Public leaderboard (like Papers with Code)
- Users submit Docker images with their models
- Automatic evaluation on standardized test sets (Jain, Harvey, Shehata)
- Public rankings and model comparisons

**Why this matters:**
- **Network effects**: If this becomes THE standard benchmark, everyone uses it
- **Citations**: Every antibody ML paper benchmarks against your platform
- **Community building**: Like epilepsybenchmarks.com, creates ecosystem
- **Foundation**: Enables all other options (inference API, UI, research)

**Similar platforms:**
- epilepsybenchmarks.com (seizure prediction)
- Papers with Code leaderboards
- GLUE/SuperGLUE (NLP)
- ImageNet (vision)

**Technical implementation:**
1. HuggingFace Space with leaderboard UI
2. Docker image submission system
3. Automated evaluation runner (already have this!)
4. Results visualization + ranking tables
5. API for programmatic result queries

**Timeline:** 3-4 weeks

**Impact:** ⭐⭐⭐⭐⭐ (Legacy project - this could be cited for years)

---

### Option 2: 🧬 Multi-PLM Support (Better Models)

**Vision:** Support multiple protein language models beyond ESM-1v

**What it is:**
- Plug-and-play architecture for different PLMs
- Support: ESM2, AntiBERTa, AbLang, ProtBERT, IgLM
- Better classification heads (MLP, attention pooling, not just LogReg)
- Comprehensive benchmark comparisons

**Why this matters:**
- **SOTA**: ESM2 (2022) is better than ESM-1v (2021)
- **Research**: "Which PLM is best for antibody prediction?" (publishable)
- **Leaderboard content**: Populates your benchmark platform with results
- **Model zoo**: Different models for different use cases

**Research questions:**
- Does ESM2's better general protein understanding help antibodies?
- Do antibody-specific models (AntiBERTa, AbLang) beat general PLMs?
- Which pooling strategy is best? (mean, max, attention, CLS token)

**Technical implementation:**
1. Abstract `ESMEmbeddingExtractor` → `PLMEmbeddingExtractor` base class
2. Implement model-specific loaders (ESM2, AntiBERTa, etc.)
3. Add advanced classifiers (MLP, attention-based)
4. Benchmark all combinations on all datasets
5. Publish results paper + leaderboard

**Timeline:** 2-3 weeks

**Impact:** ⭐⭐⭐⭐ (Research paper + feeds leaderboard)

---

### Option 3: ⚙️ Hydra + Experiment Tracking (Infrastructure) ✅ **v0.4.0 - SHIPPED**

**Vision:** Professional ML experiment management and reproducibility

**Status:** ✅ **Hydra Complete (v0.4.0)** | ⏳ W&B/Optuna TODO

**What was implemented (v0.4.0):**
- ✅ Hydra for hierarchical config management
- ✅ Structured configs with dataclasses (type-safe)
- ✅ CLI overrides: `antibody-train model.batch_size=16 classifier.C=0.5`
- ✅ Multirun sweeps: `antibody-train --multirun classifier.C=0.1,1.0,10.0`
- ✅ Automatic experiment tracking (outputs/ directory)
- ✅ Config composition (model × classifier × data)
- ✅ Reproducibility (every run saves config snapshot)

**What's still TODO:**
- ⏳ Weights & Biases (W&B) or MLflow integration
- ⏳ Hyperparameter optimization with Optuna
- ⏳ Advanced experiment templates

**Pain points solved:**
- ✅ Manual YAML editing → CLI overrides
- ✅ No experiment history → Hydra outputs/ tracking
- ✅ Hard to compare runs → Config snapshots per run
- ⏳ No hyperparameter optimization → TODO (Optuna)

**Impact:** ⭐⭐⭐⭐ (Shipped! Enables systematic research, industry-standard infrastructure)

---

### Option 4: 🚀 Inference API + HuggingFace Model Hub (Production)

**Vision:** Make predictions accessible via simple API

**What it is:**
- Publish trained models to HuggingFace Model Hub
- Lightweight inference package: `pip install antibody-predictor`
- Simple API: `model.predict(sequence)` → polyreactivity score
- Optional: FastAPI REST server for pharma companies

**Why this matters:**
- **Usability**: Non-experts can use your research
- **Citations**: People using your models cite you
- **Industry adoption**: Pharma companies can integrate easily
- **NPZ+JSON format**: You already have production-ready serialization!

**Use cases:**
- Biologist with 1000 sequences wants polyreactivity predictions
- Pharma company integrates into internal antibody design pipeline
- Researchers use as baseline for their new methods

**Technical implementation:**
1. Upload trained models to HuggingFace Model Hub
2. Create `antibody-predictor` package (inference-only, lightweight)
3. Simple Python API with 3-line usage
4. Optional: FastAPI server with REST endpoints
5. Optional: Batch prediction utilities

**API design:**
```python
from antibody_predictor import AntibodyClassifier

model = AntibodyClassifier.from_pretrained("antibody-esm1v-boughter")
score = model.predict("EVQLVESGGGLVQPGGSLRLSCAASGFTFS...")
# Returns: {"polyreactive": False, "score": 0.32, "confidence": 0.86}
```

**Timeline:** 2-3 weeks

**Impact:** ⭐⭐⭐⭐ (Industry adoption, real-world usage)

---

### Option 5: 🎨 Streamlit UI (Democratization)

**Vision:** Non-coders can use your benchmark

**What it is:**
- Web UI for sequence prediction
- Upload CSV → get predictions
- Visualize results (ROC curves, confusion matrices)
- Compare models side-by-side
- Hosted on HuggingFace Spaces (free!)

**Why this matters:**
- **Accessibility**: Biologists without coding skills can use it
- **Demos**: Easy to show stakeholders/collaborators
- **Viral potential**: Easy to share and try
- **Education**: Students can explore antibody prediction

**Use cases:**
- Biologist wants to check if their antibodies are polyreactive
- Educator demonstrates antibody ML in class
- Researcher explores model behavior interactively

**Technical implementation:**
1. Streamlit app with file upload
2. Load models from HuggingFace Hub
3. Batch prediction with progress bars
4. Interactive visualizations (plotly)
5. Model comparison tools
6. Deploy to HuggingFace Spaces

**Timeline:** 1-2 weeks

**Impact:** ⭐⭐⭐ (Broader audience, accessibility)

---

### Option 6: 🤖 MCP Server (Experimental)

**Vision:** AI agents can call antibody predictions

**What it is:**
- Model Context Protocol (MCP) server
- Exposes `predict_polyreactivity` as a tool
- Claude/agents can automatically check sequences
- Novel integration with AI workflows

**Why this matters:**
- **Cutting-edge**: Very new protocol (Anthropic, Nov 2024)
- **Viral potential**: Cool demos, Twitter engagement
- **Future-looking**: AI agent integration is coming
- **Niche**: But interesting for AI × bio community

**Use cases:**
- Chatting with Claude about antibody design
- Claude automatically checks polyreactivity predictions
- Agent workflows that include antibody validation

**Technical implementation:**
1. Implement MCP server protocol
2. Expose prediction tools with proper schemas
3. Handle authentication/rate limiting
4. Document usage for Claude Code users

**Timeline:** 1 week (experimental)

**Impact:** ⭐⭐ (Cool, viral potential, but niche)

---

## Recommended Path: 3-Phase Rollout

### Phase 1: Build the Benchmark (3-4 weeks)
**Goal:** Establish `antibodybenchmark.org` as THE standard

1. Create HuggingFace Space with leaderboard
2. Implement Docker submission system
3. Populate with ESM-1v baseline results
4. Write submission guide documentation
5. Soft launch (Twitter, Reddit, BioML communities)

**Deliverables:**
- Public leaderboard with current results
- Submission pipeline for new models
- Documentation + tutorials

---

#### Leaderboard Submission Format (Modular Architecture)

**Key insight:** Your pipeline is modular: `[Backbone] → [Embeddings] → [Head] → [Prediction]`

This means users can submit models in **three ways:**

---

##### Submission Option 1: Full Docker Model (Maximum Flexibility)

**What:** User submits complete model in Docker container

**Format:**
```dockerfile
FROM python:3.12-slim
COPY model_weights/ /app/
COPY predict.py /app/

# Must expose this API:
# python predict.py --input sequences.csv --output predictions.csv
```

**Pros:**
- Total control (custom backbone, custom head, custom preprocessing)
- Can use non-Python models (Rust, C++, etc.)
- Novel architectures welcome

**Cons:**
- Slower evaluation (download whole container)
- Harder to compare apples-to-apples

**Use case:** "I fine-tuned ESM2 end-to-end on antibody data"

---

##### Submission Option 2: Head Weights Only (Fastest)

**What:** User submits classifier head weights, uses YOUR cached embeddings

**Format:**
```json
{
  "submission_id": "user123-mlp-v1",
  "backbone": "esm1v",  // Use your pre-computed embeddings
  "head": {
    "type": "mlp",
    "architecture": "1280 -> 128 -> 2",
    "weights": "mlp_weights.npz"  // 10KB file
  },
  "metadata": {
    "author": "john@lab.edu",
    "paper": "https://arxiv.org/...",
    "description": "MLP with dropout beats LogReg"
  }
}
```

**Pros:**
- FAST evaluation (embeddings pre-computed and cached)
- Easy comparison (same backbone across submissions)
- Tiny upload (10KB weights vs 5GB Docker image)

**Cons:**
- Limited to your supported backbones (ESM-1v, ESM2, etc.)
- Can't use novel preprocessing

**Use case:** "I think MLP beats LogReg on ESM-1v embeddings"

---

##### Submission Option 3: Backbone + Head (Structured)

**What:** User submits both backbone and head in standardized format

**Format:**
```json
{
  "submission_id": "user456-antiberta-mlp",
  "backbone": {
    "name": "antiberta",
    "model_path": "alchemab/antiberta2",
    "revision": "main",
    "device": "cuda"
  },
  "head": {
    "type": "mlp",
    "config": {
      "hidden_size": 128,
      "dropout": 0.1,
      "activation": "relu"
    },
    "weights": "head_weights.npz"
  },
  "metadata": {...}
}
```

**Pros:**
- Structured (easy to parse and compare)
- Modular (can mix-and-match backbone + head)
- Fast evaluation (if backbone already cached)

**Cons:**
- Requires you to support that backbone
- Less flexible than Docker

**Use case:** "I want to test if AntiBERTa + MLP beats ESM-1v + LogReg"

---

##### Leaderboard Will Show:

| Rank | Model | Backbone | Head | Jain Acc | Harvey Acc | Params | Inference Time |
|------|-------|----------|------|----------|------------|--------|----------------|
| 1 | Fine-tuned-ESM2 | ESM2 (tuned) | Linear | **69.5%** | 63.1% | 650M | 2.3s/seq |
| 2 | Your-ESM2-MLP | ESM2 | MLP | 68.1% | 62.3% | 650M | 2.1s/seq |
| 3 | AntiBERTa-LogReg | AntiBERTa | LogReg | 67.2% | 61.9% | 420M | 1.8s/seq |
| 4 | Novo-baseline | ESM-1v | LogReg | 66.3% | 61.5% | 650M | 2.0s/seq |
| 5 | AbLang-XGBoost | AbLang | XGBoost | 65.8% | 60.2% | 120M | 0.5s/seq |

**Users can filter/sort by:**
- Backbone (compare ESM-1v vs ESM2 vs AntiBERTa)
- Head (compare LogReg vs MLP vs attention)
- Accuracy (best overall model)
- Inference speed (fastest model)
- Model size (smallest model)

---

##### Your Implementation Strategy:

**Phase 1 (Launch):**
- Support Option 1 only (Docker submissions)
- Manual review before adding to leaderboard
- Proves concept, gets first external submissions

**Phase 2 (Scale):**
- Add Option 2 (head weights only)
- Pre-compute embeddings for ESM-1v, ESM2, AntiBERTa
- Cache on disk (S3/HuggingFace Datasets)
- Fast evaluation: load embeddings → load head weights → predict

**Phase 3 (Mature):**
- Add Option 3 (structured backbone + head)
- Auto-evaluation pipeline
- Public API for programmatic submission

---

##### Why This Matters:

**For science:**
- Answers: "Which backbone is best?" (holding head constant)
- Answers: "Which head is best?" (holding backbone constant)
- Answers: "Does fine-tuning beat frozen embeddings?"

**For users:**
- Easy to submit (just head weights = 10KB upload)
- Easy to compare (standardized format)
- Easy to reproduce (public weights + configs)

**For you:**
- Efficient evaluation (cached embeddings)
- Structured data (easy to analyze)
- Compelling leaderboard (many combinations)

---

### Phase 2: Expand Model Zoo (2-3 weeks)
**Goal:** Populate leaderboard, establish model diversity

1. Implement multi-PLM support (Option 2)
2. Add Hydra + W&B for experiment tracking (Option 3)
3. Benchmark ESM2, AntiBERTa, AbLang
4. Publish results to leaderboard
5. Write blog post: "Which PLM is Best for Antibody Prediction?"

**Deliverables:**
- 5+ models on leaderboard
- Published comparison paper/blog
- Experiment tracking infrastructure

---

### Phase 3: Production API (2-3 weeks)
**Goal:** Enable real-world usage

1. Publish models to HuggingFace Model Hub (Option 4)
2. Create `antibody-predictor` inference package
3. Build Streamlit UI (Option 5)
4. Optional: FastAPI server for enterprise
5. Write usage tutorials + examples

**Deliverables:**
- `pip install antibody-predictor` working
- Public Streamlit demo
- HuggingFace Model Hub integration
- API documentation

---

### Phase 4: Advanced Features (Future Vision)
**Goal:** Transform into a drug discovery platform

**Timeline:** 6-12 months (after Phase 1-3 established)

---

#### 4.1: Drug Discovery Integration

**Sequence Triage & Optimization:**
1. **Multi-chain scoring** (VH+VL pairs)
   - Predict polyreactivity for paired sequences
   - Flag high-risk candidates early in pipeline
   - Batch processing for 1000s of variants

2. **Mutagenesis suggestions**
   - Gradient-based saliency maps (which residues contribute to polyreactivity)
   - Monte Carlo mutagenesis (suggest mutations to reduce non-specificity)
   - Conservation-aware edits (don't break CDR function)

3. **Multi-criteria ranking**
   - Polyreactivity score (your models)
   - Stability prediction (via external tools)
   - Expression yield estimates (ThermoFisher/STRIDE integrations)
   - Manufacturability filters (PTM sites, aggregation)
   - Output: ranked list optimized for pharma constraints

**Use case:** "I have 500 antibody candidates. Rank them by polyreactivity + stability + expressibility."

**Implementation:**
- Extend inference API with multi-objective scoring
- Integrate with Rosetta, FoldX, or stability predictors
- Add visualization: Pareto frontier plots (polyreactivity vs stability)

**Timeline:** 4-6 weeks

**Impact:** ⭐⭐⭐⭐⭐ (Pharma adoption, revenue potential)

---

#### 4.2: Structure-Aware Models

**Vision:** Move beyond sequence-only to 3D structure

**What it enables:**
1. **AlphaFold2/ESMFold integration**
   - Predict structure from sequence
   - Extract 3D descriptors (surface charge, hydrophobic patches)
   - Combine sequence embeddings + structural features

2. **Graph Neural Networks (GNNs)**
   - Model antibody as graph (residues = nodes, contacts = edges)
   - Learn spatial patterns (not just sequence patterns)
   - Could capture aggregation-prone patches

3. **Transformer fine-tuning**
   - Fine-tune ESM2 on antibody-specific task (not just embeddings + LogReg)
   - End-to-end learning (sequence → prediction)
   - May capture subtle patterns LogReg misses

**Research questions:**
- Does structure info improve beyond sequence?
- Do GNNs beat transformers for antibodies?
- Can we predict polyreactivity from structure alone?

**Implementation:**
1. Add AlphaFold/ESMFold structure prediction
2. Extract 3D descriptors (DSSP, surface properties)
3. Implement GNN baseline (PyTorch Geometric)
4. Fine-tune ESM2 on antibody data
5. Benchmark: sequence-only vs structure-aware

**Timeline:** 8-12 weeks (research project)

**Impact:** ⭐⭐⭐⭐ (Publishable, SOTA potential)

---

#### 4.3: Community Dataset Curation

**Vision:** Crowdsource novel antibody datasets

**What it enables:**
1. **Dataset submission portal**
   - Labs upload ELISA/PSR/BLI data
   - Standardized template (CSV with sequence + label + assay metadata)
   - Auto-validation + ingestion into benchmark

2. **Living literature tracker**
   - Maintain `docs/research/benchmarks.md` with SOTA results
   - Track new papers, datasets, methods
   - Update quarterly with "State of Antibody ML" report

3. **Novel assay types**
   - Expand beyond ELISA/PSR (e.g., BLI, SPR, immunogenicity)
   - Multi-task learning (predict across assays)
   - Meta-learning (transfer from one assay to another)

**Use case:** "I just published a paper with 200 new antibodies. How do I add them to the benchmark?"

**Implementation:**
1. Create submission template (CSV + metadata JSON)
2. Build validation pipeline (sequence QC, label checks)
3. Auto-add to leaderboard as new test set
4. Credit submitters in docs + papers

**Timeline:** 2-3 weeks

**Impact:** ⭐⭐⭐⭐ (Community growth, dataset diversity)

---

#### 4.4: Interpretability Suite

**Vision:** Explain model predictions at residue-level

**What it enables:**
1. **Per-residue contribution heatmaps**
   - Which residues drive polyreactivity predictions?
   - Integrated gradients or SHAP on embeddings
   - Visualize on antibody structure (if available)

2. **Cross-dataset diagnostics**
   - Why does model work on Jain but fail on Harvey?
   - Family-level performance analysis
   - Failure mode clustering (which sequences confuse model?)

3. **Interactive antibody inspector**
   - Streamlit/HF Space plugin
   - Upload sequence → see residue-level explanations
   - Compare predictions across models
   - Download reports for regulatory submissions

**Use case:** "Why does your model predict this antibody is polyreactive? Which residues are responsible?"

**Implementation:**
1. Add SHAP/IntegratedGradients to inference pipeline
2. Build visualization tools (matplotlib + 3D structure viewer)
3. Create Streamlit dashboard for exploration
4. Document interpretability methods in user guide

**Timeline:** 3-4 weeks

**Impact:** ⭐⭐⭐⭐ (Trust, regulatory acceptance)

---

#### 4.5: AutoML & Advanced Experiment Management

**Vision:** Automated model discovery

**What it enables:**
1. **Hyperparameter optimization at scale**
   - Optuna + Ray Tune for distributed sweeps
   - Multi-objective optimization (accuracy + inference speed)
   - Neural architecture search (find best classifier head)

2. **Model zoo with full provenance**
   - Every experiment tracked (Hydra + W&B + MLflow)
   - Reproducible configs + artifacts
   - Public dashboard: all models, all hyperparameters, all results

3. **One-click retraining**
   - `make leaderboard-submit` → trains, evaluates, uploads
   - CI/CD for model updates (new data triggers retraining)
   - Automatic deployment to production API

**Use case:** "I want to find the best ESM2 model without manual tuning."

**Implementation:**
- Already in Phase 2 (Option 3: Hydra + W&B)
- Extend with Optuna for AutoML
- Add Ray Tune for distributed sweeps

**Timeline:** Already planned for Phase 2

---

## Updated Rollout: 4-Phase Strategy

### Phase 1: Benchmark (3-4 weeks) ✅ Priority
- Leaderboard + submission system
- Baseline models (ESM-1v)

### Phase 2: Model Zoo (2-3 weeks)
- Multi-PLM support (ESM2, AntiBERTa)
- Hydra + W&B experiment tracking

### Phase 3: Production API (2-3 weeks)
- Inference package + HF Model Hub
- Streamlit demo + FastAPI server

### Phase 4: Advanced Features (6-12 months)
**4.1** Drug discovery integration (4-6 weeks)
**4.2** Structure-aware models (8-12 weeks)
**4.3** Community curation (2-3 weeks)
**4.4** Interpretability suite (3-4 weeks)
**4.5** AutoML (extends Phase 2)

---

## Why This Order?

**Phase 1 first** because:
- Creates community visibility immediately
- Establishes you as benchmark standard
- All other work feeds into the leaderboard

**Phase 2 next** because:
- Makes leaderboard interesting (multiple models)
- Generates research content (comparisons)
- Requires experiment infrastructure (Hydra)

**Phase 3 last** because:
- Requires stable models from Phase 2
- Inference is easier once you know which models work
- API can reference leaderboard results

---

## About Hydra ✅ **v0.4.0 - SHIPPED**

**Status:** ✅ **Implemented in v0.4.0** (November 2025)

**What was shipped:**
- ✅ Experiment sweeps: `antibody-train --multirun model=esm1v,esm2 classifier.C=0.1,1.0,10.0`
- ✅ Config composition: Mix and match model/classifier/dataset configs
- ✅ CLI overrides: `antibody-train model.batch_size=16` (no file editing)
- ✅ Reproducibility: Every run saves config snapshot in `outputs/`
- ✅ Structured configs: Type-safe dataclasses with validation

**What's enabled now:**
- ✅ Can run 50+ experiments with multirun
- ✅ Can compare multiple PLMs systematically (infrastructure ready)
- ⏳ Hyperparameter optimization with Optuna (TODO - easy to add)

**Next steps:**
- Add ESM2, AntiBERTa configs (just create YAML files)
- Integrate Optuna for advanced hyperparameter search
- Add W&B logging for experiment visualization

---

## Distribution Strategy

### Should We NPM Install This?

**No.** This is Python ML, not JavaScript.

**Current:** `pip install antibody-training-esm` (works via uv/pip)

**Phase 3:** `pip install antibody-predictor` (lightweight inference-only)

**Best:** HuggingFace Model Hub (no install, just `from_pretrained()`)

### What Would HuggingFace Do?

1. ✅ Publish models to Model Hub
2. ✅ Create HuggingFace Spaces (leaderboard + Streamlit demo)
3. ✅ Write blog posts + documentation
4. ✅ Engage community (Twitter, forums)
5. ✅ Open-source everything

---

## Success Metrics

**Phase 1 (Benchmark):**
- 10+ external model submissions
- 500+ leaderboard views/month
- 3+ citations in papers

**Phase 2 (Model Zoo):**
- 5+ models benchmarked
- Published comparison blog/paper
- 50+ GitHub stars

**Phase 3 (API):**
- 1000+ pip installs/month
- 10+ pharma/academic users
- Featured on HuggingFace trending

**Phase 4 (Advanced Features):**
- 5+ pharma partnerships (drug discovery integration)
- 2+ published papers (structure-aware models, GNNs)
- 20+ community-contributed datasets
- 100+ interpretability reports generated
- Industry awards/recognition (e.g., Nature Biotech spotlight)

---

## Open Questions

### Phase 1-3 Questions:
1. **Benchmark scope:** Start antibody-only or expand to general protein prediction?
2. **Test set privacy:** Keep test sets private (like ImageNet) or public?
3. **Submission format:** Docker-only or also support HF models?
4. **Compute resources:** How to fund evaluation compute for submissions?
5. **Governance:** Who maintains the benchmark long-term?

### Phase 4 Questions:
6. **Structure prediction:** AlphaFold2 (slow, accurate) or ESMFold (fast, less accurate)?
7. **GNN architecture:** Which graph representation? (k-NN, distance cutoff, learned edges)
8. **Mutagenesis strategy:** Gradient-based or evolutionary search?
9. **Pharma partnerships:** Which stability predictors to integrate? (Rosetta, FoldX, commercial tools)
10. **Interpretability standards:** What level of explanation satisfies regulatory requirements?
11. **Multi-objective optimization:** How to weight polyreactivity vs stability vs expressibility?
12. **Dataset diversity:** How to incentivize labs to contribute proprietary data?

---

## Next Steps

See `NEXT_STEPS.md` for immediate actionable tasks.

**Decision needed:** Which phase to start with? (Recommendation: Phase 1)
