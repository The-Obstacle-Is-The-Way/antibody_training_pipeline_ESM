# Ginkgo 2025 Competition: Improvement Roadmap

**Status:** 3rd place (0.500 Spearman) | Leader: 0.504 | Gap: -0.8%

**Last Updated:** 2025-11-13

---

## 🎯 Current Best Model

**Model:** ESM-1v (62.8%) + p-IgGen (36.2%) ensemble with Ridge regression (α=5.5)

**Performance:**
- Mean per-fold Spearman: **0.50043**
- Per-fold breakdown: [0.478, 0.556, 0.672, 0.216, 0.580]
- **Problem:** Fold 3 (0.216) is killing overall score

**Files:**
- Script: `scripts/generate_optimal_submission.py`
- Submissions: `ginkgo_submissions_optimal/`
- Results: `experiment_results/top3_experiments_results.csv`

---

## 📊 What We Tried (And What Worked/Failed)

### ✅ Successes

1. **VH+VL Concatenation** (0.465 → 0.476)
   - Key insight: Full antibody context beats individual chains

2. **Two-Model Ensemble** (0.476 → 0.486)
   - ESM-1v (70%) + ESM-2 (30%)
   - Different embeddings = uncorrelated errors

3. **p-IgGen Addition** (0.486 → 0.500)
   - Antibody-specific PLM beats general protein PLM
   - Optimal weights: 62.8% ESM-1v + 36.2% p-IgGen

4. **Alpha Micro-tuning** (α=7.0 → α=5.5)
   - Fine-grained regularization tuning: +0.004 gain

### ❌ Failures

1. **Boughter Transfer Learning** (0.500 → 0.491, **-1.93%**)
   - **Why it failed:**
     - ELISA (binary) ≠ PR_CHO (continuous)
     - Different assays, different antigens
     - Label distribution mismatch
   - **Lesson:** External data must match assay type AND label semantics

2. **Combined Training** (0.500 → 0.461, **-7.88%**)
   - **Why worse than transfer:**
     - Data imbalance: 914 Boughter + 197 GDPa1 = 85% Boughter
     - Ridge loss dominated by wrong patterns
     - No chance to adapt to PR_CHO
   - **Lesson:** Don't merge incompatible datasets

---

## 🔬 Root Cause Analysis: Why We're Stuck at 0.500

### Problem 1: Fold 3 Catastrophe (0.216 Spearman)

Other folds: 0.48-0.67 | Fold 3: 0.216

**If we could improve Fold 3 from 0.216 → 0.27:**
```python
mean([0.478, 0.556, 0.672, 0.270, 0.580]) = 0.511  # BEATS LEADER!
```

**Possible causes:**
- Fold 3 has different antibody types (nanobodies? unusual CDRs?)
- Ridge is too linear to capture Fold 3 patterns
- Embeddings don't capture Fold 3 features

**Next step:** Investigate Fold 3 antibodies specifically

### Problem 2: Ridge is Too Simple

**What Ridge does:**
```python
prediction = w1*embedding[0] + w2*embedding[1] + ... + w1280*embedding[1279]
```

**What it CAN'T do:**
- Capture nonlinear interactions (e.g., "VH hydrophobicity × VL charge")
- Handle feature importance (treats all embedding dims equally)
- Adapt to heterogeneous folds (same weights for all folds)

**Solution:** Try better heads (see below)

---

## 🚀 Phase 2: Better Regression Heads (Nov 2025 Edition)

### Why Better Heads Matter

Current setup:
```
Frozen PLM (ESM-1v/p-IgGen) → Dense embeddings (2304D) → Ridge → Prediction
```

**Ridge bottleneck:**
- Linear model on high-dim dense features
- No feature selection, no nonlinearity
- State-of-art for tabular regression in 2025 is NO LONGER Ridge

### The 2025 Hierarchy (Best → Simplest)

#### 1. **TabPFN v2.5** (Foundation Model for Tabular) 🔥 BLEEDING EDGE

**What:** Transformer pre-trained on 10⁸ synthetic tabular datasets, SOTA on small-N regression

**Why for us:**
- Built for N=1k regime (our sweet spot!)
- Beats tuned GBDTs on small tabular benchmarks
- Zero/minimal tuning required
- "ESM + TabPFN v2.5" is a legit 2025 combo (Nature-paper flex)

**Implementation:**
```python
from tabpfn import TabPFNRegressor

# Reduce embeddings to TabPFN's sweet spot
pca = PCA(n_components=256)
X_reduced = pca.fit_transform(embeddings)

model = TabPFNRegressor()
model.fit(X_reduced, y)
```

**Expected gain:** +2-5% (handles nonlinearity + small-N regime)

**References:**
- Paper: https://arxiv.org/abs/2207.01848
- v2.5 release: Nov 2025
- GitHub: https://github.com/automl/TabPFN

---

#### 2. **LightGBM** (Gradient Boosting) 🥇 MOST LIKELY WIN

**What:** Gradient-boosted decision trees, current king of tabular ML

**Why for us:**
- Captures nonlinear feature interactions
- Handles weird marginal distributions
- Proven winner on small-medium tabular benchmarks
- Will likely fix Fold 3 (can learn fold-specific patterns)

**Implementation:**
```python
from lightgbm import LGBMRegressor

model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    num_leaves=31,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.6,
    reg_alpha=1.0,
    reg_lambda=1.0,
    random_state=42
)

model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          early_stopping_rounds=50,
          verbose=False)
```

**Expected gain:** +3-7% (my money pick)

**Tuning strategy:**
- Start conservative: `num_leaves=16, max_depth=4`
- Use early stopping on GDPa1 folds
- Grid search: {num_leaves: [16, 31, 64], max_depth: [4, 5, 7]}

---

#### 3. **ElasticNet** (Ridge++) 🏃 FASTEST TO TRY

**What:** Ridge with L1 + L2 regularization (automatic feature selection)

**Why for us:**
- Almost free to implement (swap Ridge → ElasticNet)
- Knocks out noisy embedding dimensions
- Keeps linear model robustness
- If this doesn't beat Ridge, we know linear is at ceiling

**Implementation:**
```python
from sklearn.linear_model import ElasticNetCV

model = ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
    alphas=None,  # auto CV
    cv=5,
    max_iter=10000,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)
```

**Expected gain:** +0-2% (diagnostic test)

**If ElasticNet doesn't beat Ridge by +0.01:**
→ Linear models are maxed out, move to LightGBM/TabPFN

---

### Recommended Execution Order

```
Step 1: ElasticNet (30 min)
  ↓
If +0.01 → Submit | If not → Continue
  ↓
Step 2: LightGBM (2 hours)
  ↓
If +0.02 → Submit | If not → Continue
  ↓
Step 3: TabPFN v2.5 (4 hours - install + tune)
  ↓
Submit best
```

**Parallel option:** Run all 3 in separate tmux sessions, compare results

---

## 📦 New Datasets Discovered

### 1. Tessier Lab Cell Reports 2024 ✅ DOWNLOADED

**Paper:** "Human antibody polyreactivity is governed primarily by the heavy-chain complementarity-determining regions" (Oct 2024)

**Location:** `external_datasets/tessier_2024_polyreactivity/`

**Data:**
- **246,295 antibodies** with polyreactivity labels
  - Positive (polyreactive): 115,039
  - Negative (specific): 131,256
- CHO cell-based assays (SCP60, SMP60)
- Features: Biochemical descriptors (charge, hydrophobicity, etc.)
- Sequences: In supplemental datasets (Excel files)

**Why better than Boughter:**
- ✅ CHO cell assay (same as GDPa1's PR_CHO!)
- ✅ Massive dataset (246k vs 914)
- ❓ Labels might still be binary (need to check)

**Status:** Downloaded, needs preprocessing

**Next steps:**
1. Extract VH/VL sequences from Excel files
2. Check if labels are continuous or binary
3. If continuous CHO scores exist → THIS IS THE GOLD DATASET
4. If binary → Same problem as Boughter (but bigger)

### 2. Zenodo Big Dataset (5.8 GB) 🔍 NOT DOWNLOADED

**URL:** https://zenodo.org/doi/10.5281/zenodo.13387056

**Contents:** Unknown (need to download to inspect)

**Next step:** Download and inspect if Tessier dataset doesn't work

---

## 🧹 Repo Cleanup Plan

### Current Mess

```
├── ginkgo_submissions/              # Baseline (0.472)
├── ginkgo_submissions_ensemble/     # ESM-1v + ESM-2 (0.486)
├── ginkgo_submissions_esm1v/        # ESM-1v only
├── ginkgo_submissions_final/        # ???
├── ginkgo_submissions_optimal/      # CURRENT BEST (0.500)
├── experiment_results.csv           # Root file (wrong location)
├── experiment_results/              # Boughter transfer logs
└── combined_datasets/               # Boughter + GDPa1 merge
```

### Target Clean Structure

```
experiments/ginkgo_2025/
├── submissions/
│   ├── 01_baseline_esm1v/           # 0.472
│   ├── 02_ensemble_esm2/            # 0.486
│   ├── 03_optimal_piggen/           # 0.500 (CURRENT BEST)
│   └── 04_boughter_transfer/        # 0.491 (FAILED)
├── results/
│   ├── rapid_experiments.csv
│   ├── top3_experiments.csv
│   └── boughter_transfer.csv
└── logs/
    ├── rapid_experiments.txt
    ├── top3_experiments.txt
    └── boughter_transfer.txt

external_datasets/
└── tessier_2024_polyreactivity/     # 246k antibodies (CHO assay)

combined_datasets/                    # Boughter + GDPa1 (archived)
├── README.md                         # Document why transfer failed
├── boughter_training.csv
├── ginkgo_labeled.csv
└── boughter_ginkgo_combined.csv

train_datasets/ginkgo/                # GDPa1 competition data
├── GDPa1_v1.2_sequences.csv
└── GDPa1_v1.2_20250814.csv

test_datasets/ginkgo/                 # Private test set
└── heldout-set-sequences.csv
```

### Git Strategy

**✅ COMMIT:**
- All scripts (`.py`, `.sh`)
- Combined datasets (Boughter is ours)
- Results CSVs (small, useful)
- Documentation (`.md` files)
- External datasets (Tessier is public, MIT license)

**❌ .gitignore (DO NOT COMMIT):**
- GDPa1 raw data (`train_datasets/ginkgo/*.csv`)
- GDPa1 test set (`test_datasets/ginkgo/*.csv`)
- Submission CSVs (regenerate on demand)
- Embedding caches (too large, regenerate)
- tmux logs (`.txt` in experiments/)

**Reasoning:**
- GDPa1 is competition data (respect their terms)
- Embeddings are 100+ MB (use cache locally)
- Submissions are derivatives (code is source of truth)

---

## 🎯 Concrete Next Actions

### Immediate (Today)

1. **Clean repo structure**
   - Move files to `experiments/ginkgo_2025/`
   - Update `.gitignore`
   - Commit clean structure

2. **Implement ElasticNet head**
   - Copy `generate_optimal_submission.py` → `generate_elasticnet_submission.py`
   - Swap Ridge → ElasticNetCV
   - Run in tmux, compare to 0.500

3. **If ElasticNet fails, implement LightGBM**
   - Create `generate_lightgbm_submission.py`
   - Start conservative: `num_leaves=16, max_depth=4`
   - Run with early stopping

### Short-term (This Week)

4. **Preprocess Tessier dataset**
   - Extract sequences from Excel files
   - Check label distribution (binary vs continuous)
   - If continuous CHO scores → retry transfer learning
   - If binary → archive and move on

5. **Investigate Fold 3**
   - Extract Fold 3 antibodies
   - Analyze: CDR lengths, charge, hydrophobicity
   - Check if Fold 3 has different characteristics
   - Try fold-specific model if patterns found

6. **Implement TabPFN v2.5**
   - Install: `pip install tabpfn`
   - PCA embeddings to 256D
   - Run TabPFNRegressor with defaults
   - Compare to LightGBM

### Medium-term (Next Week)

7. **Ensemble all heads**
   - ElasticNet + LightGBM + TabPFN
   - Optimize weights with scipy
   - Final push to beat 0.504

8. **Prepare final submission**
   - Document best model in detail
   - Generate clean submission CSVs
   - Write reproducibility instructions
   - Submit to leaderboard

---

## 📚 Key Lessons Learned

### 1. Transfer Learning Needs Label Alignment

**Failed:** Boughter (binary ELISA) → GDPa1 (continuous PR_CHO)

**Why:** Different assays measure different things
- ELISA: Binding to specific antigens (DNA, insulin, LPS, ...)
- PR_CHO: Polyreactivity against CHO cell lysate (different antigens!)

**Lesson:** External data must match:
1. ✅ Assay type (both polyreactivity)
2. ✅ Assay readout (both CHO cells)  ← Tessier dataset!
3. ✅ Label distribution (both continuous)
4. ✅ Antibody format (both VH+VL IgGs, not nanobodies)

### 2. Data Imbalance Kills Combined Training

**Formula:**
```
If external_data >> target_data:
  → Loss function dominated by external patterns
  → Model never learns target task
```

**Solution:** Use transfer learning (pre-train → fine-tune) instead of combined training

### 3. Linear Models Have Ceilings

Ridge @ 0.500 is probably near the ceiling for linear models on these embeddings.

**Next frontier:** Nonlinear heads (LightGBM, TabPFN)

### 4. Fold 3 is the Key

Fold 3 (0.216) is 40% worse than other folds (0.48-0.67).

**If we fix Fold 3 → we beat the leader**

**Hypothesis:** Fold 3 has different antibody characteristics that Ridge can't capture

**Solution:** Investigate Fold 3 + use adaptive models (GBDT)

---

## 🔗 References

### Papers

1. **Sakhnini et al. (2025)** - Prediction of Antibody Non-Specificity using PLMs [bioRxiv]
2. **Boughter et al. (2020)** - Biochemical Patterns of Antibody Polyreactivity [eLife]
3. **Tessier Lab (2024)** - Human Antibody Polyreactivity Governed by Heavy Chain [Cell Reports]
4. **TabPFN (2022)** - Tabular Foundation Model [NeurIPS]

### Datasets

- **Boughter:** 914 antibodies, ELISA polyreactivity (binary)
- **GDPa1:** 246 antibodies, PR_CHO polyreactivity (continuous), 5 folds
- **Tessier:** 246k antibodies, CHO polyreactivity (binary?), public
- **Jain:** 86 antibodies, Novo parity benchmark
- **Harvey:** 141k nanobodies, PSR assay
- **Shehata:** 398 antibodies, PSR assay

### Tools

- **ESM-1v:** facebook/esm1v_t33_650M_UR90S_1
- **p-IgGen:** Exscientia/IgBert
- **ESM-2:** facebook/esm2_t33_650M_UR50D
- **LightGBM:** https://lightgbm.readthedocs.io/
- **TabPFN:** https://github.com/automl/TabPFN

---

## 🚦 Status Dashboard

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Current Score | 0.500 | 0.504 | 🟡 -0.8% |
| Fold 0 | 0.478 | 0.50 | 🟡 |
| Fold 1 | 0.556 | 0.50 | ✅ |
| Fold 2 | 0.672 | 0.50 | ✅ |
| **Fold 3** | **0.216** | **0.50** | **🔴 CRITICAL** |
| Fold 4 | 0.580 | 0.50 | ✅ |
| External Data | Tessier (246k) | CHO assay | ✅ Downloaded |
| Better Head | Ridge | GBDT/TabPFN | 🟡 Pending |
| Repo Cleanup | Messy | Clean | 🟡 In Progress |

**Next milestone:** Beat 0.504 with LightGBM or TabPFN

**Deadline:** Nov 17, 2025 (4 days remaining!)

---

**Last Updated:** 2025-11-13 07:30 PST
**Author:** Ray + Claude
**Version:** 2.0
