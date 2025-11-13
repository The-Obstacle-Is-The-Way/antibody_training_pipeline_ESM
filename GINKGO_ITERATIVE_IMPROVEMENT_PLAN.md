# Ginkgo 2025: Iterative Improvement Plan

**Current Status:**
- Our Ridge Baseline: **0.507 Spearman**
- LightGBM v1 (conservative): **0.50664 Spearman** (-0.07%)
- Leader: **0.89 Spearman**
- Gap to Close: **-38%**

**Date:** 2025-11-13

---

## 🎯 Goal Hierarchy

1. **Primary Goal:** Beat our Ridge baseline (0.507 → 0.52+) [+2.5%]
2. **Stretch Goal:** Reach 0.60 Spearman [+18%]
3. **Dream Goal:** Close gap to leader (0.89)

---

## 📋 Strategy Matrix: 5 Parallel Tracks

### Track 1: GBDT Hyperparameter Tuning 🔥 (HIGHEST PRIORITY)

**Why:** LightGBM v1 was too conservative and early stopping was broken

#### Iteration 1A: More Aggressive LightGBM
```python
# Current (failed)
num_leaves=16, max_depth=4, learning_rate=0.01

# New (aggressive)
num_leaves=64, max_depth=7, learning_rate=0.03, n_estimators=2000
```
**Expected:** 0.51-0.54 (+2-7%)
**Runtime:** ~5 min
**Risk:** Low (just tuning)

#### Iteration 1B: XGBoost Alternative
```python
# Different regularization approach
xgb.XGBRegressor(
    max_depth=6,
    learning_rate=0.03,
    n_estimators=2000,
    subsample=0.8,
    colsample_bytree=0.8,
)
```
**Expected:** 0.51-0.53 (+1-5%)
**Runtime:** ~5 min
**Risk:** Low

#### Iteration 1C: CatBoost (State-of-art 2025)
```python
# Best tabular model for small-N
CatBoostRegressor(
    iterations=1000,
    depth=6,
    learning_rate=0.03,
    l2_leaf_reg=3.0,
)
```
**Expected:** 0.52-0.55 (+3-8%)
**Runtime:** ~5 min
**Risk:** Low

---

### Track 2: Feature Engineering 🛠️

**Why:** Dense 2304D embeddings might need dimensionality reduction or augmentation

#### Iteration 2A: PCA Dimensionality Reduction
```python
# Reduce embeddings to TabPFN sweet spot
pca = PCA(n_components=256)
X_reduced = pca.fit_transform(embeddings)
# Then use LightGBM/XGBoost
```
**Expected:** 0.50-0.52 (±2%)
**Runtime:** ~3 min
**Risk:** Medium (might lose info)

#### Iteration 2B: Hand-Crafted Features + Embeddings
```python
# Add to embeddings:
features = [
    'vh_length', 'vl_length',
    'vh_hydrophobicity', 'vl_hydrophobicity',
    'vh_charge', 'vl_charge',
    'cdr_h3_length', 'cdr_l3_length',
    'vh_vl_charge_interaction',  # vh_charge * vl_charge
]
X_combined = np.concatenate([embeddings, features], axis=1)
```
**Expected:** 0.51-0.53 (+1-5%)
**Runtime:** ~10 min (need to compute features)
**Risk:** Medium (feature engineering is hit-or-miss)

#### Iteration 2C: Per-Fragment Ensemble
```python
# Train separate models on VH, VL, CDRs, FWRs
# Ensemble predictions with learned weights
models = {
    'VH': LGBMRegressor(...),
    'VL': LGBMRegressor(...),
    'CDRs': LGBMRegressor(...),
}
# Meta-learner combines predictions
```
**Expected:** 0.52-0.56 (+3-10%)
**Runtime:** ~15 min
**Risk:** High (complex, might overfit)

---

### Track 3: Embedding Optimization 🧬

**Why:** Current embeddings might not be optimal for polyreactivity prediction

#### Iteration 3A: p-IgGen Weight Tuning
```python
# Current: 62.8% ESM-1v + 36.2% p-IgGen
# Try different ratios:
weights = [
    (0.5, 0.5),   # Equal
    (0.8, 0.2),   # More ESM-1v
    (0.4, 0.6),   # More p-IgGen
    (0.7, 0.3),   # Rebalance
]
```
**Expected:** 0.50-0.52 (+1-3%)
**Runtime:** ~2 min per ratio (embeddings cached)
**Risk:** Low

#### Iteration 3B: Add ESM-2 Back (3-Model Ensemble)
```python
# ESM-1v (50%) + p-IgGen (30%) + ESM-2 (20%)
# More diversity = better generalization?
```
**Expected:** 0.51-0.54 (+2-6%)
**Runtime:** ~10 min (need ESM-2 embeddings)
**Risk:** Medium

#### Iteration 3C: Fragment-Specific Embeddings
```python
# Extract embeddings for CDRs only, FWRs only
# Hypothesis: Different regions need different attention
```
**Expected:** 0.52-0.55 (+3-8%)
**Runtime:** ~20 min (new embeddings)
**Risk:** High (experimental)

---

### Track 4: Tessier Transfer Learning 🔬 (RISKY BUT HIGH UPSIDE)

**Why:** 373k Tessier sequences could provide massive pre-training signal

#### Iteration 4A: Pre-train Regression Head on Tessier
```python
# Step 1: Train binary classifier on 298k Tessier
# Step 2: Use learned weights to initialize GDPa1 regressor
# Hypothesis: Polyreactivity patterns transfer even if labels differ
```
**Expected:** 0.48-0.60 (-5% to +18%)
**Runtime:** ~30 min
**Risk:** VERY HIGH (might hurt performance)

**Pros:**
- Massive dataset (298k vs 197)
- Same assay type (CHO cell binding)
- Could learn general polyreactivity features

**Cons:**
- Binary vs continuous labels (fundamental mismatch)
- Different label semantics (threshold vs gradient)
- Might confuse the model

#### Iteration 4B: Meta-Learning Approach
```python
# Use Tessier to learn "how to learn" polyreactivity
# Then fine-tune on GDPa1 with small learning rate
```
**Expected:** 0.50-0.62 (±2% to +22%)
**Runtime:** ~45 min
**Risk:** EXTREME (very experimental)

---

### Track 5: Advanced Architectures 🚀 (MOONSHOT)

**Why:** Ridge/GBDT might be too simple for this problem

#### Iteration 5A: Small MLP on Embeddings
```python
# Tiny neural network (2-3 layers)
nn.Sequential(
    nn.Linear(2304, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
)
# Train with heavy regularization
```
**Expected:** 0.49-0.55 (-3% to +9%)
**Runtime:** ~15 min
**Risk:** HIGH (easy to overfit on N=197)

#### Iteration 5B: TabPFN v2.5 (Foundation Model)
```python
# Pre-trained transformer for tabular data
# SOTA on small-N regression tasks
from tabpfn import TabPFNRegressor
model = TabPFNRegressor()
model.fit(X_reduced, y)
```
**Expected:** 0.52-0.58 (+3-15%)
**Runtime:** ~5 min
**Risk:** MEDIUM (requires library install)

---

## 🎬 Execution Plan: First 3 Iterations (Parallel)

### Batch 1: Low-Risk GBDT Tuning (Run in parallel)

**Iteration 1A:** Aggressive LightGBM
```bash
tmux new -s lightgbm_aggressive "python scripts/generate_lightgbm_aggressive.py"
```
- `num_leaves=64, max_depth=7, lr=0.03`
- ETA: 5 min
- Expected: 0.52-0.54

**Iteration 1B:** XGBoost
```bash
tmux new -s xgboost_v1 "python scripts/generate_xgboost_submission.py"
```
- Conservative XGBoost params
- ETA: 5 min
- Expected: 0.51-0.53

**Iteration 1C:** CatBoost
```bash
tmux new -s catboost_v1 "python scripts/generate_catboost_submission.py"
```
- Default CatBoost with CV
- ETA: 5 min
- Expected: 0.52-0.55

**Total runtime:** ~5 min (parallel)
**Best case:** One model reaches 0.54 → **+6.5%** improvement!

---

### Batch 2: Feature Engineering (Sequential)

Only run if Batch 1 disappoints (all < 0.51)

**Iteration 2A:** PCA + LightGBM
```bash
python scripts/generate_pca_lightgbm.py
```
- ETA: 3 min
- Expected: 0.50-0.52

**Iteration 2B:** Hand-crafted features
```bash
python scripts/generate_features_ensemble.py
```
- ETA: 10 min
- Expected: 0.51-0.53

---

### Batch 3: Tessier Transfer (Last Resort)

Only run if Batch 1 + 2 fail (all < 0.52)

**Iteration 4A:** Tessier pre-training
```bash
python scripts/tessier_transfer_learning.py
```
- ETA: 30 min
- Expected: 0.48-0.60 (high variance)
- RISKY: Might hurt performance

---

## 📊 Success Criteria

| Threshold | Outcome | Next Action |
|-----------|---------|-------------|
| **> 0.54** | 🎉 MAJOR WIN | Submit! Move to ensemble experiments |
| **0.52-0.54** | ✅ Success | Submit, explore Track 3 (embeddings) |
| **0.51-0.52** | ⚠️ Marginal | Try Track 2 (feature engineering) |
| **< 0.51** | ❌ Failed | Pivot to Tessier transfer (risky) |

---

## 🛠️ Implementation Checklist

### Batch 1 Scripts (Create 3 files)
- [ ] `scripts/generate_lightgbm_aggressive.py` (copy from v1, update params)
- [ ] `scripts/generate_xgboost_submission.py` (new)
- [ ] `scripts/generate_catboost_submission.py` (new)

### Batch 2 Scripts
- [ ] `scripts/generate_pca_lightgbm.py` (new)
- [ ] `scripts/generate_features_ensemble.py` (new)

### Batch 3 Scripts
- [ ] `scripts/tessier_transfer_learning.py` (new)

---

## 📝 Notes

**Why This Order?**
1. **GBDT tuning first** → Fastest, highest success rate, low risk
2. **Feature engineering second** → More work, moderate risk
3. **Tessier transfer last** → Slow, high risk, but huge upside

**Parallelization Strategy:**
- Batch 1: 3 tmux sessions (5 min wall time)
- Batch 2: Sequential (stop if Batch 1 wins)
- Batch 3: Only if desperate

**Expected Timeline:**
- Best case: 5 min (Batch 1 wins)
- Typical case: 15 min (Batch 1 + Batch 2)
- Worst case: 45 min (all 3 batches)

---

## 🎯 Final Thoughts

**Most Likely to Win:** Track 1 (GBDT tuning)
**Highest Upside:** Track 4 (Tessier transfer)
**Safest Bet:** Track 1C (CatBoost)

Let's start with Batch 1 (3 parallel GBDT experiments) and see what happens!
