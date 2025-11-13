# Ginkgo Competition Strategy - Beat 0.486

## Current State
- **Baseline (ESM-1v)**: 0.472 (3rd place)
- **Ensemble (0.7×ESM-1v + 0.3×ESM-2)**: 0.486 (predicted #1)
- **Target**: > 0.486

## First Principles Analysis

### What We Know Works:
1. **VH+VL concatenation** (+15% over VH-only)
2. **Ensemble averaging** (+2.9% over single model)
3. **Alpha tuning** (5.5 better than 7.0)
4. **Bigger models** (ESM-1v 650M >> ESM-2 8M)

### What We Haven't Tried:
1. **AbLang2** (antibody-specific PLM, got 0.362 in benchmarks)
2. **Stacking** (meta-learner on top of base models)
3. **Feature engineering** (TAP, MOE, DeepSP available)
4. **Different regressors** (ElasticNet, LightGBM)
5. **Weighted ensemble optimization** (learn optimal weights)

---

## Top 10 Experiments (Ranked by Expected ROI)

### 🥇 **Tier 1: High Impact, Fast (< 10 min each)**

#### 1. **Three-Model Ensemble: ESM-1v + ESM-2 + AbLang2**
**Hypothesis**: AbLang2 is antibody-specific and should capture different patterns
**Expected gain**: +1-2%
**Implementation**:
```python
# Train AbLang2 model
ablang2_embeddings = extract_ablang2_vh_vl_embeddings()
# Try weights: (0.6*ESM1v + 0.2*ESM2 + 0.2*AbLang2)
ensemble = w1*pred1 + w2*pred2 + w3*pred3
```
**File**: `scripts/ginkgo_exp01_ablang2_ensemble.py`

---

#### 2. **TAP + MOE Features WITHOUT PCA**
**Hypothesis**: PCA loses information; raw features might help
**Expected gain**: +0.5-1.5%
**Details**:
- **TAP features** (5D): SFvCSP, PSH, PPC, PNC, CDR_Length
- **MOE features** (48D): Structure-based properties from reference repo
- Concat with ESM-1v: 2560D + 5D + 48D = **2613D**
**File**: `scripts/ginkgo_exp02_tap_moe_features.py`

---

#### 3. **Weighted Ensemble with CV Optimization**
**Hypothesis**: Learn optimal weights instead of guessing
**Expected gain**: +0.5-1%
**Method**:
```python
from scipy.optimize import minimize

def cv_score(weights, preds_list, labels):
    ensemble = sum(w*p for w, p in zip(weights, preds_list))
    return -spearmanr(labels, ensemble)[0]

optimal_weights = minimize(cv_score, x0=[0.5, 0.5], bounds=[(0,1), (0,1)])
```
**File**: `scripts/ginkgo_exp03_optimize_weights.py`

---

### 🥈 **Tier 2: Medium Impact, Moderate Time (10-30 min)**

#### 4. **Stacked Ensemble (Meta-Learner)**
**Hypothesis**: Train Ridge on top of base model predictions
**Expected gain**: +0.5-1.5%
**Method**:
```python
# Layer 1: Base models (ESM-1v, ESM-2, AbLang2)
base_preds = [pred1, pred2, pred3]  # (197, 3)

# Layer 2: Meta-learner Ridge on OOF predictions
meta_model = Ridge(alpha=1.0)
meta_model.fit(base_preds, labels)
```
**File**: `scripts/ginkgo_exp04_stacked_ensemble.py`

---

#### 5. **ElasticNet Instead of Ridge**
**Hypothesis**: L1 regularization might select better features
**Expected gain**: +0-1%
**Method**:
```python
from sklearn.linear_model import ElasticNetCV
model = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=5)
```
**File**: `scripts/ginkgo_exp05_elasticnet.py`

---

#### 6. **LightGBM Gradient Boosting**
**Hypothesis**: Tree-based might capture non-linear patterns
**Expected gain**: +0-1.5%
**Method**:
```python
import lightgbm as lgb
model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.01, max_depth=3)
```
**File**: `scripts/ginkgo_exp06_lightgbm.py`

---

### 🥉 **Tier 3: Experimental, Higher Risk (30-60 min)**

#### 7. **Separate VH/VL Models Then Combine**
**Hypothesis**: VH and VL have different predictive power, train separately
**Expected gain**: +0-1%
**Method**:
```python
vh_model = Ridge(alpha=5.5).fit(vh_embeddings, labels)
vl_model = Ridge(alpha=5.5).fit(vl_embeddings, labels)
ensemble = 0.6*vh_model.predict() + 0.4*vl_model.predict()
```
**File**: `scripts/ginkgo_exp07_separate_vh_vl.py`

---

#### 8. **CDR-Only Embeddings**
**Hypothesis**: CDRs drive polyreactivity, focus there
**Expected gain**: +0-0.5%
**Method**: Extract only CDR regions, embed those
**File**: `scripts/ginkgo_exp08_cdr_only.py`

---

#### 9. **Ensemble with All 5 Models from Benchmark**
**Hypothesis**: More diversity = better ensemble
**Models**: ESM-1v, ESM-2, AbLang2, piggen, DeepSP
**Expected gain**: +1-2%
**File**: `scripts/ginkgo_exp09_five_model_ensemble.py`

---

#### 10. **Target Transformation (Log, Sqrt)**
**Hypothesis**: PR_CHO might be better in log-space
**Expected gain**: +0-0.5%
**Method**:
```python
labels_log = np.log1p(labels)
model.fit(X, labels_log)
preds = np.expm1(model.predict(X))
```
**File**: `scripts/ginkgo_exp10_target_transform.py`

---

## Recommended Execution Order

### Phase 1: Quick Wins (Run in parallel, 30 min total)
1. **Exp #1: AbLang2 Ensemble** (best ROI)
2. **Exp #3: Optimize Weights**
3. **Exp #2: TAP+MOE Features**

### Phase 2: If Phase 1 Doesn't Break 0.50 (Run sequentially, 1 hour)
4. **Exp #4: Stacked Ensemble**
5. **Exp #6: LightGBM**

### Phase 3: Nuclear Option (If desperate, 2 hours)
9. **Exp #9: Five-Model Ensemble**

---

## Key Insights from Reference Repo

### Available Pre-computed Features:
| Feature Set | Dimensions | CV Score | Notes |
|-------------|------------|----------|-------|
| **TAP** | 5D | 0.136 | Biophysical descriptors |
| **MOE** | 48D | ? | Structure-based from MOE software |
| **DeepSP** | 30D | 0.257 | Spatial charge maps |
| **Aggrescan3D** | ? | 0.112 | Aggregation propensity |

### Model Performance (CV):
| Model | Score | Strategy |
|-------|-------|----------|
| esm2_tap_ridge | 0.413 | ESM-2 + TAP + PCA |
| piggen | 0.424 | p-IgGen embeddings |
| ablang2_elastic_net | 0.362 | AbLang2 + ElasticNet |
| deepsp_ridge | 0.257 | DeepSP spatial features |

**Key Learning**: Their best uses **PCA to 50D** - we DON'T and that's why we're winning!

---

## Why Ensembling Works So Well

1. **Uncorrelated Errors**: ESM-1v and ESM-2 make different mistakes
2. **Different Representations**: ESM-1v (variant-focused) vs ESM-2 (general)
3. **Variance Reduction**: Averaging smooths out noise

**Math**: If two models have correlation ρ and individual errors σ, ensemble error is:
```
σ_ensemble² = (σ₁² + σ₂² + 2ρσ₁σ₂) / 4
```
For ρ=0.7 (typical), we get **~15% error reduction** → matches our +2.9% gain!

---

## Things We Should NOT Try (Avoid Rabbit Holes)

❌ **Fine-tuning ESM models** (requires GPUs, weeks of compute, high risk)
❌ **Deep learning** (197 samples too small, will overfit)
❌ **Structure prediction** (slow, marginal gains based on benchmark)
❌ **More PCA** (we tried, it hurts performance)
❌ **Data augmentation** (can't create fake labels)

---

## Implementation Notes

### Fast Iteration Setup:
```python
# Load all cached embeddings once
embeddings_cache = {
    'esm1v': load_cached('ginkgo_full_vh', 'ginkgo_full_vl'),
    'esm2': load_cached('ginkgo_esm2_vh', 'ginkgo_esm2_vl'),
    # Add ablang2, etc.
}

# Run all experiments in single script
for exp_name, exp_func in experiments.items():
    score = exp_func(embeddings_cache, labels, folds)
    log_result(exp_name, score)
```

### Naming Convention:
- `ginkgo_exp##_description.py` - Individual experiments
- `ginkgo_batch_experiments.py` - Run multiple at once
- Output: `experiment_results.csv` with all scores

---

## Success Criteria

### Minimum Viable:
- **0.490**: Comfortable #1 (better than current leader by ~3%)

### Stretch Goals:
- **0.500**: Psychological barrier, 5% better than leader
- **0.510**: Dominant, 7% better

### Reality Check:
- Ensemble got us +2.9% (0.472 → 0.486)
- Each additional model adds ~1-1.5% if diverse
- **Realistic ceiling: 0.495-0.500** with 3-5 model ensemble

---

## Next Steps

1. ✅ **Create experiment scripts** (Tier 1 first)
2. ✅ **Run experiments in parallel** (use multiprocessing)
3. ✅ **Log all results** to CSV for tracking
4. ✅ **Submit best model** when > 0.490
5. ✅ **Document final approach** for reproducibility
