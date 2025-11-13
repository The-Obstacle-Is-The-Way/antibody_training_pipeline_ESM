# Tessier→GDPa1 Transfer Learning: Route Plan (2025 Best Practices)

**Created**: 2025-11-13
**Goal**: Leverage 298k Tessier pre-training data to improve GDPa1 predictions (197 sequences)
**Current Best**: 0.507 Spearman (Ridge baseline)
**Leader**: 0.89 Spearman
**Failed Attempt**: 0.480 Spearman (broken Ridge implementation)

---

## Executive Summary

**Why the first attempt failed:**
- sklearn `Ridge.fit()` does NOT support warm start
- Pre-trained weights were completely discarded
- Effectively trained two independent models

**What 2025 research says:**
- ✅ Transfer learning WORKS for protein embeddings with small datasets
- ✅ Multi-task learning is "immensely powerful" when training data is scarce
- ✅ Mean pooling + regularized regression is standard (we already use this)
- ⚠️ Large embedding spaces (3328D) may be underutilized with N=197

**Best bet**: Multi-task learning OR neural network with proper fine-tuning

---

## Research Foundation (2025 Sources)

### Key Finding 1: Transfer Learning for High-Dimensional Linear Regression
**Source**: PLOS Computational Biology (Jan 2025)
**Method**: Trans-Lasso (transfer learning extension of Lasso)
**Result**: "Significant improvement when number of target samples is small"
**Key Quote**: "Knowledge from informative auxiliary samples can be transferred to improve learning performance"

### Key Finding 2: Multi-Task Learning Addresses Data Scarcity
**Source**: Scientific Reports (2022, still cited in 2025)
**Result**: Multi-task model achieves same performance as single-task with **1/8th the data**
**Key Quote**: "Multi-task setup becomes even more important when training set is reduced"

### Key Finding 3: Medium-Sized PLMs Are Sufficient
**Source**: Scientific Reports (2025)
**Result**: ESM-2 650M performs as well as larger models for moderately sized datasets (<10k)
**Key Quote**: "Larger embedding spaces cannot be fully taken advantage of by datasets of fewer than 10^4 observations"

### Key Finding 4: Mean Pooling Is Best for Compression
**Source**: Scientific Reports (2025)
**Result**: Mean pooling outperforms all other compression methods for protein embeddings
**Implication**: We're already doing this correctly with ESM-1v + p-IgGen

---

## Route Options (Ranked by Likelihood of Success)

### 🥇 Route A: Multi-Task Learning (RECOMMENDED)
**Estimated LOC**: 150 lines
**Estimated Time**: 3-4 hours (no re-computation needed)
**Risk**: Low (well-supported by literature)
**Expected Gain**: +3-5% Spearman (0.52-0.53)

#### Approach
Train a **single model** on BOTH datasets simultaneously:
- Tessier: 298k sequences, binary labels (polyreactive yes/no)
- GDPa1: 197 sequences, continuous labels (PR_CHO values)

#### Implementation
```python
import torch
import torch.nn as nn

class MultiTaskHead(nn.Module):
    def __init__(self, input_dim=3328):
        super().__init__()
        self.shared = nn.Linear(input_dim, 512)
        self.tessier_head = nn.Linear(512, 1)  # Binary classification
        self.gdpa1_head = nn.Linear(512, 1)    # Continuous regression

    def forward(self, x, task='gdpa1'):
        shared_repr = torch.relu(self.shared(x))
        if task == 'tessier':
            return torch.sigmoid(self.tessier_head(shared_repr))
        else:
            return self.gdpa1_head(shared_repr)

# Training loop
for epoch in range(100):
    # Sample batches from both datasets
    tessier_batch = sample(tessier_loader)
    gdpa1_batch = sample(gdpa1_loader)

    # Compute losses
    loss_tessier = bce_loss(model(tessier_batch, task='tessier'), tessier_labels)
    loss_gdpa1 = mse_loss(model(gdpa1_batch, task='gdpa1'), gdpa1_labels)

    # Weighted combination (tune α)
    total_loss = α * loss_tessier + (1-α) * loss_gdpa1
    total_loss.backward()
```

#### Why This Works
- Shared representation learns general antibody properties from Tessier
- Task-specific heads adapt to binary vs continuous labels
- Both datasets trained jointly, not sequentially
- Gradient updates propagate through shared layers

#### Hyperparameters to Tune
- `α` (task weighting): Start with 0.5, try [0.3, 0.5, 0.7]
- Hidden layer size: 512 (or try 256, 1024)
- Learning rate: 0.001 (or try 0.0001, 0.01)
- Regularization: L2 weight decay (try 0.01, 0.001)

---

### 🥈 Route B: Neural Network with Proper Fine-Tuning
**Estimated LOC**: 120 lines
**Estimated Time**: 2-3 hours
**Risk**: Medium (requires careful tuning)
**Expected Gain**: +2-4% Spearman (0.52-0.54)

#### Approach
1. Pre-train MLP on Tessier (binary classification)
2. Fine-tune on GDPa1 (continuous regression) with LOW learning rate

#### Implementation
```python
import torch.nn as nn

class TransferMLP(nn.Module):
    def __init__(self, input_dim=3328):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.head = nn.Linear(256, 1)

    def forward(self, x):
        return self.head(self.features(x))

# Stage 1: Pre-train on Tessier (binary classification)
model = TransferMLP()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train(model, tessier_data, binary_labels, epochs=50)

# Stage 2: Fine-tune on GDPa1 (continuous regression)
# Option A: Freeze features, train only head
for param in model.features.parameters():
    param.requires_grad = False
optimizer = torch.optim.Adam(model.head.parameters(), lr=0.01)
train(model, gdpa1_data, continuous_labels, epochs=50)

# Option B: Unfreeze all, low learning rate
for param in model.parameters():
    param.requires_grad = True
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # 10x lower LR
train(model, gdpa1_data, continuous_labels, epochs=100)
```

#### Why This Might Work
- Feature extractor learns antibody representations from Tessier
- Fine-tuning adapts to continuous labels with low LR
- Dropout prevents overfitting on small GDPa1 dataset

#### Pitfalls to Avoid
- ❌ High learning rate during fine-tuning (will destroy pre-trained features)
- ❌ Too many epochs on GDPa1 (overfitting on 197 samples)
- ❌ Binary loss for regression task (use MSE for GDPa1)

---

### 🥉 Route C: Trans-Lasso (Linear Transfer Learning)
**Estimated LOC**: 200 lines (custom implementation)
**Estimated Time**: 4-6 hours
**Risk**: High (no off-the-shelf sklearn implementation)
**Expected Gain**: +1-3% Spearman (0.51-0.53)

#### Approach
Implement Trans-Lasso algorithm from literature:
1. Train Lasso on Tessier
2. Use Tessier coefficients as "informative prior" for GDPa1
3. Regularize GDPa1 model toward Tessier coefficients

#### Mathematical Formulation
```
β_gdpa1 = argmin_β { ||y - Xβ||² + λ₁||β|| + λ₂||β - β_tessier||² }
              ↑ MSE loss      ↑ Lasso    ↑ Transfer penalty
```

#### Implementation Sketch
```python
from sklearn.linear_model import Lasso
import numpy as np

# Step 1: Train Lasso on Tessier
lasso_tessier = Lasso(alpha=5.5)
lasso_tessier.fit(tessier_embeddings, tessier_labels)
beta_source = lasso_tessier.coef_

# Step 2: Custom optimizer for GDPa1 with transfer penalty
def trans_lasso_objective(beta, X, y, lambda1, lambda2, beta_source):
    mse = np.mean((y - X @ beta) ** 2)
    lasso = lambda1 * np.sum(np.abs(beta))
    transfer = lambda2 * np.sum((beta - beta_source) ** 2)
    return mse + lasso + transfer

# Optimize using scipy.optimize.minimize
from scipy.optimize import minimize
beta_init = beta_source  # Warm start from Tessier
result = minimize(
    trans_lasso_objective,
    beta_init,
    args=(gdpa1_embeddings, gdpa1_labels, lambda1, lambda2, beta_source)
)
beta_target = result.x
```

#### Challenges
- No sklearn implementation exists (must write from scratch)
- Tuning λ₁ and λ₂ requires grid search
- May not handle binary→continuous label mismatch well

---

### ❌ Route D: SGDRegressor with partial_fit (NOT RECOMMENDED)
**Estimated LOC**: 80 lines
**Estimated Time**: 1-2 hours
**Risk**: Medium
**Expected Gain**: +0-2% Spearman (0.50-0.52)

#### Why NOT Recommended
- Literature search found ZERO mentions of SGDRegressor for protein transfer learning
- sklearn `partial_fit()` is for **online learning**, not transfer learning
- No guarantee that incrementally adding GDPa1 samples helps

#### Implementation (for completeness)
```python
from sklearn.linear_model import SGDRegressor

# Pre-train on Tessier
sgd = SGDRegressor(alpha=5.5, max_iter=1000, random_state=42)
sgd.fit(tessier_embeddings, tessier_labels)

# "Fine-tune" by adding GDPa1 samples incrementally
for i in range(100):  # Multiple passes over GDPa1
    sgd.partial_fit(gdpa1_embeddings, gdpa1_labels)
```

#### Problems
- `partial_fit()` expects same label distribution (binary → continuous breaks this)
- No control over learning rate decay
- Not designed for this use case

---

## Recommended Implementation Plan

### Phase 1: Multi-Task Learning (Route A)
**Timeline**: Week 1
**Deliverables**:
- `scripts/tessier_gdpa1_multitask.py` (new implementation)
- Uses existing embeddings (no re-computation needed!)
- 5-fold CV on GDPa1 with multi-task training

**Steps**:
1. ✅ Load pre-computed embeddings (already cached)
2. ✅ Implement `MultiTaskHead` PyTorch model
3. ✅ Create balanced dataloaders for Tessier + GDPa1
4. ✅ Train with weighted loss (tune α)
5. ✅ Evaluate on 5-fold CV
6. ✅ Compare to 0.507 baseline

**Success Criteria**: Spearman > 0.52 (+0.013 improvement)

### Phase 2: Neural Network Fine-Tuning (Route B)
**Timeline**: Week 2 (if Route A fails)
**Deliverables**:
- `scripts/tessier_gdpa1_finetune.py`
- Two-stage training: pre-train → fine-tune

**Steps**:
1. ✅ Implement `TransferMLP` with frozen/unfrozen modes
2. ✅ Pre-train on Tessier (binary BCE loss)
3. ✅ Fine-tune on GDPa1 (continuous MSE loss)
4. ✅ Tune learning rate schedule
5. ✅ Evaluate on 5-fold CV

**Success Criteria**: Spearman > 0.52

### Phase 3: Trans-Lasso (Route C)
**Timeline**: Week 3 (if both A and B fail)
**Only pursue if**: We have evidence that linear models work better than neural nets

---

## Implementation Checklist: Multi-Task Learning (Route A)

### Prerequisites
- [x] Tessier embeddings extracted (298k sequences)
- [x] GDPa1 embeddings extracted (197 sequences)
- [x] Both cached in `embeddings_cache/`
- [ ] PyTorch installed (`uv add torch`)
- [ ] Dataloader utility functions

### Code Structure
```
scripts/tessier_gdpa1_multitask.py
├── MultiTaskHead (PyTorch model)
├── MultiTaskDataset (combines Tessier + GDPa1)
├── train_multitask() (training loop)
├── evaluate_cv() (5-fold cross-validation)
└── main() (orchestration)
```

### Hyperparameter Grid Search
```python
HYPERPARAMS = {
    'alpha': [0.3, 0.5, 0.7],       # Task weighting
    'hidden_dim': [256, 512, 1024], # Hidden layer size
    'lr': [0.0001, 0.001, 0.01],   # Learning rate
    'weight_decay': [0.001, 0.01], # L2 regularization
    'dropout': [0.1, 0.2, 0.3],    # Dropout rate
}
```

### Expected Runtime
- Single run (one hyperparameter set): ~5 minutes
- Full grid search (3×3×3×2×3 = 162 runs): ~13 hours
- **Recommendation**: Start with defaults (α=0.5, hidden=512, lr=0.001, wd=0.01, dropout=0.2)

---

## Risk Analysis

### High-Confidence Predictions
✅ Multi-task learning WILL help (strong literature support)
✅ PyTorch fine-tuning WILL work (standard approach)
✅ Our embeddings are good (ESM-1v + p-IgGen is state-of-art)

### Uncertainties
⚠️ How much improvement? (Research shows 1-5%, depends on dataset similarity)
⚠️ Optimal α (task weighting)? (Need to tune empirically)
⚠️ Binary→continuous transfer difficulty? (Task mismatch may limit gains)

### Failure Modes
❌ Tessier data is too different from GDPa1 (polyreactivity ≠ CHO binding)
❌ N=197 is too small to benefit from transfer (unlikely per literature)
❌ Overfitting on GDPa1 during fine-tuning (use strong regularization)

---

## Comparison to Original Ridge Implementation

### What Was Broken
```python
# ❌ BROKEN: Two independent models
ridge_pretrain = Ridge(alpha=5.5)
ridge_pretrain.fit(tessier_embeddings, tessier_labels)  # ← Learns weights

ridge_finetune = Ridge(alpha=5.5)  # ← NEW Ridge, starts from ZERO!
ridge_finetune.fit(gdpa1_embeddings, gdpa1_labels)  # ← IGNORES pre-trained weights!
```

### What Will Work (Multi-Task)
```python
# ✅ CORRECT: Single model, joint training
model = MultiTaskHead(input_dim=3328)
for epoch in range(100):
    # Both datasets update the SAME shared weights
    loss_tessier = train_on_tessier(model)
    loss_gdpa1 = train_on_gdpa1(model)
    total_loss = α * loss_tessier + (1-α) * loss_gdpa1
    total_loss.backward()  # ← Gradients flow through shared layers!
```

### Key Difference
- **Ridge**: No weight initialization from pre-training
- **Multi-Task**: Shared weights learn from BOTH datasets simultaneously
- **Fine-Tuning**: Explicit weight transfer via model initialization

---

## Success Metrics

### Minimum Viable Success
- Spearman > 0.507 (beat baseline)
- Transfer > Baseline in at least 3/5 CV folds

### Target Success
- Spearman > 0.52 (+2.5% improvement)
- Consistent across all 5 CV folds

### Stretch Goal
- Spearman > 0.55 (+8% improvement)
- Narrow gap to leader (0.89 - 0.55 = 0.34)

---

## Lessons Learned from Failed Attempt

### What We Validated ✅
- Tessier preprocessing pipeline works (373k → 298k train sequences)
- Embedding extraction scales to 300k sequences with MPS
- ANARCI annotation handles large-scale data
- 80/20 train/val split is stratified correctly

### What We Learned ❌
- sklearn Ridge does NOT support warm start
- Binary→continuous label mismatch requires careful handling
- Sequential training (pre-train → fine-tune) needs custom code

### What to Try Next ✅
- Multi-task learning (joint training, not sequential)
- Neural networks with proper initialization
- Stronger regularization to prevent overfitting on N=197

---

## References

1. **Transfer Learning for High-Dimensional Linear Regression**
   PLOS Computational Biology (2025)
   https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012739

2. **Medium-sized Protein Language Models for Transfer Learning**
   Scientific Reports (2025)
   https://www.nature.com/articles/s41598-025-05674-x

3. **Multi-task Learning for PPI Prediction**
   Scientific Reports (2022)
   https://www.nature.com/articles/s41598-022-13951-2

4. **Multi-modal Representation Learning in Low-Data Settings**
   arXiv (2024)
   https://arxiv.org/html/2412.08649v1

---

## Next Actions

### Immediate (Today)
- [ ] Install PyTorch: `uv add torch`
- [ ] Create `scripts/tessier_gdpa1_multitask.py`
- [ ] Implement `MultiTaskHead` class
- [ ] Test on single fold (sanity check)

### This Week
- [ ] Full 5-fold CV with multi-task learning
- [ ] Hyperparameter tuning (α, lr, hidden_dim)
- [ ] Compare to 0.507 baseline
- [ ] Submit to leaderboard if > 0.52

### Backup Plan
- [ ] If multi-task fails, try Route B (fine-tuning)
- [ ] If both fail, document why and move to Track 2/3 from roadmap

---

**Status**: 🟢 READY TO IMPLEMENT
**Confidence**: 🔥🔥🔥 HIGH (strong literature support)
**Expected Timeline**: 3-5 days to first results
**Go/No-Go Decision**: Implement Route A (Multi-Task Learning)
