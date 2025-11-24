# Protein Language Model Zoo Research (November 2025)

**Date**: 2025-11-23
**Purpose**: Identify candidate protein language models (PLMs) and classifier heads that could outperform or complement ESM-1v for antibody non-specificity prediction using frozen backbone + simple head architecture.

## Executive Summary

### Key Findings from Deep Web Research (Nov 2025)

1. **Medium-sized models are the efficiency sweet spot** - ESM-2 650M and ESM C 600M offer 95-99% of the performance of 15B parameter models for transfer learning, making them ideal for production pipelines.

2. **Antibody-specific PLMs (IgBERT, AntiBERTy) are not guaranteed winners** - Recent studies (AbbVie 2024) show they often perform *similarly* to general PLMs for biophysical tasks like polyreactivity. However, **IgBERT** (2024) trained on 2B+ sequences represents a significant leap in data scale that remains untested against ESM-1v.

3. **Structure-Aware Models (SaProt) represent the next frontier** - SaProt explicitly encodes 3D structure tokens (Foldseek) alongside amino acids. Since polyreactivity is a surface property, this could be the "breaking" innovation to beat sequence-only models like ESM-1v.

4. **Classifier Heads Matter** - While Novo Nordisk stuck to Logistic Regression, **Linear SVMs** and **Ridge Classifiers** are theoretically superior for high-dimensional frozen embeddings and should be part of the standard benchmark.

5. **The "Gold Standard" Strategy**: To beat Novo Nordisk's 71% accuracy, we shouldn't just swap backbones. We must test:
    *   **Better Backbones**: IgBERT (Scale), AMPLIFY (Data Quality), SaProt (Structure).
    *   **Better Simple Heads**: Linear SVM (Margin maximization), Ridge (L2 regularization).

### Recommendation: The "Banger" Model Zoo

We will implement a **Tiered Model Zoo** prioritizing models with distinct inductive biases (Scale, Data Quality, Structure, Evolution).

#### Tier 1 (High Priority - The "Evo-Scale" Contenders)
1. **IgBERT** (Antibody-Specific Scale) - Trained on 2B+ OAS sequences.
2. **AMPLIFY 350M** (Data Quality) - Trained on curated data; 43x more efficient.
3. **ESM-2 650M** (Modern General) - The efficient modern standard.

#### Tier 2 (Structure & Specialized)
4. **SaProt 650M** (Structure-Aware) - Uses Foldseek tokens; potential to capture surface patches.
5. **AntiBERTy** (Proven Antibody) - Well-benchmarked baseline.

#### Tier 3 (Experimental)
6. **ESM-3** (Multimodal) - Cutting edge but complex.
7. **ProtT5-XL** (The Old Guard) - Valid baseline.

---

## 1. The Baseline: Novo Nordisk's ESM-1v

| Model | Performance | Why it Won |
|-------|-------------|------------|
| **ESM-1v** | **Best** | Masked language modeling on *evolutionary variants*. Captures "fitness" landscape better than general sequence modeling. |
| **Head** | **LogReg** | Simple, interpretable, prevents overfitting on small data (914 sequences). |

**The Challenge**: Beat **71% Accuracy** (Jain Dataset) and **~0.75-0.80 AUC**.

---

## 2. The Contenders: "Banger" Backbones

### 2.1 IgBERT (2024) - The "Big Data" Bet 🔥
*   **The Alpha**: Largest antibody-specific model ever (2B+ sequences).
*   **Why it could win**: Size matters. 2 billion antibody sequences cover the "Observed Antibody Space" far better than ESM-1v's general protein training.
*   **Hypothesis**: Will capture rare CDR patterns linked to specificity that ESM-1v misses.
*   **Frozen Backbone**: ✅ Yes (Initialized from ProtBERT).

### 2.2 AMPLIFY 350M (2024) - The "Efficiency" Bet 🔥
*   **The Alpha**: "Is Scaling Necessary?" paper argues data quality > quantity.
*   **Why it could win**: Trained on highly curated data including OAS. If Novo's dataset is noisy, AMPLIFY's clean representations might denoise the signal.
*   **Hypothesis**: Matches ESM-1v performance at 1/2 the inference cost.
*   **Frozen Backbone**: ✅ Yes.

### 2.3 SaProt (2024) - The "Structure" Bet 🧠
*   **The Alpha**: Structure-aware Protein Language Model. Uses a "structure vocabulary" (Foldseek tokens) + amino acids.
*   **Why it could win**: **Polyreactivity is a surface property** (charge patches, hydrophobicity). Sequence-only models guess structure; SaProt *knows* it (or learns it explicitly).
*   **Constraint**: Requires running Foldseek to generate structure tokens from sequence (can use predicted structures from ESMFold/AlphaFold).
*   **Verdict**: The highest "high risk, high reward" candidate.

### 2.4 ESM-2 650M - The "Modern" Bet
*   **The Alpha**: Facebook's upgrade to ESM-1v.
*   **Why it could win**: Better attention mechanisms, cleaner training data than ESM-1v.
*   **Caveat**: Novo found it worse. But did they use the 650M or the tiny/huge versions? We test the 650M sweet spot.

---

## 3. The Heads: Beyond Logistic Regression

Novo Nordisk found that complex heads (XGBoost, RF) overfitted. They stuck to Logistic Regression. But we can do better *within* the realm of simple linear models.

### 3.1 Linear SVM (Support Vector Machine)
*   **Why**: SVMs optimize the **geometric margin** between classes. In high-dimensional embedding spaces (1280d+), a max-margin separator often generalizes better than LogReg's probability fit, especially with limited data (914 samples).
*   **Config**: `kernel='linear'`, `C=1.0` (tune C).

### 3.2 Ridge Classifier (L2 Regularization)
*   **Why**: Highly robust to **multicollinearity** in embeddings. PLM dimensions are often correlated; Ridge handles this explicitly.
*   **Config**: `alpha=1.0` (tune alpha).

### 3.3 KNN (k-Nearest Neighbors)
*   **Why**: Non-parametric. If the embedding space is truly good, similar sequences (neighbors) should have similar properties. Serves as a "manifold quality" check.

---

## 4. Implementation Strategy: The "Model Zoo"

We will refactor `src/antibody_training_esm` to support swappable backbones and heads.

### 4.1 Architecture
```
src/antibody_training_esm/
├── models/
│   ├── base.py           # AbstractPLM
│   ├── esm1v.py          # Current
│   ├── igbert.py         # New
│   ├── amplify.py        # New
│   ├── saprot.py         # New (Structure-aware)
│   └── registry.py       # get_model("igbert")
└── heads/
    ├── logistic.py
    ├── svm.py            # LinearSVC
    └── ridge.py          # RidgeClassifier
```

### 4.2 Benchmarking Plan (The "Banger" Test)

Run `scripts/benchmark_zoo.py`:

1.  **Embed** Boughter (Train) + Jain (Test) with ALL backbones.
2.  **Train** ALL heads (LogReg, SVM, Ridge) on embeddings.
3.  **Evaluate** on Jain Parity Benchmark.

**Success Matrix**:

| Backbone | Head | Jain Acc | Status |
|----------|------|----------|--------|
| ESM-1v | LogReg | 71.0% | **Baseline** |
| ESM-1v | SVM | ? | **Quick Win?** |
| IgBERT | LogReg | ? | **Scale Win?** |
| SaProt | SVM | ? | **Structure Win?** |

---

## 5. Conclusion

To build the "Gold Standard" pipeline:
1.  **Keep ESM-1v** as the rock-solid baseline.
2.  **Add IgBERT** to test if 2B antibody sequences beat general evolution.
3.  **Add SaProt** to test if structure tokens unlock surface property prediction.
4.  **Add Linear SVM** to squeeze more generalization out of frozen embeddings.

This approach is scientifically rigorous, covers all inductive biases (Sequence, Evolution, Structure, Data Scale), and provides the best possible chance of beating the state-of-the-art.