# Novo Nordisk Training Methodology - Official Specification

**Document Purpose:** This document extracts the exact training methodology described in Sakhnini et al. 2025 (Novo Nordisk/Cambridge) to serve as the ground truth specification for our implementation.

**Last Updated:** 2025-11-05
**Paper DOI:** 10.1016/j.cell.2024.12.025
**Supplementary Materials:** novo-media-1

---

## 1. Data Preparation (Section 4.3, Lines 395-405)

### 1.1 Dataset Parsing

The Boughter dataset was parsed into **three groups** following Boughter et al. (2020):

| Group | ELISA Flags | Used in Training? |
|-------|-------------|-------------------|
| **Specific** | 0 flags | **YES (Class 0)** |
| **Mildly poly-reactive** | 1-3 flags | **NO (excluded)** |
| **Poly-reactive** | >3 flags | **YES (Class 1)** |

**Critical Point:** The mildly poly-reactive group (1-3 flags) was **explicitly left out** from training.

### 1.2 Sequence Annotation

**Quote from paper (Lines 238-239):**
> "The primary sequences were annotated in the CDRs using ANARCI following the IMGT numbering scheme."

- Tool: ANARCI
- Numbering Scheme: IMGT
- Reference: Dunbar & Deane, Bioinformatics 32:298-300 (2016)

### 1.3 Fragment Assembly

**Quote from paper (Lines 239-241):**
> "Following this, 16 different antibody fragment sequences were assembled..."

**Table 4 from paper** defines the 16 fragment types:

| Fragment Type | Description |
|--------------|-------------|
| VL, VH | Individual variable domains |
| L-CDR1, L-CDR2, L-CDR3 | Individual light chain CDRs |
| H-CDR1, H-CDR2, H-CDR3 | Individual heavy chain CDRs |
| VH/VL joined | Concatenated variable domains |
| L-CDRs joined | Concatenated light CDRs |
| H-CDRs joined | Concatenated heavy CDRs |
| H/L-CDRs joined | All CDRs concatenated |

---

## 2. Feature Embedding (Lines 240-245)

### 2.1 Protein Language Models Tested

The following PLMs were used for embedding:

| Model | Reference | Notes |
|-------|-----------|-------|
| **ESM 1v** | Meier et al., PNAS 118:e2016239118 (2021) | **Top performer** |
| ESM 1b | Rives et al., PNAS 118:e2016239118 (2021) | |
| ESM 2 | (Reference not in paper) | |
| Protbert bfd | Elnaggar et al., IEEE TPAMI 44:7112-7127 (2022) | |
| AntiBERTy | (Reference to be checked) | Antibody-specific |
| AbLang2 | Olsen et al., bioRxiv (2024) | Antibody-specific |

### 2.2 Pooling Strategy

**Quote from paper (Line 244):**
> "For the embeddings from the PLMs, *mean* (average of all token vectors) was used."

**CRITICAL SPECIFICATION:**
- Pooling method: **Mean pooling**
- Definition: Average of all token vectors
- This is applied to the **final layer hidden states** of the model

### 2.3 Additional Encoders Tested

- Z-scale descriptor (Sandberg et al., J Med Chem 41:2481-2491 (1998))
- 68 sequence-based biophysical descriptors (see Table S1 in supplementary)

---

## 3. Model Training (Lines 241-245)

### 3.1 Classification Algorithms

**Quote from paper (Lines 242-245):**
> "The vectorised embeddings were served as features for training of binary classification models (e.g. LogisticReg, RandomForest, GaussianProcess, GradeintBoosting and SVM algorithms) for non-specificity"

**Algorithms tested:**
- **LogisticReg** (Logistic Regression) ← **Top performer**
- RandomForest
- GaussianProcess
- GradientBoosting
- SVM

**Binary Classification Labels:**
- Class 0: Specific group (0 flags)
- Class 1: Poly-reactive group (>3 flags)

---

## 4. Validation Strategy (Lines 260-266)

### 4.1 Cross-Validation Schemes

**Quote from paper (Lines 260-263):**
> "The trained classification models were validated by (i) 3, 5 and 10-Fold cross-validation (CV), (ii) Leave-One Family-Out validation..."

**Four validation approaches:**

| Method | Description |
|--------|-------------|
| **3-Fold CV** | Standard k-fold cross-validation |
| **5-Fold CV** | Standard k-fold cross-validation |
| **10-Fold CV** | Standard k-fold cross-validation (primary metric) |
| **Leave-One-Family-Out** | Train on HIV + Influenza, test on mouse IgA (and permutations) |

### 4.2 External Validation

**Quote from paper (Line 266):**
> "...and (iv) testing on the Jain dataset."

- External test set: Jain dataset (137 clinical-stage antibodies)
- Same parsing applied: 0 flags vs >3 flags (1-3 flags excluded)

---

## 5. Evaluation Metrics (Lines 266-287)

**Quote from paper (Lines 266-267):**
> "The evaluation metrics included accuracy, sensitivity and specificity (Eqs. 1-3)."

### 5.1 Definitions

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)

Sensitivity = TP / P

Specificity = TN / N
```

Where:
- TP = True Positive (correctly predicted poly-reactive)
- TN = True Negative (correctly predicted specific)
- FP = False Positive (incorrectly predicted poly-reactive)
- FN = False Negative (incorrectly predicted specific)
- P = All positives (all poly-reactive)
- N = All negatives (all specific)

### 5.2 Reported Performance

**Top model: ESM 1v mean-mode VH-based LogisticReg**
- 10-fold CV accuracy: **71%**
- Jain dataset accuracy: **69%**
- Harvey dataset: Not applicable (different assay)

---

## 6. What the Paper DOES NOT Specify

### 6.1 Critical Missing Details

The following details are **NOT explicitly stated** in the paper:

| Detail | Status | Impact |
|--------|--------|--------|
| Random seed/state | **Not specified** | Affects reproducibility |
| Stratified vs regular K-fold | **Not specified** | Could affect class balance in folds |
| LogisticReg hyperparameters | **Not specified** | Default sklearn settings assumed |
| StandardScaler usage | **Not specified** | **CRITICAL - potential data leakage** |
| Class weights/balancing | **Not specified** | May affect imbalanced classes |
| Exact scikit-learn version | Version not specified (just referenced) | API changes possible |
| How to handle "joined" sequences | **Not specified** | Concatenation with separator? Direct join? |
| BOS/EOS token handling | **Not specified** | Included in mean pooling or excluded? |

### 6.2 StandardScaler - The Elephant in the Room

**The paper does NOT mention StandardScaler anywhere.**

This is critical because:
- ESM embeddings are already in a normalized space (from transformer outputs)
- LogisticReg in sklearn **benefits from scaling** for L2 regularization
- **Correct**: Fit scaler on train folds only, transform train and test
- **Incorrect**: Fit scaler on all data before CV (data leakage)
- **Incorrect**: Fit scaler separately on each test fold (wrong distribution)

**Question for verification:** Does the current codebase have the StandardScaler bug?

---

## 7. Python Implementation Details (Section 4.2)

### 7.1 Software Stack

From **Table 3** in the paper:

| Module | Version | URL |
|--------|---------|-----|
| NumPy | Not specified | https://numpy.org |
| SciPy | Not specified | https://www.scipy.org |
| Statsmodels | Not specified | https://www.statsmodels.org |
| Pandas | Not specified | https://pandas.pydata.org |
| Matplotlib | Not specified | https://matplotlib.org |
| Seaborn | Not specified | https://seaborn.pydata.org |
| **Scikit-Learn** | Not specified | https://scikit-learn.org |
| Json | Python 3.9.2 stdlib | Standard library |
| Collections | Python 3.9.2 stdlib | Standard library |
| Itertools | Python 3.9.2 stdlib | Standard library |
| ANARCI | Not specified | https://github.com/oxpig/ANARCI |

**IDE:** Spyder IDE and Jupyter Notebook (Anaconda)
**Python Version:** 3.9.2 (inferred from stdlib references)

---

## 8. Comparison Checklist - Current Implementation

Use this checklist to verify our codebase matches Novo methodology:

### Data Preparation
- [ ] Boughter dataset parsed into 0, 1-3, >3 flag groups
- [ ] Only 0 and >3 groups used for training
- [ ] 1-3 flag group completely excluded from training
- [ ] ANARCI annotation with IMGT numbering
- [ ] 16 fragment types correctly assembled

### Feature Embedding
- [ ] ESM-1v model used (which variant? esm1v_t33_650M_UR90S_1-5?)
- [ ] Mean pooling applied to token embeddings
- [ ] Final layer hidden states used (not logits)
- [ ] BOS/EOS tokens handled correctly

### Model Training
- [ ] LogisticReg from sklearn
- [ ] Binary classification (0 vs 1)
- [ ] Hyperparameters documented (or defaults stated)
- [ ] **StandardScaler applied correctly (if at all)**
- [ ] No data leakage in preprocessing

### Validation
- [ ] 10-fold cross-validation implemented
- [ ] Stratified K-fold? (check if needed for balance)
- [ ] Random state fixed for reproducibility
- [ ] Accuracy, Sensitivity, Specificity calculated correctly

### Expected Performance
- [ ] VH-only model achieves ~71% 10-fold CV accuracy
- [ ] H-CDRs model achieves ~71% 10-fold CV accuracy
- [ ] Results reproducible across runs

---

## 9. Known Discrepancies

### 9.1 StandardScaler Bug (User Reported)

**User Statement:**
> "they had a severe bug and that codebase still has a severe bug of standard scaler in the codebase"

**Action Required:** Investigate where StandardScaler is applied in current code and verify:
1. Is it fit on train data only?
2. Is it applied inside or outside CV loop?
3. Does it cause data leakage?

### 9.2 10-Fold CV Replication Issues

**User Statement:**
> "when we haven't been able to replicate their 10cv, I realized that I didn't look at the actual model training pipeline aside from realizing that they had a bug"

**Possible causes:**
- StandardScaler placement
- Random state not fixed
- Stratified vs non-stratified K-fold
- Different sklearn version
- Different ESM model variant
- BOS/EOS token handling in mean pooling

---

## 10. Questions for Senior Review

Before proceeding with training, we need clarification on:

1. **StandardScaler:** Should we use it? If yes, where exactly in the pipeline?
2. **ESM-1v variant:** Which of the 5 ESM-1v models did Novo use? Or ensemble?
3. **Mean pooling:** Should BOS and EOS tokens be included or excluded?
4. **K-fold stratification:** Should we use stratified K-fold for class balance?
5. **LogisticReg hyperparameters:** Use sklearn defaults or tune?
6. **Random seed:** What value should we use for reproducibility?
7. **Current implementation:** Does it correctly implement the above methodology?

---

## 11. References

**Primary Paper:**
- Sakhnini, L.I., et al. (2025). Prediction of Antibody Non-Specificity using Protein Language Models and Biophysical Parameters. *Cell* (in press). DOI: 10.1016/j.cell.2024.12.025

**Supplementary Materials:**
- novo-media-1 (Supporting Information document)

**Original Boughter Paper:**
- Boughter, C.T., et al. (2020). Biochemical patterns of antibody polyreactivity revealed through a bioinformatics-based analysis of CDR loops. *eLife* 9:e61393.

**ESM Model:**
- Meier, J., et al. (2021). Language models enable zero-shot prediction of the effects of mutations on protein function. *PNAS* 118:e2016239118.

---

## Document Status

- [x] Extracted all explicit training details from Methods section 4.3
- [x] Documented missing/ambiguous details
- [x] Created comparison checklist
- [x] Identified known discrepancies
- [x] Prepared questions for senior review
- [ ] **PENDING:** Senior review and approval
- [ ] **PENDING:** Verification against current codebase
- [ ] **PENDING:** Resolution of StandardScaler bug

**Next Steps:**
1. Senior review of this document for accuracy
2. Line-by-line comparison with current training code
3. Document exact discrepancies found
4. Fix StandardScaler bug if confirmed
5. Re-run training with corrected pipeline
6. Verify 71% accuracy is achievable
